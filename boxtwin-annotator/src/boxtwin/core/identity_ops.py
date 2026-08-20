"""
BoxTwin - Operaciones de identidad, reversibles.

POR QUE EXISTE
  El problema real del dominio no es la caja mal ubicada, es el intercambio de identidad.
  Corregirlo son reescrituras estructurales: partir un intervalo, truncar otro, insertar
  dos nuevos. Calcular la inversa exacta de cada una a mano es donde se cuelan los errores,
  y un error aca no se ve: el esqueleto sigue estando sobre un cuerpo, solo que se exporta
  con el peleador equivocado.

  Por eso estas operaciones deshacen con instantanea del bloque de identidad y no con una
  inversa calculada. Es lo contrario de lo que hacen los comandos de evento, y la razon es
  el tamano: los eventos con sus metricas son varios megas y copiarlos cincuenta veces
  costaria cientos, mientras que las asignaciones son unas decenas de registros y una copia
  no se nota. Donde la copia es barata, conviene la opcion que no se puede equivocar.

QUE HACE
  Asignar un rol a un track sobre un rango, corregir un intercambio desde un cuadro,
  re-sembrar sobre una caja dibujada a mano, marcar tramos no confiables y aceptar la union
  de dos tracks.

USO
  pila.do(SwapFromFrame(4120, annotator="lucas"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from boxtwin.core.annotations import new_id
from boxtwin.core.identity import rol_de_track
from boxtwin.core.interpolation import GapCandidate
from boxtwin.core.schema import (
    AnnotationDoc,
    Assignment,
    Identity,
    Interpolation,
    ManualBox,
    ManualTrack,
    Origin,
    UnreliableSegment,
)
from boxtwin.core.types import (
    AssignmentOp,
    FighterId,
    InterpolationMethod,
    TrackRole,
    UnreliableReason,
    UnreliableSource,
)

__all__ = [
    "AssignRole",
    "SwapFromFrame",
    "Reseed",
    "MarkUnreliable",
    "AcceptJoin",
    "FillInternalGaps",
    "IgnoreSmallTracks",
    "boundary_after",
]

# Operaciones cuyo inicio corta el alcance de un intercambio: son decisiones explicitas del
# anotador y un swap posterior no puede pisarlas.
OPS_MANUALES = (AssignmentOp.MANUAL, AssignmentOp.SWAP, AssignmentOp.RESEED)


def boundary_after(doc: AnnotationDoc, frame: int) -> int:
    """
    Hasta donde llega una correccion que arranca en `frame`.

    Es el inicio de la proxima decision manual posterior, o el fin del video. Sin este
    limite, un intercambio corregido en el minuto dos pisaria la correccion que ya se habia
    hecho en el minuto cinco.
    """
    posteriores = [
        a.start_frame
        for a in doc.identity.assignments
        if a.start_frame > frame and a.origin.op in OPS_MANUALES
    ]
    return min(posteriores) if posteriores else doc.video.total_frames


class _SnapshotCommand:
    """
    Base de las operaciones de identidad: guarda el bloque entero antes de tocarlo.

    Ver el razonamiento del modulo. La instantanea se toma al ejecutar, no al construir, asi
    que rehacer despues de deshacer parte siempre del estado correcto.
    """

    label: str = "identidad"
    _antes: Identity | None = None

    def _aplicar(self, doc: AnnotationDoc) -> None:  # pragma: no cover - abstracto
        raise NotImplementedError

    def do(self, doc: AnnotationDoc) -> None:
        self._antes = doc.identity.model_copy(deep=True)
        self._aplicar(doc)

    def undo(self, doc: AnnotationDoc) -> None:
        if self._antes is None:
            raise RuntimeError("no se puede deshacer una operacion que nunca se ejecuto")
        doc.identity = self._antes.model_copy(deep=True)


def _recortar(
    asignaciones: list[Assignment], track_id: int, desde: int, hasta: int
) -> list[Assignment]:
    """
    Saca el tramo [desde, hasta) de los intervalos de un track, partiendo lo que haga falta.

    Los intervalos de un mismo track no se pueden solapar: el resolver busca uno solo y con
    dos superpuestos el resultado dependeria del orden de la lista.
    """
    salida: list[Assignment] = []
    for a in asignaciones:
        if a.track_id != track_id or a.end_frame_excl <= desde or a.start_frame >= hasta:
            salida.append(a)
            continue
        if a.start_frame < desde:
            salida.append(a.model_copy(update={"end_frame_excl": desde}))
        if a.end_frame_excl > hasta:
            salida.append(a.model_copy(update={"start_frame": hasta}))
        # El tramo que queda dentro se descarta: es lo que se esta reemplazando.
    return salida


def _origen(doc: AnnotationDoc, op: AssignmentOp, frame: int, annotator: str, **extra) -> Origin:
    return Origin(
        op=op,
        op_id=new_id(doc, "op"),
        at_frame=frame,
        annotator=annotator,
        created_at=datetime.now().astimezone(),
        **extra,
    )


# ---------------------------------------------------------------------------


@dataclass
class AssignRole(_SnapshotCommand):
    """
    Asigna un rol a un track sobre un rango, reemplazando lo que hubiera de ESE track.

    NO le quita el rol a los demas tracks, y se probo lo contrario. Parecia obvio que un
    peleador es uno solo y que asignar un track nuevo deberia truncar al anterior; sobre el
    round anotado de Pacquiao vs Margarito eso elimina las 1165 colisiones de rol, pero los
    cuadros sin pose dentro de eventos suben de 51 a 218 y los eventos afectados de 6 a 34.

    La razon es que los solapamientos hacen de respaldo. El tracker fragmenta a un peleador
    en varios tracks dentro del mismo plano, y cuando uno vuelve a aparecer la asignacion
    vieja lo sigue cubriendo. Truncarla deja esos cuadros sin nadie.

    El ruido de validacion que esto genera se resuelve del otro lado, en validate_document,
    reportando solo las colisiones donde los dos tracks tienen detecciones a la vez.
    """

    track_id: int
    role: TrackRole
    start_frame: int
    end_frame_excl: int
    annotator: str = "desconocido"
    label: str = ""
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"asignar track {self.track_id} a {self.role.value}"

    def _aplicar(self, doc: AnnotationDoc) -> None:
        origen = _origen(doc, AssignmentOp.MANUAL, self.start_frame, self.annotator)
        asignaciones = _recortar(
            doc.identity.assignments, self.track_id, self.start_frame, self.end_frame_excl
        )

        asignaciones.append(
            Assignment(
                id=new_id(doc, "assignment"),
                track_id=self.track_id,
                role=self.role,
                start_frame=self.start_frame,
                end_frame_excl=self.end_frame_excl,
                origin=origen,
            )
        )
        doc.identity.assignments = asignaciones


@dataclass
class SwapFromFrame(_SnapshotCommand):
    """
    Corrige un intercambio de identidad a partir de un cuadro.

    Los dos assignments nuevos comparten op_id: en un diff se lee como un solo gesto y no
    como dos cambios sueltos que hay que correlacionar a ojo.
    """

    frame: int
    annotator: str = "desconocido"
    label: str = ""
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"intercambiar identidad desde {self.frame}"

    def _aplicar(self, doc: AnnotationDoc) -> None:
        hasta = boundary_after(doc, self.frame)
        vigentes: dict[TrackRole, int] = {}
        for a in doc.identity.assignments:
            if a.covers(self.frame) and a.role in (TrackRole.A, TrackRole.B):
                vigentes[a.role] = a.track_id
        if len(vigentes) != 2:
            raise ValueError(
                f"en el cuadro {self.frame} no hay un track por peleador "
                f"(hay {len(vigentes)}); no se puede intercambiar lo que no esta asignado"
            )

        origen = _origen(doc, AssignmentOp.SWAP, self.frame, self.annotator)
        asignaciones = list(doc.identity.assignments)
        for track in vigentes.values():
            asignaciones = _recortar(asignaciones, track, self.frame, hasta)

        for rol, track in vigentes.items():
            nuevo = TrackRole.B if rol is TrackRole.A else TrackRole.A
            asignaciones.append(
                Assignment(
                    id=new_id(doc, "assignment"),
                    track_id=track,
                    role=nuevo,
                    start_frame=self.frame,
                    end_frame_excl=hasta,
                    origin=origen,
                )
            )
        doc.identity.assignments = asignaciones


@dataclass
class Reseed(_SnapshotCommand):
    """
    Re-siembra un peleador sobre una caja dibujada a mano.

    Dos modos y la diferencia importa. Si la caja se superpone con un track existente, se le
    asigna el rol a ese track: es el caso comun, porque el tracker no perdio al peleador
    sino que le cambio el id. Si no se superpone con nada, se crea un track manual sin
    keypoints, y esos cuadros se marcan solos como no confiables porque no hay pose que
    exportar.
    """

    frame: int
    role: TrackRole
    bbox: list[float]
    track_id: int | None = None  # si ya se resolvio afuera
    iou: float | None = None
    annotator: str = "desconocido"
    label: str = ""
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"re-sembrar {self.role.value} en {self.frame}"

    def _aplicar(self, doc: AnnotationDoc) -> None:
        hasta = boundary_after(doc, self.frame)
        origen = _origen(
            doc, AssignmentOp.RESEED, self.frame, self.annotator,
            seed_box_xyxy=list(self.bbox), seed_iou=self.iou,
        )

        if self.track_id is not None:
            asignaciones = _recortar(
                doc.identity.assignments, self.track_id, self.frame, hasta
            )
            asignaciones.append(
                Assignment(
                    id=new_id(doc, "assignment"),
                    track_id=self.track_id,
                    role=self.role,
                    start_frame=self.frame,
                    end_frame_excl=hasta,
                    origin=origen,
                )
            )
            doc.identity.assignments = asignaciones
            return

        # Sin track debajo: caja manual, negativa para no colisionar con las de BoT-SORT.
        negativos = [m.track_id for m in doc.identity.manual_tracks]
        nuevo_id = (min(negativos) - 1) if negativos else -1
        doc.identity.manual_tracks = [
            *doc.identity.manual_tracks,
            ManualTrack(
                track_id=nuevo_id,
                role=self.role,
                boxes=[ManualBox(frame=self.frame, xyxy=list(self.bbox))],
                notes="re-seed sin track debajo",
                annotator=self.annotator,
                created_at=origen.created_at,
            ),
        ]
        if self.role in (TrackRole.A, TrackRole.B):
            # Sin keypoints no hay nada que exportar: se declara y no se descubre despues.
            doc.unreliable_segments = [
                *doc.unreliable_segments,
                UnreliableSegment(
                    id=new_id(doc, "unreliable"),
                    fighter=FighterId(self.role.value),
                    start_frame=self.frame,
                    end_frame_excl=self.frame + 1,
                    reason=UnreliableReason.POSE_UNRELIABLE,
                    notes="caja manual sin keypoints",
                    source=UnreliableSource.AUTO_MANUAL_TRACK,
                    annotator=self.annotator,
                    created_at=origen.created_at,
                ),
            ]


@dataclass
class MarkUnreliable(_SnapshotCommand):
    """
    Marca un tramo como no confiable. No borra la pose: la excluye del export por defecto.

    Se guarda la decision y no su consecuencia. Que un tramo entre o no al dataset es
    politica del export, y esa politica puede cambiar sin volver a mirar el video.
    """

    fighter: FighterId
    start_frame: int
    end_frame_excl: int
    reason: UnreliableReason = UnreliableReason.OCCLUDED
    notes: str = ""
    annotator: str = "desconocido"
    label: str = ""
    _antes_segmentos: list[UnreliableSegment] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"marcar {self.reason.value} en {self.fighter.value}"

    def do(self, doc: AnnotationDoc) -> None:
        self._antes_segmentos = [s.model_copy(deep=True) for s in doc.unreliable_segments]
        doc.unreliable_segments = [
            *doc.unreliable_segments,
            UnreliableSegment(
                id=new_id(doc, "unreliable"),
                fighter=self.fighter,
                start_frame=self.start_frame,
                end_frame_excl=self.end_frame_excl,
                reason=self.reason,
                notes=self.notes,
                source=UnreliableSource.MANUAL,
                annotator=self.annotator,
                created_at=datetime.now().astimezone(),
            ),
        ]

    def undo(self, doc: AnnotationDoc) -> None:
        if self._antes_segmentos is None:
            raise RuntimeError("no se puede deshacer una operacion que nunca se ejecuto")
        doc.unreliable_segments = list(self._antes_segmentos)


@dataclass
class AcceptJoin(_SnapshotCommand):
    """
    Acepta que dos tracks son la misma persona.

    Con hueco se registra la interpolacion; sin hueco solo se extiende el rol, porque no
    hay ningun cuadro que inventar y declarar una interpolacion vacia seria decir que se
    sintetizo algo que no se sintetizo.
    """

    candidato: GapCandidate
    role: TrackRole
    annotator: str = "desconocido"
    label: str = ""
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = (
                f"unir tracks {self.candidato.from_track_id} y {self.candidato.to_track_id}"
            )

    def _aplicar(self, doc: AnnotationDoc) -> None:
        cand = self.candidato
        hasta = boundary_after(doc, cand.first_frame)
        origen = _origen(
            doc, AssignmentOp.INTERPOLATION, cand.first_frame, self.annotator,
            seed_iou=cand.iou,
        )
        asignaciones = _recortar(doc.identity.assignments, cand.to_track_id, cand.first_frame, hasta)
        asignaciones.append(
            Assignment(
                id=new_id(doc, "assignment"),
                track_id=cand.to_track_id,
                role=self.role,
                start_frame=cand.first_frame,
                end_frame_excl=hasta,
                origin=origen,
            )
        )
        doc.identity.assignments = asignaciones

        if cand.gap_len > 0:
            doc.identity.interpolations = [
                *doc.identity.interpolations,
                Interpolation(
                    id=new_id(doc, "interpolation"),
                    role=self.role,
                    from_track_id=cand.from_track_id,
                    to_track_id=cand.to_track_id,
                    gap_start_frame=cand.gap_start,
                    gap_end_frame_excl=cand.gap_end_excl,
                    gap_len=cand.gap_len,
                    method=InterpolationMethod.LINEAR,
                    iou_at_join=cand.iou,
                    accepted_by=self.annotator,
                    created_at=origen.created_at,
                ),
            ]


@dataclass
class FillInternalGaps(_SnapshotCommand):
    """
    Rellena de una vez los huecos internos de los tracks que ya son un peleador.

    Va en lote y las uniones no, y la diferencia no es de comodidad. Unir dos tracks afirma
    que dos ids son la misma persona, y equivocarse ahi mete keypoints del peleador
    equivocado en el dataset sin que se vea en el overlay. Un hueco interno no afirma nada:
    el id es el mismo a los dos lados, ya lo dijo el tracker, y lo unico que se agrega son
    los cuadros del medio. El riesgo que justifica confirmar de a uno no existe aca, y sobre
    material real son 253 huecos, o sea 253 confirmaciones que no deciden nada.

    Solo toca tracks con rol de peleador: interpolar un track ignorado seria inventar pose
    para alguien que se decidio dejar afuera. No duplica interpolaciones ya declaradas, asi
    que correrlo dos veces es inofensivo.
    """

    candidatos: list[GapCandidate]
    annotator: str = "desconocido"
    label: str = ""
    aplicados: int = 0
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"rellenar {len(self.candidatos)} huecos internos"

    def _aplicar(self, doc: AnnotationDoc) -> None:
        ya = {(i.to_track_id, i.gap_start_frame) for i in doc.identity.interpolations}
        ahora = datetime.now().astimezone()
        nuevas: list[Interpolation] = []

        for c in self.candidatos:
            if not c.es_mismo_track:
                raise ValueError(
                    f"el candidato {c.from_track_id}->{c.to_track_id} une tracks distintos; "
                    "eso es una afirmacion de identidad y va por AcceptJoin"
                )
            if c.gap_len <= 0 or (c.to_track_id, c.gap_start) in ya:
                continue
            rol = rol_de_track(doc, c.from_track_id, c.last_frame)
            if rol not in (TrackRole.A, TrackRole.B):
                continue
            nuevas.append(
                Interpolation(
                    id=new_id(doc, "interpolation"),
                    role=rol,
                    from_track_id=c.from_track_id,
                    to_track_id=c.to_track_id,
                    gap_start_frame=c.gap_start,
                    gap_end_frame_excl=c.gap_end_excl,
                    gap_len=c.gap_len,
                    method=InterpolationMethod.LINEAR,
                    iou_at_join=c.iou,
                    accepted_by=self.annotator,
                    created_at=ahora,
                )
            )
            ya.add((c.to_track_id, c.gap_start))

        self.aplicados = len(nuevas)
        doc.identity.interpolations = [*doc.identity.interpolations, *nuevas]


@dataclass
class IgnoreSmallTracks(_SnapshotCommand):
    """
    Marca como `ignore` a los tracks que nunca superan cierto alto. El publico, basicamente.

    En metraje de transmision el detector encuentra a todo el mundo. Medido sobre 20 s de una
    pelea profesional a 1080p: 9,4 personas por cuadro y 108 tracks distintos, de los cuales
    dos son los boxeadores. Asignarlos de a uno no es trabajo, es imposible.

    Va en lote por la misma razon que el relleno de huecos: no decide nada sobre quien es
    quien, solo declara que un track no es ninguno de los dos peleadores. Y el umbral es una
    decision del anotador que queda registrada como asignaciones, no un filtro horneado en el
    cache: el cache guarda lo que el modelo observo, y si el umbral resulta mal elegido se
    deshace con Ctrl+Z en vez de reprocesar horas de video.

    NUNCA pisa un track que ya tiene rol de peleador en algun lado. Un boxeador puede quedar
    chico durante un plano abierto, y una operacion en lote que le borre el rol por eso es
    justo el tipo de error que este sistema no puede permitirse, porque no se ve.
    """

    track_ids: list[int]
    total_frames: int          # fin del rango, exclusivo
    start_frame: int = 0       # inicio del rango: 0 para todo el video, o el corte de plano
    umbral: float = 0.0
    annotator: str = "desconocido"
    label: str = ""
    aplicados: int = 0
    _antes: Identity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"ignorar {len(self.track_ids)} tracks chicos"

    def _aplicar(self, doc: AnnotationDoc) -> None:
        peleadores = {TrackRole.A, TrackRole.B}
        con_rol = {
            a.track_id for a in doc.identity.assignments if a.role in peleadores
        }
        ya_ignorados = {
            a.track_id
            for a in doc.identity.assignments
            if a.role is TrackRole.IGNORE
            and a.start_frame <= self.start_frame
            and a.end_frame_excl >= self.total_frames
        }
        origen = _origen(doc, AssignmentOp.MANUAL, self.start_frame, self.annotator)

        nuevos: list[Assignment] = []
        for tid in sorted(set(self.track_ids)):
            if tid in con_rol or tid in ya_ignorados:
                continue
            nuevos.append(
                Assignment(
                    id=new_id(doc, "assignment"),
                    track_id=tid,
                    role=TrackRole.IGNORE,
                    start_frame=self.start_frame,
                    end_frame_excl=self.total_frames,
                    origin=origen,
                )
            )
        self.aplicados = len(nuevos)
        doc.identity.assignments = [*doc.identity.assignments, *nuevos]
