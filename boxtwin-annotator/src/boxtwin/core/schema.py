"""
BoxTwin - Modelo de datos del archivo de anotacion.

POR QUE EXISTE
  El annot.json es la fuente de verdad del dataset propio y tiene que sobrevivir a meses
  de anotacion, a cambios de esquema y a revision en git. Eso impone tres cosas que un
  dict suelto no da: que un campo mal escrito falle al abrir en vez de perderse en
  silencio, que el orden de las claves y el redondeo de los floats sean estables para que
  los diffs sirvan, y que las invariantes estructurales se verifiquen en un solo lugar.
  La separacion contra validation.py es deliberada: aca van las invariantes duras, las
  que hacen que el archivo no tenga sentido si se violan. Todo lo que sea advertencia va
  alla, porque un archivo con solapamientos raros tiene que poder abrirse igual.

QUE HACE
  Define los modelos pydantic del documento completo: video, referencia al cache de pose,
  peleadores con su guardia, asignaciones de identidad por rango, tramos no confiables,
  eventos, correcciones de combinacion, metricas de proceso y el snapshot de parametros
  que afectan el significado de lo anotado.
  El orden de declaracion de los campos es el orden canonico en el archivo.

USO
  from boxtwin.core.schema import AnnotationDoc, Event, SCHEMA_VERSION
  doc = AnnotationDoc.model_validate(raw_dict)
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

from boxtwin.core.types import (
    ArmRole,
    AssignmentOp,
    ComboOp,
    Completeness,
    FighterId,
    FpsSource,
    GloveSource,
    Guard,
    InterpolationMethod,
    KeypointFormat,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
    UnreliableReason,
    UnreliableSource,
    arm_role,
)

__all__ = [
    "SCHEMA_VERSION",
    "DOC_KIND",
    "BoxTwinModel",
    "Generator",
    "Counters",
    "VideoInfo",
    "PoseRef",
    "GuardOverride",
    "FighterDef",
    "Origin",
    "Assignment",
    "ManualBox",
    "ManualTrack",
    "Interpolation",
    "Identity",
    "UnreliableSegment",
    "Event",
    "ComboOverride",
    "AnnotatorInfo",
    "SessionMetrics",
    "Totals",
    "EventMetrics",
    "Process",
    "ProxySettings",
    "SettingsSnapshot",
    "AnnotationDoc",
]

SCHEMA_VERSION = 1
DOC_KIND = "boxtwin.annot"


# ---------------------------------------------------------------------------
# Tipos base y helpers de validacion
# ---------------------------------------------------------------------------


def _round2(v: float) -> float:
    return round(v, 2)


def _round4(v: float) -> float:
    return round(v, 4)


def _round3(v: float) -> float:
    return round(v, 3)


def _check_bbox_order(v: list[float]) -> list[float]:
    x1, y1, x2, y2 = v
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"caja invalida, se esperaba x1<x2 y y1<y2: {v}")
    return v


def _require_half_open(start: int, end_excl: int) -> None:
    """Rango semiabierto [start, end_excl). Un rango vacio o invertido no tiene sentido."""
    if end_excl <= start:
        raise ValueError(f"rango invalido, se esperaba start_frame < end_frame_excl: [{start}, {end_excl})")


# El redondeo se aplica al validar y no al serializar, asi el objeto en memoria y el
# archivo coinciden siempre y un round-trip no mueve ningun digito.
Coord = Annotated[float, AfterValidator(_round2)]
Unit = Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_round4)]
Seconds = Annotated[float, Field(gt=0.0), AfterValidator(_round3)]
Frame = Annotated[int, Field(ge=0)]
Millis = Annotated[int, Field(ge=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

BBox = Annotated[
    list[Coord],
    Field(min_length=4, max_length=4),
    AfterValidator(_check_bbox_order),
]

# timespec="seconds" mantiene los timestamps legibles y con ancho fijo. La precision fina
# la lleva active_ms, que es lo que se mide de verdad.
#
# Los microsegundos se truncan AL VALIDAR y no solo al serializar, por el mismo motivo que
# el redondeo de coordenadas: si el objeto en memoria tuviera mas precision que el archivo,
# cargar lo que se acaba de guardar daria un documento distinto del que se tenia, y las
# comparaciones de round-trip fallarian por una diferencia que nunca llego al disco.
Timestamp = Annotated[
    AwareDatetime,
    AfterValidator(lambda v: v.replace(microsecond=0)),
    PlainSerializer(lambda v: v.isoformat(timespec="seconds"), return_type=str, when_used="json"),
]


class BoxTwinModel(BaseModel):
    """
    Base de todos los modelos del esquema.

    extra="forbid" es intencional: un campo desconocido significa archivo de otra version
    y tiene que fallar para que lo agarre el migrador. Si se colara en silencio, un
    archivo viejo se abriria perdiendo datos sin avisar.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------


class Generator(BoxTwinModel):
    app: str = "boxtwin-annotator"
    app_version: str
    created_at: Timestamp
    updated_at: Timestamp


class Counters(BoxTwinModel):
    """
    Contadores de IDs ya emitidos.

    Existen para que un ID nunca se reuse. Derivar el proximo del maximo existente falla
    al borrar el ultimo evento: el siguiente tomaria el mismo ID y un reanno.json que
    referencie ev_0043 pasaria a apuntar a otro evento.
    """

    event: int = Field(default=0, ge=0)
    assignment: int = Field(default=0, ge=0)
    unreliable: int = Field(default=0, ge=0)
    interpolation: int = Field(default=0, ge=0)
    combo: int = Field(default=0, ge=0)
    session: int = Field(default=0, ge=0)
    op: int = Field(default=0, ge=0)


class VideoInfo(BoxTwinModel):
    """
    Identidad y metadatos reales del video.

    fps y total_frames son los medidos decodificando. Los campos _declared guardan lo que
    dijo el contenedor, que puede ser None: los nueve videos de BoxingVI no declaran
    nb_frames. Guardar los dos permite auditar despues cuanto minten los contenedores sin
    volver a abrir los videos.
    size_bytes y mtime son un prechequeo barato para no rehashear al abrir la GUI.
    """

    path: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    mtime: Timestamp
    fps: float = Field(gt=0.0)
    fps_declared: float | None = Field(default=None, gt=0.0)
    fps_source: FpsSource
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    total_frames: int = Field(gt=0)
    total_frames_declared: int | None = Field(default=None, ge=0)
    duration_s: Seconds
    codec: str


class PoseRef(BoxTwinModel):
    """
    Referencia al cache de pose, que es inmutable y que la anotacion nunca modifica.

    meta_sha256 detecta que el video se reproceso con otra configuracion y que las
    anotaciones quedaron colgando de una pose distinta de la que se uso para hacerlas.
    """

    npz_path: str
    meta_path: str
    meta_sha256: Sha256
    keypoint_format: KeypointFormat = KeypointFormat.COCO17
    keypoint_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Peleadores
# ---------------------------------------------------------------------------


class GuardOverride(BoxTwinModel):
    """Tramo en que el peleador cambia de guardia. El evento hereda la vigente en su inicio."""

    start_frame: Frame
    end_frame_excl: Frame
    guard: Guard
    notes: str = ""

    @model_validator(mode="after")
    def _check_range(self) -> GuardOverride:
        _require_half_open(self.start_frame, self.end_frame_excl)
        return self


class FighterDef(BoxTwinModel):
    label: str = ""
    guard: Guard
    guard_overrides: list[GuardOverride] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Identidad
# ---------------------------------------------------------------------------


class Origin(BoxTwinModel):
    """
    De que operacion nacio un assignment.

    op_id es compartido por todos los assignments que salieron del mismo gesto del
    anotador, asi un swap se lee en el diff como una sola cosa y no como dos cambios
    sueltos que hay que correlacionar a ojo.
    """

    op: AssignmentOp
    op_id: str
    at_frame: Frame
    annotator: str
    created_at: Timestamp
    seed_box_xyxy: BBox | None = None
    seed_iou: Unit | None = None


class Assignment(BoxTwinModel):
    """Mapeo track_id -> rol, valido sobre [start_frame, end_frame_excl)."""

    id: str
    track_id: int
    role: TrackRole
    start_frame: Frame
    end_frame_excl: Frame
    origin: Origin

    @model_validator(mode="after")
    def _check_range(self) -> Assignment:
        _require_half_open(self.start_frame, self.end_frame_excl)
        return self

    def covers(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame_excl


class ManualBox(BoxTwinModel):
    frame: Frame
    xyxy: BBox


class ManualTrack(BoxTwinModel):
    """
    Cajas dibujadas a mano donde no hay ningun track al que engancharse.

    No traen keypoints, asi que los frames que cubren se marcan pose_unreliable y no
    entran a los exports de pose. Sirven para dejar registrado que el peleador estaba ahi
    aunque el tracker lo haya perdido.
    El track_id es negativo por convencion y no colisiona con los que emite BoT-SORT.
    """

    track_id: int = Field(lt=0)
    role: TrackRole
    boxes: list[ManualBox] = Field(min_length=1)
    interpolate_between: bool = True
    notes: str = ""
    annotator: str
    created_at: Timestamp

    @property
    def start_frame(self) -> int:
        return min(b.frame for b in self.boxes)

    @property
    def end_frame_excl(self) -> int:
        return max(b.frame for b in self.boxes) + 1


class Interpolation(BoxTwinModel):
    """Union de dos tracks a traves de un hueco corto, aceptada explicitamente por el anotador."""

    id: str
    role: TrackRole
    from_track_id: int
    to_track_id: int
    gap_start_frame: Frame
    gap_end_frame_excl: Frame
    gap_len: int = Field(gt=0)
    method: InterpolationMethod = InterpolationMethod.LINEAR
    iou_at_join: Unit
    accepted_by: str
    created_at: Timestamp

    @model_validator(mode="after")
    def _check_range(self) -> Interpolation:
        _require_half_open(self.gap_start_frame, self.gap_end_frame_excl)
        expected = self.gap_end_frame_excl - self.gap_start_frame
        if self.gap_len != expected:
            raise ValueError(f"gap_len={self.gap_len} no coincide con el rango, se esperaba {expected}")
        return self


class Identity(BoxTwinModel):
    assignments: list[Assignment] = Field(default_factory=list)
    manual_tracks: list[ManualTrack] = Field(default_factory=list)
    interpolations: list[Interpolation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tramos no confiables y eventos
# ---------------------------------------------------------------------------


class UnreliableSegment(BoxTwinModel):
    """Se excluye del export por defecto pero queda registrado. Nunca borra pose."""

    id: str
    fighter: FighterId
    start_frame: Frame
    end_frame_excl: Frame
    reason: UnreliableReason
    notes: str = ""
    source: UnreliableSource = UnreliableSource.MANUAL
    annotator: str
    created_at: Timestamp

    @model_validator(mode="after")
    def _check_range(self) -> UnreliableSegment:
        _require_half_open(self.start_frame, self.end_frame_excl)
        return self

    def overlaps_inclusive(self, start_frame: int, end_frame: int) -> bool:
        """Cruce contra un rango inclusivo, que es como se guardan los eventos."""
        return self.start_frame <= end_frame and start_frame < self.end_frame_excl


class Event(BoxTwinModel):
    """
    Un evento de golpe.

    start_frame y end_frame son INCLUSIVOS, a diferencia de los rangos de identidad, que
    son semiabiertos y se llaman end_frame_excl. La diferencia de nombre es la que evita
    confundirlos leyendo el JSON.

    Fronteras, definicion operacional que la GUI muestra en panel fijo:
      start_frame  primer cuadro en que el puno inicia el desplazamiento hacia el
                   objetivo, con el codo empezando a extenderse o el hombro rotando.
                   No el cuadro en que se carga el peso.
      peak_frame   maxima extension del brazo o contacto, lo que ocurra primero.
      end_frame    cuadro en que el puno retrocedio aproximadamente la mitad del camino
                   de vuelta a la guardia.
    """

    id: str
    fighter: FighterId
    start_frame: Frame
    peak_frame: Frame | None = None
    end_frame: Frame
    side: Side
    punch_type: PunchType
    target: Target
    completeness: Completeness
    landed: Landed = Landed.UNKNOWN
    guard: Guard
    quality: Quality = Quality.CLEAN
    notes: str = ""

    @model_validator(mode="after")
    def _check_frames(self) -> Event:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"evento invalido, se esperaba start_frame < end_frame: [{self.start_frame}, {self.end_frame}]"
            )
        if self.peak_frame is not None and not (self.start_frame <= self.peak_frame <= self.end_frame):
            raise ValueError(
                f"peak_frame={self.peak_frame} fuera de [{self.start_frame}, {self.end_frame}]"
            )
        return self

    @property
    def duration_frames(self) -> int:
        """Los limites son inclusivos, por eso el +1."""
        return self.end_frame - self.start_frame + 1

    @property
    def arm_role(self) -> ArmRole:
        """Derivado, nunca se serializa. Ver types.arm_role."""
        return arm_role(self.guard, self.side)

    def overlaps(self, other: Event) -> bool:
        return self.start_frame <= other.end_frame and other.start_frame <= self.end_frame

    def overlap_frames(self, other: Event) -> int:
        return max(0, min(self.end_frame, other.end_frame) - max(self.start_frame, other.start_frame) + 1)


class ComboOverride(BoxTwinModel):
    """
    Correccion manual a la deteccion automatica de combinaciones.

    Las combinaciones se derivan de los tiempos, no se anotan: derivarlas cuesta cero por
    evento y se recalculan si se mueve el umbral. Esto es solo para los casos en que la
    heuristica se equivoca.
    """

    id: str
    op: ComboOp
    event_ids: list[str] = Field(min_length=2)
    reason: str = ""
    annotator: str
    created_at: Timestamp


# ---------------------------------------------------------------------------
# Metricas de proceso
# ---------------------------------------------------------------------------


class AnnotatorInfo(BoxTwinModel):
    id: str
    name: str = ""


class SessionMetrics(BoxTwinModel):
    id: str
    annotator: str
    started_at: Timestamp
    ended_at: Timestamp | None = None
    active_ms: Millis = 0
    events_created: int = Field(default=0, ge=0)
    events_edited: int = Field(default=0, ge=0)
    app_version: str


class Totals(BoxTwinModel):
    active_ms: Millis = 0
    events: int = Field(default=0, ge=0)
    median_ms_per_event: int | None = Field(default=None, ge=0)


class EventMetrics(BoxTwinModel):
    """
    Metricas por evento. Alimentan el analisis metodologico, no son opcionales.

    active_ms cuenta solo tiempo con la app en foco y sin idle. replays es cuantas vueltas
    del loop de preview hubo antes de confirmar, que es el proxy directo de dificultad de
    la decision.
    """

    annotator: str
    session_id: str
    created_at: Timestamp
    confirmed_at: Timestamp | None = None
    active_ms: Millis = 0
    replays: int = Field(default=0, ge=0)
    edits: int = Field(default=0, ge=0)
    last_edited_at: Timestamp | None = None


class Process(BoxTwinModel):
    """
    Bloque aparte y no inline en cada evento a proposito.

    active_ms cambia cada vez que se toca un evento. Inline, todo diff de etiquetas
    vendria contaminado con ruido de cronometro y dejaria de servir para auditar
    decisiones de anotacion.
    """

    annotators: list[AnnotatorInfo] = Field(default_factory=list)
    sessions: list[SessionMetrics] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    event_metrics: dict[str, EventMetrics] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuracion que afecta el significado de lo anotado
# ---------------------------------------------------------------------------


class ProxySettings(BoxTwinModel):
    """
    Proxy de baja resolucion para el reproductor.

    Medido sobre los videos del proyecto: decodificar 4K a un hilo da 41 fps, contra 919
    del proxy a 960 px. Sin proxy, retroceder cuadro a cuadro sobre 4K es inviable.
    GOP 12 cuesta 42% mas de tamano y baja el peor caso de seek de 249 cuadros a 11.
    """

    width: int = Field(default=960, gt=0)
    gop: int = Field(default=12, gt=0)
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "veryfast"


class SettingsSnapshot(BoxTwinModel):
    """
    Parametros que afectan el significado de lo anotado, congelados en el archivo.

    Si manana cambia el umbral de score o el criterio de frontera, los eventos viejos se
    anotaron bajo otro criterio y hay que poder saberlo sin adivinar por fecha.
    """

    # 60 y no 5. Con 5 el detector de uniones no propone nada: medido sobre Sparring.mp4
    # (5531 cuadros, 42 tracks) da 0 candidatos con hueco <= 5 y 3 correctos con hueco <= 60,
    # porque BoT-SORT con track_buffer 120 recupera al peleador bastante despues de perderlo.
    # Evidencia de un solo video, asi que es un default, no una constante del dominio.
    interp_max_gap_frames: int = Field(default=60, gt=0)
    interp_min_iou: Unit = 0.5
    kp_score_threshold: Unit = 0.3
    event_max_duration_frames: int = Field(default=40, gt=0)
    boundary_definitions_version: int = Field(default=1, ge=1)

    # Combinaciones: un golpe puede empezar antes de que termine el anterior.
    same_side_overlap_max_frames: int = Field(default=3, ge=0)
    refire_min_gap_frames: int = Field(default=4, ge=0)
    combo_gap_max_frames: int = Field(default=12, ge=0)
    background_margin_frames: int = Field(default=5, ge=0)

    # Guantes en los indices 17 y 18.
    glove_source: GloveSource = GloveSource.DERIVED
    glove_extrapolation_k: float = Field(default=0.35, ge=0.0)

    proxy: ProxySettings = Field(default_factory=ProxySettings)
    player_buffer_mb: int = Field(default=512, gt=0)


# ---------------------------------------------------------------------------
# Documento
# ---------------------------------------------------------------------------


class AnnotationDoc(BoxTwinModel):
    """
    Fuente de verdad de la anotacion de un video.

    El orden de declaracion de estos campos es el orden en que salen al archivo. No se usa
    sort_keys: alfabetico separaria start_frame de end_frame y el archivo dejaria de
    leerse solo.
    """

    schema_version: int = Field(ge=1)
    kind: Literal["boxtwin.annot"] = DOC_KIND
    generator: Generator
    counters: Counters = Field(default_factory=Counters)
    video: VideoInfo
    pose: PoseRef
    fighters: dict[FighterId, FighterDef]
    identity: Identity = Field(default_factory=Identity)
    unreliable_segments: list[UnreliableSegment] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    combo_overrides: list[ComboOverride] = Field(default_factory=list)
    process: Process = Field(default_factory=Process)
    settings_snapshot: SettingsSnapshot = Field(default_factory=SettingsSnapshot)

    @model_validator(mode="after")
    def _check_doc(self) -> AnnotationDoc:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={self.schema_version} distinto del actual {SCHEMA_VERSION}; "
                "el documento tiene que pasar por migrations.migrate antes de parsearse"
            )
        faltan = {FighterId.A, FighterId.B} - set(self.fighters)
        if faltan:
            raise ValueError(f"faltan peleadores en el documento: {sorted(f.value for f in faltan)}")
        return self

    # -- accesos derivados -------------------------------------------------

    def guard_at(self, fighter: FighterId, frame: int) -> Guard:
        """Guardia vigente del peleador en un frame. Los overrides ganan sobre el default."""
        fd = self.fighters[fighter]
        for ov in fd.guard_overrides:
            if ov.start_frame <= frame < ov.end_frame_excl:
                return ov.guard
        return fd.guard

    def events_of(self, fighter: FighterId) -> list[Event]:
        return [e for e in self.events if e.fighter is fighter]

    def event_by_id(self, event_id: str) -> Event | None:
        for e in self.events:
            if e.id == event_id:
                return e
        return None

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def new_document(
    *,
    app_version: str,
    now: datetime,
    video: VideoInfo,
    pose: PoseRef,
    guard_a: Guard,
    guard_b: Guard,
    label_a: str = "",
    label_b: str = "",
) -> AnnotationDoc:
    """Documento vacio y valido para un video recien preprocesado."""
    return AnnotationDoc(
        schema_version=SCHEMA_VERSION,
        generator=Generator(app_version=app_version, created_at=now, updated_at=now),
        video=video,
        pose=pose,
        fighters={
            FighterId.A: FighterDef(label=label_a, guard=guard_a),
            FighterId.B: FighterDef(label=label_b, guard=guard_b),
        },
    )
