"""
BoxTwin - Deteccion de huecos de track e interpolacion de keypoints.

POR QUE EXISTE
  BoT-SORT pierde el track en cada oclusion y lo recupera con un id nuevo. En boxeo eso
  pasa en cada clinch, varias veces por round, y cada id nuevo es trabajo manual de
  re-asignacion. La mayoria de esos cortes son huecos de dos o tres cuadros donde el
  peleador no se fue a ningun lado: la caja de antes y la de despues se superponen casi
  por completo.

  Se propone, no se aplica solo. Unir dos tracks es afirmar que son la misma persona, y en
  un clinch esa afirmacion puede ser falsa justamente cuando mas se parecen las cajas. El
  costo de equivocarse es un tramo de keypoints atribuido al peleador equivocado, que no se
  ve al mirar el overlay porque el esqueleto sigue estando sobre un cuerpo.

  La interpolacion es lineal y los cuadros que produce quedan marcados. No se pretende que
  sean observacion: en tres cuadros de un golpe rapido la muneca recorre bastante y una
  recta no es la trayectoria. Sirven para que el tramo no tenga agujeros, y el export
  decide si los usa.

  Hay dos clases de hueco y confundirlas fue el error de la primera version. Un hueco ENTRE
  DOS TRACKS DISTINTOS es una afirmacion de identidad y necesita confirmacion humana. Un
  hueco INTERNO de un mismo track no lo es: el id es el mismo a los dos lados, el tracker ya
  afirmo que es la misma persona y no hay a quien atribuirle mal los keypoints. Por eso los
  internos se pueden rellenar en lote y los otros no.

  Medido sobre Sparring.mp4 (5531 cuadros, 42 tracks, 2 peleadores), que es lo que motivo
  esta reescritura:
    - 325 huecos internos, mediana de 1 cuadro. 253 en tracks con rol de peleador, 767
      cuadros. La cobertura de "los dos peleadores presentes" sube de 80% a ~94%.
    - Uniones entre tracks distintos, deduplicadas por par: 3 candidatos con IoU >= 0,5 y
      hueco de hasta 60 en todo el video, los tres correctos. Con el default viejo de
      max_gap=5 eran 0, o sea el detector no proponia nada y las 42 reasignaciones se
      hicieron a mano.
  O sea: el volumen esta en los huecos internos, no en las uniones.

QUE HACE
  Encuentra huecos internos de cada track y pares de tracks candidatos a ser el mismo,
  ordenados por confianza, e interpola linealmente las cajas y los keypoints del hueco.

USO
  from boxtwin.core.interpolation import detectar_huecos, detectar_huecos_internos
  internos = detectar_huecos_internos(cache)
  uniones = detectar_huecos(cache, max_gap=60, min_iou=0.5, roles=roles_por_track(doc))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from boxtwin.core.posecache import PoseCache
from boxtwin.core.types import TrackRole

__all__ = [
    "GapCandidate",
    "RolEnCuadro",
    "iou",
    "detectar_huecos",
    "detectar_huecos_internos",
    "interpolar_tramo",
    "tramos",
    "GAP_INTERNO_MAX",
    "GAP_INTERNO_MIN_IOU",
]

# "De quien es el track T en el cuadro F". Se recibe como funcion y no como documento para que
# este modulo siga sin saber nada del esquema de anotacion.
RolEnCuadro = Callable[[int, int], TrackRole | None]

# Tope de los huecos internos que se proponen para relleno automatico.
#
# 20 y no mas alto porque la interpolacion es lineal y a partir de ahi deja de ser una
# aproximacion razonable: en 20 cuadros (0,7 s a 30 fps) una muneca recorre un golpe entero y
# una recta no es esa trayectoria. Sobre Sparring.mp4 el corte en 20 toma 249 de los 253
# huecos y 618 de los 767 cuadros; los 4 que quedan afuera suman 149 cuadros y miden entre 21
# y 53, donde inventar la pose seria afirmar mas de lo que se sabe. Esos se marcan a mano
# como tramo no confiable, que es lo que son.
GAP_INTERNO_MAX = 20

# Umbral bajo a proposito. Aca el IoU no decide identidad, eso ya lo decidio el tracker al
# reusar el id: es solo un chequeo de cordura contra un salto absurdo. Exigir 0,5 como en las
# uniones descartaria huecos legitimos de un peleador que se desplazo durante la oclusion.
GAP_INTERNO_MIN_IOU = 0.2


@dataclass(frozen=True)
class GapCandidate:
    """
    Dos tracks que podrian ser la misma persona.

    gap_len 0 significa que no falta ningun cuadro y solo cambio el id: ahi no hay nada que
    interpolar, alcanza con extender la asignacion de rol. Es una operacion distinta y por
    eso se distingue.
    """

    from_track_id: int
    to_track_id: int
    last_frame: int  # ultimo cuadro del track que se corta
    first_frame: int  # primer cuadro del que aparece
    iou: float

    @property
    def gap_start(self) -> int:
        return self.last_frame + 1

    @property
    def gap_end_excl(self) -> int:
        return self.first_frame

    @property
    def gap_len(self) -> int:
        return self.first_frame - self.last_frame - 1

    @property
    def es_continuacion(self) -> bool:
        return self.gap_len == 0

    @property
    def es_mismo_track(self) -> bool:
        """Hueco interno: no afirma nada sobre identidad, solo rellena cuadros que faltan."""
        return self.from_track_id == self.to_track_id


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Interseccion sobre union de dos cajas xyxy."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extremos(cache: PoseCache) -> dict[int, tuple[int, int]]:
    """Primer y ultimo cuadro de cada track."""
    salida: dict[int, tuple[int, int]] = {}
    for tid in cache.track_ids().tolist():
        frames = cache.frames_of_track(int(tid))
        if frames.size:
            salida[int(tid)] = (int(frames.min()), int(frames.max()))
    return salida


def tramos(cache: PoseCache) -> dict[int, list[tuple[int, int]]]:
    """
    Tramos contiguos de detecciones de cada track, inclusivos a los dos lados.

    Un track no es un intervalo. Con track_buffer alto BoT-SORT mantiene vivo un id a traves
    de oclusiones cortas, asi que un mismo track_id aparece, desaparece y vuelve. Mirar solo
    el primer y ultimo cuadro (lo que hace `_extremos`) da un intervalo que finge continuidad:
    sobre Sparring.mp4, 39 de 42 tracks estan partidos y los 42 tracks son en realidad 367
    tramos. Los huecos entre esos tramos son lo que este modulo rellena.
    """
    salida: dict[int, list[tuple[int, int]]] = {}
    for tid in cache.track_ids().tolist():
        frames = np.sort(cache.frames_of_track(int(tid)))
        if not frames.size:
            continue
        cortes = np.flatnonzero(np.diff(frames) > 1)
        bordes = [0, *(cortes + 1).tolist(), int(frames.size)]
        salida[int(tid)] = [
            (int(frames[bordes[i]]), int(frames[bordes[i + 1] - 1]))
            for i in range(len(bordes) - 1)
        ]
    return salida


def _caja(cache: PoseCache, frame: int, track_id: int) -> np.ndarray | None:
    dets = cache.detections(frame)
    i = dets.index_of_track(track_id)
    return dets.bbox[i] if i is not None else None


def detectar_huecos_internos(
    cache: PoseCache,
    *,
    max_gap: int = GAP_INTERNO_MAX,
    min_iou: float = GAP_INTERNO_MIN_IOU,
) -> list[GapCandidate]:
    """
    Huecos de un mismo track: el id es igual a los dos lados y falta la deteccion del medio.

    Estos NO son una afirmacion de identidad y por eso se pueden aplicar en lote, a
    diferencia de las uniones entre tracks distintos. Salen en orden cronologico porque se
    revisan y se aplican de corrido, no de a uno eligiendo el mejor.
    """
    candidatos: list[GapCandidate] = []
    for tid, segmentos in tramos(cache).items():
        for (_, fin), (ini, _) in zip(segmentos, segmentos[1:]):
            hueco = ini - fin - 1
            if not 1 <= hueco <= max_gap:
                continue
            caja_a, caja_b = _caja(cache, fin, tid), _caja(cache, ini, tid)
            if caja_a is None or caja_b is None:
                continue
            solape = iou(caja_a, caja_b)
            if solape < min_iou:
                continue
            candidatos.append(
                GapCandidate(
                    from_track_id=tid, to_track_id=tid,
                    last_frame=fin, first_frame=ini, iou=round(solape, 4),
                )
            )

    candidatos.sort(key=lambda c: (c.last_frame, c.from_track_id))
    return candidatos


def _roles_incompatibles(
    rol_en: RolEnCuadro | None, a: int, fin_a: int, b: int, ini_b: int
) -> bool:
    """
    True si los dos tracks ya son peleadores distintos EN EL CUADRO DONDE SE UNIRIAN.

    Es preventivo, no correctivo. En un clinch las cajas de los dos boxeadores se superponen
    y la geometria no distingue una union buena de una que fusionaria a las dos personas; lo
    unico que las distingue es que el anotador ya dijo de quien es cada track. Sobre
    Sparring.mp4 con los umbrales de trabajo no descarta nada, porque las propuestas que
    sobreviven a IoU >= 0,5 ya son correctas; bajando a IoU >= 0 con hueco 120 descarta 3.
    Sirve como red para material donde el tracker se equivoque mas, no para arreglar un caso
    observado.

    Se consulta en `fin_a` y en `ini_b`, no en abstracto: un track puede cambiar de peleador
    a lo largo del video cuando un swap corrige un intercambio del tracker, y preguntarle "de
    quien es este track" sin decir cuando devuelve los dos roles y no filtra nada. El track 3
    de Sparring.mp4 es fighter_B hasta el cuadro 614 y fighter_A despues.
    """
    if rol_en is None:
        return False
    peleadores = (TrackRole.A, TrackRole.B)
    ra, rb = rol_en(a, fin_a), rol_en(b, ini_b)
    return ra in peleadores and rb in peleadores and ra is not rb


def detectar_huecos(
    cache: PoseCache,
    *,
    max_gap: int = 60,
    min_iou: float = 0.5,
    rol_en: RolEnCuadro | None = None,
) -> list[GapCandidate]:
    """
    Pares de tracks DISTINTOS separados por a lo sumo `max_gap` cuadros y con cajas compatibles.

    Se devuelven ordenados por IoU descendente: el mas parecido primero, que es el que el
    anotador va a querer confirmar sin pensar. Los dudosos quedan al final, que es donde
    corresponde mirarlos.

    `rol_en(track_id, frame)` responde de quien es un track en un cuadro; con eso se descartan
    las propuestas que unirian a los dos peleadores. Ver `_roles_incompatibles`.

    Para los huecos internos de un mismo track usar `detectar_huecos_internos`: son otra cosa
    y se aplican distinto.
    """
    extremos = _extremos(cache)
    candidatos: list[GapCandidate] = []

    for a, (_, fin_a) in extremos.items():
        caja_a = _caja(cache, fin_a, a)
        if caja_a is None:
            continue
        for b, (ini_b, _) in extremos.items():
            if b == a:
                continue
            hueco = ini_b - fin_a - 1
            if not 0 <= hueco <= max_gap:
                continue
            if _roles_incompatibles(rol_en, a, fin_a, b, ini_b):
                continue
            caja_b = _caja(cache, ini_b, b)
            if caja_b is None:
                continue
            solape = iou(caja_a, caja_b)
            if solape < min_iou:
                continue
            candidatos.append(
                GapCandidate(
                    from_track_id=a, to_track_id=b,
                    last_frame=fin_a, first_frame=ini_b, iou=round(solape, 4),
                )
            )

    candidatos.sort(key=lambda c: (-c.iou, c.last_frame, c.from_track_id))
    return candidatos


def interpolar_tramo(
    cache: PoseCache, cand: GapCandidate
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Cuadros sinteticos del hueco: (frame, bbox, keypoints, kp_score).

    Interpolacion lineal entre el ultimo cuadro del track que se corta y el primero del que
    aparece. El score se toma como el minimo de los dos extremos: un punto inventado no
    puede tener mas confianza que los datos con que se invento.
    """
    if cand.gap_len <= 0:
        return []

    dets_a = cache.detections(cand.last_frame)
    dets_b = cache.detections(cand.first_frame)
    ia = dets_a.index_of_track(cand.from_track_id)
    ib = dets_b.index_of_track(cand.to_track_id)
    if ia is None or ib is None:
        raise ValueError("los tracks del candidato no estan en los cuadros que declara")

    bbox_a, bbox_b = dets_a.bbox[ia], dets_b.bbox[ib]
    kp_a, kp_b = dets_a.keypoints[ia], dets_b.keypoints[ib]
    sc = np.minimum(dets_a.kp_score[ia], dets_b.kp_score[ib])

    pasos = cand.first_frame - cand.last_frame
    salida = []
    for f in range(cand.gap_start, cand.gap_end_excl):
        t = (f - cand.last_frame) / pasos
        salida.append(
            (
                f,
                ((1 - t) * bbox_a + t * bbox_b).astype(np.float32),
                ((1 - t) * kp_a + t * kp_b).astype(np.float32),
                sc.astype(np.float32),
            )
        )
    return salida
