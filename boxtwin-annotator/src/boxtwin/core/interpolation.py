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

QUE HACE
  Encuentra pares de tracks candidatos a ser el mismo, ordenados por confianza, e interpola
  linealmente las cajas y los keypoints del hueco.

USO
  from boxtwin.core.interpolation import detectar_huecos, interpolar_tramo
  candidatos = detectar_huecos(cache, max_gap=5, min_iou=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxtwin.core.posecache import PoseCache

__all__ = ["GapCandidate", "iou", "detectar_huecos", "interpolar_tramo"]


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


def _caja(cache: PoseCache, frame: int, track_id: int) -> np.ndarray | None:
    dets = cache.detections(frame)
    i = dets.index_of_track(track_id)
    return dets.bbox[i] if i is not None else None


def detectar_huecos(
    cache: PoseCache, *, max_gap: int = 5, min_iou: float = 0.5
) -> list[GapCandidate]:
    """
    Pares de tracks separados por a lo sumo `max_gap` cuadros y con cajas compatibles.

    Se devuelven ordenados por IoU descendente: el mas parecido primero, que es el que el
    anotador va a querer confirmar sin pensar. Los dudosos quedan al final, que es donde
    corresponde mirarlos.
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
