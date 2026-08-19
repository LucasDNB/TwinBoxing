#!/usr/bin/env python3
"""
BoxTwin - Compara corridas del tracker contra la anotacion de identidad.

POR QUE EXISTE
  configs/botsort.yaml elige sus valores con argumentos razonables pero sin medirlos, y lo
  dice: "Lo que NO esta medido: si el ReID efectivamente reduce los intercambios de
  identidad. Eso necesita ground truth de identidad, que es justamente lo que el anotador va
  a producir." Este script es la otra mitad de esa frase.

  Sirve para cualquier cambio del tracker, no solo el ReID: track_buffer, match_thresh,
  gmc_method. La receta es la misma, cambiar una linea del yaml y volver a correrlo.

  Como usarlo sin romper nada: el preproceso escribe en cache/ del proyecto que infiere del
  video, y si el video es un symlink lo sigue hasta el original. Se corre sobre una COPIA
  del video en un proyecto aparte y con --project explicito, o se pisa el cache del que
  cuelga la anotacion, y con el los track_id de todas las asignaciones.

QUE HACE
  El ground truth son las cajas de cada peleador cuadro a cuadro, que salen de las
  asignaciones manuales sobre el cache anotado. Para cada corrida se busca, en cada cuadro,
  que track se superpone con esa caja, y se mide cuantas veces cambia el id, cuantos ids
  distintos hacen falta y cuantas veces el mismo track se le atribuye a los dos peleadores.

  No se comparan ids entre corridas: cambiando el tracker los ids son otros. Lo que se
  compara es cuanto le cuesta a cada corrida seguir a la MISMA persona, definida por la
  anotacion.

USO
  python tools/comparar_tracker.py proyecto_anotado/ dir_con_las_corridas/
  donde el segundo argumento tiene subdirectorios <nombre>/cache/<video>.pose.npz
"""
import sys
from pathlib import Path

import numpy as np

from boxtwin.core.annotations import load
from boxtwin.core.identity import IdentityResolver
from boxtwin.core.interpolation import iou
from boxtwin.core.posecache import PoseCache
from boxtwin.core.types import FighterId

P = Path(sys.argv[1])
doc, _ = load(P / "annotations/Sparring.annot.json")
original = PoseCache.open(P / "cache/Sparring.pose.npz")
res = IdentityResolver(doc, original)
T = doc.video.total_frames

# Ground truth: solo detecciones observadas, sin las interpoladas.
gt: dict[FighterId, dict[int, np.ndarray]] = {FighterId.A: {}, FighterId.B: {}}
for f in range(T):
    for pel, pose in res.by_fighter(f).items():
        if pose is not None and not pose.interpolated:
            gt[pel][f] = np.asarray(pose.bbox, dtype=np.float64)

print(f"ground truth: {len(gt[FighterId.A])} cuadros de A, {len(gt[FighterId.B])} de B\n")


def evaluar(cache: PoseCache, nombre: str) -> None:
    asignado: dict[FighterId, dict[int, int]] = {FighterId.A: {}, FighterId.B: {}}
    for pel, cajas in gt.items():
        for f, caja in cajas.items():
            dets = cache.detections(f)
            mejor, mejor_iou = None, 0.5  # umbral MOT habitual
            for i in range(len(dets)):
                v = iou(caja, np.asarray(dets.bbox[i], dtype=np.float64))
                if v > mejor_iou:
                    mejor, mejor_iou = int(dets.track_id[i]), v
            if mejor is not None:
                asignado[pel][f] = mejor

    print(f"=== {nombre} ===")
    print(f"  tracks totales: {len(cache.track_ids())}   detecciones: {cache.n_detections}")
    total_sw = 0
    for pel in (FighterId.A, FighterId.B):
        a = asignado[pel]
        frames = sorted(a)
        ids = {a[f] for f in frames}
        sw = sum(1 for x, y in zip(frames, frames[1:]) if a[x] != a[y])
        total_sw += sw
        cob = 100 * len(frames) / max(1, len(gt[pel]))
        print(f"  {pel.value}: {len(ids):>3} ids distintos | {sw:>3} cambios de id | "
              f"cobertura {cob:.1f}%")
    # Confusion: el mismo track atribuido a los dos peleadores en el mismo cuadro.
    conf = sum(
        1 for f in range(T)
        if f in asignado[FighterId.A] and f in asignado[FighterId.B]
        and asignado[FighterId.A][f] == asignado[FighterId.B][f]
    )
    print(f"  cambios de id totales: {total_sw}   cuadros con los dos peleadores en un track: {conf}\n")


S = Path(sys.argv[2])
evaluar(original, "cache original de la anotacion")
evaluar(PoseCache.open(S / "con_reid/cache/Sparring.pose.npz"), "CON ReID (config actual)")
evaluar(PoseCache.open(S / "sin_reid/cache/Sparring.pose.npz"), "SIN ReID")
