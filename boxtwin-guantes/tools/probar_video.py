#!/usr/bin/env python3
"""
BoxTwin - Prueba el detector de guantes sobre un video anotado.

POR QUE EXISTE
  El mAP que imprime el entrenamiento se midio sobre fotografia de producto: la validacion
  del dataset publico no tiene una sola imagen de ring, porque sus 853 cuadros de video son
  un solo clip y el reparto por procedencia los manda enteros a train. Ese numero no dice
  nada sobre rendimiento en ringside y hace falta verlo sobre material propio.

  No vuelve a correr pose. El cache del proyecto ya tiene la caja y el track de cada persona
  en cada cuadro, que es exactamente lo que el detector necesita de entrada, asi que
  reprocesar seria pagar dos veces por lo mismo. Ademas el cache es la unica fuente de
  track_id consistente con las asignaciones de identidad.

QUE HACE
  Recorta cada persona del cache, le pasa el detector de guantes, y produce dos cosas: un
  video con el resultado dibujado, para mirar, y la fraccion de cuadros con guante detectado
  POR TRACK, que es la medida que decide si esto sirve como filtro de identidad. Si el
  detector sirve, los tracks de peleador deberian separarse de los intrusos -arbitro,
  entrenador, cronometrista- por esa fraccion.

  Si el proyecto tiene anotacion, marca que tracks son peleador segun las asignaciones
  manuales, asi la separacion se lee de una en vez de tener que cruzarla a mano.

USO
  python tools/probar_video.py ~/Proyectos/TwinBoxing/anotacion-spar-01 \
      --modelo modelos/guantes.pt --hasta 1500 --out /tmp/prueba.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _resolver(proyecto: Path) -> tuple[Path, Path, Path | None]:
    """Devuelve (video, npz, annot) a partir del directorio del proyecto."""
    videos = sorted((proyecto / "videos").glob("*.mp4")) if (proyecto / "videos").is_dir() else []
    if not videos:
        videos = sorted(proyecto.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"error: no encontre ningun mp4 en {proyecto}")
    video = videos[0]
    npz = proyecto / "cache" / f"{video.stem}.pose.npz"
    if not npz.exists():
        raise SystemExit(f"error: falta el cache de pose {npz}")
    annot = proyecto / "annotations" / f"{video.stem}.annot.json"
    return video, npz, (annot if annot.exists() else None)


def _peleadores(annot: Path | None) -> dict[int, str]:
    """track_id -> rol, segun las asignaciones manuales. Vacio si no hay anotacion."""
    if annot is None:
        return {}
    from boxtwin.core.annotations import load

    doc, _ = load(annot)
    roles: dict[int, str] = {}
    for a in doc.identity.assignments:
        # Un track puede cambiar de rol; para este reporte alcanza con el ultimo que tuvo.
        roles[a.track_id] = a.role.value
    return roles


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("proyecto", type=Path, help="directorio del proyecto anotado")
    p.add_argument("--modelo", default="modelos/guantes.pt", help="pesos del detector")
    p.add_argument("--desde", type=int, default=0)
    p.add_argument("--hasta", type=int, default=None, help="cuadro final, exclusivo")
    p.add_argument("--paso", type=int, default=1, help="procesar uno cada N cuadros")
    p.add_argument("--conf", type=float, default=0.25, help="umbral del detector")
    p.add_argument("--margen", type=float, default=0.08, help="igual que en los recortes")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--out", default=None, help="mp4 de salida; sin esto no escribe video")
    p.add_argument("--min-detecciones", type=int, default=30, dest="min_det",
                   help="tracks con menos detecciones que esto no entran al reporte")
    args = p.parse_args()

    from boxtwin.core.posecache import PoseCache
    from ultralytics import YOLO

    video, npz, annot = _resolver(args.proyecto)
    cache = PoseCache.open(npz)
    roles = _peleadores(annot)
    modelo = YOLO(args.modelo)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"error: no se pudo abrir {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fin = min(args.hasta or len(cache), len(cache))

    escritor = None
    if args.out:
        escritor = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, args.paso), (ancho, alto)
        )

    vistos = defaultdict(int)     # track -> cuadros en que aparecio
    con_guante = defaultdict(int)  # track -> cuadros con al menos un guante
    guantes_tot = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.desde)
    for f in range(args.desde, fin):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - args.desde) % args.paso:
            continue
        dets = cache.detections(f)
        for i in range(len(dets)):
            tid = int(dets.track_id[i])
            x0, y0, x1, y1 = (float(v) for v in dets.bbox[i])
            ph, pw = y1 - y0, x1 - x0
            if ph < 16 or pw < 8:
                continue
            m = args.margen
            cx0 = max(0, int(x0 - m * pw)); cy0 = max(0, int(y0 - m * ph))
            cx1 = min(ancho, int(x1 + m * pw)); cy1 = min(alto, int(y1 + m * ph))
            recorte = frame[cy0:cy1, cx0:cx1]
            if recorte.size == 0:
                continue

            vistos[tid] += 1
            r = modelo.predict(recorte, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            if n:
                con_guante[tid] += 1
                guantes_tot += n

            if escritor is not None:
                # La persona en gris, el guante en verde: se lee de un vistazo quien tiene.
                color = (0, 220, 0) if n else (140, 140, 140)
                cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), color, 2)
                etiqueta = f"{tid}"
                if roles.get(tid) in ("fighter_A", "fighter_B"):
                    etiqueta += f" {roles[tid][-1]}"
                cv2.putText(frame, etiqueta, (cx0, max(12, cy0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if n:
                    for gx0, gy0, gx1, gy1 in r.boxes.xyxy.cpu().numpy():
                        cv2.rectangle(frame, (cx0 + int(gx0), cy0 + int(gy0)),
                                      (cx0 + int(gx1), cy0 + int(gy1)), (0, 0, 255), 2)
        if escritor is not None:
            escritor.write(frame)
        if (f - args.desde) % 300 == 0:
            print(f"  cuadro {f}/{fin}", file=sys.stderr)

    cap.release()
    if escritor is not None:
        escritor.release()

    # -- reporte ------------------------------------------------------------
    filas = [
        (tid, vistos[tid], con_guante[tid], con_guante[tid] / vistos[tid])
        for tid in vistos if vistos[tid] >= args.min_det
    ]
    filas.sort(key=lambda r: -r[3])

    print(f"\n{video.name}  cuadros {args.desde}-{fin} paso {args.paso}")
    print(f"{guantes_tot} guantes detectados sobre {sum(vistos.values())} recortes\n")
    print(f"{'track':>7} {'rol':>10} {'recortes':>9} {'con guante':>11} {'fraccion':>9}")
    for tid, v, c, frac in filas:
        rol = roles.get(tid, "-")
        rol = {"fighter_A": "A", "fighter_B": "B", "ignore": "ignore"}.get(rol, rol)
        print(f"{tid:>7} {rol:>10} {v:>9} {c:>11} {frac:>9.3f}")

    if roles:
        pel = [f for f in filas if roles.get(f[0]) in ("fighter_A", "fighter_B")]
        otros = [f for f in filas if roles.get(f[0]) not in ("fighter_A", "fighter_B")]
        if pel and otros:
            fp = np.array([f[3] for f in pel]); fo = np.array([f[3] for f in otros])
            print(f"\nseparacion, que es lo que decide si esto sirve como filtro:")
            print(f"  peleadores ({len(pel):>3} tracks): fraccion mediana {np.median(fp):.3f}  "
                  f"min {fp.min():.3f}")
            print(f"  el resto   ({len(otros):>3} tracks): fraccion mediana {np.median(fo):.3f}  "
                  f"max {fo.max():.3f}")
            if fp.min() > fo.max():
                print(f"  SEPARAN LIMPIO. Cualquier umbral entre {fo.max():.3f} y "
                      f"{fp.min():.3f} los parte sin error")
            else:
                solapan = sum(1 for x in fo if x >= fp.min())
                print(f"  se solapan: {solapan} tracks no peleadores llegan o superan al "
                      f"peleador mas bajo")
    if args.out:
        print(f"\nvideo en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
