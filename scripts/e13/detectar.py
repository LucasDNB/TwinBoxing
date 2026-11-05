"""
BoxTwin - E13: guarda las detecciones crudas de pose, antes del tracker (3-11-2026).

POR QUE EXISTE
  E13 re-trackea offline "las mismas detecciones de raiz-v8, guardadas por cuadro"
  (docs/specs/2026-10-30-mejoras-post-e8/06-e13-tracker.md §2). Pero esas detecciones no estan
  guardadas: _inferir solo persiste lo que salio del tracker. Dos cosas lo impiden: las
  detecciones sin track se descartan, y la caja del cache es el estado del Kalman (STrack.result),
  no la caja detectada. Con el cache como entrada, H13-2 no podria dar 100% por construccion.

QUE HACE
  Por fuente, recorre el video con la misma lectura, modelo y parametros de raiz-v8
  (sha del modelo y parametros salen del meta del cache y se verifican) y llama a predict, que es
  el mismo paso que track sin el callback del tracker. Guarda en <e13>/det/<fuente>.npz:
  frame_index, xyxy, conf, cls, keypoints y kp_score. Antes de cada bloque de 500 cuadros lee las
  colas de A y BoxTwinZ y, si hay trabajos, suelta el candado y espera. Una fuente terminada no se
  repite. No escribe nada en raiz-v8.

USO
  python scripts/e13/detectar.py            # las siete fuentes, con el candado de GPU
  python scripts/e13/detectar.py --fuente 01-sparring
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import comun_e13 as c  # noqa: E402

BLOQUE = 500


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", nargs="*", default=None)
    a = ap.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    c.DET.mkdir(parents=True, exist_ok=True)
    candado = open(c.CANDADO, "a+")
    fcntl.flock(candado, fcntl.LOCK_EX)
    c.log("detectar: candado tomado")

    modelo = None
    for fuente in a.fuente or c.ORDEN:
        destino = c.DET / f"{fuente}.npz"
        if destino.is_file():
            continue
        meta = c.meta_de(fuente)
        inf = meta["inference"]
        peso = Path(inf["model_path"])
        if c.sha256(peso) != inf["model_sha256"]:
            raise SystemExit(f"el modelo de {fuente} no es el del cache")
        if modelo is None:
            modelo = YOLO(str(peso))
        cap = cv2.VideoCapture(str(c.video_de(fuente)))
        off, xyxy, conf, cls, kp, sc = [0], [], [], [], [], []
        f, t0 = 0, time.time()
        while True:
            if f % BLOQUE == 0:
                fcntl.flock(candado, fcntl.LOCK_UN)
                c.esperar_colas()
                fcntl.flock(candado, fcntl.LOCK_EX)
            ok, cuadro = cap.read()
            if not ok:
                break
            r = modelo.predict(cuadro, imgsz=inf["imgsz"], conf=inf["conf"], iou=inf["iou"],
                               device=inf["device"], half=inf["half"], verbose=False)[0]
            b = r.boxes
            n = 0 if b is None else len(b)
            if n:
                xyxy.append(b.xyxy.cpu().numpy().astype(np.float32))
                conf.append(b.conf.cpu().numpy().astype(np.float32))
                cls.append(b.cls.cpu().numpy().astype(np.float32))
                kp.append(r.keypoints.xy.cpu().numpy().astype(np.float32))
                sc.append(r.keypoints.conf.cpu().numpy().astype(np.float32))
            off.append(off[-1] + n)
            f += 1
        cap.release()
        if f != meta["video"]["total_frames"]:
            raise SystemExit(f"{fuente}: {f} cuadros, el cache tiene {meta['video']['total_frames']}")
        tmp = destino.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, frame_index=np.array(off), xyxy=np.concatenate(xyxy), conf=np.concatenate(conf),
                            cls=np.concatenate(cls), keypoints=np.concatenate(kp), kp_score=np.concatenate(sc))
        tmp.rename(destino)
        c.log(f"detectar {fuente}: {f} cuadros, {off[-1]} detecciones, {time.time() - t0:.0f} s")
    (c.DET / "hecho.json").write_text(json.dumps({"fuentes": sorted(p.stem for p in c.DET.glob("*.npz"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
