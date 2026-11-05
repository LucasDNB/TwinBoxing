"""
BoxTwin - R-11: cronometra una celda sobre un video (3-11-2026).

POR QUE EXISTE
  Los benchmarks de inferencia mienten si se promedia desde el primer cuadro (CLAUDE.md §6.1).
  Cada celda corre en su propio proceso para que un motor no herede memoria ni caches del
  anterior, con el protocolo de docs/specs/2026-11-03-r11-perfiles/00-preregistro.md §4.

QUE HACE
  Lee la lista de cuadros de cuadros.py y recorre los tramos en serie, en un hilo: t_dec
  (decodificar) y t_pose (del cuadro BGR a keypoints en CPU). Los primeros 200 son warm-up: se
  corren igual y no entran en los estadisticos. Guarda los tiempos, las detecciones y el proveedor
  real del backend en <salida>.npz y <salida>.json. Si ONNX cae a CPU, falla: no seria la celda.

USO
  python scripts/r11/medir_celda.py --modelo engine.engine --imgsz 640 --formato engine \
      --cuadros cuadros/sparring-3.json -o crudo/x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regla  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--formato", required=True, choices=("engine", "onnx", "pytorch"))
    ap.add_argument("--cuadros", type=Path, required=True)
    ap.add_argument("-o", "--salida", type=Path, required=True)
    a = ap.parse_args(argv)

    import cv2
    import torch  # noqa: F401  (carga cuDNN antes que onnxruntime)
    from ultralytics import YOLO

    lista = json.loads(a.cuadros.read_text())
    modelo = YOLO(a.modelo, task="pose")
    cap = cv2.VideoCapture(lista["video"])
    if not cap.isOpened():
        raise SystemExit(f"no abre {lista['video']}")

    t_dec, t_pose, idx, personas, muro = [], [], [], [], 0.0
    cajas, kps, scs = [], [], []
    perf = time.perf_counter
    for ini, fin in lista["tramos"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
        t_tramo = perf()
        for f in range(ini, fin):
            t0 = perf()
            ok, cuadro = cap.read()
            t1 = perf()
            if not ok:
                raise SystemExit(f"no decodifica el cuadro {f}")
            r = modelo.predict(cuadro, imgsz=a.imgsz, conf=0.25, iou=0.7, device=0, verbose=False)[0]
            caja = r.boxes.xyxy.cpu().numpy()
            kp = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 2))
            sc = (r.keypoints.conf.cpu().numpy() if r.keypoints is not None and r.keypoints.conf is not None
                  else np.zeros((0, 17)))
            t2 = perf()
            t_dec.append((t1 - t0) * 1e3)
            t_pose.append((t2 - t1) * 1e3)
            idx.append(f)
            personas.append(len(caja))
            cajas.append(caja.astype(np.float32))
            kps.append(kp.astype(np.float32))
            scs.append(sc.astype(np.float32))
        muro += perf() - t_tramo
    cap.release()

    proveedor = a.formato
    backend = getattr(getattr(modelo, "predictor", None), "model", None)
    sesion = getattr(backend, "session", None)
    if sesion is not None:
        proveedor = sesion.get_providers()[0]
        if a.formato == "onnx" and proveedor != "CUDAExecutionProvider":
            raise SystemExit(f"ONNX corrio en {proveedor}, no en CUDA")

    t_dec, t_pose = np.array(t_dec), np.array(t_pose)
    w = regla.WARMUP
    personas = np.array(personas)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    largo = np.array([len(c) for c in cajas])
    np.savez_compressed(
        f"{a.salida}.npz", t_dec=t_dec, t_pose=t_pose, cuadro=np.array(idx), personas=personas,
        n=largo, cajas=np.concatenate(cajas), kp=np.concatenate(kps), sc=np.concatenate(scs))
    Path(f"{a.salida}.json").write_text(json.dumps({
        "proveedor": proveedor,
        "sha_lista": lista["sha256"],
        "fps_sostenidos": float(len(t_dec) / muro),
        "k_r11a_dos_personas_warmup": float((personas[:w] >= 2).mean()),
        "fraccion_dos_personas": float((personas[w:] >= 2).mean()),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
