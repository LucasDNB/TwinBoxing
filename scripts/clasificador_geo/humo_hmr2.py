"""
BoxTwin - E7-1: prueba de humo de HMR2.0 (4D-Humans) sobre un golpe de BoxTwin.

POR QUE EXISTE
  El diseno de E7 deja tres cosas sin saber hasta probar en la estacion: cuanta VRAM usa el
  ViT-H de HMR2 en la 2080 de 8 GB, a que velocidad corre con nuestras cajas y que claves y
  formas devuelve el modelo. Si no entra con lote 1, E7 queda no realizada (diseno 1). Este
  script lo mide con un solo golpe, antes de escribir elevar_hmr2.py.

QUE HACE
  Toma el primer golpe de una fuente en golpes.csv de E4, lee los cuadros de su ventana
  [ini, fin] del video de la anotacion, arma la caja del atacante con los keypoints COCO de
  score >= 0,3 (pre-registro E7, E2), corre HMR2 con el ViTDetDataset por defecto y saca las 24
  articulaciones de SMPL con el forward de smplx.SMPL (sin el reordenamiento a OpenPose del
  envoltorio de 4D-Humans). Mide VRAM pico, tiempo por recorte con lote 1 y con el lote pedido,
  y escribe ~/boxtwin/e7-4dhumans/humo.json con claves, formas y numeros.
  No produce ningun numero de E7: solo decide si E7 es viable y con que lote.

USO (en boxtwin_4dhumans)
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python scripts/clasificador_geo/humo_hmr2.py
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python scripts/clasificador_geo/humo_hmr2.py --fuente pacquiao_margarito --lote 16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elevar_hmr2 import caja  # noqa: E402
from geo import TRABAJO, TRABAJO_E7, comun  # noqa: E402


def cuadros(video: Path, ini: int, fin: int) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    salida = []
    for _ in range(ini, fin + 1):
        ok, img = cap.read()
        if not ok:
            break
        salida.append(img)
    cap.release()
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuente", default="02-sparring")
    ap.add_argument("--lote", type=int, default=8)
    args = ap.parse_args(argv)

    import smplx
    import torch
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
    from hmr2.utils import recursive_to

    with (TRABAJO / "datos" / "golpes.csv").open(encoding="utf-8") as fh:
        g = next(r for r in csv.DictReader(fh) if r["fuente"] == args.fuente)
    z = np.load(TRABAJO / "datos" / "series" / f"{args.fuente}.npz")
    p, ini, fin = int(g["p"]), int(g["ini"]), int(g["fin"])
    video = Path(json.loads(comun.ruta_annot(args.fuente).read_text())["video"]["path"])
    imgs = cuadros(video, ini, fin)

    dev = torch.device("cuda")
    t0 = time.time()
    modelo, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    modelo = modelo.to(dev).eval()
    t_carga = time.time() - t0
    torch.cuda.reset_peak_memory_stats()

    items, sin_caja = [], 0
    for i, img in enumerate(imgs):
        b = caja(z["kp"][p, ini + i], z["sc"][p, ini + i])
        if b is None:
            sin_caja += 1
            continue
        items.append(ViTDetDataset(cfg, img, b[None])[0])

    def correr(lote: int):
        salidas, t = [], time.time()
        for k in range(0, len(items), lote):
            batch = torch.utils.data.default_collate(items[k:k + lote])
            batch = recursive_to(batch, dev)
            with torch.no_grad():
                out = modelo(batch)
                prm = out["pred_smpl_params"]
                n = prm["body_pose"].shape[0]
                crudo = smplx.SMPL.forward(modelo.smpl,
                                           global_orient=prm["global_orient"].reshape(n, -1, 3, 3),
                                           body_pose=prm["body_pose"].reshape(n, -1, 3, 3),
                                           betas=prm["betas"].reshape(n, -1), pose2rot=False)
            torch.cuda.synchronize()
            salidas.append((out, crudo))
        return salidas, (time.time() - t) / max(len(items), 1)

    correr(1)                                   # calentamiento
    _, s_lote1 = correr(1)
    salidas, s_lote = correr(args.lote)
    out, crudo = salidas[0]
    J24 = torch.cat([c.joints[:, :24] for _, c in salidas]).cpu().numpy()
    hombros = np.linalg.norm(J24[:, 16] - J24[:, 17], axis=1)
    r = {
        "golpe": g["frame_dir"], "fuente": args.fuente, "ventana": [ini, fin], "cuadros": len(imgs),
        "recortes": len(items), "sin_caja": sin_caja, "lote": args.lote,
        "carga_s": round(t_carga, 1), "s_por_recorte_lote1": round(s_lote1, 4),
        "s_por_recorte_lote": round(s_lote, 4),
        "vram_pico_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        "claves_salida": sorted(out.keys()),
        "formas": {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")},
        "formas_smpl": {k: list(v.shape) for k, v in out["pred_smpl_params"].items()},
        "joints_smplx": list(crudo.joints.shape),
        "ancho_hombros_smpl_mediana_m": round(float(np.median(hombros)), 3),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    TRABAJO_E7.mkdir(parents=True, exist_ok=True)
    (TRABAJO_E7 / "humo.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
