"""
BoxTwin - E7: 3D del atacante con HMR2.0 (4D-Humans) en lugar de MotionBERT.

POR QUE EXISTE
  En BM, el mismo codigo de BoxMind pierde 0,139 de F1 de tecnica al pasar de la pose de
  BoxingWeb (4D-Humans) a la de BoxTwin (YOLOv8l-pose + MotionBERT), y E5 mostro que el 3D de
  MotionBERT no mejora el hook. E7 cambia solo el estimador 3D y deja todo lo demas de E5 fijo
  (docs/specs/2026-10-24-e7-4dhumans/01-preregistro.md). Este script produce el pose3d.npz de
  E7 con el mismo formato que el de E5, para que features3d.py, entrenar.py y evaluar.py no
  cambien de calculo.

QUE HACE
  Por golpe de golpes.csv (los 1047 de E4):
  1. Cuadros de la ventana [ini, fin] de E4, con el paso de E5 (uno de cada dos por encima de
     45 fps), tomados de la misma grilla de contexto de E5 (cuadros_ventana).
  2. Caja del atacante por cuadro: minimo y maximo de los keypoints COCO con score >= 0,3 en los
     cuadros con pose; huecos de hasta 10 cuadros interpolados sobre la caja con
     elevar3d.rellenar_huecos; los mas largos quedan con mascara 0 (pre-registro, E2 y E4).
  3. HMR2.0b por cuadro con el ViTDetDataset por defecto, sin suavizado (E1).
  4. Las 24 articulaciones de SMPL con el forward de smplx.SMPL y los parametros que predice
     HMR2, llevadas a H36M-17 (smpl_a_h36m, E3); las que features3d no usa quedan en cero.
  5. Remuestreo a los 32 pasos con elevar3d.a_ventana y mascara con elevar3d.mascara_ventana.
  Escribe ~/boxtwin/e7-4dhumans/datos/pose3d.npz (frame_dir, P3 (N, 32, 17, 3), M3 (N, 32)) y
  pose3d.meta.json (checkpoint, commit del clon, pasos con 3D, tiempo). Reanudable por fuente:
  cada fuente se guarda en parciales/<fuente>.npz y se saltea si ya esta.

USO (en boxtwin_4dhumans, con la GPU libre)
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python scripts/clasificador_geo/elevar_hmr2.py
  ~/miniforge3/envs/boxtwin_4dhumans/bin/python scripts/clasificador_geo/elevar_hmr2.py --fuentes 02-sparring --lote 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elevar3d import a_ventana, indices_contexto, mascara_ventana, paso_de, rellenar_huecos  # noqa: E402
from geo import FUENTES, PASOS, TRABAJO, TRABAJO_E7, comun  # noqa: E402

SCORE_MIN = 0.3
LOTE = 8                     # E7-1: 2,8 GB de VRAM pico en la 2080
# H36M <- SMPL (pre-registro E7, E3). Torax <- cuello de SMPL, a la altura de los hombros.
SMPL_A_H36M = {0: 0, 1: 2, 4: 1, 8: 12, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21}
CLON = Path("~/externos/4D-Humans").expanduser()


def smpl_a_h36m(J: np.ndarray) -> np.ndarray:
    """(..., 24, 3) de SMPL -> (..., 17, 3) en el orden H36M de features3d; el resto en cero."""
    Q = np.zeros(J.shape[:-2] + (17, 3), np.float32)
    for h, s in SMPL_A_H36M.items():
        Q[..., h, :] = J[..., s, :]
    return Q


def caja(kp: np.ndarray, sc: np.ndarray, score_min: float = SCORE_MIN) -> np.ndarray | None:
    """xyxy desde los keypoints con score >= score_min; None si no hay ninguno."""
    ok = sc >= score_min
    if not ok.any():
        return None
    p = kp[ok]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)


def cajas_de(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    kp (T, 17, 2), sc (T, 17), valid (T,) del atacante -> cajas (T, 4) y mascara (T,). Un cuadro
    tiene caja propia si tiene pose y algun keypoint con score suficiente; los huecos se rellenan
    con la misma regla que la pose en E5.
    """
    T = len(kp)
    crudas = np.zeros((T, 2, 2), np.float32)
    presente = np.zeros(T, bool)
    for t in np.flatnonzero(valid):
        b = caja(kp[t], sc[t])
        if b is not None:
            crudas[t], presente[t] = b.reshape(2, 2), True
    rellenas, mascara = rellenar_huecos(crudas, presente)
    return rellenas.reshape(T, 4), mascara


def cuadros_ventana(ini: int, fin: int, T: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cuadros de la grilla de contexto de E5 que cubren [ini, fin] (uno afuera de cada lado como
    mucho), sin recortar y recortados al video.
    """
    paso = paso_de(fps)
    crudo, _ = indices_contexto(ini, fin, T, paso)
    a = np.flatnonzero(crudo <= ini)[-1]
    b = np.flatnonzero(crudo >= fin)[0]
    sel = crudo[a:b + 1]
    return sel, np.clip(sel, 0, T - 1)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_golpes(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for g in filas:
        for k in ("p", "ini", "fin", "start_frame", "end_frame"):
            g[k] = int(g[k])
        g["fps"] = float(g["fps"])
    return filas


class Lector:
    """Cuadros de un video por indice, leyendo hacia adelante y saltando solo si hace falta."""

    def __init__(self, ruta: Path):
        import cv2
        self.cv2, self.cap, self.pos, self.ultimo = cv2, cv2.VideoCapture(str(ruta)), -1, None

    def cuadro(self, i: int):
        if i == self.pos:
            return self.ultimo
        if i < self.pos or i > self.pos + 30:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i - 1
        while self.pos < i:
            ok, self.ultimo = self.cap.read()
            self.pos += 1
            if not ok:
                raise RuntimeError(f"no se pudo leer el cuadro {i}")
        return self.ultimo


class Elevador:
    def __init__(self, lote: int):
        import smplx
        import torch
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
        from hmr2.utils import recursive_to
        self.torch, self.smplx, self.Dataset, self.recursive_to = torch, smplx, ViTDetDataset, recursive_to
        self.dev = torch.device("cuda")
        self.modelo, self.cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.modelo = self.modelo.to(self.dev).eval()
        self.checkpoint, self.lote = Path(DEFAULT_CHECKPOINT), lote

    def j24(self, imagenes: list, cajas: list) -> np.ndarray:
        """(n, 24, 3) de SMPL para cada par imagen / caja xyxy."""
        torch = self.torch
        items = [self.Dataset(self.cfg, img, b[None])[0] for img, b in zip(imagenes, cajas)]
        salida = []
        for k in range(0, len(items), self.lote):
            batch = self.recursive_to(torch.utils.data.default_collate(items[k:k + self.lote]), self.dev)
            with torch.no_grad():
                prm = self.modelo(batch)["pred_smpl_params"]
                n = prm["body_pose"].shape[0]
                crudo = self.smplx.SMPL.forward(self.modelo.smpl,
                                                global_orient=prm["global_orient"].reshape(n, -1, 3, 3),
                                                body_pose=prm["body_pose"].reshape(n, -1, 3, 3),
                                                betas=prm["betas"].reshape(n, -1), pose2rot=False)
            salida.append(crudo.joints[:, :24].float().cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros((0, 24, 3), np.float32)


def golpe_3d(g: dict, kp, sc, valid, T: int, fps: float, lector: Lector, elev: Elevador,
             cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """P3 (32, 17, 3) y M3 (32,) de un golpe."""
    p = g["p"]
    if p not in cache:
        cache[p] = cajas_de(kp[p], sc[p], valid[p])
    cajas, mascara = cache[p]
    sel, idx = cuadros_ventana(g["ini"], g["fin"], T, fps)
    ok = mascara[idx]
    X = np.zeros((len(sel), 17, 3), np.float32)
    if ok.any():
        usar = np.flatnonzero(ok)
        J = elev.j24([lector.cuadro(int(idx[i])) for i in usar], [cajas[idx[i]] for i in usar])
        X[usar] = smpl_a_h36m(J)
        # sin caja: el vecino con 3D mas cercano (queda con mascara 0, como en E5)
        for i in np.flatnonzero(~ok):
            X[i] = X[usar[np.argmin(np.abs(usar - i))]]
    return a_ventana(X, sel, g["ini"], g["fin"]), mascara_ventana(mascara, g["ini"], g["fin"]) & ok.any()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=FUENTES)
    ap.add_argument("--lote", type=int, default=LOTE)
    args = ap.parse_args(argv)
    d, dest = TRABAJO / "datos", TRABAJO_E7 / "datos"
    (dest / "parciales").mkdir(parents=True, exist_ok=True)
    golpes = leer_golpes(d / "golpes.csv")
    elev = Elevador(args.lote)
    t0 = time.time()
    for fuente in args.fuentes:
        parcial = dest / "parciales" / f"{fuente}.npz"
        if parcial.exists():
            print(f"{fuente}: ya estaba", flush=True)
            continue
        gs = [g for g in golpes if g["fuente"] == fuente]
        with np.load(d / "series" / f"{fuente}.npz") as z:
            kp, sc, valid, fps = z["kp"], z["sc"], z["valid"], float(z["fps"])
        video = Path(json.loads(comun.ruta_annot(fuente).read_text())["video"]["path"])
        lector, cache, P3, M3 = Lector(video), {}, [], []
        for g in sorted(gs, key=lambda g: g["ini"]):
            P, M = golpe_3d(g, kp, sc, valid, kp.shape[1], fps, lector, elev, cache)
            P3.append(P)
            M3.append(M)
        orden = sorted(gs, key=lambda g: g["ini"])
        np.savez_compressed(parcial, frame_dir=np.array([g["frame_dir"] for g in orden]),
                            P3=np.stack(P3), M3=np.stack(M3))
        print(f"{fuente}: {len(gs)} golpes, paso {paso_de(fps)}, pasos con 3D {np.stack(M3).mean():.3f}, "
              f"{time.time() - t0:.0f} s acumulados", flush=True)
    partes = [np.load(dest / "parciales" / f"{f}.npz") for f in FUENTES if (dest / "parciales" / f"{f}.npz").exists()]
    fd = np.concatenate([z["frame_dir"] for z in partes])
    P3, M3 = np.concatenate([z["P3"] for z in partes]), np.concatenate([z["M3"] for z in partes])
    assert P3.shape[1] == PASOS
    np.savez_compressed(dest / "pose3d.npz", frame_dir=fd, P3=P3, M3=M3)
    commit = subprocess.run(["git", "-C", str(CLON), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (dest / "pose3d.meta.json").write_text(json.dumps({
        "estimador": "HMR2.0b (4D-Humans)", "commit_4dhumans": commit, "checkpoint": str(elev.checkpoint),
        "checkpoint_sha256": _sha(elev.checkpoint), "lote": args.lote, "score_min": SCORE_MIN,
        "smpl_a_h36m": SMPL_A_H36M, "golpes": int(len(fd)), "pasos_con_mascara": round(float(M3.mean()), 4),
        "segundos": round(time.time() - t0, 1)}, indent=2) + "\n")
    print(f"pose3d.npz: {len(fd)} golpes, pasos con 3D {M3.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
