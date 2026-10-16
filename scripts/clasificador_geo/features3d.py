"""
BoxTwin - Features 3D del brazo que golpea en un marco del cuerpo, para E5.

POR QUE EXISTE
  La pose 3D de MotionBERT viene en coordenadas de camara: un hook visto de frente y uno visto
  de costado dan numeros distintos, que es justo el problema del 2D. Expresar el brazo en un
  marco pegado al cuerpo (origen en el torax, eje vertical pelvis -> torax, eje x sobre la linea
  de caderas) hace a las features independientes de donde esta la camara. El brazo derecho se
  espeja, como en el detector 2D, para leer siempre el izquierdo.

QUE HACE
  Por paso de la ventana (diseno 4.2): espejo si el golpe es del brazo derecho, origen en el
  torax, escala pelvis-torax, rotacion alrededor del eje vertical del cuerpo para que la linea
  de caderas quede sobre x. Features: muneca, codo y hombro xyz (9), angulo del codo, elevacion
  del codo respecto del tronco, azimut de la muneca alrededor del hombro en el plano horizontal
  (cos, sin), velocidad de la muneca en el plano horizontal y en vertical, velocidad del azimut.
  Son 16: el diseno dice 18 pero su lista nombra 16, y no se inventan dos (ver TAREAS.md,
  decisiones). Las velocidades van por segundo. Pasos sin 3D (M3 = 0) en cero.
  Escribe <trabajo>/datos/features3d.npz (frame_dir, X3 (N, 32, 16), M3) y features3d.meta.json.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/features3d.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import PASOS, TRABAJO  # noqa: E402

PELVIS, CAD_DER, CAD_IZQ, TORAX = 0, 1, 4, 8
HOMBRO, CODO, MUNECA = 11, 12, 13           # brazo izquierdo
PARES_LR_H36M = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
NOMBRES = [
    "muneca3_x", "muneca3_y", "muneca3_z", "codo3_x", "codo3_y", "codo3_z",
    "hombro3_x", "hombro3_y", "hombro3_z",
    "angulo_codo3", "elevacion_codo3", "azimut3_cos", "azimut3_sin",
    "v_muneca3_horizontal", "v_muneca3_vertical", "v_azimut3",
]
_EPS = 1e-6


def espejar(P: np.ndarray) -> np.ndarray:
    """x -> -x y permutacion izquierda/derecha en H36M. P (..., 17, 3)."""
    Q = np.array(P, copy=True)
    Q[..., 0] = -Q[..., 0]
    for a, b in PARES_LR_H36M:
        Q[..., [a, b], :] = Q[..., [b, a], :]
    return Q


def marco_cuerpo(P: np.ndarray) -> np.ndarray:
    """
    P (T, 17, 3) -> mismo esqueleto en el marco del cuerpo: origen torax, escala pelvis-torax,
    y = pelvis -> torax, x = linea de caderas (derecha -> izquierda) sin componente vertical,
    z = x cruz y.
    """
    arriba = P[:, TORAX] - P[:, PELVIS]
    escala = np.linalg.norm(arriba, axis=-1)
    y = arriba / np.maximum(escala, _EPS)[:, None]
    cad = P[:, CAD_IZQ] - P[:, CAD_DER]
    cad = cad - (cad * y).sum(-1, keepdims=True) * y
    x = cad / np.maximum(np.linalg.norm(cad, axis=-1), _EPS)[:, None]
    z = np.cross(x, y)
    R = np.stack([x, y, z], 1)                                 # (T, 3, 3): filas = ejes
    rel = (P - P[:, TORAX][:, None]) / np.maximum(escala, _EPS)[:, None, None]
    return np.einsum("tij,tkj->tki", R, rel)


def features_de(P: np.ndarray, side: str, duracion_s: float) -> np.ndarray:
    """P (T, 17, 3) del atacante en la grilla de la ventana -> (T, 16)."""
    if side == "right":
        P = espejar(P)
    B = marco_cuerpo(np.asarray(P, np.float64))
    m, c, h = B[:, MUNECA], B[:, CODO], B[:, HOMBRO]
    a, b = c - h, m - c
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), _EPS)
    angulo_codo = np.arccos(np.clip(cos, -1, 1))
    elev = np.arccos(np.clip(-a[:, 1] / np.maximum(np.linalg.norm(a, axis=-1), _EPS), -1, 1))
    v = m - h
    hz = np.linalg.norm(v[:, [0, 2]], axis=-1)
    az_cos = np.where(hz > _EPS, v[:, 0] / np.maximum(hz, _EPS), 0.0)
    az_sin = np.where(hz > _EPS, v[:, 2] / np.maximum(hz, _EPS), 0.0)
    T = len(B)
    dt = max(duracion_s, _EPS) / max(T - 1, 1)
    d = (lambda s: np.gradient(s, dt, axis=0)) if T > 1 else (lambda s: np.zeros_like(s))
    vm = d(m)
    v_h = np.linalg.norm(vm[:, [0, 2]], axis=-1)
    v_v = vm[:, 1]
    v_az = d(np.unwrap(np.arctan2(az_sin, az_cos)))
    f = np.concatenate([m, c, h, np.stack([angulo_codo, elev, az_cos, az_sin, v_h, v_v, v_az], -1)], -1)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def construir(golpes: list[dict], P3: np.ndarray, M3: np.ndarray) -> np.ndarray:
    X = []
    for g, P, msk in zip(golpes, P3, M3):
        dur = (g["fin"] - g["ini"]) / g["fps"]
        f = features_de(P, g["side"], dur)
        f[~msk.astype(bool)] = 0.0
        X.append(f)
    return np.stack(X)


def main(argv=None) -> int:
    from datos import leer_csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    d = TRABAJO / "datos"
    por_fd = {g["frame_dir"]: g for g in leer_csv(d / "golpes.csv")}
    with np.load(d / "pose3d.npz") as z:
        fd, P3, M3 = z["frame_dir"], z["P3"], z["M3"]
    golpes = [por_fd[n] for n in fd]
    X3 = construir(golpes, P3, M3)
    assert X3.shape[1] == PASOS
    np.savez_compressed(d / "features3d.npz", frame_dir=fd, X3=X3, M3=M3)
    (d / "features3d.meta.json").write_text(json.dumps(
        {"features": NOMBRES, "F": len(NOMBRES), "golpes": int(len(fd)),
         "nota": "el diseno 4.2 dice 18; su lista nombra 16 y son estas"}, indent=2) + "\n")
    print(f"X3 {X3.shape}, pasos con 3D {M3.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
