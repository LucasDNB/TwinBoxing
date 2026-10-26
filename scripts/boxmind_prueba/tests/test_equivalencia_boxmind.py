"""
La representacion reimplementada contra las funciones originales de dataset_youtube.py@e06add2.

Las funciones se leen del clon externo con ast y se ejecutan solo con numpy (el modulo entero
importa torch, cv2 y sklearn). Si el clon no esta, la prueba se saltea; en la estacion tiene
que correr antes de cualquier corrida de BM.
"""

import ast
import os
from pathlib import Path

import numpy as np
import pytest

import representacion as rp

CLON = Path(os.environ.get("BOXTWIN_BOXMIND_REPO", "~/externos/BoxingWeb")).expanduser()
FUENTE = CLON / "dataset" / "dataset_youtube.py"
pytestmark = pytest.mark.skipif(not FUENTE.is_file(), reason="sin el clon de BoxMind")


def _originales():
    arbol = ast.parse(FUENTE.read_text())
    piezas = [n for n in arbol.body
              if (isinstance(n, ast.FunctionDef) and n.name in ("convert_pose_3d", "get_region", "normalize"))
              or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "JOINT_INDEX" for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=piezas, type_ignores=[]), str(FUENTE), "exec"), ns)
    return ns


def _smpl(rng, T=6):
    P = rng.normal(0, 0.3, (T, 25, 3))
    P[:, 1], P[:, 2] = [-0.15, 0, 0], [0.15, 0, 0]
    P[:, 16], P[:, 17] = [-0.2, 0.05, 0.5], [0.2, -0.05, 0.5]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return P @ q.T + rng.normal(size=3)


def test_marco_cuerpo_igual_a_convert_pose_3d():
    ns = _originales()
    rng = np.random.default_rng(0)
    a, b = _smpl(rng), _smpl(rng)
    suyo = ns["convert_pose_3d"](a, b)                        # (T, 28, 3)
    idx = list(ns["JOINT_INDEX"])
    np.testing.assert_allclose(rp.marco_cuerpo(a[:, idx], escalar=False), suyo[:, :14], atol=1e-5)
    np.testing.assert_allclose(rp.marco_cuerpo(b[:, idx], escalar=False), suyo[:, 14:], atol=1e-5)


def test_caja_igual_a_get_region():
    ns = _originales()
    rng = np.random.default_rng(1)
    p = rng.uniform(100, 500, (5, 25, 2))
    suya = ns["get_region"](p.copy(), 360, 640, expansion_rate=0.4)
    assert rp.caja_atacante(p[:, list(ns["JOINT_INDEX"])], 640, 360) == suya
