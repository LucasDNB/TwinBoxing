"""
tools/elevar_hmr2_sesion.py (docs/specs/2026-10-25-e7-modo-academico, M2): de una sesion a
pose3d_hmr2.npz, un registro por segmento. HMR2 y el video se reemplazan por dobles: lo que se
prueba es que cada segmento sale con su id, su ventana y su mascara. Que las piezas copiadas den
lo mismo que scripts/clasificador_geo lo prueba CA-E7a (test_equivalencia_e7.py).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERRAMIENTA = Path(__file__).resolve().parents[1] / "tools" / "elevar_hmr2_sesion.py"


def _h():
    spec = importlib.util.spec_from_file_location("elevar_hmr2_sesion", HERRAMIENTA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Elevador:
    """Devuelve un SMPL de 24 cuyo valor es el indice de la caja, para seguirle la pista."""

    def __init__(self):
        self.llamadas = 0

    def j24(self, imagenes, cajas):
        self.llamadas += 1
        return np.stack([np.full((24, 3), float(b[0]), np.float32) for b in cajas])


class _Lector:
    def cuadro(self, i):
        return np.zeros((8, 8, 3), np.uint8)


def _sesion(tmp_path, T=120):
    d = tmp_path / "ses"
    d.mkdir()
    kp = np.zeros((2, T, 17, 2), np.float32)
    kp[..., 0] = np.arange(T)[None, :, None] + np.arange(17)[None, None, :]
    kp[..., 1] = 100
    sc = np.full((2, T, 17), 0.9, np.float32)
    valid = np.ones((2, T), bool)
    valid[1, :] = False                       # B sin pose: sus golpes quedan con mascara 0
    np.savez_compressed(d / "poses.npz", keypoints=kp, kp_score=sc, valid=valid)
    segs = [{"peleador": "A", "brazo": "left", "inicio": 20, "fin": 30},
            {"peleador": "A", "brazo": "right", "inicio": 100, "fin": 118},
            {"peleador": "B", "brazo": "left", "inicio": 50, "fin": 60}]
    (d / "segmentos.json").write_text(json.dumps({
        "version": "0.1", "poses": "poses.npz", "segmentos": segs,
        "video": {"ruta": str(d / "v.mp4"), "fps": 30.0, "total_frames": T, "ancho": 640, "alto": 480}}))
    return d


def test_gid_es_el_de_la_fight_card():
    h = _h()
    assert h.gid({"peleador": "A", "brazo": "left", "inicio": 20}) == "A-izq-20"
    assert h.gid({"peleador": "B", "brazo": "der", "inicio": 7}) == "B-der-7"


def test_elevar_sesion_escribe_un_registro_por_segmento(tmp_path):
    h = _h()
    d = _sesion(tmp_path)
    r = h.elevar_sesion(d, _Elevador(), _Lector())
    with np.load(d / "pose3d_hmr2.npz") as z:
        gids, P3, M3 = list(z["gid"]), z["P3"], z["M3"]
    assert gids == ["A-izq-20", "A-der-100", "B-izq-50"]
    assert P3.shape == (3, 32, 17, 3) and M3.shape == (3, 32)
    assert M3[0].all() and M3[1].all() and not M3[2].any()
    # el hombro izquierdo H36M (11) sale de la caja: x minimo de los keypoints del cuadro
    ini, fin = h.ventana(20, 30, 120)
    assert np.allclose(P3[0, :, 11, 0], np.linspace(ini, fin, 32))
    meta = json.loads((d / "pose3d_hmr2.meta.json").read_text())
    assert meta["segmentos"] == 3 and meta["sin_3d"] == 1 and r["sin_3d"] == 1


def test_sin_ruta_de_video_falla_claro(tmp_path):
    h = _h()
    d = _sesion(tmp_path)
    seg = json.loads((d / "segmentos.json").read_text())
    del seg["video"]["ruta"]
    (d / "segmentos.json").write_text(json.dumps(seg))
    try:
        h.elevar_sesion(d, _Elevador())
    except ValueError as e:
        assert "ruta" in str(e)
    else:
        raise AssertionError("tenia que fallar sin la ruta del video")
