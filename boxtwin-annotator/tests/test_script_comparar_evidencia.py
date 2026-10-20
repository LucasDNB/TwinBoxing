"""La comparacion de evidencias del arreglo de lectura (C5, 20-10)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "comparar_evidencia", Path(__file__).resolve().parents[1] / "scripts" / "comparar_evidencia.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _t(tid, **k):
    base = {"track_id": tid, "primer_frame": 0, "ultimo_frame": 10, "alto_max": 0.5,
            "recortes": 3, "con_guante": 2, "colores": [[0, [10.0, 200.0, 150.0]]],
            "frames_con_guante": [0, 5], "xs": [[0, 100.0]]}
    base.update(k)
    return base


def test_identicas_con_ruido_de_gpu_debajo_de_la_tolerancia():
    a = {"tracks": [_t(1), _t(2)]}
    b = {"tracks": [_t(1, colores=[[0, [10.0004, 200.0, 150.0]]]), _t(2)]}
    r = m.comparar(a, b)
    assert r["identicas"] and r["iguales"] == 2


def test_detecta_campos_distintos_y_tracks_que_faltan():
    a = {"tracks": [_t(1), _t(2), _t(3)]}
    b = {"tracks": [_t(1, recortes=4), _t(2, frames_con_guante=[0]), _t(9)]}
    r = m.comparar(a, b)
    assert not r["identicas"]
    assert r["distintos"] == {"1": ["recortes"], "2": ["frames_con_guante"]}
    assert r["solo_en_vieja"] == [3] and r["solo_en_nueva"] == [9] and r["iguales"] == 0
