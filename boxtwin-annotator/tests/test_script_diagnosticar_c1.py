"""Las partes puras del diagnostico de C1: aristas espurias y voto de color de oraculo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_dir = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_dir))
_spec = importlib.util.spec_from_file_location("diagnosticar_c1", _dir / "diagnosticar_c1.py")
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

A, B = "fighter_A", "fighter_B"
ROJO, AZUL = (2.0, 200.0, 180.0), (110.0, 200.0, 180.0)


def test_aristas_espurias_separa_mismo_rol_y_no_peleador_y_cuenta_cuadros():
    ady = {1: {2, 3}, 2: {1}, 3: {1, 9}, 9: {3}}
    frames = {1: {0, 1, 2, 3}, 2: {0, 1, 2}, 3: {1, 2, 3}, 9: {2, 3}}
    verdad = {1: A, 2: B, 3: A}            # 1-3 mismo rol, 3-9 contra no peleador
    mismo, no_pel = d.aristas_espurias(ady, frames, verdad)
    assert mismo == [[1, 3, 3]]
    assert no_pel == [[3, 9]]


def test_aristas_espurias_no_cuenta_dos_veces_ni_las_legitimas():
    ady = {1: {2}, 2: {1}}
    mismo, no_pel = d.aristas_espurias(ady, {1: {0}, 2: {0}}, {1: A, 2: B})
    assert mismo == [] and no_pel == []


def test_voto_oraculo_elige_el_perfil_mas_cercano():
    assert d.voto_oraculo([ROJO] * 5 + [AZUL], [ROJO] * 10, [AZUL] * 10) == A
    assert d.voto_oraculo([AZUL] * 4, [ROJO] * 10, [AZUL] * 10) == B


def test_voto_oraculo_empate_o_sin_perfil_no_decide():
    assert d.voto_oraculo([ROJO, AZUL], [ROJO] * 10, [AZUL] * 10) is None
    assert d.voto_oraculo([ROJO], [], [AZUL]) is None
