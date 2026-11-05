"""Pruebas de los recortes de C5 (docs/specs/2026-11-04-cierre/00-preregistro.md §3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recortes_c5 as r  # noqa: E402


def s(i, factor, cuenta="x", duracion=200.0, llega=True):
    return {"id": i, "factor": factor, "cuenta": cuenta, "duracion_s": duracion, "llego_a_fightcard": llega}


def test_recorte_mediana_y_umbral_incluido():
    assert r.recorte([s("a", 1.0), s("b", 2.0), s("c", 3.0)]) == {"n": 3, "mediana": 2.0, "maximo": 3.0, "pasa": True}
    assert not r.recorte([s("a", 2.1)])["pasa"]
    assert not r.recorte([])["pasa"]


def test_recortes_separan_cuenta_transmision_y_largo():
    x = r.recortes([s("a", 1.0, r.CUENTA_FUENTES), s(r.TRANSMISION, 3.0), s("c", 2.0, duracion=20.0), s("d", 9.0, llega=False)])
    assert x["todas"]["n"] == 3
    assert x["otras_cuentas"]["n"] == 2
    assert x["sin_transmision"]["n"] == 2
    assert x["tres_minutos_o_mas"]["n"] == 2
    assert x["tres_minutos_o_mas_sin_transmision"]["n"] == 1


def test_recorte_de_las_nuevas_de_gimnasio():
    x = r.recortes([s(r.NUEVAS_GIMNASIO[0], 9.0), s(r.NUEVAS_GIMNASIO[1], 4.0), s("c", 1.0)])
    assert x["nuevas_de_gimnasio"] == {"n": 2, "mediana": 6.5, "maximo": 9.0, "pasa": False}
    assert x["sin_las_nuevas_de_gimnasio"]["n"] == 1

