"""Pruebas de la numeracion, H13-2, la regla de P18a y el emparejado de salidas (06-e13-tracker.md)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comun_e13 as c  # noqa: E402
from retrack import emparejar_salida  # noqa: E402


def test_ids_siguen_a_writer_en_los_cortes():
    i = c.Ids()
    assert i.global_([1, 2]).tolist() == [1, 2]
    i.corte()
    assert i.global_([1]).tolist() == [3]
    i.corte()
    i.corte()              # dos cortes seguidos sin ids nuevos no mueven el offset de mas
    assert i.global_([1]).tolist() == [4]


def test_comparar_ids_igual():
    off = np.array([0, 2, 3])
    ids = np.array([1, 2, 1])
    r = c.comparar_ids(off, ids, off, ids)
    assert r["distintos"] == 0 and r["fraccion"] == 1.0


def test_comparar_ids_detecta_orden_y_faltante():
    r = c.comparar_ids(np.array([0, 2, 3]), np.array([1, 2, 1]), np.array([0, 2, 2]), np.array([2, 1]))
    assert r["distintos"] == 2 and r["primer_distinto"] == 0


def test_regla_p18a_borde_y_rechazos():
    assert c.regla_p18a(10, 7, 10.0, 12.0)["mejora"]
    assert not c.regla_p18a(10, 8, 10.0, 10.0)["mejora"]        # baja 20%
    assert not c.regla_p18a(10, 5, 10.0, 12.1)["mejora"]        # suben los cambios
    assert not c.regla_p18a(0, 0, 10.0, 10.0)["mejora"]


def test_cambios_por_minuto_junta_a_y_b():
    t = {"A": {"cambios": 6, "cambios_por_minuto": 3.0}, "B": {"cambios": 2, "cambios_por_minuto": 1.0}}
    assert c.cambios_por_minuto(t) == pytest.approx(2.0)


def test_emparejar_salida_sin_repetir():
    d = np.array([[0, 0, 10, 10], [100, 100, 110, 110.0]])
    s = np.array([[101, 100, 111, 110], [1, 0, 11, 10], [1, 1, 10, 10.0]])
    assert emparejar_salida(s, d) == [1, -1, 0] or emparejar_salida(s, d) == [1, 0, -1]


def test_emparejar_salida_sin_solape():
    assert emparejar_salida(np.array([[0, 0, 1, 1.0]]), np.array([[5, 5, 6, 6.0]])) == [-1]
