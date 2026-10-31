"""S7 (P14d): largos de los huecos cerrados por articulacion, con la definicion de E10 §2."""

import numpy as np
import pytest

import huecos

U = {a: 0.3 for a in ("nariz", "hombro", "codo", "muneca", "cadera")}


def test_largos_de_huecos_cerrados():
    sc = np.ones((40, 17))
    sc[5:8, 9] = 0.1          # 3, cerrado
    sc[20:30, 10] = 0.1       # 10, cerrado
    sc[0:2, 9] = 0.1          # abierto: no cuenta
    sc[12, 7] = 0.1           # codo, 1
    r = huecos.largos(sc, np.ones(40, bool), U)
    assert sorted(r["muneca"]) == [3, 10] and r["codo"] == [1]
    assert r["muneca_perdidos"] == 15     # los abiertos tambien son cuadros perdidos


def test_hueco_sin_deteccion_no_es_hueco():
    sc = np.ones((20, 17))
    sc[5:8, 9] = 0.1
    presente = np.ones(20, bool)
    presente[6] = False
    assert huecos.largos(sc, presente, U)["muneca"] == []


def test_fraccion_bajo_el_tope():
    r = huecos.resumen({"muneca": [2, 3, 10], "muneca_perdidos": 20, "codo": [], "codo_perdidos": 0}, fps=30)
    assert r["muneca"]["fraccion_perdidos_en_huecos_cortos"] == pytest.approx(5 / 20)
    assert r["muneca"]["histograma_s"]["0.1"] == 2


def test_el_total_usa_el_tope_de_cada_fuente():
    # auditoria de E10: llevar los largos a cuadros de 30 dejaba afuera los huecos de 3 de
    # Sparring (29,643 fps) y los de 6 de Pacquiao (59,94), que el relleno si toma
    fuentes = [({"muneca": [3, 4], "muneca_perdidos": 10, "codo": [], "codo_perdidos": 0}, 29.643),
               ({"muneca": [6, 7], "muneca_perdidos": 20, "codo": [], "codo_perdidos": 0}, 59.94)]
    t = huecos.total(fuentes)
    assert t["muneca"]["fraccion_perdidos_en_huecos_cortos"] == pytest.approx((3 + 6) / 30)
    assert t["muneca"]["huecos"] == 4
