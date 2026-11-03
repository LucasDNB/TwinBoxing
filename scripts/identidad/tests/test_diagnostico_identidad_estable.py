"""Diagnostico de IE6: forma de los errores (intercambiado, mixto, tercero), con verdad sintetica."""

import numpy as np

import diagnostico_identidad_estable as d
from boxtwin.mvp.revision import Tramo

A, B = d.A, d.B
# 1 es A, 2 es B, 3 es A hasta el 50 y B despues, 4 es A hasta el 80 y un tercero despues.
VERDAD = {1: [(0, 100, A)], 2: [(0, 100, B)], 3: [(0, 50, A), (50, 100, B)],
          4: [(0, 80, A), (80, 100, "ignore")]}
FRAMES = {t: np.arange(100) for t in VERDAD}


def test_sin_errores_no_hay_formas():
    tramos = [Tramo(1, 0, 100, "A"), Tramo(2, 0, 100, "B")]
    assert d.formas_de_error(VERDAD, tramos, [], FRAMES.get) == {}


def test_intercambiado_mixto_y_tercero_fuera_de_la_cola():
    tramos = [Tramo(1, 0, 100, "B"), Tramo(3, 0, 100, "A"), Tramo(4, 0, 100, "A"),
              Tramo(2, 0, 100, "")]
    assert d.formas_de_error(VERDAD, tramos, [], FRAMES.get) == {
        "fuera/intercambiado": 100, "fuera/mixto": 50, "fuera/tercero": 20}


def test_la_cola_separa_lo_visto_de_lo_silencioso():
    cola = [{"track": 3, "desde": 0, "hasta": 70}]
    assert d.formas_de_error(VERDAD, [Tramo(3, 0, 100, "A")], cola, FRAMES.get) == {
        "cola/mixto": 20, "fuera/mixto": 30}
