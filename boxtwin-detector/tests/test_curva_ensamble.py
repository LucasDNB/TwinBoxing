"""
Gemelo YOLO26, E8 (docs/specs/2026-10-27-gemelo-yolo26, 02-preregistro-e8.md §4, P12a): el
F1 del ensamble se lee en el umbral 0,80 y la curva de 0,70 a 0,90 se informa. `folds` solo
guardaba el mejor umbral; `agregar_curva` suma los conteos de todas las fuentes por umbral.
"""

import pytest

from boxtwin_detector.cli import agregar_curva


def _c(golpes, predichos, emparejados):
    return {"golpes": golpes, "predichos": predichos, "emparejados": emparejados}


def test_suma_conteos_por_umbral_y_no_promedia():
    us = [0.7, 0.8]
    curvas = [[_c(10, 8, 6), _c(10, 5, 4)], [_c(30, 20, 10), _c(30, 10, 8)]]
    a, b = agregar_curva(us, curvas)
    assert a["umbral"] == 0.7 and a["golpes"] == 40 and a["emparejados"] == 16
    assert a["recall"] == pytest.approx(16 / 40, abs=1e-4) and a["precision"] == pytest.approx(16 / 28, abs=1e-4)
    assert b["f1"] == pytest.approx(2 * (12 / 40) * (12 / 15) / (12 / 40 + 12 / 15), abs=1e-4)


def test_sin_predichos_da_precision_cero_y_no_divide_por_cero():
    (x,) = agregar_curva([0.9], [[_c(5, 0, 0)]])
    assert x["precision"] == 0.0 and x["f1"] == 0.0


def test_la_grilla_de_folds_incluye_el_0_80():
    import numpy as np
    assert 0.8 in [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
