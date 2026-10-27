"""El procedimiento de medicion de B8 (ampliacion 02 §2): planilla cada 2 s y su cuenta."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "medir_individual", Path(__file__).resolve().parents[1] / "scripts" / "medir_individual.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _planilla(tmp_path, filas):
    p = tmp_path / "p.csv"
    p.write_text("t,en_cuadro,marca_A\n" + "\n".join(",".join(map(str, f)) for f in filas) + "\n")
    return p


def test_saltos_de_dos_segundos_incluyen_el_cero_y_no_pasan_la_duracion():
    assert m.saltos(5.0) == [0.0, 2.0, 4.0]
    assert m.saltos(180.0)[-1] == 180.0 and len(m.saltos(180.0)) == 91


def test_el_porcentaje_es_sobre_los_saltos_con_la_persona_en_cuadro(tmp_path):
    filas = [(i * 2.0, "si", "si") for i in range(95)] + [(190.0 + i * 2, "si", "no") for i in range(5)]
    filas += [(400.0, "no", "")]
    r = m.resultado(_planilla(tmp_path, filas))
    assert (r["en_cuadro"], r["con_marca"], r["fraccion"]) == (100, 95, 0.95)
    assert r["pasa"] is True


def test_un_video_corto_no_pasa_aunque_el_porcentaje_si(tmp_path):
    r = m.resultado(_planilla(tmp_path, [(0.0, "si", "si"), (2.0, "si", "si")]))
    assert r["fraccion"] == 1.0 and r["pasa"] is False


def test_una_planilla_a_medias_no_es_una_medicion(tmp_path):
    with pytest.raises(ValueError, match="a medias"):
        m.resultado(_planilla(tmp_path, [(0.0, "si", "si"), (2.0, "", "")]))
    with pytest.raises(ValueError, match="falta marca_A"):
        m.resultado(_planilla(tmp_path, [(0.0, "si", "")]))


def test_wilson():
    lo, hi = m.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-3) and hi == pytest.approx(0.9785, abs=1e-3)
