"""Las cuentas de C3 (pre-registro del 20-10, §4): sorteo, lectura del indicador y acuerdo."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "medir_c3", Path(__file__).resolve().parents[1] / "scripts" / "medir_c3.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _ev(i, completeness="full"):
    return {"id": f"ev_{i:04d}", "fighter": "fighter_A", "side": "left",
            "start_frame": 10 * i, "end_frame": 10 * i + 5, "completeness": completeness}


def test_sorteo_solo_completos_determinista_y_sin_repetidos():
    evs = [_ev(i) for i in range(300)] + [_ev(i, "feint") for i in range(300, 320)]
    a = m.sortear(evs, n=100, semilla=7)
    assert [e["id"] for e in a] == [e["id"] for e in m.sortear(list(reversed(evs)), 100, 7)]
    assert len({e["id"] for e in a}) == 100
    assert all(e["completeness"] == "full" for e in a)


def test_sorteo_falla_si_no_alcanzan():
    with pytest.raises(ValueError):
        m.sortear([_ev(i) for i in range(5)], n=10)


def test_golpe_de_traduce_peleador_brazo_y_tiempo():
    g = m.golpe_de({"id": "ev_1", "fighter": "fighter_B", "side": "right",
                    "start_frame": 30, "end_frame": 36}, fps=30.0)
    assert (g.peleador, g.brazo, g.inicio, g.fin, g.t_inicio) == ("B", "right", 30, 36, 1.0)


def _evg(medible=True, retorno=None, caida=None):
    return SimpleNamespace(medible=medible, retorno_ms=retorno, mano_opuesta_caida=caida)


def test_lectura_del_indicador():
    assert m.lectura_indicador(_evg(True, 200.0, False)) == {"volvio": "si", "caida": "no"}
    assert m.lectura_indicador(_evg(True, None, True)) == {"volvio": "no", "caida": "si"}
    assert m.lectura_indicador(_evg(False, None, None)) == {"volvio": None, "caida": None}


def test_acuerdo_excluye_no_se_ve_y_no_medibles_y_arma_la_confusion():
    lucas = ["si", "si", "no", "no", "no_se_ve", "si"]
    ind = ["si", "no", "no", "si", "si", None]
    r = m.acuerdo(lucas, ind)
    assert (r["n"], r["aciertos"], r["acuerdo"]) == (4, 2, 0.5)
    assert r["lucas_no_se_ve"] == 1 and r["indicador_no_medible"] == 1
    assert r["confusion"] == {"lucas_si_indicador_si": 1, "lucas_si_indicador_no": 1,
                              "lucas_no_indicador_si": 1, "lucas_no_indicador_no": 1}
    assert r["kappa"] == 0.0 and r["pasa"] is False


def test_acuerdo_umbral_es_la_proporcion_puntual():
    r = m.acuerdo(["si"] * 8 + ["no"] * 2, ["si"] * 8 + ["si"] * 2)
    assert r["acuerdo"] == 0.8 and r["pasa"] is True


def test_kappa_perfecto_y_de_una_sola_clase():
    assert m.kappa([("si", "si"), ("no", "no")]) == 1.0
    assert m.kappa([("si", "si"), ("si", "si")]) is None


def test_faltan_pide_las_dos_respuestas_validas():
    muestra = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    marcas = {"a": {"volvio": "si", "caida": "no"}, "b": {"volvio": "si"},
              "c": {"volvio": "x", "caida": "no"}}
    assert m.faltan(muestra, marcas) == ["b", "c"]
