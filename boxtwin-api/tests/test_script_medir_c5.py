"""Las cuentas de C5 (pre-registro del 20-10): que trabajos cuentan, solapamiento y mediana."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "medir_c5", Path(__file__).resolve().parents[1] / "scripts" / "medir_c5.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

T0 = datetime(2026, 10, 27, 10, 0, 0)


def _t(etapa, ini, fin, estado="listo", sesion="s"):
    return {"sesion_id": sesion, "etapa": etapa, "estado": estado,
            "tomado_en": T0 + timedelta(seconds=ini),
            "terminado_en": None if fin is None else T0 + timedelta(seconds=fin)}


def test_cuenta_el_primer_procesar_completar_y_clasificar_en_orden():
    trs = [_t("procesar", 0, 100), _t("completar", 300, 310), _t("clasificar", 311, 313),
           _t("render", 314, 350), _t("completar", 400, 420), _t("clasificar", 421, 430)]
    f = m.medir_sesion(trs, duracion_s=60.0, otros=[])
    assert f["etapas"] == {"procesar": 100.0, "completar": 10.0, "clasificar": 2.0}
    assert f["segundos"] == 112.0 and f["factor"] == round(112 / 60, 3)
    assert f["render_s"] == 36.0 and not f["sin_clasificar"]


def test_ignora_fallos_y_un_completar_anterior_al_procesar():
    trs = [_t("completar", -50, -40), _t("procesar", 0, 20, estado="fallo"),
           _t("procesar", 30, 90), _t("completar", 100, 105)]
    f = m.medir_sesion(trs, duracion_s=30.0, otros=[])
    assert f["etapas"] == {"procesar": 60.0, "completar": 5.0}
    assert f["sin_clasificar"] and f["factor"] == round(65 / 30, 3)


def test_sin_completar_no_llega_a_la_fightcard():
    f = m.medir_sesion([_t("procesar", 0, 50)], duracion_s=30.0, otros=[])
    assert f["llego_a_fightcard"] is False and "factor" not in f


def test_solapada_si_otra_sesion_corre_a_la_vez():
    trs = [_t("procesar", 0, 100), _t("completar", 200, 210)]
    assert m.medir_sesion(trs, 60.0, [_t("procesar", 50, 150, sesion="x")])["solapada"]
    assert not m.medir_sesion(trs, 60.0, [_t("procesar", 100, 150, sesion="x")])["solapada"]
    assert m.medir_sesion(trs, 60.0, [_t("procesar", 205, None, estado="tomado",
                                          sesion="x")])["solapada"]


def test_resumen_mediana_umbral_y_sin_solapadas():
    filas = [{"llego_a_fightcard": True, "factor": 1.5, "solapada": False},
             {"llego_a_fightcard": True, "factor": 2.5, "solapada": True},
             {"llego_a_fightcard": True, "factor": 1.9, "solapada": False},
             {"llego_a_fightcard": False, "solapada": False}]
    r = m.resumir(filas)
    assert r["mediana_factor"] == 1.9 and r["pasa"] is True
    assert r["mediana_sin_solapadas"] == 1.7 and r["llegaron_a_fightcard"] == 3
    assert m.resumir([{"llego_a_fightcard": True, "factor": 2.01, "solapada": False}])["pasa"] is False
