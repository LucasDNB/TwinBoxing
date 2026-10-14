import pytest

import comun
from extraer_tiras import asignar_cupos, muestrear


def test_wilson_valores_conocidos():
    # 108 de 132 es el recall de cuerpo de C8; el doc del 13-10 da 0,744-0,875
    bajo, alto = comun.wilson(108, 132)
    assert (round(bajo, 3), round(alto, 3)) == (0.744, 0.875)
    assert comun.wilson(0, 0) is None


def test_balanceada_cuenta_prediccion_fuera_de_clase_como_error():
    real = ["head", "head", "body", "body"]
    pred = ["head", "no_se", "body", "head"]
    assert comun.balanceada(real, pred, comun.ALTURAS) == pytest.approx(0.5)


def test_kappa_acuerdo_perfecto_y_azar():
    assert comun.kappa(list("aabb"), list("aabb")) == pytest.approx(1.0)
    assert comun.kappa(list("abab"), list("aabb")) == pytest.approx(0.0)


def test_partes_de_clase_en_los_tres_espacios():
    assert comun.partes_de_clase("hook-left-body") == {"tipo": "hook", "eje": "left", "altura": "body"}
    assert comun.partes_de_clase("straight-lead") == {"tipo": "straight", "eje": "lead", "altura": None}
    assert comun.partes_de_clase("uppercut-head") == {"tipo": "uppercut", "eje": None, "altura": "head"}
    assert comun.clase_tipo_altura("uppercut-right-head") == "uppercut-head"
    with pytest.raises(ValueError):
        comun.partes_de_clase("feint")


def test_cupos_respetan_minimo_y_total():
    conteos = {("straight", "head"): 600, ("hook", "head"): 220, ("hook", "body"): 70,
               ("straight", "body"): 70, ("uppercut", "head"): 60, ("uppercut", "body"): 6}
    cupo = asignar_cupos(conteos, 150, 10)
    assert sum(cupo.values()) == 150
    assert cupo[("uppercut", "body")] == 6          # tiene menos que el minimo: entran todos
    assert all(v >= 10 for c, v in cupo.items() if conteos[c] >= 10)
    assert cupo[("straight", "head")] > cupo[("hook", "head")]


def test_cupos_minimo_imposible():
    with pytest.raises(ValueError):
        asignar_cupos({"a": 50, "b": 50}, 10, 10)


def test_muestreo_determinista_y_sin_repetidos():
    evs = [{"fuente": f"f{i % 3}", "event_id": f"ev_{i:04d}", "punch_type": "hook" if i % 4 else "straight",
            "target": "body" if i % 5 == 0 else "head"} for i in range(300)]
    a = muestrear(evs, 60, 5, seed=7)
    b = muestrear(evs, 60, 5, seed=7)
    assert a == b
    assert len({(e["fuente"], e["event_id"]) for e in a}) == 60
    assert muestrear(evs, 60, 5, seed=8) != a
