"""Z6: metricas contra casos calculados a mano (CZ5)."""

import pytest

from boxtwinz_api.services.metrics import (
    GolpeT, calcular_sesion, combinaciones, curva, exposicion, iniciativa, metricas_de,
    segmentos, sosten_segmento, tendencia,
)


def G(*pares):
    return [GolpeT(t, m) for t, m in pares]


def test_segmentos_de_180_y_resto_corto_se_suma():
    assert segmentos(400) == [(0, 180), (180, 400)]           # resto 40 s < 60 -> se suma
    assert segmentos(420) == [(0, 180), (180, 360), (360, 420)]  # resto 60 s se queda
    assert segmentos(100) == [(0, 100)]
    assert segmentos(0) == []


def test_segmentos_con_rounds_declarados():
    assert segmentos(600, 3) == [(0, 200), (200, 400), (400, 600)]


def test_combinaciones_encadenadas_cuentan_una():
    # 0.0, 0.3, 0.6, 0.9 es una serie de 4 (cada salto < 0,5): UNA combinacion.
    # 2.0, 2.4 otra. 5.0 suelto. 7.0 y 7.5: salto = 0,5, NO es < 0,5 -> sueltos.
    ts = [0.0, 0.3, 0.6, 0.9, 2.0, 2.4, 5.0, 7.0, 7.5]
    assert combinaciones(ts) == (2, 6)


def test_combinaciones_vacio_y_uno():
    assert combinaciones([]) == (0, 0)
    assert combinaciones([1.0]) == (0, 0)


def test_volumen_balance_y_mano_adelantada_ortodoxa():
    g = G((1, "izquierda"), (2, "izquierda"), (3, "izquierda"), (4, "derecha"))
    m = metricas_de(g, 0, 120, "ortodoxa")
    assert m["volumen"] == 2.0              # 4 golpes en 2 min
    assert m["pct_izquierda"] == 0.75
    assert m["pct_mano_adelantada"] == 0.75


def test_zurdo_invierte_la_mano_adelantada():
    g = G((1, "izquierda"), (2, "izquierda"), (3, "izquierda"), (4, "derecha"))
    m = metricas_de(g, 0, 120, "zurda")
    assert m["pct_izquierda"] == 0.75
    assert m["pct_mano_adelantada"] == 0.25


def test_sosten_a_mano():
    # Segmento de 180 s: 10 golpes en el primer minuto (10/min), 16 en los 2 restantes (8/min).
    ts = [i * 5.0 for i in range(10)] + [60 + i * 7.0 for i in range(16)]
    assert sosten_segmento(ts, 0, 180) == pytest.approx(0.8)


def test_sosten_no_se_calcula_en_segmento_corto():
    assert sosten_segmento([1, 2, 3, 70], 0, 89.9) is None
    assert sosten_segmento([70, 80], 0, 120) is None    # primer minuto sin golpes


def test_sosten_total_es_promedio_de_segmentos():
    # 360 s, dos segmentos. Seg 1: 6 en el 1er min, 6 en el resto -> (6/2)/6 = 0.5.
    # Seg 2: 4 en el 1er min, 8 en el resto -> (8/2)/4 = 1.0. Promedio 0.75.
    ts = ([i * 10.0 for i in range(6)] + [60 + i * 20.0 for i in range(6)]
          + [180 + i * 15.0 for i in range(4)] + [240 + i * 15.0 for i in range(8)])
    m = calcular_sesion(G(*[(t, "izquierda") for t in ts]), 360)
    assert [s["sosten_ritmo"] for s in m["segmentos"]] == pytest.approx([0.5, 1.0])
    assert m["total"]["sosten_ritmo"] == pytest.approx(0.75)


def test_curva_por_bloques_de_10s():
    assert curva([0, 9.9, 10, 25, 29.9], 30) == [2, 1, 2]
    assert curva([34], 35) == [0, 0, 0, 1]


def test_iniciativa():
    assert iniciativa(30, 10) == 0.75
    assert iniciativa(0, 0) is None


def test_exposicion_cruza_los_lados():
    e = exposicion(G((1, "izquierda"), (2, "izquierda"), (3, "derecha")))
    assert e == {"rival_izquierda": 2, "rival_derecha": 1,
                 "lado_izquierdo_usuario": 1, "lado_derecho_usuario": 2}


def test_tendencia_con_banda():
    assert tendencia([10, 11, 12, 13]) == "subiendo"   # pendiente 1, *3/11,5 = 0,26
    assert tendencia([10, 10.1, 10, 10.1]) == "estable"
    assert tendencia([20, 18, 16]) == "bajando"
    assert tendencia([5]) is None


def test_sesion_sin_golpes_no_rompe():
    m = calcular_sesion([], 200)
    assert m["total"]["volumen"] == 0 and m["total"]["pct_izquierda"] is None
