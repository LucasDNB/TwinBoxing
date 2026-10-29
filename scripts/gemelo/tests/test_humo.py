"""T11: humo.py, las partes de H1 a H3 que dan un numero (01-percepcion.md §3)."""

import numpy as np
import pytest


def test_las_permutaciones_izquierda_derecha_son_las_255_no_triviales(gemelo):
    import humo
    perms = humo.permutaciones_lr()
    assert len(perms) == 2 ** 8 - 1
    assert all(sorted(p) == list(range(17)) and p[0] == 0 for p in perms)   # la nariz no se mueve
    assert all(list(p) != list(range(17)) for p in perms)
    espejo = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    assert espejo in [list(p) for p in perms]
    assert all(p[p[i]] == i for p in perms for i in range(17))               # involuciones


def test_las_parejas_salen_de_los_nombres_coco(gemelo):
    import humo
    assert humo.PAREJAS_LR[0] == (1, 2) and (9, 10) in humo.PAREJAS_LR and len(humo.PAREJAS_LR) == 8


def test_emparejar_personas_por_iou(gemelo):
    import humo
    a = np.array([[0, 0, 100, 100], [500, 0, 600, 100]], float)
    b = np.array([[505, 0, 605, 100], [300, 0, 400, 100], [2, 0, 102, 100]], float)
    assert sorted(humo.emparejar_personas(a, b)) == [(0, 2), (1, 0)]
    assert humo.emparejar_personas(a, np.zeros((0, 4))) == []


def _cuerpo(rng):
    return rng.uniform(0, 500, (17, 2))


def test_orden_propio_pasa_y_espejado_falla(gemelo):
    import humo
    rng = np.random.default_rng(0)
    pares = []
    for _ in range(20):
        kp = _cuerpo(rng)
        pares.append((kp, kp + rng.normal(0, 2, kp.shape)))
    v = humo.veredicto_orden(pares)
    assert v["pasa"] and v["mediana_propia"] < v["mediana_mejor_permutacion"]
    espejo = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    v = humo.veredicto_orden([(a, b[espejo]) for a, b in pares])
    assert not v["pasa"] and v["mejor_permutacion"] == espejo


def test_sin_pares_no_pasa(gemelo):
    import humo
    assert humo.veredicto_orden([])["pasa"] is False


def test_tiempo_saca_el_calentamiento(gemelo):
    import humo
    ms = [1000.0] * 100 + list(range(1, 1001))
    e = humo.estadisticas_tiempo(ms, calentamiento=100)
    assert e["n"] == 1000 and e["mediana_ms"] == pytest.approx(500.5)
    assert e["p95_ms"] == pytest.approx(np.percentile(np.arange(1, 1001), 95))
    with pytest.raises(ValueError):
        humo.estadisticas_tiempo([1.0] * 50, calentamiento=100)


def test_tramo_con_los_dos_primeros_cortes(gemelo):
    import humo
    assert humo.tramo_de_cortes([718, 1712, 2208], margen=60) == (658, 1772, [718, 1712])
    with pytest.raises(ValueError):
        humo.tramo_de_cortes([718], margen=60)


def test_cuadros_equiespaciados(gemelo):
    import humo
    c = humo.cuadros_equiespaciados(5053, 50)
    assert len(c) == 50 and c[0] == 0 and c[-1] < 5053 and len(set(c)) == 50


def test_veredicto_general(gemelo):
    import humo
    bien = {"H1": {"pasa": True}, "H2": {"pasa": True}, "H3": {"pasa": True}}
    assert humo.codigo_de_salida(bien) == 0
    assert humo.codigo_de_salida({**bien, "H1": {"pasa": False}}) == 1
    assert humo.codigo_de_salida({**bien, "H2": {"pasa": False}}) == 1
