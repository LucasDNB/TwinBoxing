"""T7: medir_trackeo.py, P12c sobre el denominador comun (01-percepcion.md §5)."""

import pytest

from sinteticos import caja, cache, documento

FPS = 30.0


def _a(**por_pel):
    return {p: dict(v) for p, v in por_pel.items()}


def test_cambios_ids_y_mixtos_sobre_una_secuencia(gemelo):
    import medir_trackeo as mt
    # A: 1 1 1 5 5 1 -> 2 cambios, 2 ids; B: 2 2 5 5 -> 1 cambio; el 5 es de A y de B: mixto
    asign = _a(A=enumerate([1, 1, 1, 5, 5, 1]), B=enumerate([2, 2, 5, 5]))
    m = mt.metricas(asign, {"A": set(range(6)), "B": set(range(4))}, FPS)
    assert m["A"]["cambios"] == 2 and m["A"]["tracks_distintos"] == 2
    assert m["B"]["cambios"] == 1 and m["B"]["tracks_distintos"] == 2
    assert m["tracks_mixtos"] == 1 and m["cuadros_en_tracks_mixtos"] == 4
    assert m["A"]["cambios_por_minuto"] == pytest.approx(2 / (6 / FPS / 60))


def test_un_cuadro_fuera_del_denominador_no_cuenta(gemelo):
    import medir_trackeo as mt
    asign = _a(A=enumerate([1, 1, 9, 1]), B={})
    m = mt.metricas(asign, {"A": {0, 1, 3}, "B": set()}, FPS)
    assert m["A"]["cambios"] == 0 and m["A"]["tracks_distintos"] == 1 and m["A"]["cuadros"] == 3


def test_el_denominador_es_la_interseccion(gemelo):
    import medir_trackeo as mt
    base = _a(A={0: 1, 1: 1, 2: 1, 3: 1}, B={0: 2, 1: 2})
    y26 = _a(A={1: 7, 2: 7, 3: 8, 4: 8}, B={1: 9, 2: 9})
    r = mt.comparar(base, y26, FPS)
    assert r["denominador"] == {"A": 3, "B": 1}
    assert r["base"]["A"]["cuadros"] == r["y26"]["A"]["cuadros"] == 3
    assert r["y26"]["A"]["cambios"] == 1 and r["base"]["A"]["cambios"] == 0


def test_v8_contra_si_mismo_da_cero_diferencias(gemelo, tmp_path):
    import medir_trackeo as mt
    from boxtwin.core.posecache import PoseCache
    A, B = caja(300), caja(800)
    cuadros = [[(1, A), (2, B)] if f < 6 else [(3, A), (2, B)] for f in range(12)]
    npz = cache(tmp_path / "c" / "spar.pose.npz", cuadros)
    doc = documento(npz, 12, [(1, "A", 0, 6), (3, "A", 6, 12), (2, "B", 0, 12)])
    c = PoseCache.open(npz)
    r = mt.medir_fuente(doc, c, c, c)
    assert r["base"] == r["y26"]
    assert all(v == 0 for v in r["diferencia"].values())
    assert r["base"]["A"]["cambios"] == 1 and r["tracks_totales"] == {"base": 3, "y26": 3}


def test_total_suma_las_fuentes(gemelo):
    import medir_trackeo as mt
    una = mt.comparar(_a(A=enumerate([1, 1, 2]), B=enumerate([3, 3])),
                      _a(A=enumerate([5, 5, 5]), B=enumerate([6, 7])), FPS)
    tot = mt.total([una, una], FPS)
    assert tot["base"]["A"]["cambios"] == 2 and tot["y26"]["B"]["cambios"] == 2
    assert tot["denominador"] == {"A": 6, "B": 4}


def test_k3_fraccion_de_munecas_con_puntaje(gemelo, tmp_path):
    """K3: munecas (9 y 10) con puntaje >= 0,3 en los cuadros del denominador comun."""
    import medir_trackeo as mt
    from boxtwin.core.posecache import PoseCache
    A, B = caja(300), caja(800)
    npz = cache(tmp_path / "c" / "spar.pose.npz", [[(1, A), (2, B)] for _ in range(4)])
    doc = documento(npz, 4, [(1, "A", 0, 4), (2, "B", 0, 4)])
    c = PoseCache.open(npz)
    r = mt.medir_fuente(doc, c, c, c)
    assert r["k3"] == {"base": 1.0, "y26": 1.0, "diferencia": 0.0, "munecas": 16}
    assert mt.fraccion_munecas(c, {"A": {0: 1}, "B": {}}, {"A": {0}, "B": set()}, umbral=0.95) == (0.0, 2)
