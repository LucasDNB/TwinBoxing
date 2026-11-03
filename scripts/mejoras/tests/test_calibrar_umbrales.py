"""T2: la regla de E8b §2 (igualar la fraccion aceptada por articulacion) y K-cal."""

import numpy as np
import pytest

import calibrar_umbrales as cu

ARTIC = ("nariz", "hombro", "codo", "muneca", "cadera")


def _poblaciones(n=20000, semilla=0):
    rng = np.random.default_rng(semilla)
    v8 = {a: rng.uniform(0, 1, n) for a in ARTIC}                    # fraccion >= 0,3: 0,70
    y26 = {a: rng.uniform(0, 1, n) ** (1 + i) for i, a in enumerate(ARTIC)}   # cada una corrida distinto
    return v8, y26


def test_umbral_con_cuantil_conocido():
    # y26 uniforme al cuadrado: P(u^2 >= t) = 1 - sqrt(t). Para igualar 0,70: t = 0,09
    v8, y26 = _poblaciones()
    r = cu.calibrar(v8, y26)
    assert r["umbrales"]["hombro"] == pytest.approx(0.09, abs=0.01)
    assert r["umbrales"]["nariz"] == pytest.approx(0.30, abs=0.01)
    assert r["fraccion_v8"]["muneca"] == pytest.approx(0.70, abs=0.01)


def test_k_cal_pasa_en_el_sintetico():
    v8, y26 = _poblaciones()
    r = cu.calibrar(v8, y26)
    k = cu.k_cal(y26, r["umbrales"], r["fraccion_v8"])
    assert k["pasa"]
    for a in ARTIC:
        assert abs(k["por_articulacion"][a]["diferencia"]) <= 0.005


def test_k_cal_falla_con_umbrales_de_mas():
    v8, y26 = _poblaciones()
    r = cu.calibrar(v8, y26)
    malos = {**r["umbrales"], "muneca": r["umbrales"]["muneca"] + 0.1}
    assert not cu.k_cal(y26, malos, r["fraccion_v8"])["pasa"]


def test_un_umbral_por_encima_de_0_3_se_usa_igual():
    rng = np.random.default_rng(1)
    v8 = {a: rng.uniform(0, 1, 5000) for a in ARTIC}
    y26 = {a: rng.uniform(0, 1, 5000) ** 0.5 for a in ARTIC}       # puntua mas alto que v8
    r = cu.calibrar(v8, y26)
    assert all(u > 0.3 for u in r["umbrales"].values())


def test_con_empates_la_fraccion_queda_cerca():
    v8 = {a: np.repeat([0.1, 0.5, 0.9], 1000) for a in ARTIC}      # 2/3 aceptadas
    y26 = {a: np.round(np.linspace(0, 1, 3000), 3) for a in ARTIC}
    r = cu.calibrar(v8, y26)
    assert cu.k_cal(y26, r["umbrales"], r["fraccion_v8"])["pasa"]


def test_puntajes_junta_izquierda_y_derecha():
    class Dets:
        def __init__(self, sc):
            self.kp_score = sc

        def index_of_track(self, t):
            return 0 if t == 7 else None

    class Cache:
        def detections(self, f):
            return Dets(np.arange(17, dtype=np.float32)[None] / 100)

    asign = {"A": {0: 7, 1: 7}, "B": {0: 9}}
    frames = {"A": {0, 1}, "B": {0}}
    p = cu.puntajes(Cache(), asign, frames)
    assert sorted(p["muneca"].tolist()) == pytest.approx([0.09, 0.09, 0.10, 0.10])
    assert len(p["nariz"]) == 2                                       # B no tiene el track
