"""T4 (mejoras post-E8, E10 §2): relleno de huecos cortos bajo umbral."""

import numpy as np
import pytest

from boxtwin_detector.features import ARTICULACIONES
from boxtwin_detector.relleno import maximo_cuadros, rellenar

L_MUN = 9
U = {a: 0.3 for a in ARTICULACIONES}


def serie(T=40, semilla=0):
    rng = np.random.default_rng(semilla)
    kp = rng.normal(size=(T, 17, 2)) * 20 + 200
    sc = np.ones((T, 17))
    return kp, sc


def hueco(sc, a, n, j=L_MUN):
    sc[a:a + n, j] = 0.1


@pytest.mark.parametrize("fps,largo,se_rellena", [
    (30, 3, True), (30, 4, False), (60, 6, True), (60, 7, False), (30, 1, True),
])
def test_tope_de_0_10_s(fps, largo, se_rellena):
    kp, sc = serie()
    hueco(sc, 10, largo)
    kp2, sc2, r = rellenar(kp, sc, U, fps)
    assert r[10:10 + largo, L_MUN].all() == se_rellena
    assert r.sum() == (largo if se_rellena else 0)
    if se_rellena:
        assert (sc2[10:10 + largo, L_MUN] == 0.3).all()
    else:
        assert np.array_equal(sc2, sc) and np.array_equal(kp2, kp)


def test_el_tope_va_al_entero_mas_cercano():
    # decision 9: 3 cuadros a 29,97 y 6 a 59,94, como 30 y 60; hacia abajo daban 2 y 5
    assert maximo_cuadros(30, 0.10) == 3
    assert maximo_cuadros(60, 0.10) == 6
    assert maximo_cuadros(29.97, 0.10) == 3
    assert maximo_cuadros(59.94, 0.10) == 6
    assert maximo_cuadros(30, 0.20) == 6


@pytest.mark.parametrize("fps,largo,se_rellena", [(29.97, 3, True), (29.97, 4, False),
                                                   (59.94, 6, True), (59.94, 7, False)])
def test_tope_a_29_97_y_59_94(fps, largo, se_rellena):
    kp, sc = serie()
    hueco(sc, 10, largo)
    _, _, r = rellenar(kp, sc, U, fps)
    assert r.any() == se_rellena


def test_interpolacion_lineal_entre_los_extremos():
    kp, sc = serie()
    kp[9, L_MUN] = [100.0, 0.0]
    kp[13, L_MUN] = [140.0, 40.0]
    hueco(sc, 10, 3)
    kp2, _, _ = rellenar(kp, sc, U, 30)
    np.testing.assert_allclose(kp2[10:13, L_MUN], [[110, 10], [120, 20], [130, 30]])


@pytest.mark.parametrize("inicio", [0, 37])
def test_huecos_abiertos_no_se_rellenan(inicio):
    kp, sc = serie()
    hueco(sc, inicio, 3)
    _, _, r = rellenar(kp, sc, U, 30)
    assert not r.any()


def test_las_coordenadas_medidas_no_cambian():
    kp, sc = serie()
    sc[5, 7] = 0.2                    # un codo bajo umbral fuera de cualquier hueco de muneca
    hueco(sc, 10, 2)
    kp2, sc2, r = rellenar(kp, sc, U, 30)
    fuera = ~r
    assert np.array_equal(kp2[fuera], kp[fuera])
    assert np.array_equal(sc2[fuera], sc[fuera])
    assert r[5, 7]                    # el codo tambien era un hueco de 1, y se rellena


def test_hueco_con_la_deteccion_ausente_no_se_rellena():
    kp, sc = serie()
    hueco(sc, 10, 3)
    presente = np.ones(len(kp), bool)
    presente[11] = False
    _, _, r = rellenar(kp, sc, U, 30, presente=presente)
    assert not r.any()


def test_extremo_sin_deteccion_no_cuenta_como_aceptado():
    kp, sc = serie()
    hueco(sc, 10, 3)
    presente = np.ones(len(kp), bool)
    presente[13] = False
    _, _, r = rellenar(kp, sc, U, 30, presente=presente)
    assert not r.any()


def test_cada_articulacion_con_su_umbral():
    kp, sc = serie()
    sc[10, L_MUN] = 0.4
    _, _, r = rellenar(kp, sc, U, 30)
    assert not r.any()
    _, _, r = rellenar(kp, sc, {**U, "muneca": 0.5}, 30)
    assert r[10, L_MUN] and r.sum() == 1


def test_no_toca_articulaciones_que_no_usan_las_features():
    kp, sc = serie()
    sc[10, 15] = 0.1                  # tobillo
    _, _, r = rellenar(kp, sc, U, 30)
    assert not r.any()


def test_no_modifica_las_entradas():
    kp, sc = serie()
    hueco(sc, 10, 2)
    k0, s0 = kp.copy(), sc.copy()
    rellenar(kp, sc, U, 30)
    assert np.array_equal(kp, k0) and np.array_equal(sc, s0)


def test_en_float32_el_cuadro_rellenado_queda_aceptado():
    kp, sc = serie()
    sc = sc.astype(np.float32)
    umbral = 0.2147
    sc[10, L_MUN] = 0.1
    _, sc2, r = rellenar(kp, sc, {**U, "muneca": umbral}, 30)
    assert r[10, L_MUN] and float(sc2[10, L_MUN]) >= umbral
