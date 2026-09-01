import numpy as np
import pytest

from boxtwin_detector.features import (
    MIN_SCORE, NOMBRES, N_FEATURES, espejar, features_de,
)

L_HOM, R_HOM, L_COD, L_MUN = 5, 6, 7, 9


def esqueleto(T=30, semilla=0):
    rng = np.random.default_rng(semilla)
    kp = rng.normal(size=(T, 17, 2)) * 30 + 250
    # hombros bien separados, para que la escala no sea degenerada
    kp[:, L_HOM] = kp[:, R_HOM] + np.array([60.0, 0.0])
    return kp, np.ones((T, 17))


def test_cantidad_de_features_coincide_con_los_nombres():
    kp, sc = esqueleto()
    f, _ = features_de(kp, sc, "left")
    assert f.shape == (len(kp), N_FEATURES) == (len(kp), len(NOMBRES))


def test_invariante_a_la_traslacion():
    kp, sc = esqueleto()
    a, _ = features_de(kp, sc, "left")
    b, _ = features_de(kp + np.array([137.0, -412.0]), sc, "left")
    assert np.abs(a - b).max() < 1e-4


def test_invariante_a_la_escala():
    kp, sc = esqueleto()
    a, _ = features_de(kp, sc, "left")
    b, _ = features_de(kp * 3.7, sc, "left")
    assert np.abs(a - b).max() < 1e-4


def test_el_brazo_derecho_es_el_izquierdo_del_esqueleto_espejado():
    # Es la razon de ser del espejado: un jab y un cross tienen que verse iguales.
    kp, sc = esqueleto(semilla=3)
    der, ok_d = features_de(kp, sc, "right")
    izq, ok_i = features_de(*espejar(kp, sc), "left")
    assert np.abs(der - izq).max() == 0.0
    assert (ok_d == ok_i).all()


def test_espejar_dos_veces_es_la_identidad():
    kp, sc = esqueleto()
    kp2, sc2 = espejar(*espejar(kp, sc))
    assert np.abs(kp - kp2).max() == 0.0
    assert np.abs(sc - sc2).max() == 0.0


def test_score_bajo_marca_el_cuadro_y_sus_vecinos():
    kp, sc = esqueleto(T=10)
    sc[5, L_MUN] = MIN_SCORE - 0.01
    f, ok = features_de(kp, sc, "left")
    assert not ok[5]
    # los vecinos caen porque su derivada cruza el agujero
    assert not ok[4] and not ok[6]
    assert ok[3] and ok[7]
    assert np.abs(f[5]).max() == 0.0


def test_hombros_colapsados_no_produce_nan():
    kp, sc = esqueleto(T=8)
    kp[3, L_HOM] = kp[3, R_HOM]
    f, ok = features_de(kp, sc, "left")
    assert np.isfinite(f).all()
    assert not ok[3]


def test_todo_en_cero_no_produce_nan():
    f, ok = features_de(np.zeros((6, 17, 2)), np.ones((6, 17)), "left")
    assert np.isfinite(f).all()
    assert not ok.any()


def test_el_angulo_de_hombros_no_salta_al_dar_la_vuelta():
    # Codificado como (cos, sin) justamente para esto: a -pi y a +pi el cuerpo esta igual.
    T = 361
    ang = np.deg2rad(np.arange(T))
    kp = np.zeros((T, 17, 2))
    for i in range(17):
        kp[:, i] = [0.0, 0.0]
    kp[:, R_HOM] = np.stack([-30 * np.cos(ang), -30 * np.sin(ang)], -1)
    kp[:, L_HOM] = np.stack([30 * np.cos(ang), 30 * np.sin(ang)], -1)
    kp[:, L_COD] = kp[:, L_HOM] * 1.3
    kp[:, L_MUN] = kp[:, L_HOM] * 1.6
    f, ok = features_de(kp, np.ones((T, 17)), "left")
    i_cos, i_sin = NOMBRES.index("hombros_cos"), NOMBRES.index("hombros_sin")
    salto = np.abs(np.diff(f[:, [i_cos, i_sin]], axis=0)).max()
    assert salto < 0.05, f"salto de {salto} en la codificacion del angulo"


def test_extension_crece_al_estirar_el_brazo():
    T = 20
    kp, sc = esqueleto(T=T)
    for t in range(T):
        kp[t, L_MUN] = kp[t, L_HOM] + np.array([0.0, -20.0 - 5.0 * t])
    f, _ = features_de(kp, sc, "left")
    ext = f[:, NOMBRES.index("extension")]
    assert (np.diff(ext) > 0).all()


def test_brazo_invalido_es_error():
    kp, sc = esqueleto()
    with pytest.raises(ValueError):
        features_de(kp, sc, "izquierda")


def test_un_solo_cuadro_no_rompe_las_derivadas():
    kp, sc = esqueleto(T=1)
    f, ok = features_de(kp, sc, "left")
    assert f.shape == (1, N_FEATURES)
    assert np.isfinite(f).all()
