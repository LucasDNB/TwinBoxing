import numpy as np

import features3d


def _esqueleto(T=32, seed=0):
    """Esqueleto H36M de pie con eje vertical y; el brazo izquierdo hace un arco."""
    rng = np.random.default_rng(seed)
    P = np.zeros((T, 17, 3))
    P[:, 0] = (0, 0, 0)
    P[:, 1] = (-0.15, 0, 0); P[:, 4] = (0.15, 0, 0)
    P[:, 2] = (-0.15, -0.5, 0); P[:, 5] = (0.15, -0.5, 0)
    P[:, 3] = (-0.15, -1, 0); P[:, 6] = (0.15, -1, 0)
    P[:, 7] = (0, 0.25, 0); P[:, 8] = (0, 0.5, 0); P[:, 9] = (0, 0.6, 0); P[:, 10] = (0, 0.7, 0)
    P[:, 11] = (0.2, 0.5, 0); P[:, 14] = (-0.2, 0.5, 0)
    t = np.linspace(0, np.pi / 2, T)
    P[:, 12] = P[:, 11] + np.stack([0.25 * np.cos(t), -0.1 + 0 * t, 0.25 * np.sin(t)], -1)
    P[:, 13] = P[:, 12] + np.stack([-0.25 * np.sin(t), 0.05 + 0 * t, 0.25 * np.cos(t)], -1)
    P[:, 15] = P[:, 14] + (-0.1, -0.25, 0.05)
    P[:, 16] = P[:, 15] + (0.05, 0.25, 0.1)
    return P + rng.normal(0, 1e-3, P.shape)


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_invariancia_a_rotacion_alrededor_del_eje_vertical():
    P = _esqueleto()
    f0 = features3d.features_de(P, "left", 0.5)
    for th in (0.3, 1.2, np.pi, -2.0):
        Q = P @ _rot_y(th).T + np.array([3.0, 0.1, -2.0])     # rotacion y traslacion
        assert np.allclose(features3d.features_de(Q, "left", 0.5), f0, atol=1e-4)
    # y escala
    assert np.allclose(features3d.features_de(2.5 * P, "left", 0.5), f0, atol=1e-4)


def test_espejo_de_brazo_derecho_da_las_del_izquierdo():
    P = _esqueleto(seed=1)
    Q = features3d.espejar(P)               # el mismo golpe dado con el brazo derecho
    assert np.allclose(features3d.features_de(Q, "right", 0.5), features3d.features_de(P, "left", 0.5), atol=1e-5)
    # y rotado ademas
    Q2 = Q @ _rot_y(0.8).T
    assert np.allclose(features3d.features_de(Q2, "right", 0.5), features3d.features_de(P, "left", 0.5), atol=1e-4)


def test_cantidad_y_valores_basicos():
    P = _esqueleto()
    f = features3d.features_de(P, "left", 0.5)
    assert f.shape == (32, len(features3d.NOMBRES)) == (32, 16)
    # hombro izquierdo: a la izquierda (x > 0) y a la altura del torax (y = 0)
    assert np.all(f[:, 6] > 0) and np.allclose(f[:, 7], 0, atol=1e-2)
    # el arco del codo hace girar el azimut: velocidad de azimut no nula
    assert np.abs(f[:, 15]).mean() > 0.1


def test_construir_pone_ceros_sin_3d():
    P3 = np.stack([_esqueleto(), _esqueleto(seed=2)])
    M3 = np.ones((2, 32), bool)
    M3[1, :10] = False
    gs = [{"side": "left", "ini": 0, "fin": 15, "fps": 30.0}, {"side": "right", "ini": 0, "fin": 15, "fps": 30.0}]
    X = features3d.construir(gs, P3, M3)
    assert X.shape == (2, 32, 16) and np.all(X[1, :10] == 0) and np.any(X[1, 10:] != 0)
