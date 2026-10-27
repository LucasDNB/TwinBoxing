import numpy as np

import representacion as rp


def _coco(T=5, semilla=0):
    rng = np.random.default_rng(semilla)
    base = np.array([[50, 10], [48, 8], [52, 8], [45, 9], [55, 9],        # cara
                     [40, 30], [60, 30], [35, 45], [65, 45], [30, 60], [70, 60],  # brazos
                     [44, 70], [56, 70], [43, 90], [57, 90], [42, 110], [58, 110]], np.float32)
    return base[None] + rng.normal(0, 0.5, (T, 17, 2)).astype(np.float32)


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _h36m(rng):
    P = rng.normal(0, 0.3, (17, 3)).astype(np.float32)
    P[1], P[4] = [0.15, 0, 0], [-0.15, 0, 0]        # caderas d / i
    P[14], P[11] = [0.2, 0, 0.5], [-0.2, 0, 0.5]    # hombros d / i
    return P


def test_coco_a_14_usa_el_orden_de_boxmind():
    kp = _coco(1)
    q = rp.coco_a_14(kp)
    assert q.shape == (1, 14, 2)
    np.testing.assert_allclose(q[0, rp.HOMBRO_D], kp[0, 6])
    np.testing.assert_allclose(q[0, rp.MUNECA_I], kp[0, 9])
    np.testing.assert_allclose(q[0, rp.CADERA_D], kp[0, 12])
    np.testing.assert_allclose(q[0, 1], 0.5 * (kp[0, 5] + kp[0, 6]))


def test_cabeza_ignora_puntos_de_score_bajo_y_cae_a_la_nariz():
    kp = _coco(2)
    sc = np.ones((2, 17), np.float32)
    sc[0, 1:5] = 0.0
    sc[1, 0:5] = 0.0
    q = rp.coco_a_14(kp, sc)
    np.testing.assert_allclose(q[0, 0], kp[0, 0])
    np.testing.assert_allclose(q[1, 0], kp[1, 0])     # ninguno pasa: la nariz sola


def test_h36m_a_14():
    P = np.arange(17 * 3, dtype=np.float32).reshape(1, 17, 3)
    q = rp.h36m_a_14(P)
    np.testing.assert_allclose(q[0, 0], P[0, 10])
    np.testing.assert_allclose(q[0, rp.HOMBRO_D], P[0, 14])
    np.testing.assert_allclose(q[0, rp.CADERA_I], P[0, 4])


def test_marco_cuerpo_no_depende_de_rotacion_traslacion_ni_escala():
    rng = np.random.default_rng(1)
    P = np.stack([_h36m(rng) for _ in range(4)])
    P14 = rp.h36m_a_14(P)
    R, t, s = _rot(rng), rng.normal(size=3), 3.7
    Q14 = s * P14 @ R.T + t
    a, b = rp.marco_cuerpo(P14), rp.marco_cuerpo(Q14)
    np.testing.assert_allclose(a, b, atol=1e-4)
    origen = 0.5 * (a[:, rp.CADERA_D] + a[:, rp.CADERA_I])
    np.testing.assert_allclose(origen, 0, atol=1e-5)
    hombros = 0.5 * (a[:, rp.HOMBRO_D] + a[:, rp.HOMBRO_I])
    np.testing.assert_allclose(hombros[:, :2], 0, atol=1e-5)    # el torso queda sobre z
    assert (hombros[:, 2] > 0).all()
    assert (a[:, rp.CADERA_D, 0] > 0).all()                     # cadera derecha en x positivo


def test_norm2d_deja_al_atacante_dentro_del_recorte():
    kp = _coco(6)
    q = rp.coco_a_14(kp)
    caja = rp.caja_atacante(q, 1000, 1000)
    n = rp.norm2d(q, caja)[:, :rp.CADERA_D + 1]    # piernas fuera del recorte, como en BoxMind
    assert n.min() >= 0 and n.max() <= 1


def test_caja_degenerada_da_none():
    q = np.zeros((3, 14, 2), np.float32)
    assert rp.caja_atacante(q, 640, 360) is None


def test_espejar_dos_veces_es_la_identidad_y_cambia_lados():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(7, 28, 5)).astype(np.float32)
    np.testing.assert_allclose(rp.espejar(rp.espejar(x)), x, atol=1e-6)
    y = rp.espejar(x)
    np.testing.assert_allclose(y[:, rp.HOMBRO_I, 1], x[:, rp.HOMBRO_D, 1])
    np.testing.assert_allclose(y[:, rp.HOMBRO_I, 0], 1 - x[:, rp.HOMBRO_D, 0])
    np.testing.assert_allclose(y[:, 14 + rp.HOMBRO_I, 2], -x[:, 14 + rp.HOMBRO_D, 2])


def test_ejemplo_forma_y_ceros_sin_3d():
    kp = _coco(5)
    x = rp.ejemplo(kp, None, kp + 30, None, None, None, 640, 360)
    assert x.shape == (5, 28, 5)
    assert np.all(x[..., 2:] == 0) and np.any(x[..., :2] != 0)
    rng = np.random.default_rng(3)
    P = np.stack([_h36m(rng) for _ in range(5)])
    x3 = rp.ejemplo(kp, None, kp + 30, None, P, P, 640, 360)
    assert np.any(x3[:, :14, 2:] != 0) and np.any(x3[:, 14:, 2:] != 0)
