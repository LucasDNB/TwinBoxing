import numpy as np
import pytest

import elevar3d


def test_huecos_cortos_se_interpolan_y_largos_quedan_con_mascara_cero():
    T = 40
    kp = np.zeros((T, 17, 2), np.float32)
    kp[:, :, 0] = np.arange(T)[:, None]          # x = cuadro
    presente = np.ones(T, bool)
    presente[5:10] = False                       # 5 cuadros: se interpola
    presente[20:32] = False                      # 12 cuadros: no
    presente[:2] = False                         # borde: no
    kp[~presente] = -99
    r, m = elevar3d.rellenar_huecos(kp, presente)
    assert np.allclose(r[5:10, 0, 0], np.arange(5, 10)) and m[5:10].all()
    assert not m[20:32].any() and not m[:2].any()
    assert np.allclose(r[20:26, 0, 0], 19) and np.allclose(r[26:32, 0, 0], 32)   # vecino mas cercano
    assert np.allclose(r[:2, 0, 0], 2)
    assert (r != -99).all()


def test_contexto_centrado_con_borde_y_paso():
    crudo, idx = elevar3d.indices_contexto(10, 40, T=1000, paso=1)
    assert len(crudo) == 243 and crudo[121] == 25 and idx.min() == 0 and crudo[0] < 0
    crudo, idx = elevar3d.indices_contexto(500, 540, T=1000, paso=2)
    assert crudo[1] - crudo[0] == 2 and crudo[0] <= 500 and crudo[-1] >= 540
    assert elevar3d.paso_de(59.94) == 2 and elevar3d.paso_de(29.97) == 1


def test_a_ventana_interpola_en_cuadros():
    crudo = np.arange(0, 243 * 2, 2)
    X = np.zeros((243, 17, 3), np.float32)
    X[..., 0] = crudo[:, None]                   # x = numero de cuadro
    out = elevar3d.a_ventana(X, crudo, 101, 132)
    assert out.shape == (32, 17, 3) and np.allclose(out[:, 0, 0], np.linspace(101, 132, 32))


def test_cv_huesos_de_brazo_rigido_es_cero():
    P = np.zeros((32, 17, 3), np.float32)
    ang = np.linspace(0, 1, 32)
    P[:, 11] = 0
    P[:, 12] = np.stack([np.cos(ang), np.sin(ang), 0 * ang], -1)
    P[:, 13] = P[:, 12] + np.stack([np.cos(2 * ang), 0 * ang, np.sin(2 * ang)], -1) * 0.8
    ant, bra = elevar3d.cv_huesos(P, np.ones(32, bool), "left")
    assert ant < 1e-5 and bra < 1e-5


def test_elevador_con_mmpose():
    pytest.importorskip("mmpose")
    pytest.skip("la elevacion real se corre en boxtwin_mmaction (G7), no en las pruebas")
