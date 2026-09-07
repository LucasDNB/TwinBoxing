import pytest

torch = pytest.importorskip("torch")

from boxtwin_detector.modelo import DILATACIONES, TCN, campo_receptivo


def test_campo_receptivo_cubre_dos_segundos():
    # 63 cuadros a 30 fps son 2,1 s, ~9 veces la duracion mediana de un golpe
    assert campo_receptivo() == 63
    assert campo_receptivo((1, 2)) == 7


def test_la_salida_conserva_el_largo():
    m = TCN(n_features=20).eval()
    x = torch.randn(2, 20, 300)
    assert m(x).shape == (2, 3, 300)


def test_anda_con_secuencias_mas_cortas_que_el_campo_receptivo():
    m = TCN(n_features=20).eval()
    assert m(torch.randn(1, 20, 10)).shape == (1, 3, 10)


def test_forma_de_entrada_equivocada_es_error():
    m = TCN(n_features=20)
    with pytest.raises(ValueError, match="batch, features, tiempo"):
        m(torch.randn(20, 300))
    with pytest.raises(ValueError, match="20 features"):
        m(torch.randn(1, 7, 300))


def test_ve_el_futuro_y_el_pasado():
    # Es la razon de que sea no causal: la decision sobre el cuadro 161 depende del 166.
    m = TCN(n_features=4, canales=8).eval()
    T, centro = 200, 100
    x = torch.zeros(1, 4, T)
    base = m(x)[0, :, centro].clone()

    adelante = x.clone()
    adelante[0, :, centro + 5] = 10.0
    assert not torch.allclose(m(adelante)[0, :, centro], base, atol=1e-5)

    atras = x.clone()
    atras[0, :, centro - 5] = 10.0
    assert not torch.allclose(m(atras)[0, :, centro], base, atol=1e-5)


def test_fuera_del_campo_receptivo_no_influye():
    m = TCN(n_features=4, canales=8).eval()
    T, centro = 400, 200
    r = m.campo_receptivo // 2
    x = torch.zeros(1, 4, T)
    base = m(x)[0, :, centro].clone()
    lejos = x.clone()
    lejos[0, :, centro + r + 5] = 100.0
    assert torch.allclose(m(lejos)[0, :, centro], base, atol=1e-5)


def test_es_determinista_en_evaluacion():
    m = TCN(n_features=20).eval()
    x = torch.randn(1, 20, 100)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))


def test_los_gradientes_llegan_a_la_primera_capa():
    m = TCN(n_features=20)
    m(torch.randn(2, 20, 128)).sum().backward()
    g = m.entrada.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
