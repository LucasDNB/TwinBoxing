import torch

from modelo import ClasificadorGolpe


def test_forma_de_salida():
    m = ClasificadorGolpe(26).eval()
    lt, la = m(torch.randn(5, 32, 26), torch.ones(5, 32))
    assert lt.shape == (5, 3) and la.shape == (5, 2)


def test_la_mascara_importa():
    torch.manual_seed(0)
    m = ClasificadorGolpe(26).eval()
    x = torch.randn(4, 32, 26)
    mask = torch.ones(4, 32)
    mask[:, 10:20] = 0
    mask[2] = 0                                   # un golpe sin pasos usables
    a = m(x, mask)
    x2 = x.clone()
    x2[:, 10:20] = 1000 * torch.randn(4, 10, 26)  # cambiar solo los pasos con M = 0
    x2[2] = torch.randn(32, 26)
    b = m(x2, mask)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
    # y cambiar un paso usable si cambia la salida
    x3 = x.clone()
    x3[0, 5] += 10
    assert not torch.allclose(m(x3, mask)[0][0], a[0][0])
    assert torch.isfinite(a[0]).all()
