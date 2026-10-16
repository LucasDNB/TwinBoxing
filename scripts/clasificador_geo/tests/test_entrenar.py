import json

import numpy as np
import torch

import entrenar


def _separable(n=96, T=32, F=6, seed=0):
    rng = np.random.default_rng(seed)
    yt = np.arange(n) % 3
    ya = (np.arange(n) // 3) % 2
    X = rng.normal(0, 0.3, (n, T, F)).astype(np.float32)
    X[:, :, 0] += 3 * yt[:, None]
    X[:, :, 1] += 3 * ya[:, None]
    M = np.ones((n, T), bool)
    return X, M, yt, ya


def test_pesos_de_clase_media_uno():
    w = entrenar.pesos_de_clase(np.array([0, 0, 0, 1, 2, 2]), 3)
    assert np.isclose(w.mean(), 1.0) and w[1] > w[2] > w[0]


def test_sobreajusta_un_lote_separable():
    X, M, yt, ya = _separable()
    idx = np.arange(len(X))
    modelo, media, desvio = entrenar.entrenar_modelo(X, M, yt, ya, idx, semilla=0, epocas=60)
    pt, pa = entrenar.predecir(modelo, X, M, media, desvio)
    assert (pt.argmax(1) == yt).mean() == 1.0
    assert (pa.argmax(1) == ya).mean() == 1.0


def _trabajo_sintetico(trabajo, n_por_fuente=40):
    fuentes = ["F1", "F2", "F3"]
    X, M, yt, ya = _separable(n=n_por_fuente * 3)
    d = trabajo / "datos"
    d.mkdir(parents=True)
    fuente = np.repeat(fuentes, n_por_fuente)
    np.savez(d / "golpes.npz", X=np.concatenate([X, np.zeros(X.shape[:2] + (20,), np.float32)], -1),
             M=M, y_tipo=yt, y_altura=ya, y_rol=yt % 2,
             frame_dir=np.array([f"{f}_ev_{i:04d}" for i, f in enumerate(fuente)]), fuente=fuente)
    (d / "golpes2d.meta.json").write_text(json.dumps({"features": [f"f{i}" for i in range(26)]}))
    return fuentes


def test_misma_semilla_mismo_pred_en_cpu(trabajo):
    _trabajo_sintetico(trabajo)
    d = entrenar.cargar_entrada("E4", trabajo)
    a = entrenar.correr("E4", "F1", 0, d, "cpu", trabajo)
    pa = dict(np.load(a / "pred.npz"))
    (a / "pred.npz").unlink()
    entrenar.correr("E4", "F1", 0, d, "cpu", trabajo)
    pb = dict(np.load(a / "pred.npz"))
    assert np.array_equal(pa["p_tipo"], pb["p_tipo"]) and np.array_equal(pa["p_altura"], pb["p_altura"])
    assert all(n.startswith("F1_") for n in pa["frame_dir"]) and len(pa["frame_dir"]) == 40
    cfg = json.loads((a / "config.json").read_text())
    assert cfg["n_train"] == 80 and cfg["hiper"]["epocas"] == 60


def test_reanudable(trabajo):
    _trabajo_sintetico(trabajo)
    d = entrenar.cargar_entrada("E4a", trabajo)
    a = entrenar.correr("E4a", "F2", 1, d, "cpu", trabajo)
    t = (a / "pred.npz").stat().st_mtime_ns
    entrenar.correr("E4a", "F2", 1, d, "cpu", trabajo)
    assert (a / "pred.npz").stat().st_mtime_ns == t
    assert d["X"].shape[-1] == 20


def test_la_fuente_de_prueba_no_entra_al_entrenamiento_ni_a_la_estandarizacion():
    X, M, yt, ya = _separable(n=90)
    fuente = np.repeat(["F1", "F2", "F3"], 30)
    idx_train = np.flatnonzero(fuente != "F1")
    m1, med1, des1 = entrenar.entrenar_modelo(X, M, yt, ya, idx_train, semilla=3, epocas=3)
    X2, yt2, ya2, M2 = X.copy(), yt.copy(), ya.copy(), M.copy()
    X2[fuente == "F1"] = 1e3                      # la fuente de prueba cambia entera
    yt2[fuente == "F1"] = 0
    ya2[fuente == "F1"] = 1
    M2[fuente == "F1", :5] = False
    m2, med2, des2 = entrenar.entrenar_modelo(X2, M2, yt2, ya2, idx_train, semilla=3, epocas=3)
    assert np.array_equal(med1, med2) and np.array_equal(des1, des2)
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k


def test_crop_resize_de_largo_completo_es_identidad():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1).repeat(2, 1, 3)
    m = torch.ones(2, 32)
    g = torch.Generator().manual_seed(0)
    xn, mn = entrenar.crop_resize(x, m, g, lo=1.0, hi=1.0)
    assert torch.allclose(xn, x, atol=1e-5) and torch.equal(mn, m)
    xn, _ = entrenar.crop_resize(x, m, g, lo=0.8, hi=0.8)
    paso = xn[0, 1, 0] - xn[0, 0, 0]
    assert torch.allclose(paso, torch.tensor(0.8), atol=1e-4)      # sigue siendo una rampa, mas lenta
