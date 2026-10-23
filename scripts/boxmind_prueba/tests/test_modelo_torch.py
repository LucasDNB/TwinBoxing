"""
Humo del entrenamiento con la arquitectura real de BoxMind: forma de las salidas, una corrida
corta de punta a punta con conjuntos sinteticos y el formato de pred.npz que lee evaluar.py.
Se saltea sin torch o sin el clon.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
CLON = Path(os.environ.get("BOXTWIN_BOXMIND_REPO", "~/externos/BoxingWeb")).expanduser()
pytestmark = pytest.mark.skipif(not (CLON / "model" / "mfusion.py").is_file(), reason="sin el clon de BoxMind")


def _conjunto(fuentes, n_por, L, rng, con_bw=False):
    fs = [f for f in fuentes for _ in range(n_por)]
    n = len(fs)
    largo = rng.integers(4, L + 1, n)
    X = np.zeros((n, L, 28, 5), np.float32)
    for i, l in enumerate(largo):
        X[i, :l] = rng.normal(size=(l, 28, 5))
    return {"X": X, "largo": largo, "frame_dir": np.array([f"{f}_{i}" for i, f in enumerate(fs)]),
            "fuente": np.array(fs), "particion": np.array(["train" if i % 4 else "test" for i in range(n)] if con_bw else [""] * n),
            "y_tipo": rng.integers(0, 3, n), "y_altura": rng.integers(0, 2, n),
            "y_distancia": rng.integers(0, 3, n) if con_bw else -np.ones(n, int),
            "y_efecto": rng.integers(0, 2, n) if con_bw else -np.ones(n, int)}


def test_salidas_de_mfusion(trabajo, monkeypatch):
    monkeypatch.setenv("BOXTWIN_BOXMIND_REPO", str(CLON))
    import importlib

    import bm
    importlib.reload(bm)
    import entrenar
    importlib.reload(entrenar)
    m = entrenar.construir_modelo().eval()
    x = torch.randn(3, 11, 28, 5)
    tech, tgt, dis, eff, aux = m(None, x, torch.zeros(3, 1))
    assert tech.shape == (3, 3) and tgt.shape == (3, 3) and dis.shape == (3, 3) and eff.shape == (3, 1)
    torch.testing.assert_close(tech.sum(1), torch.ones(3))


def test_perdida_ignora_etiquetas_ausentes(trabajo):
    import entrenar
    p = torch.softmax(torch.randn(4, 3), 1)
    y = torch.tensor([-1, -1, -1, -1])
    assert float(entrenar.focal_multiclase(p, y, (0.3, 0.3, 0.4), 2.0)) == 0.0
    assert float(entrenar.focal_binaria(torch.rand(4, 1), y, (0.3, 0.7), 2.0)) == 0.0
    y = torch.tensor([0, 1, 2, -1])
    assert float(entrenar.focal_multiclase(p, y, (0.3, 0.3, 0.4), 2.0)) > 0


def test_corrida_corta_de_punta_a_punta(trabajo, monkeypatch):
    monkeypatch.setenv("BOXTWIN_BOXMIND_REPO", str(CLON))
    import importlib

    import bm
    importlib.reload(bm)
    import entrenar
    importlib.reload(entrenar)
    monkeypatch.setitem(entrenar.HIPER, "epocas", 1)
    rng = np.random.default_rng(0)
    datos = bm.TRABAJO_BM / "datos"
    datos.mkdir(parents=True)
    np.savez(datos / "conjunto_boxtwin.npz", **_conjunto(bm.geo.FUENTES, 6, 9, rng))
    np.savez(datos / "conjunto_boxingweb.npz", **_conjunto([f"r{i}" for i in range(4)], 8, 12, rng, con_bw=True))
    assert entrenar.main(["--exp", "BM-2", "--semillas", "0", "--dispositivo", "cpu"]) == 0
    assert entrenar.main(["--exp", "BM-1", "--semillas", "0", "--dispositivo", "cpu"]) == 0
    for exp in ("BM-2", "BM-1"):
        for f in bm.geo.FUENTES:
            with np.load(bm.TRABAJO_BM / "corridas" / exp / f / "s0" / "pred.npz") as z:
                assert len(z["frame_dir"]) == 6 and z["p_tipo"].shape == (6, 3) and z["p_altura"].shape == (6, 2)
                np.testing.assert_allclose(z["p_altura"].sum(1), 1, atol=1e-5)
                assert all(str(n).startswith(f) for n in z["frame_dir"])
