"""entrenar_final.py: guarda pesos, estandarizacion y sha256, y excluye la fuente pedida."""

import json

import numpy as np
import torch


def _datos(n=40, f=26):
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(n, 32, f)).astype(np.float32),
        "M": np.ones((n, 32), bool),
        "y_tipo": rng.integers(0, 3, n), "y_altura": rng.integers(0, 2, n),
        "fuente": np.array(["Sparring"] * 10 + ["01-sparring"] * (n - 10)),
        "frame_dir": np.array([f"g{i}" for i in range(n)]),
        "nombres": [f"f{i}" for i in range(f)],
    }


def test_completo_y_excluido(trabajo):
    import entrenar_final as ef
    d = _datos()
    out = ef.entrenar_final(d, trabajo / "m", None, semillas=(0, 1), epocas=1, metricas={"x": 1})
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_train"] == 40 and meta["n_features"] == 26 and meta["pasos"] == 32 and meta["margen"] == 8
    assert set(meta["semillas"]) == {"s0", "s1"}
    for s, info in meta["semillas"].items():
        assert ef.sha256(out / info["archivo"]) == info["sha256"]
        assert len(info["media"]) == 26 and len(info["desvio"]) == 26
        assert isinstance(torch.load(out / info["archivo"]), dict)
    sin = ef.entrenar_final(d, trabajo / "m", "Sparring", semillas=(0,), epocas=1, metricas={})
    assert sin.name == "sin_Sparring"
    meta = json.loads((sin / "meta.json").read_text())
    assert meta["n_train"] == 30 and meta["fuentes_train"] == ["01-sparring"]


def test_reanudable(trabajo):
    import entrenar_final as ef
    d = _datos()
    out = ef.entrenar_final(d, trabajo / "m", None, semillas=(0,), epocas=1, metricas={})
    antes = (out / "s0.pt").stat().st_mtime_ns
    ef.entrenar_final(d, trabajo / "m", None, semillas=(0, 1), epocas=1, metricas={})
    assert (out / "s0.pt").stat().st_mtime_ns == antes
    assert set(json.loads((out / "meta.json").read_text())["semillas"]) == {"s0", "s1"}


def test_metricas_e4_de_json(tmp_path):
    import entrenar_final as ef
    p = tmp_path / "e4.json"
    p.write_text(json.dumps({"ensamble_total": {"golpes": 1047, "familia": {
        "exactitud": 0.7268, "wilson95": [0.699, 0.753], "balanceada": 0.64, "recall": {"hook": 0.62}}}}))
    m = ef.metricas_e4(p)
    assert m["familia_fuente_no_vista"] == 0.7268 and m["golpes"] == 1047
    assert ef.metricas_e4(tmp_path / "no.json") == {}


def test_variante_e4a_sin_rival(trabajo):
    """16-10: el modelo de bolsa y sombra tiene otro nombre y declara 20 features y E4a."""
    import entrenar_final as ef
    d = _datos(f=20)
    out = ef.entrenar_final(d, trabajo / "m", None, semillas=(0,), epocas=1, metricas={}, exp="E4a")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["exp"] == "E4a" and meta["nombre"] == "clasificador_geo_v1_sin_rival"
    assert meta["n_features"] == 20 and len(meta["semillas"]["s0"]["media"]) == 20
    assert ef.VARIANTES["E4a"]["salida"].name == "clasificador_geo_v1_sin_rival"
    assert ef.VARIANTES["E4a"]["metricas"].name.endswith("E4a.json")
    # sin exp sigue siendo el de E4, como antes
    meta = json.loads((ef.entrenar_final(_datos(), trabajo / "n", None, semillas=(0,), epocas=1,
                                         metricas={}) / "meta.json").read_text())
    assert meta["exp"] == "E4" and meta["nombre"] == "clasificador_geo_v1"


def test_metricas_reales_de_e4a():
    import entrenar_final as ef
    m = ef.metricas_e4(ef.VARIANTES["E4a"]["metricas"])
    assert m["golpes"] == 1047 and round(m["familia_fuente_no_vista"], 3) == 0.724

