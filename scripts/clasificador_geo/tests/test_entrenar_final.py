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



def test_variante_e7_modo_academico(trabajo):
    """25-10 (docs/specs/2026-10-25-e7-modo-academico): 42 features, metodo geometrico-e7 y licencia."""
    import entrenar_final as ef
    d = _datos(f=42)
    out = ef.entrenar_final(d, trabajo / "m", None, semillas=(0,), epocas=1, metricas={}, exp="E7")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["exp"] == "E7" and meta["nombre"] == "clasificador_e7_v1"
    assert meta["metodo"] == "geometrico-e7" and meta["n_features"] == 42
    assert "SMPL" in meta["licencia"] and "no comercial y sin fines de lucro" in meta["licencia"]
    assert ef.VARIANTES["E7"]["salida"].name == "clasificador_e7_v1"
    # E4 sigue sin licencia y con su metodo
    meta4 = json.loads((ef.entrenar_final(_datos(), trabajo / "n", None, semillas=(0,), epocas=1,
                                          metricas={}) / "meta.json").read_text())
    assert meta4["metodo"] == "geometrico-v1" and "licencia" not in meta4


def test_metricas_reales_de_e7():
    import entrenar_final as ef
    m = ef.metricas_e4(ef.VARIANTES["E7"]["metricas"], ef.VARIANTES["E7"]["referencia"])
    assert m["golpes"] == 1047 and round(m["familia_fuente_no_vista"], 4) == 0.7775
    assert m["referencia"].endswith("2026-10-24-e7-4dhumans.md")


# -- T4 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6) ------


def test_variantes_y26_rutas_propias(trabajo):
    """E4-y26 y E4a-y26 leen el trabajo de E8 y escriben en ~/boxtwin-gemelo/modelos."""
    from pathlib import Path
    import entrenar_final as ef
    gemelo = Path("~/boxtwin-gemelo").expanduser()
    for exp, nombre, base in (("E4-y26", "clasificador_geo_y26_v1", "E4"),
                              ("E4a-y26", "clasificador_geo_y26_v1_sin_rival", "E4a")):
        v = ef.VARIANTES[exp]
        assert v["nombre"] == nombre and v["columnas"] == base
        assert v["salida"] == gemelo / "modelos" / nombre
        assert v["trabajo"] == gemelo / "e8" / "y26" / "clasificador-geo"
        assert v["metricas"].name.endswith("-e8-yolo26-E4.json")
        assert v["seccion"] == ("y26", base)


def test_e4_y_e4a_con_las_rutas_de_hoy(trabajo):
    from pathlib import Path
    import entrenar_final as ef
    modelos = Path("~/boxtwin/modelos").expanduser()
    assert ef.VARIANTES["E4"]["salida"] == modelos / "clasificador_geo_v1"
    assert ef.VARIANTES["E4a"]["salida"] == modelos / "clasificador_geo_v1_sin_rival"
    assert ef.VARIANTES["E4"]["metricas"].name == "2026-10-16-clasificador-geo-E4.json"
    assert ef.VARIANTES["E4a"]["metricas"].name == "2026-10-16-clasificador-geo-E4a.json"
    for exp in ("E4", "E4a", "E7"):
        assert "trabajo" not in ef.VARIANTES[exp] and "seccion" not in ef.VARIANTES[exp]
    m = ef.metricas_e4(ef.VARIANTES["E4"]["metricas"])
    if m:                                    # el json real esta en el repo
        assert round(m["familia_fuente_no_vista"], 3) == 0.727


def test_metricas_de_una_seccion_del_json_de_e8(tmp_path):
    import entrenar_final as ef
    p = tmp_path / "2026-11-01-e8-yolo26-E4.json"
    total = lambda fam: {"ensamble_total": {"golpes": 1047, "familia": {"exactitud": fam}}}
    p.write_text(json.dumps({"base": {"E4": total(0.727)},
                             "y26": {"E4": total(0.741), "E4a": total(0.719)}}))
    assert ef.metricas_e4(p, seccion=("y26", "E4"))["familia_fuente_no_vista"] == 0.741
    assert ef.metricas_e4(p, seccion=("y26", "E4a"))["familia_fuente_no_vista"] == 0.719


def test_la_variante_y26_usa_las_columnas_de_su_base(trabajo):
    import entrenar_final as ef
    out = ef.entrenar_final(_datos(f=20), trabajo / "m", None, semillas=(0,), epocas=1,
                            metricas={}, exp="E4a-y26", comando="entrenar_final.py --exp E4a-y26")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["exp"] == "E4a-y26" and meta["nombre"] == "clasificador_geo_y26_v1_sin_rival"
    assert meta["metodo"] == "geometrico-v1"           # el producto lo carga como el de A
    assert meta["comando"] == "entrenar_final.py --exp E4a-y26"   # CG9


def test_y26_sin_el_json_de_e8_se_niega(trabajo, monkeypatch, tmp_path):
    """CG5: un modelo de C sin las metricas de E8 mostraria nada, o las de A. No se arma."""
    import entrenar_final as ef
    monkeypatch.setitem(ef.VARIANTES["E4-y26"], "metricas", tmp_path / "no-esta-e8-yolo26-E4.json")
    llamado = []
    monkeypatch.setattr(ef.entrenar, "cargar_entrada", lambda *a, **k: llamado.append(a))
    assert ef.main(["--exp", "E4-y26", "--dispositivo", "cpu", "--salida", str(tmp_path / "m")]) == 2
    assert not llamado and not (tmp_path / "m").exists()
