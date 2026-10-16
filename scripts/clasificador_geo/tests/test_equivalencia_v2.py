"""
CA1 y CA2 del tipo v2 (docs/specs/2026-10-16-fightcard-tipo-v2/01-requisitos.md seccion 3).

CA1: el camino de produccion (boxtwin.mvp.clasificador_geo.features_golpe) da el mismo X y M
que golpes.npz, que es la entrada con la que se midio E4.

CA2: la inferencia de produccion (ClasificadorGeo) da las mismas probabilidades que
entrenar.predecir sobre el mismo modelo guardado por entrenar_final.py. La comparacion contra
el pred.npz de E4 no puede ser exacta: E4 se entreno en cuda y ese entrenamiento no es
determinista (decision 1 del registro de tareas), asi que contra E4 se reporta, no se exige.

Las pruebas con datos sinteticos corren siempre. Las de datos reales se saltean si no estan
golpes.npz, las series o el modelo final.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boxtwin_detector")
cg = pytest.importorskip("boxtwin.mvp.clasificador_geo")

TRABAJO_REAL = Path("~/boxtwin/clasificador-geo-2026-10").expanduser()
MODELO_REAL = Path("~/boxtwin/modelos/clasificador_geo_v1").expanduser()
TOL_X = 1e-5
TOL_P = 1e-4


def _sinteticos(T=150, semilla=3):
    rng = np.random.default_rng(semilla)
    kp = (rng.normal(size=(2, T, 17, 2)) * 25 + 300).astype(np.float32)
    sc = rng.uniform(0.1, 1.0, size=(2, T, 17)).astype(np.float32)
    valid = rng.uniform(size=(2, T)) > 0.2
    return kp, sc, valid


def test_ca1_logica_igual_a_features2d():
    import features2d
    kp, sc, valid = _sinteticos()
    golpes = []
    for p, side, a, b in [(0, "left", 3, 12), (1, "right", 60, 75), (0, "right", 140, 149)]:
        ini, fin = max(0, a - 8), min(149, b + 8)
        golpes.append({"fuente": "s", "p": p, "side": side, "ini": ini, "fin": fin, "tipo": "hook",
                       "altura": "head", "rol": "lead", "frame_dir": f"g{a}", "a": a, "b": b})
    ref = features2d.construir(golpes, {"s": {"kp": kp, "sc": sc, "valid": valid}})
    for i, g in enumerate(golpes):
        X, M = cg.features_golpe(kp, sc, valid, g["p"], g["side"], g["a"], g["b"])
        assert np.abs(X - ref["X"][i]).max() <= TOL_X
        assert np.array_equal(M, ref["M"][i])


def test_ca2_logica_igual_a_predecir(tmp_path):
    import entrenar
    import entrenar_final
    rng = np.random.default_rng(0)
    n = 30
    d = {"X": rng.normal(size=(n, 32, 26)).astype(np.float32), "M": rng.uniform(size=(n, 32)) > 0.1,
         "y_tipo": rng.integers(0, 3, n), "y_altura": rng.integers(0, 2, n),
         "fuente": np.array(["a"] * 10 + ["b"] * 20), "frame_dir": np.arange(n).astype(str),
         "nombres": [str(i) for i in range(26)]}
    out = entrenar_final.entrenar_final(d, tmp_path, "a", semillas=(0,), epocas=2, metricas={})
    clf = cg.ClasificadorGeo.cargar(out)
    info = json.loads((out / "meta.json").read_text())["semillas"]["s0"]
    net = entrenar.ClasificadorGolpe(26)
    net.load_state_dict(torch.load(out / "s0.pt"))
    net.eval()
    t = d["fuente"] == "a"
    ref, _ = entrenar.predecir(net, d["X"][t], d["M"][t], np.array(info["media"], np.float32),
                               np.array(info["desvio"], np.float32))
    assert np.abs(clf.familia(d["X"][t], d["M"][t]) - ref).max() <= TOL_P


# ---------------------------------------------------------------------------
# Datos reales
# ---------------------------------------------------------------------------

def _hay_reales():
    return (TRABAJO_REAL / "datos" / "golpes.npz").is_file() and (TRABAJO_REAL / "datos" / "golpes.csv").is_file()


@pytest.mark.skipif(not _hay_reales(), reason="sin golpes.npz de E4")
def test_ca1_real():
    from datos import leer_csv
    d = TRABAJO_REAL / "datos"
    golpes = leer_csv(d / "golpes.csv")
    with np.load(d / "golpes.npz") as z:
        Xr, Mr, fd = z["X"], z["M"], z["frame_dir"]
    assert len(golpes) == len(Xr) == 1047
    series, cache = {}, {}
    dx, dm = 0.0, 0
    for i, g in enumerate(golpes):
        assert g["frame_dir"] == fd[i]
        if g["fuente"] not in series:
            with np.load(d / "series" / f"{g['fuente']}.npz") as z:
                series[g["fuente"]] = (z["kp"], z["sc"], z["valid"])
        kp, sc, valid = series[g["fuente"]]
        clave = (g["fuente"], g["p"], g["side"])
        if clave not in cache:
            cache[clave] = cg.features_serie(kp, sc, valid, g["p"], g["side"])
        X, M = cg.features_golpe(kp, sc, valid, g["p"], g["side"], g["start_frame"], g["end_frame"],
                                 serie=cache[clave])
        dx = max(dx, float(np.abs(X - Xr[i]).max()))
        dm += int((M != Mr[i]).sum())
    print(f"CA1: 1047 golpes, diferencia absoluta maxima de X {dx:.2e}, pasos de M distintos {dm}")
    assert dx <= TOL_X and dm == 0


@pytest.mark.skipif(not (_hay_reales() and (MODELO_REAL / "sin_Sparring" / "meta.json").is_file()),
                    reason="sin golpes.npz o sin el modelo sin_Sparring")
def test_ca2_real_sparring():
    import entrenar
    d = entrenar.cargar_entrada("E4", TRABAJO_REAL)
    t = d["fuente"] == "Sparring"
    dir_ = MODELO_REAL / "sin_Sparring"
    clf = cg.ClasificadorGeo.cargar(dir_)
    info = json.loads((dir_ / "meta.json").read_text())["semillas"]["s0"]
    net = entrenar.ClasificadorGolpe(26)
    net.load_state_dict(torch.load(dir_ / "s0.pt"))
    net.eval()
    ref, _ = entrenar.predecir(net, d["X"][t], d["M"][t], np.array(info["media"], np.float32),
                               np.array(info["desvio"], np.float32))
    s0 = cg.ClasificadorGeo(clf.modelos[:1], clf.medias[:1], clf.desvios[:1], clf.meta)
    prod = s0.familia(d["X"][t], d["M"][t])
    dif = float(np.abs(prod - ref).max())
    y = d["y_tipo"][t]
    linea = f"CA2: Sparring {int(t.sum())} golpes, produccion contra predecir (s0) {dif:.2e}"
    pred_e4 = TRABAJO_REAL / "corridas" / "E4" / "Sparring" / "s0" / "pred.npz"
    if pred_e4.is_file():
        z = np.load(pred_e4)
        assert np.array_equal(z["frame_dir"], d["frame_dir"][t])
        pe = z["p_tipo"]
        linea += (f"; contra E4 s0 (descriptivo): dif max {np.abs(prod - pe).max():.3f}, acuerdo de argmax "
                  f"{(prod.argmax(1) == pe.argmax(1)).mean():.3f}, exactitud {(prod.argmax(1) == y).mean():.3f} "
                  f"contra {(pe.argmax(1) == y).mean():.3f} de E4")
    ens = clf.familia(d["X"][t], d["M"][t])
    linea += f"; ensamble de 5 semillas {(ens.argmax(1) == y).mean():.3f}"
    print(linea)
    assert dif <= TOL_P
