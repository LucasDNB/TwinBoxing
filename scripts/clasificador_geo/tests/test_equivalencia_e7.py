"""
CA-E7a y CA2-E7 del modo academico (docs/specs/2026-10-25-e7-modo-academico/01-plan.md, seccion 3).

CA-E7a: lo que el producto copio de scripts/ da lo mismo que el original. La herramienta de
elevacion por sesion (boxtwin-annotator/tools/elevar_hmr2_sesion.py) contra elevar_hmr2.py y
elevar3d.py, y las features 3D de boxtwin.mvp.clasificador_geo contra features3d.py, sobre
los 1047 golpes con el 3D real de E7.

CA2-E7: la inferencia de produccion (ClasificadorGeo con 42 features) da las mismas
probabilidades que entrenar.predecir sobre el modelo guardado por entrenar_final.py --exp E7.
Contra el pred.npz de E7 se reporta, no se exige (entrenamiento en cuda, decision 1 de E4).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import elevar3d
import elevar_hmr2
import features3d
from test_features3d import _esqueleto

torch = pytest.importorskip("torch")
pytest.importorskip("boxtwin_detector")
cg = pytest.importorskip("boxtwin.mvp.clasificador_geo")

REPO = Path(__file__).resolve().parents[3]
HERRAMIENTA = REPO / "boxtwin-annotator" / "tools" / "elevar_hmr2_sesion.py"
TRABAJO_REAL = Path("~/boxtwin/clasificador-geo-2026-10").expanduser()
TRABAJO_E7_REAL = Path("~/boxtwin/e7-4dhumans").expanduser()
MODELO_E7_REAL = Path("~/boxtwin/modelos/clasificador_e7_v1").expanduser()
TOL_P = 1e-4


def _herramienta():
    spec = importlib.util.spec_from_file_location("elevar_hmr2_sesion", HERRAMIENTA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- CA-E7a: la herramienta ---------------------------------------------------------------

def test_herramienta_copia_las_piezas_de_elevar_hmr2():
    h = _herramienta()
    rng = np.random.default_rng(0)
    J = rng.normal(size=(5, 24, 3)).astype(np.float32)
    assert np.array_equal(h.smpl_a_h36m(J), elevar_hmr2.smpl_a_h36m(J))
    T = 60
    kp = rng.uniform(0, 500, size=(T, 17, 2)).astype(np.float32)
    sc = rng.uniform(0, 1, size=(T, 17)).astype(np.float32)
    valid = rng.uniform(size=T) > 0.3
    valid[20:35] = False
    for a, b in zip(h.cajas_de(kp, sc, valid), elevar_hmr2.cajas_de(kp, sc, valid)):
        assert np.array_equal(a, b)
    for fps, ini, fin in ((29.97, 101, 132), (59.94, 501, 540), (24.0, 0, 20), (59.94, 0, 30)):
        for x, y in zip(h.cuadros_ventana(ini, fin, 5000, fps), elevar_hmr2.cuadros_ventana(ini, fin, 5000, fps)):
            assert np.array_equal(x, y)
    sel, _ = elevar_hmr2.cuadros_ventana(501, 540, 5000, 59.94)
    X = rng.normal(size=(len(sel), 17, 3)).astype(np.float32)
    assert np.array_equal(h.a_ventana(X, sel, 501, 540), elevar3d.a_ventana(X, sel, 501, 540))
    m = rng.uniform(size=5000) > 0.5
    assert np.array_equal(h.mascara_ventana(m, 501, 540), elevar3d.mascara_ventana(m, 501, 540))


def test_herramienta_usa_la_ventana_de_e4():
    h = _herramienta()
    for ini, fin, T in ((3, 12, 150), (140, 149, 150), (60, 75, 150)):
        assert h.ventana(ini, fin, T) == cg.ventana(ini, fin, T)


# -- CA-E7a: features 3D del producto -----------------------------------------------------

def test_features_3d_del_producto_igual_a_features3d():
    for semilla in range(3):
        P = _esqueleto(seed=semilla).astype(np.float32)
        for side in ("left", "right"):
            assert np.array_equal(cg.features_3d(P, side, 0.6), features3d.features_de(P, side, 0.6))


def _hay_e7_real():
    return ((TRABAJO_E7_REAL / "datos" / "pose3d.npz").is_file()
            and (TRABAJO_E7_REAL / "datos" / "features3d.npz").is_file()
            and (TRABAJO_REAL / "datos" / "golpes.csv").is_file())


@pytest.mark.skipif(not _hay_e7_real(), reason="sin el 3D de E7")
def test_ca_e7a_real_features_3d_de_los_1047():
    from datos import leer_csv
    golpes = {g["frame_dir"]: g for g in leer_csv(TRABAJO_REAL / "datos" / "golpes.csv")}
    with np.load(TRABAJO_E7_REAL / "datos" / "pose3d.npz") as z:
        fd, P3, M3 = z["frame_dir"], z["P3"], z["M3"]
    with np.load(TRABAJO_E7_REAL / "datos" / "features3d.npz") as z:
        pos = {n: i for i, n in enumerate(z["frame_dir"])}
        X3 = z["X3"]
    dif = 0.0
    for n, P, M in zip(fd, P3, M3):
        g = golpes[str(n)]
        f = cg.features_3d_golpe(P, M, g["side"], g["ini"], g["fin"], g["fps"])
        dif = max(dif, float(np.abs(f - X3[pos[str(n)]]).max()))
    print(f"CA-E7a: 1047 golpes, features 3D del producto contra features3d.npz de E7 {dif:.2e}")
    assert len(fd) == 1047 and dif <= 1e-5


# -- CA2-E7 -------------------------------------------------------------------------------

def test_ca2_e7_logica_igual_a_predecir(tmp_path):
    import entrenar
    import entrenar_final
    rng = np.random.default_rng(2)
    n = 30
    d = {"X": rng.normal(size=(n, 32, 42)).astype(np.float32), "M": rng.uniform(size=(n, 32)) > 0.1,
         "y_tipo": rng.integers(0, 3, n), "y_altura": rng.integers(0, 2, n),
         "fuente": np.array(["a"] * 10 + ["b"] * 20), "frame_dir": np.arange(n).astype(str),
         "nombres": [str(i) for i in range(42)]}
    out = entrenar_final.entrenar_final(d, tmp_path, "a", semillas=(0,), epocas=2, metricas={}, exp="E7")
    clf = cg.ClasificadorGeo.cargar(out)
    assert clf.con_3d and not clf.sin_rival
    info = json.loads((out / "meta.json").read_text())["semillas"]["s0"]
    net = entrenar.ClasificadorGolpe(42)
    net.load_state_dict(torch.load(out / "s0.pt"))
    net.eval()
    t = d["fuente"] == "a"
    ref, _ = entrenar.predecir(net, d["X"][t], d["M"][t], np.array(info["media"], np.float32),
                               np.array(info["desvio"], np.float32))
    assert np.abs(clf.familia(d["X"][t], d["M"][t]) - ref).max() <= TOL_P


@pytest.mark.skipif(not (_hay_e7_real() and (MODELO_E7_REAL / "sin_Sparring" / "meta.json").is_file()),
                    reason="sin el 3D de E7 o sin el modelo E7 sin_Sparring")
def test_ca2_e7_real_sparring():
    import entrenar
    d = entrenar.cargar_entrada("E7", TRABAJO_REAL)
    t = d["fuente"] == "Sparring"
    dir_ = MODELO_E7_REAL / "sin_Sparring"
    clf = cg.ClasificadorGeo.cargar(dir_)
    info = clf.meta["semillas"]["s0"]
    net = entrenar.ClasificadorGolpe(42)
    net.load_state_dict(torch.load(dir_ / info["archivo"]))
    net.eval()
    ref, _ = entrenar.predecir(net, d["X"][t], d["M"][t], np.array(info["media"], np.float32),
                               np.array(info["desvio"], np.float32))
    prod = clf.familia(d["X"][t], d["M"][t])
    dif = float(np.abs(prod - ref).max())
    linea = f"CA2-E7: Sparring {int(t.sum())} golpes, produccion contra predecir (s0) {dif:.2e}"
    pred_e7 = TRABAJO_E7_REAL / "corridas" / "E7" / "Sparring" / "s0" / "pred.npz"
    if pred_e7.is_file():
        z = np.load(pred_e7)
        assert np.array_equal(z["frame_dir"], d["frame_dir"][t])
        y = d["y_tipo"][t]
        linea += (f"; contra E7 s0 (descriptivo): acuerdo de argmax {(prod.argmax(1) == z['p_tipo'].argmax(1)).mean():.3f}, "
                  f"exactitud {(prod.argmax(1) == y).mean():.3f} contra {(z['p_tipo'].argmax(1) == y).mean():.3f}")
    print(linea)
    assert dif <= TOL_P
