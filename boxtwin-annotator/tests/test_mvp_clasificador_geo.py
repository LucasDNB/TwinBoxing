"""
Tipo de golpe v2 (02-diseno.md seccion 2 y 7): features, carga con sha256, ensamble y rol.

Datos sinteticos: lo que se prueba es la logica (ventana, rival, mascara, verificacion del
modelo). Que las features y las predicciones sean las de E4 lo prueban CA1 y CA2 contra los
datos reales, en scripts/clasificador_geo/tests/test_equivalencia_v2.py.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boxtwin_detector")

from boxtwin.mvp import clasificador_geo as cg  # noqa: E402

T = 120


def _poses(n_peleadores=2, semilla=0):
    rng = np.random.default_rng(semilla)
    kp = (rng.normal(size=(n_peleadores, T, 17, 2)) * 20 + 200).astype(np.float32)
    sc = np.full((n_peleadores, T, 17), 0.9, np.float32)
    valid = np.ones((n_peleadores, T), bool)
    return kp, sc, valid


def _modelo(dir_, semillas=(0, 1), n_features=26):
    dir_.mkdir(parents=True, exist_ok=True)
    meta = {"nombre": "prueba", "n_features": n_features, "semillas": {},
            "metricas_e4": {"familia_fuente_no_vista": 0.727, "referencia": "docs/x.md"}}
    for s in semillas:
        torch.manual_seed(s)
        m = cg.ClasificadorGolpe(n_features)
        pt = dir_ / f"s{s}.pt"
        torch.save(m.state_dict(), pt)
        meta["semillas"][f"s{s}"] = {"archivo": pt.name, "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
                                     "media": [0.0] * n_features, "desvio": [1.0] * n_features}
    (dir_ / "meta.json").write_text(json.dumps(meta))
    return dir_


def test_features_golpe_forma_y_ventana():
    kp, sc, valid = _poses()
    X, M = cg.features_golpe(kp, sc, valid, 0, "left", 40, 50)
    assert X.shape == (32, 26) and M.shape == (32,) and X.dtype == np.float32
    serie, us = cg.features_serie(kp, sc, valid, 0, "left")
    # la ventana es [32, 58]: el primer y el ultimo paso son esos cuadros
    np.testing.assert_allclose(X[0], serie[32], atol=1e-6)
    np.testing.assert_allclose(X[-1], serie[58], atol=1e-6)


def test_ventana_en_los_bordes():
    assert cg.ventana(3, 10, T) == (0, 18)
    assert cg.ventana(110, 118, T) == (102, T - 1)
    kp, sc, valid = _poses()
    serie, _ = cg.features_serie(kp, sc, valid, 1, "der")
    X, _ = cg.features_golpe(kp, sc, valid, 1, "der", 2, 6)
    np.testing.assert_allclose(X[0], serie[0], atol=1e-6)
    np.testing.assert_allclose(X[-1], serie[14], atol=1e-6)


def test_rival_ausente_va_en_cero():
    kp, sc, valid = _poses()
    valid[1, :] = False
    X, _ = cg.features_golpe(kp, sc, valid, 0, "left", 40, 50)
    assert np.all(X[:, 20:] == 0)
    valid[1, :] = True
    X, _ = cg.features_golpe(kp, sc, valid, 0, "left", 40, 50)
    assert np.all(X[:, 25] == 1)


def test_un_solo_peleador():
    kp, sc, valid = _poses(n_peleadores=1)
    X, M = cg.features_golpe(kp, sc, valid, 0, "right", 40, 50)
    assert X.shape == (32, 26) and np.all(X[:, 20:] == 0) and M.any()


def test_mascara_sin_pasos_usables():
    kp, sc, valid = _poses()
    sc[0, :, :] = 0.0
    _, M = cg.features_golpe(kp, sc, valid, 0, "left", 40, 50)
    assert not M.any()


def test_carga_y_ensamble(tmp_path):
    d = _modelo(tmp_path / "m")
    clf = cg.ClasificadorGeo.cargar(d)
    assert len(clf.modelos) == 2
    kp, sc, valid = _poses()
    X, M = cg.features_golpe(kp, sc, valid, 0, "left", 40, 50)
    p = clf.familia(np.stack([X, X]), np.stack([M, M]))
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    # el ensamble es el promedio de las semillas
    uno = [cg.ClasificadorGeo([m], [a], [b], {}).familia(X[None], M[None])
           for m, a, b in zip(clf.modelos, clf.medias, clf.desvios)]
    np.testing.assert_allclose(p[0], np.mean(uno, axis=0)[0], atol=1e-6)


def test_carga_con_sha256_incorrecto_falla(tmp_path):
    d = _modelo(tmp_path / "m")
    meta = json.loads((d / "meta.json").read_text())
    meta["semillas"]["s1"]["sha256"] = "0" * 64
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        cg.ClasificadorGeo.cargar(d)


@pytest.mark.parametrize("guardia,brazo,esperado", [
    ("ortodoxa", "izq", "lead"), ("ortodoxa", "der", "rear"),
    ("zurda", "izq", "rear"), ("zurda", "der", "lead"),
])
def test_rol_en_los_cuatro_casos(guardia, brazo, esperado):
    assert cg.rol(guardia, brazo) == esperado
    assert cg.rol(guardia, {"izq": "left", "der": "right"}[brazo]) == esperado


def test_tipo_de():
    assert cg.tipo_de("straight", "lead") == "jab"
    assert cg.tipo_de("straight", "rear") == "cross"
    assert cg.tipo_de("hook", "lead") == "hook"
    assert cg.tipo_de("uppercut", "rear") == "uppercut"
