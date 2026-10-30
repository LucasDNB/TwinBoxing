"""T12: K-reuso compara el contenido de los tensores, no los bytes del zip."""

import json
import time

import numpy as np

import kreuso


def _dir(raiz, nombre, X, extra=None):
    d = raiz / nombre
    (d / "det").mkdir(parents=True)
    np.savez_compressed(d / "det" / "F.det.npz", features=X, usable=X > 0)
    (d / "clasificador-geo" / "datos").mkdir(parents=True)
    np.savez_compressed(d / "clasificador-geo" / "datos" / "golpes.npz", X=X[:2], **(extra or {}))
    return d


def test_dos_directorios_iguales(tmp_path):
    X = np.random.default_rng(0).normal(size=(4, 50, 20)).astype(np.float32)
    a = _dir(tmp_path, "a", X)
    time.sleep(1.1)                                   # otra fecha en el zip, mismo contenido
    b = _dir(tmp_path, "b", X.copy())
    r = kreuso.comparar(a, b)
    assert r["pasa"] and r["archivos"] == 2 and not r["diferencias"]


def test_un_byte_cambiado(tmp_path):
    X = np.random.default_rng(0).normal(size=(4, 50, 20)).astype(np.float32)
    a = _dir(tmp_path, "a", X)
    Y = X.copy()
    Y.reshape(-1).view(np.uint8)[123] ^= 1
    b = _dir(tmp_path, "b", Y)
    r = kreuso.comparar(a, b)
    assert not r["pasa"]
    assert {(d["archivo"], d["clave"]) for d in r["diferencias"]} == {
        ("det/F.det.npz", "features"), ("clasificador-geo/datos/golpes.npz", "X")}


def test_un_arreglo_de_mas_o_un_archivo_faltante_tambien_falla(tmp_path):
    X = np.ones((2, 3, 4), np.float32)
    a = _dir(tmp_path, "a", X)
    b = _dir(tmp_path, "b", X, extra={"R": np.zeros(3, bool)})
    assert not kreuso.comparar(a, b)["pasa"]
    (b / "det" / "F.det.npz").unlink()
    r = kreuso.comparar(a, b)
    assert any(d["clave"] == "*" for d in r["diferencias"])


def test_main_escribe_el_json(tmp_path):
    X = np.ones((2, 3, 4), np.float32)
    a, b = _dir(tmp_path, "a", X), _dir(tmp_path, "b", X)
    out = tmp_path / "k.json"
    assert kreuso.main([str(a), str(b), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["pasa"]
    b2 = _dir(tmp_path, "c", X * 2)
    assert kreuso.main([str(a), str(b2), "--out", str(out)]) == 2
