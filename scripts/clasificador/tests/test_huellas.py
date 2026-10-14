import pytest

import huellas


def _repo(tmp_path):
    d = tmp_path / "repo" / "anotacion-x" / "annotations"
    d.mkdir(parents=True)
    (d / "x.annot.json").write_text('{"events": []}')
    return tmp_path / "repo"


def test_tomar_y_verificar(tmp_path):
    repo = _repo(tmp_path)
    archivo = tmp_path / "h.json"
    huellas.tomar(archivo, repo)
    assert huellas.verificar(archivo, repo) == []
    (repo / "anotacion-x" / "annotations" / "x.annot.json").write_text('{"events": [1]}')
    assert huellas.verificar(archivo, repo) == ["cambio anotacion-x/annotations/x.annot.json"]
    (repo / "anotacion-x" / "annotations" / "y.reanno.json").write_text("{}")
    assert any(p.startswith("nuevo") for p in huellas.verificar(archivo, repo))


def test_no_pisa_huellas(tmp_path):
    repo = _repo(tmp_path)
    archivo = tmp_path / "h.json"
    huellas.tomar(archivo, repo)
    with pytest.raises(SystemExit):
        huellas.tomar(archivo, repo)
