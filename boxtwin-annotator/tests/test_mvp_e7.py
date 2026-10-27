"""
Modo academico E7 (docs/specs/2026-10-25-e7-modo-academico, M3 y M5): un modelo de 42
features clasifica con el 3D de pose3d_hmr2.npz, falla claro sin el archivo (no cae a E4 a
escondidas) y la Fight-Card lleva el metodo geometrico-e7 con la leyenda de licencia.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boxtwin_detector")

from boxtwin.mvp import clasificador_geo as cg  # noqa: E402
from boxtwin.mvp.orquesta import aplicar_tipos_a_sesion  # noqa: E402

from tests.test_mvp_clasificador_geo import _modelo  # noqa: E402
from tests.test_mvp_tipos_v2 import sesion_dir  # noqa: E402,F401

LICENCIA = "modo academico: 3D de 4D-Humans con SMPL, licencia de investigacion no comercial"


def _modelo_e7(dir_):
    out = _modelo(dir_, n_features=42)
    meta = json.loads((out / "meta.json").read_text())
    meta.update({"metodo": "geometrico-e7", "exp": "E7", "nombre": "clasificador_e7_v1", "licencia": LICENCIA})
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def _pose3d(directorio, gids):
    rng = np.random.default_rng(0)
    P3 = rng.normal(size=(len(gids), 32, 17, 3)).astype(np.float32)
    P3[:, :, 8] = P3[:, :, 0] + [0, 1, 0]                 # torax arriba de la pelvis
    np.savez_compressed(directorio / "pose3d_hmr2.npz", gid=np.array(gids), P3=P3,
                        M3=np.ones((len(gids), 32), bool))


def test_features_3d_forma_y_mascara():
    P = np.random.default_rng(1).normal(size=(32, 17, 3)).astype(np.float32)
    M = np.ones(32, bool)
    M[:5] = False
    f = cg.features_3d_golpe(P, M, "left", 10, 41, 30.0)
    assert f.shape == (32, cg.N_3D) and cg.N_3D == 16
    assert np.all(f[:5] == 0) and np.any(f[5:] != 0)


def test_modelo_e7_es_con_3d(tmp_path):
    clf = cg.ClasificadorGeo.cargar(_modelo_e7(tmp_path / "m"))
    assert clf.con_3d and not clf.sin_rival
    assert not cg.ClasificadorGeo.cargar(_modelo(tmp_path / "e4")).con_3d


def test_e7_sin_pose3d_falla_y_no_cae_a_e4(sesion_dir, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="pose3d"):
        cg.clasificar_sesion(sesion_dir, _modelo_e7(tmp_path / "m"), "zurda", None)
    assert not (sesion_dir / "tipos.json").exists()


def test_e7_con_pose3d_escribe_metodo_y_licencia(sesion_dir, tmp_path):  # noqa: F811
    _pose3d(sesion_dir, ["A-izq-20", "A-der-50", "B-izq-80"])
    d = cg.clasificar_sesion(sesion_dir, _modelo_e7(tmp_path / "m"), "zurda", None,
                             pose3d=sesion_dir / "pose3d_hmr2.npz")
    assert d["metodo"] == "geometrico-e7" and d["licencia"] == LICENCIA
    assert d["checkpoint"] == "clasificador_e7_v1" and d["segmentos_sin_3d"] == 0
    assert set(d["tipos"]) == {"A-izq-20", "A-der-50", "B-izq-80"}


def test_e7_segmento_sin_3d_entra_con_el_3d_en_cero(sesion_dir, tmp_path):  # noqa: F811
    _pose3d(sesion_dir, ["A-izq-20", "A-der-50"])
    d = cg.clasificar_sesion(sesion_dir, _modelo_e7(tmp_path / "m"), None, None,
                             pose3d=sesion_dir / "pose3d_hmr2.npz")
    assert d["segmentos_sin_3d"] == 1 and "B-izq-80" in d["tipos"]


def test_e4_sigue_igual_sin_pose3d_y_sin_licencia(sesion_dir, tmp_path):  # noqa: F811
    d = cg.clasificar_sesion(sesion_dir, _modelo(tmp_path / "m"), "zurda", None)
    assert d["metodo"] == "geometrico-v1" and "licencia" not in d and "segmentos_sin_3d" not in d


def test_fightcard_lleva_metodo_y_licencia_y_valida(sesion_dir, tmp_path):  # noqa: F811
    import jsonschema

    _pose3d(sesion_dir, ["A-izq-20", "A-der-50", "B-izq-80"])
    cg.clasificar_sesion(sesion_dir, _modelo_e7(tmp_path / "m"), "zurda", None,
                         pose3d=sesion_dir / "pose3d_hmr2.npz")
    fc = aplicar_tipos_a_sesion(sesion_dir)
    assert fc["clasificador"]["metodo"] == "geometrico-e7"
    assert fc["clasificador"]["licencia"] == LICENCIA
    schema = json.loads((Path(__file__).resolve().parents[2]
                         / "prototipo/contrato/fightcard-0.3.schema.json").read_text())
    jsonschema.validate(fc, schema)
    # volver a E4 saca la licencia
    cg.clasificar_sesion(sesion_dir, _modelo(tmp_path / "e4"), "zurda", None)
    fc = aplicar_tipos_a_sesion(sesion_dir)
    assert fc["clasificador"]["metodo"] == "geometrico-v1" and "licencia" not in fc["clasificador"]


def test_cli_clasificar_con_pose3d(sesion_dir, tmp_path, capsys):  # noqa: F811
    from boxtwin.cli import main

    _pose3d(sesion_dir, ["A-izq-20", "A-der-50", "B-izq-80"])
    rc = main(["clasificar", str(sesion_dir), "--modelo", str(_modelo_e7(tmp_path / "m")),
               "--pose3d", str(sesion_dir / "pose3d_hmr2.npz")])
    assert rc == 0
    assert json.loads((sesion_dir / "tipos.json").read_text())["metodo"] == "geometrico-e7"
    assert "modo academico" in capsys.readouterr().out
