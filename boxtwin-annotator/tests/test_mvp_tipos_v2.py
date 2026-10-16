"""
Tipo de golpe v2 (02-diseno.md seccion 3 y 7): `clasificar` escribe tipos.json 0.2,
aplicar_tipos con 0.1 hace lo de siempre y con 0.2 copia los campos nuevos, y CA6: una
correccion del entrenador sobrevive a la reclasificacion con v2.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boxtwin_detector")

from boxtwin.mvp.fightcard import aplicar_tipos, construir  # noqa: E402
from boxtwin.mvp.orquesta import aplicar_tipos_a_sesion, escribir_fightcard, registrar_correccion  # noqa: E402
from boxtwin.mvp.sesion import Sesion  # noqa: E402

from tests.test_mvp_clasificador_geo import _modelo, _poses  # noqa: E402

VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": 4.0, "total_frames": 120, "alto": 480, "ancho": 640}


class _Golpe:
    def __init__(self, peleador, brazo, inicio, fin):
        self.peleador, self.brazo, self.inicio, self.fin = peleador, brazo, inicio, fin
        self.t_inicio, self.t_fin, self.score = inicio / 30.0, fin / 30.0, 0.9

    def a_dict(self):
        return {"peleador": self.peleador, "brazo": self.brazo, "inicio": self.inicio, "fin": self.fin,
                "t_inicio": self.t_inicio, "t_fin": self.t_fin, "score": self.score, "cuadros_modelo": []}


GOLPES = [_Golpe("A", "left", 20, 30), _Golpe("A", "right", 50, 60), _Golpe("B", "left", 80, 90)]


@pytest.fixture
def sesion_dir(tmp_path):
    d = tmp_path / "ses"
    d.mkdir()
    ses = Sesion.nueva(d, VIDEO, round_s=180.0, descanso_s=20.0)
    kp, sc, valid = _poses()
    np.savez_compressed(d / "poses.npz", keypoints=kp, kp_score=sc, valid=valid)
    (d / "segmentos.json").write_text(json.dumps({
        "version": "0.1", "video": VIDEO, "poses": "poses.npz",
        "segmentos": [g.a_dict() for g in GOLPES]}))
    escribir_fightcard(d, construir(ses, GOLPES))
    return d


def test_clasificar_escribe_tipos_0_2(sesion_dir, tmp_path):
    from boxtwin.mvp.clasificador_geo import FAMILIAS, clasificar_sesion

    m = _modelo(tmp_path / "m")
    d = clasificar_sesion(sesion_dir, m, "zurda", None)
    escrito = json.loads((sesion_dir / "tipos.json").read_text())
    assert escrito == d
    assert d["version"] == "0.2" and d["metodo"] == "geometrico-v1"
    assert d["exactitud_familia_fuente_no_vista"] == 0.727 and d["referencia"] == "docs/x.md"
    assert d["guardia"] == {"A": "zurda", "B": "ortodoxa"}
    assert d["guardia_supuesta"] == {"A": False, "B": True}
    assert d["segmentos_sin_clasificar"] == 0
    assert set(d["tipos"]) == {"A-izq-20", "A-der-50", "B-izq-80"}
    for gid, t in d["tipos"].items():
        assert t["crudo"] in FAMILIAS and len(t["p_familia"]) == 3
        assert abs(sum(t["p_familia"]) - 1) < 1e-3
        assert t["confianza"] == max(t["p_familia"])
        # zurda en A: izquierda atrasada; B ortodoxo supuesto: izquierda adelantada
        assert t["rol"] == {"A-izq-20": "rear", "A-der-50": "lead", "B-izq-80": "lead"}[gid]
        if t["crudo"] == "straight":
            assert t["tipo"] == ("jab" if t["rol"] == "lead" else "cross")
        else:
            assert t["tipo"] == t["crudo"]


def test_segmento_sin_pasos_usables_no_se_clasifica(sesion_dir, tmp_path):
    from boxtwin.mvp.clasificador_geo import clasificar_sesion

    z = dict(np.load(sesion_dir / "poses.npz"))
    z["kp_score"][1] = 0.0
    np.savez_compressed(sesion_dir / "poses.npz", **z)
    d = clasificar_sesion(sesion_dir, _modelo(tmp_path / "m"))
    assert d["segmentos_sin_clasificar"] == 1 and "B-izq-80" not in d["tipos"]


def test_aplicar_tipos_0_1_igual_que_antes(sesion_dir):
    fc = json.loads((sesion_dir / "fightcard.json").read_text())
    tipos = {"A-izq-20": {"tipo": "jab", "confianza": 0.7, "crudo": "jab"}}
    antes = copy.deepcopy(fc)
    out = aplicar_tipos(fc, tipos, "poseC3D.pth", 0.745)
    g = out["peleadores"]["A"]["golpes"][0]
    assert (g["tipo"], g["confianza_tipo"], g["tipo_clasificador"]) == ("jab", 0.7, "jab")
    assert "p_familia" not in g and "rol" not in g
    assert not {"metodo", "referencia", "guardia_supuesta"} & set(out["clasificador"])
    # nada mas cambio respecto de lo que hacia aplicar_tipos antes de v2
    antes["clasificador"].update(checkpoint="poseC3D.pth", exactitud_familia_fuente_no_vista=0.745, estimado=True)
    ga = antes["peleadores"]["A"]["golpes"][0]
    ga.update(tipo="jab", confianza_tipo=0.7, tipo_clasificador="jab")
    out.pop("combinaciones", None), antes.pop("combinaciones", None)
    for p in ("A", "B"):
        out["peleadores"][p].pop("combinaciones", None), antes["peleadores"][p].pop("combinaciones", None)
    assert out == antes


def test_aplicar_tipos_0_2_copia_los_campos_y_volver_a_0_1_los_saca(sesion_dir, tmp_path):
    from boxtwin.mvp.clasificador_geo import clasificar_sesion

    clasificar_sesion(sesion_dir, _modelo(tmp_path / "m"), None, "zurda")
    fc = aplicar_tipos_a_sesion(sesion_dir)
    c = fc["clasificador"]
    assert c["metodo"] == "geometrico-v1" and c["referencia"] == "docs/x.md"
    assert c["guardia_supuesta"] == {"A": True, "B": False}
    assert c["exactitud_familia_fuente_no_vista"] == 0.727 and c["checkpoint"] == "prueba"
    g = fc["peleadores"]["A"]["golpes"][0]
    assert len(g["p_familia"]) == 3 and g["rol"] == "lead"
    # vuelta atras a v1: los rotulos de v2 no quedan colgados
    (sesion_dir / "tipos.json").write_text(json.dumps({
        "version": "0.1", "checkpoint": "poseC3D.pth",
        "tipos": {"A-izq-20": {"tipo": "cross", "confianza": 0.5, "crudo": "cross"}}}))
    fc = aplicar_tipos_a_sesion(sesion_dir)
    assert "metodo" not in fc["clasificador"] and "guardia_supuesta" not in fc["clasificador"]
    assert fc["clasificador"]["exactitud_familia_fuente_no_vista"] is None
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["tipo"] == "cross" and "p_familia" not in g and "rol" not in g


def test_ca6_la_correccion_sobrevive_a_v2(sesion_dir, tmp_path):
    from boxtwin.mvp.clasificador_geo import clasificar_sesion

    registrar_correccion(sesion_dir, "A-izq-20", tipo="uppercut", por="lucas")
    d = clasificar_sesion(sesion_dir, _modelo(tmp_path / "m"), "ortodoxa", "ortodoxa")
    fc = aplicar_tipos_a_sesion(sesion_dir)
    g = next(g for g in fc["peleadores"]["A"]["golpes"] if g["id"] == "A-izq-20")
    assert g["tipo"] == "uppercut" and g["corregido"]["tipo"] == "uppercut"
    assert g["tipo_modelo_nuevo"] == d["tipos"]["A-izq-20"]["tipo"]
    assert g["confianza_tipo"] is None
    otro = next(g for g in fc["peleadores"]["A"]["golpes"] if g["id"] == "A-der-50")
    assert otro["tipo"] == d["tipos"]["A-der-50"]["tipo"]


def test_cli_clasificar(sesion_dir, tmp_path, capsys):
    from boxtwin.cli import main

    m = _modelo(tmp_path / "m")
    assert main(["clasificar", str(sesion_dir), "--modelo", str(m), "--guardia-a", "zurda"]) == 0
    d = json.loads((sesion_dir / "tipos.json").read_text())
    assert d["guardia_supuesta"] == {"A": False, "B": True}
    assert "guardia supuesta ortodoxa en B" in capsys.readouterr().out
    assert main(["tipos", str(sesion_dir)]) == 0
    fc = json.loads((sesion_dir / "fightcard.json").read_text())
    assert fc["clasificador"]["metodo"] == "geometrico-v1"
