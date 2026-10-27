"""
B11: la etapa `enriquecer` (prototipo/specs/ampliacion-13-10/03 §3 y §5).

CA-14.4: sobre sesiones ya hechas no corre pose (se ve en el registro de etapas) y las deja
en 0.3. CA-14.1: cabeza + cuerpo + sin dato = detectados, por sesion y por round. CA-14.3, del
lado del anotador: corregir la altura cambia el conteo sin reprocesar y sobrevive a reenriquecer.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from boxtwin.mvp.fightcard import construir
from boxtwin.mvp.objetivo import REGLA, TAU
from boxtwin.mvp.orquesta import enriquecer, enriquecer_todas, leer_fightcard, registrar_correccion
from boxtwin.mvp.sesion import Sesion

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "prototipo/contrato/fightcard-0.3.schema.json").read_text())
T = 600
VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": T / 30.0, "total_frames": T}


class _Golpe:
    def __init__(self, peleador, brazo, f, dur=6):
        self.peleador, self.brazo = peleador, brazo
        self.inicio, self.fin = f, f + dur
        self.t_inicio, self.t_fin = f / 30.0, (f + dur) / 30.0
        self.score = 0.9


def _poses(directorio: Path, alturas: dict[int, float], rival_ausente: set[int] = frozenset()):
    """A en x=300 y B en x=500; en el cuadro f la muneca izquierda de A esta a y=alturas[f]."""
    kp = np.zeros((2, T, 17, 2), np.float32)
    for p, cx in ((0, 300.0), (1, 500.0)):
        kp[p, :, 5] = (cx - 40, 200)
        kp[p, :, 6] = (cx + 40, 200)
        kp[p, :, 9] = (cx - 30, 220)
        kp[p, :, 10] = (cx + 30, 220)
        kp[p, :, 11] = (cx - 30, 300)
        kp[p, :, 12] = (cx + 30, 300)
    for f, y in alturas.items():
        kp[0, f, 9] = (440.0, y)
    sc = np.full((2, T, 17), 0.9, np.float32)
    valid = np.ones((2, T), bool)
    for f in rival_ausente:
        valid[1, f - 3:f + 4] = False
    np.savez_compressed(directorio / "poses.npz", keypoints=kp, kp_score=sc, valid=valid)


@pytest.fixture
def sesion(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    s = Sesion.nueva(d, VIDEO, round_s=10.0, descanso_s=0.0)
    s.anotar_etapa("pose", 12.0)
    s.guardar()
    # tres golpes de A: cabeza (h = -0,3), cuerpo (h = 0,5) y uno sin rival
    _poses(d, {103: 170.0, 403: 250.0, 503: 170.0}, rival_ausente={503})
    fc = construir(s, [_Golpe("A", "left", 100), _Golpe("A", "left", 400), _Golpe("A", "left", 500)],
                   identidad={"cobertura_A": 1.0, "cobertura_B": 1.0, "sin_asignar": 0.0})
    fc["version"] = "0.2"                          # una sesion procesada antes de la ampliacion
    del fc["detector"]["medido_en"]
    for p in fc["peleadores"].values():
        del p["combinaciones"]
    (d / "fightcard.json").write_text(json.dumps(fc))
    return d


def test_CA_14_4_lleva_a_03_sin_correr_pose_y_registra_su_etapa(sesion):
    fc = enriquecer(sesion)
    assert fc["version"] == "0.3"
    assert fc["metodos"]["objetivo"]["regla"] == REGLA and fc["metodos"]["objetivo"]["tau"] == TAU
    assert fc["metodos"]["objetivo"]["medido"]["pasa"] is False
    assert fc["detector"]["medido_en"] == "sparring"
    assert "combinaciones" in fc["peleadores"]["A"]
    etapas = [e["etapa"] for e in Sesion.cargar(sesion).etapas]
    assert etapas == ["pose", "enriquecer"], "solo se suma enriquecer"
    jsonschema.validate(leer_fightcard(sesion), SCHEMA)


def test_cada_golpe_trae_su_altura_y_su_motivo(sesion):
    golpes = enriquecer(sesion)["peleadores"]["A"]["golpes"]
    assert [g["objetivo"] for g in golpes] == ["cabeza", "cuerpo", None]
    assert [g["objetivo_motivo"] for g in golpes] == [None, None, "sin_rival"]
    assert golpes[0]["objetivo_confianza"] > 0.9


def test_CA_14_1_cabeza_mas_cuerpo_mas_sin_dato_es_el_total_por_sesion_y_por_round(sesion):
    fc = enriquecer(sesion)
    for p in fc["peleadores"].values():
        t = p["total"]
        assert t["cabeza"] + t["cuerpo"] + t["objetivo_sin_dato"] == t["total"]
        for r in p["por_round"]:
            assert r["cabeza"] + r["cuerpo"] + r["objetivo_sin_dato"] == r["total"]
    a = fc["peleadores"]["A"]
    assert (a["total"]["cabeza"], a["total"]["cuerpo"], a["total"]["objetivo_sin_dato"]) == (1, 1, 1)
    assert [(r["cabeza"], r["cuerpo"]) for r in a["por_round"]] == [(1, 0), (0, 1)]


def test_es_reanudable(sesion):
    assert enriquecer(sesion) is not None
    assert enriquecer(sesion) is None
    assert enriquecer(sesion, forzar=True) is not None


def test_CA_14_3_corregir_la_altura_cambia_el_conteo_y_sobrevive_a_reenriquecer(sesion):
    fc = enriquecer(sesion)
    gid = fc["peleadores"]["A"]["golpes"][0]["id"]
    fc = registrar_correccion(sesion, gid, objetivo="cuerpo", por="lucas")
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["objetivo"] == "cuerpo" and g["objetivo_corregido"]["objetivo_original"] == "cabeza"
    assert fc["peleadores"]["A"]["total"]["cuerpo"] == 2
    fc = enriquecer(sesion, forzar=True)
    assert fc["peleadores"]["A"]["golpes"][0]["objetivo"] == "cuerpo"
    linea = json.loads((sesion / "correcciones.jsonl").read_text().splitlines()[-1])
    assert linea["campo"] == "objetivo" and linea["objetivo_h"] == pytest.approx(-0.3)


def test_tipo_y_altura_se_corrigen_por_separado(sesion):
    fc = enriquecer(sesion)
    gid = fc["peleadores"]["A"]["golpes"][0]["id"]
    registrar_correccion(sesion, gid, objetivo="cuerpo")
    fc = registrar_correccion(sesion, gid, tipo="jab")
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["tipo"] == "jab" and g["objetivo"] == "cuerpo"
    with pytest.raises(ValueError):
        registrar_correccion(sesion, gid, objetivo="pierna")
    with pytest.raises(ValueError):
        registrar_correccion(sesion, gid, tipo="jab", objetivo="cuerpo")


def test_DA_5_en_bolsa_no_se_calcula(sesion):
    fc = json.loads((sesion / "fightcard.json").read_text())
    del fc["peleadores"]["B"]
    (sesion / "fightcard.json").write_text(json.dumps(fc))
    golpes = enriquecer(sesion)["peleadores"]["A"]["golpes"]
    assert {g["objetivo"] for g in golpes} == {None}
    assert {g["objetivo_motivo"] for g in golpes} == {"individual"}


def test_todas_recorre_y_no_frena_por_una_rota(sesion):
    rota = sesion.parent / "s2"
    rota.mkdir()
    (rota / "fightcard.json").write_text("{}")
    r = enriquecer_todas(sesion.parent)
    assert r == {"s1": "enriquecida", "s2": "sin poses.npz"}
    assert enriquecer_todas(sesion.parent)["s1"] == "ya estaba"


def test_B15_5_una_sesion_sin_rounds_sale_con_segmentos_y_suman_el_total(sesion):
    # Una Fight-Card procesada sin rounds declarados (antes del 14-10 quedaba sin agregado).
    fc = json.loads((sesion / "fightcard.json").read_text())
    fc["video"]["rounds"] = []
    fc["video"].pop("rounds_origen", None)
    for p in fc["peleadores"].values():
        p["por_round"] = []
    (sesion / "fightcard.json").write_text(json.dumps(fc))
    fc = enriquecer(sesion)
    assert fc["video"]["rounds_origen"] == "segmentos_180s"
    assert len(fc["video"]["rounds"]) == 1, "20 s de video: un solo segmento"
    a = fc["peleadores"]["A"]
    assert sum(r["total"] for r in a["por_round"]) == a["total"]["total"] == 3
    assert sum(r["cabeza"] + r["cuerpo"] + r["objetivo_sin_dato"] for r in a["por_round"]) == 3
    jsonschema.validate(fc, SCHEMA)
    assert enriquecer(sesion) is None, "reanudable: la segunda pasada no hace nada"
