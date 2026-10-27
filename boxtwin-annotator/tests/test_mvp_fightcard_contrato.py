"""
B4 y el contrato 0.2 de la Fight-Card (prototipo/specs/05-api-y-datos.md, B4 y §4).

"No fue un golpe" es la primera senal que el detector recibe del producto: una etiqueta de
falso positivo. Lo que no puede hacer es cambiar el conteo de detectados, porque ese numero
es lo que el sistema detecto y su recall medido lo califica a el, no a la correccion.

El contrato se valida contra el mismo JSON Schema que valida la web
(prototipo/contrato/fightcard-0.3.schema.json): si uno de los dos lados cambia el documento
sin tocar el schema, falla.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from boxtwin.mvp.fightcard import (
    NO_GOLPE,
    aplicar_correcciones,
    construir,
    correccion,
)
from boxtwin.mvp.sesion import Sesion

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "prototipo/contrato/fightcard-0.3.schema.json").read_text()
)
VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": 400.0, "total_frames": 12000}


class _Golpe:
    def __init__(self, peleador, brazo, t, dur=0.25, score=0.9, fps=30.0):
        self.peleador, self.brazo = peleador, brazo
        self.t_inicio, self.t_fin = t, t + dur
        self.inicio, self.fin = int(t * fps), int((t + dur) * fps)
        self.score = score


@pytest.fixture
def fc(tmp_path):
    s = Sesion.nueva(tmp_path, VIDEO, round_s=180.0, descanso_s=20.0)
    return construir(
        s, [_Golpe("A", "left", 10.0), _Golpe("A", "right", 20.0), _Golpe("B", "left", 30.0)],
        identidad={"cobertura_A": 0.8, "cobertura_B": 0.9, "sin_asignar": 0.1,
                   "separacion_de_perfiles": 1.375},
    )


def _corregir(fc, gid, tipo):
    previas = fc.setdefault("_correcciones", [])
    previas.append(correccion(gid, tipo, fc))
    return aplicar_correcciones(fc, previas)


def test_lo_que_produce_construir_cumple_el_contrato(fc):
    jsonschema.validate(fc, SCHEMA)
    assert fc["version"] == "0.3"


def test_construir_trae_los_campos_nuevos_en_su_valor_neutro(fc):
    assert fc["peleadores"]["A"]["descartados"] == 0
    assert all(g["descartado"] is False for g in fc["peleadores"]["A"]["golpes"])
    assert fc["identidad"]["tramos_total"] is None, "sin medir es nulo, no cero"
    assert fc["identidad"]["separacion_color"] == 1.375


def test_no_fue_un_golpe_marca_descartado_y_no_cambia_el_total(fc):
    gid = fc["peleadores"]["A"]["golpes"][0]["id"]
    fc = _corregir(fc, gid, NO_GOLPE)
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["descartado"] is True
    assert g["corregido"]["tipo"] == "no_golpe"
    assert fc["peleadores"]["A"]["descartados"] == 1
    assert fc["peleadores"]["A"]["total"]["total"] == 2, "el total es lo que se detecto"
    assert len(fc["peleadores"]["A"]["golpes"]) == 2, "el golpe sigue en la lista"
    jsonschema.validate({k: v for k, v in fc.items() if k != "_correcciones"}, SCHEMA)


def test_no_golpe_y_despues_jab_deja_el_golpe_sin_descartar_y_con_tipo_jab(fc):
    gid = fc["peleadores"]["A"]["golpes"][0]["id"]
    fc = _corregir(fc, gid, NO_GOLPE)
    fc = _corregir(fc, gid, "jab")
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["descartado"] is False
    assert g["tipo"] == "jab"
    assert fc["peleadores"]["A"]["descartados"] == 0
    assert fc["peleadores"]["A"]["total"]["total"] == 2


def test_un_documento_con_conectados_no_cumple_el_contrato(fc):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**fc, "conectados": 3}, SCHEMA)


def test_una_02_sin_descartados_no_cumple_el_contrato(fc):
    roto = json.loads(json.dumps(fc))
    del roto["peleadores"]["A"]["descartados"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(roto, SCHEMA)


def test_una_01_vieja_sigue_cumpliendo(fc):
    vieja = json.loads(json.dumps(fc))
    vieja["version"] = "0.1"
    for p in vieja["peleadores"].values():
        p.pop("descartados")
        for g in p["golpes"]:
            g.pop("descartado")
    for k in ("tramos_total", "tramos_asignados_auto", "tramos_asignados_manual", "separacion_color"):
        vieja["identidad"].pop(k)
    jsonschema.validate(vieja, SCHEMA)
