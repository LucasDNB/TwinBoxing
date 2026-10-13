"""
B8 de la web entrenador: sesiones de un solo peleador (prototipo/specs/05-api-y-datos.md).

El cableado se prueba aca; si la identidad de una sola persona llega al 95% de los cuadros
en material real se mide sobre dos videos propios antes de encender la bandera, y eso no es
una prueba unitaria: es el criterio de B8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from boxtwin.core.identidad_auto import ConfigIdentidadAuto, EvidenciaTrack, guardar_evidencia  # noqa: E402
from boxtwin.core.types import TrackRole  # noqa: E402
from boxtwin.mvp.individual import proponer_individual  # noqa: E402
from boxtwin.mvp.orquesta import completar  # noqa: E402
from boxtwin.mvp.sesion import Sesion  # noqa: E402
from tests.test_mvp_tramos import AZUL, MITAD, ROJO, T, detector, sesion  # noqa: E402,F401

import jsonschema  # noqa: E402

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "prototipo/contrato/fightcard-0.2.schema.json").read_text())


def _ev(t, primero, ultimo, color, n_colores=None):
    frames = list(range(primero, ultimo + 1, 5))
    cols = frames if n_colores is None else frames[:n_colores]
    return EvidenciaTrack(t, primero, ultimo, 0.61, len(frames), len(frames),
                          [(f, color) for f in cols], frames, [(f, 300.0) for f in frames])


def test_lo_que_coexiste_con_el_es_otro_y_lo_del_mismo_color_es_el():
    ev = {1: _ev(1, 0, 199, ROJO), 2: _ev(2, 0, 399, AZUL), 3: _ev(3, 200, 399, ROJO),
          4: _ev(4, 200, 399, AZUL)}
    prop = proponer_individual(ev, ConfigIdentidadAuto(), T, 30.0, semilla=1)
    assert prop.roles[1] is TrackRole.A
    assert prop.roles[2] is TrackRole.IGNORE, "coexiste con la semilla"
    assert prop.roles[3] is TrackRole.A, "mismo color y no coexiste"
    assert prop.roles[4] is TrackRole.IGNORE, "coexiste con el 3, que ya es A"


def test_sin_color_para_votar_queda_sin_asignar():
    ev = {1: _ev(1, 0, 199, ROJO), 3: _ev(3, 200, 399, ROJO, n_colores=2)}
    prop = proponer_individual(ev, ConfigIdentidadAuto(), T, 30.0, semilla=1)
    assert prop.sin_asignar == [3]


def test_completar_con_una_sola_semilla_da_una_fightcard_solo_con_A(sesion, detector):  # noqa: F811
    # En la sesion de prueba el 3 no tiene color: queda sin asignar, y el 2 coexiste: ninguno.
    fc = completar(sesion, semilla_a=1, semilla_b=None, detector=detector, umbral=0.8, device="cpu")
    assert set(fc["peleadores"]) == {"A"}
    assert fc["lectura"]["volumen_relativo"]["B"] is None
    jsonschema.validate(fc, SCHEMA)
    assert Sesion.cargar(sesion).siembra == {"track_a": 1, "track_b": None}
