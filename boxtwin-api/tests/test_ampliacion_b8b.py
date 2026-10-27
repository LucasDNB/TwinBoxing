"""
B8b de la ampliacion del 13-10: la revision de tramos con una sola persona
(prototipo/specs/ampliacion-13-10/02 §2, prueba 3 del lado de la API).

Lo que hace el completar con una semilla y la regla de coexistencia (espejo, reentrada, el que
pasa por atras) se prueba en boxtwin-annotator/tests/test_mvp_individual.py.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


@pytest.fixture
def bolsa_lista(cliente, registrado, entorno):
    sid = subir_video(cliente, contexto="bolsa").json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps({
        "estado": "listo", "candidatos": [], "siembra": {"track_a": 3, "track_b": None},
        "identidad": {"tramos": [
            {"track": 26, "t_desde": 50.0, "t_hasta": 58.0, "cuadro": 1600,
             "motivo": "color_insuficiente", "asignado": None, "recorte": None},
        ]},
    }))
    (d / "fightcard.json").write_text(json.dumps({
        "identidad": {"cobertura_A": 0.81, "cobertura_B": 0.0, "sin_asignar": 0.12},
        "peleadores": {},
    }))
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        s.estado, s.semilla_a, s.semilla_b = "listo", 3, None
        db.commit()
    return sid


def test_la_cobertura_trae_solo_a_la_persona(cliente, bolsa_lista):
    d = cliente.get(f"/jobs/{bolsa_lista}/tramos").json()
    assert d["cobertura"] == {"A": 0.81, "sin_asignar": 0.12}


def test_un_tramo_asignado_a_B_es_422(cliente, bolsa_lista):
    r = cliente.put(f"/jobs/{bolsa_lista}/tramos", json={"asignaciones": [{"track": 26, "rol": "B"}]})
    assert r.status_code == 422 and "una sola persona" in r.json()["detail"]


def test_asignar_a_A_encola_el_completar_manual_sin_segunda_semilla(cliente, bolsa_lista, entorno):
    r = cliente.put(f"/jobs/{bolsa_lista}/tramos",
                    json={"asignaciones": [{"track": 26, "rol": "A"}]})
    assert r.status_code == 202, r.text
    from boxtwin_api.modelos import Sesion, Trabajo
    from boxtwin_api.worker import comando_de

    with entorno["db"].hacer_sesion() as db:
        t = db.query(Trabajo).filter(Trabajo.etapa == "completar").one()
        cmd = comando_de(t, db.get(Sesion, bolsa_lista))[0]
    assert "--manual" in cmd and "--semilla-b" not in cmd
