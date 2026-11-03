"""
IE3 (docs/specs/2026-11-02-identidad-estable): la API de la revision de identidad.

La lista ahora trae asignados sospechosos (y con ?todos=1, todos los asignados); las
decisiones pueden ir por rango, sobre cualquier track de la lista; un track partido tiene un
recorte por pedazo; y las dos banderas llegan al worker. Lo que hace `completar` con todo esto
se prueba en boxtwin-annotator/tests/test_mvp_identidad_estable.py.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


@pytest.fixture
def lista_nueva(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "tramos").mkdir(parents=True)
    (d / "tramos" / "track_12-900.jpg").write_bytes(b"\xff\xd8\xff")
    base = {"partes": [], "asignado": None, "recorte": None}
    (d / "sesion.json").write_text(json.dumps({
        "estado": "listo", "candidatos": [], "siembra": {"track_a": 3, "track_b": 7},
        "identidad": {"tramos": [
            {**base, "clave": "12-900", "track": 12, "desde": 900, "hasta": 1500,
             "t_desde": 30.0, "t_hasta": 50.0, "cuadro": 1200, "tipo": "sospechoso",
             "motivo": "conflicto", "rol_actual": "A", "revision_pendiente": True,
             "recorte": "tramos/track_12-900.jpg"},
            {**base, "clave": "26", "track": 26, "desde": 5760, "hasta": 6612,
             "t_desde": 192.0, "t_hasta": 220.4, "cuadro": 6200, "tipo": "sin_asignar",
             "motivo": "color_insuficiente", "rol_actual": None, "revision_pendiente": True},
            {**base, "clave": "3", "track": 3, "desde": 0, "hasta": 3000,
             "t_desde": 0.0, "t_hasta": 100.0, "cuadro": 1500, "tipo": "asignado",
             "motivo": None, "rol_actual": "A", "revision_pendiente": False},
        ]},
    }))
    (d / "fightcard.json").write_text(json.dumps({
        "identidad": {"cobertura_A": 0.7, "cobertura_B": 0.6, "sin_asignar": 0.1,
                      "separacion_color": 1.4, "revision_pendiente": 1},
        "peleadores": {},
    }))
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        s.estado, s.semilla_a, s.semilla_b = "listo", 3, 7
        db.commit()
    return sid


def _a_listo(sid):
    import boxtwin_api.db as db_mod
    from boxtwin_api.modelos import Sesion

    with db_mod.hacer_sesion() as db:
        db.get(Sesion, sid).estado = "listo"
        db.commit()


def test_la_lista_trae_los_sospechosos_primero_y_los_asignados_solo_si_se_piden(cliente, lista_nueva):
    d = cliente.get(f"/jobs/{lista_nueva}/tramos").json()
    assert [(t["clave"], t["tipo"]) for t in d["tramos"]] == [("12-900", "sospechoso"),
                                                              ("26", "sin_asignar")]
    assert d["tramos"][0]["rol_actual"] == "A" and d["revision_pendiente"] == 1
    assert d["tramos"][0]["recorte_url"] == f"/jobs/{lista_nueva}/tramos/12-900/recorte"
    todos = cliente.get(f"/jobs/{lista_nueva}/tramos?todos=1").json()["tramos"]
    assert [t["clave"] for t in todos] == ["12-900", "26", "3"]


def test_el_recorte_de_un_pedazo_entra_por_su_clave(cliente, lista_nueva):
    ticket = cliente.get(f"/videos/{lista_nueva}/ticket").json()["ticket"]
    r = cliente.get(f"/jobs/{lista_nueva}/tramos/12-900/recorte?t={ticket}")
    assert r.status_code == 200
    for mala in ("12-900.jpg", "..-1", "12-"):
        assert cliente.get(f"/jobs/{lista_nueva}/tramos/{mala}/recorte?t={ticket}").status_code == 404


def test_partir_y_corregir_un_asignado_va_a_correcciones_por_rango(cliente, lista_nueva, entorno):
    r = cliente.put(f"/jobs/{lista_nueva}/tramos", json={"asignaciones": [
        {"track": 12, "rol": "A", "desde": 30.0, "hasta": 41.5},
        {"track": 12, "rol": "B", "desde": 41.5, "hasta": 50.0},
        {"track": 3, "rol": "ninguno", "desde": 0.0, "hasta": 2.0},
    ]})
    assert r.status_code == 202, r.text
    disco = json.loads((entorno["cfg"].dir_sesion(lista_nueva) / "sesion.json").read_text())
    assert disco["identidad"]["correcciones"] == [
        {"track": 12, "desde": 30.0, "hasta": 41.5, "rol": "A"},
        {"track": 12, "desde": 41.5, "hasta": 50.0, "rol": "B"},
        {"track": 3, "desde": 0.0, "hasta": 2.0, "rol": "ninguno"},
    ]
    from boxtwin_api.modelos import AsignacionManual

    with entorno["db"].hacer_sesion() as db:
        filas = db.query(AsignacionManual).filter(AsignacionManual.track == 12).all()
        assert sorted((f.desde, f.hasta, f.rol) for f in filas) == [(30.0, 41.5, "A"),
                                                                    (41.5, 50.0, "B")]


def test_la_misma_decision_sobre_el_mismo_rango_reemplaza_a_la_anterior(cliente, lista_nueva, entorno):
    for rol in ("A", "B"):
        cliente.put(f"/jobs/{lista_nueva}/tramos",
                    json={"asignaciones": [{"track": 12, "rol": rol, "desde": 30.0, "hasta": 50.0}]})
        _a_listo(lista_nueva)
    # despues de una correccion por rango, el track entero tambien va a la lista nueva
    cliente.put(f"/jobs/{lista_nueva}/tramos", json={"asignaciones": [{"track": 26, "rol": "B"}]})
    disco = json.loads((entorno["cfg"].dir_sesion(lista_nueva) / "sesion.json").read_text())
    assert disco["identidad"]["correcciones"] == [
        {"track": 12, "desde": 30.0, "hasta": 50.0, "rol": "B"},
        {"track": 26, "desde": None, "hasta": None, "rol": "B"},
    ]
    assert disco["identidad"]["asignaciones_manuales"] == {}


def test_un_rango_vacio_se_rechaza(cliente, lista_nueva):
    r = cliente.put(f"/jobs/{lista_nueva}/tramos",
                    json={"asignaciones": [{"track": 12, "rol": "A", "desde": 40.0, "hasta": 40.0}]})
    assert r.status_code == 422


def test_las_banderas_llegan_al_completar(cliente, lista_nueva, entorno, monkeypatch):
    from boxtwin_api.modelos import Sesion, Trabajo
    from boxtwin_api.worker import comando_de

    cliente.put(f"/jobs/{lista_nueva}/tramos", json={"asignaciones": [{"track": 26, "rol": "A"}]})
    with entorno["db"].hacer_sesion() as db:
        t = db.query(Trabajo).filter(Trabajo.etapa == "completar").one()
        s = db.get(Sesion, lista_nueva)
        cmd = comando_de(t, s)[0]
        assert "--cortar-mixtos" not in cmd and "--estricta" not in cmd
        assert "--apariencia" not in cmd
        monkeypatch.setattr(entorno["cfg"], "corte_mixtos", True)
        monkeypatch.setattr(entorno["cfg"], "identidad_estricta", True)
        monkeypatch.setattr(entorno["cfg"], "identidad_apariencia", True)
        cmd = comando_de(t, s)[0]
    assert "--cortar-mixtos" in cmd and "--estricta" in cmd and "--apariencia" in cmd


def test_la_bandera_se_lee_del_entorno(monkeypatch):
    from boxtwin_api.config import Config

    monkeypatch.setenv("BOXTWIN_CORTE_MIXTOS", "1")
    monkeypatch.setenv("BOXTWIN_IDENTIDAD_ESTRICTA", "quizas")
    c = Config()
    assert c.corte_mixtos is True and c.identidad_estricta is False
    assert c.identidad_apariencia is False
    monkeypatch.setenv("BOXTWIN_IDENTIDAD_APARIENCIA", "1")
    assert Config().identidad_apariencia is True
