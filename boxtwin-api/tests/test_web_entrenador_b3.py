"""
B3 de la web entrenador del lado de la API: listar tramos, recorte con ticket, y asignar a
mano encolando el completar manual (prototipo/specs/05-api-y-datos.md).

Lo que hace el completar manual con las asignaciones -subir la cobertura, sumar los golpes,
no correr la pose- se prueba en boxtwin-annotator/tests/test_mvp_tramos.py, que es donde
vive. Aca se prueba el cableado.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


@pytest.fixture
def lista_con_tramos(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "tramos").mkdir(parents=True)
    (d / "tramos" / "track_26.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "sesion.json").write_text(json.dumps({
        "estado": "listo", "candidatos": [], "siembra": {"track_a": 3, "track_b": 7},
        "identidad": {"tramos": [
            {"track": 26, "t_desde": 192.0, "t_hasta": 220.4, "cuadro": 6200,
             "motivo": "color_insuficiente", "asignado": None, "recorte": "tramos/track_26.jpg"},
            {"track": 29, "t_desde": 10.0, "t_hasta": 12.5, "cuadro": 340,
             "motivo": "sin_coexistencia", "asignado": None, "recorte": None},
        ]},
    }))
    (d / "fightcard.json").write_text(json.dumps({
        "identidad": {"cobertura_A": 0.62, "cobertura_B": 0.58, "sin_asignar": 0.18,
                      "separacion_color": 1.375},
        "peleadores": {},
    }))
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        s.estado, s.semilla_a, s.semilla_b = "listo", 3, 7
        db.commit()
    return sid


def test_lista_los_tramos_con_cobertura_y_motivo(cliente, lista_con_tramos):
    d = cliente.get(f"/jobs/{lista_con_tramos}/tramos").json()
    assert d["cobertura"] == {"A": 0.62, "B": 0.58, "sin_asignar": 0.18}
    assert d["separacion_color"] == 1.375
    t26, t29 = d["tramos"]
    assert t26 == {"track": 26, "t_desde": 192.0, "t_hasta": 220.4, "motivo": "color_insuficiente",
                   "recorte_url": f"/jobs/{lista_con_tramos}/tramos/26/recorte", "asignado": None}
    assert t29["recorte_url"] is None


def test_el_recorte_del_tramo_entra_con_ticket(cliente, lista_con_tramos):
    ticket = cliente.get(f"/videos/{lista_con_tramos}/ticket").json()["ticket"]
    token = cliente.headers.pop("Authorization")
    try:
        r = cliente.get(f"/jobs/{lista_con_tramos}/tramos/26/recorte?t={ticket}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert cliente.get(f"/jobs/{lista_con_tramos}/tramos/26/recorte").status_code == 401
    finally:
        cliente.headers["Authorization"] = token


def test_asignar_guarda_en_disco_y_en_la_base_y_encola_el_completar_manual(cliente, lista_con_tramos, entorno):
    r = cliente.put(f"/jobs/{lista_con_tramos}/tramos",
                    json={"asignaciones": [{"track": 26, "rol": "A"}, {"track": 29, "rol": "ninguno"}]})
    assert r.status_code == 202, r.text
    assert r.json()["estado"] == "en_cola"

    disco = json.loads((entorno["cfg"].dir_sesion(lista_con_tramos) / "sesion.json").read_text())
    assert disco["identidad"]["asignaciones_manuales"] == {"26": "A", "29": "ninguno"}

    from boxtwin_api.modelos import AsignacionManual, Trabajo
    from boxtwin_api.worker import comando_de
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        filas = db.query(AsignacionManual).order_by(AsignacionManual.track).all()
        assert [(f.track, f.rol) for f in filas] == [(26, "A"), (29, "ninguno")]
        t = db.query(Trabajo).filter(Trabajo.etapa == "completar").one()
        assert t.parametros == {"track_a": 3, "track_b": 7, "manual": True}
        cmd = comando_de(t, db.get(Sesion, lista_con_tramos))[0]
    assert "completar" in cmd and "--manual" in cmd
    assert "procesar" not in cmd, "no se vuelve a correr la pose"


def test_la_asignacion_se_ve_en_la_lista(cliente, lista_con_tramos):
    cliente.put(f"/jobs/{lista_con_tramos}/tramos", json={"asignaciones": [{"track": 26, "rol": "B"}]})
    from boxtwin_api.modelos import Sesion  # noqa: F401
    # La sesion queda en cola: se vuelve a lista para mirar.
    import boxtwin_api.db as db_mod

    with db_mod.hacer_sesion() as db:
        db.get(Sesion, lista_con_tramos).estado = "listo"
        db.commit()
    tramos = cliente.get(f"/jobs/{lista_con_tramos}/tramos").json()["tramos"]
    assert tramos[0]["asignado"] == "B"


def test_un_track_que_no_es_tramo_de_la_sesion_se_rechaza(cliente, lista_con_tramos):
    r = cliente.put(f"/jobs/{lista_con_tramos}/tramos", json={"asignaciones": [{"track": 999, "rol": "A"}]})
    assert r.status_code == 422


def test_un_rol_inventado_se_rechaza(cliente, lista_con_tramos):
    r = cliente.put(f"/jobs/{lista_con_tramos}/tramos", json={"asignaciones": [{"track": 26, "rol": "arbitro"}]})
    assert r.status_code == 422


def test_una_sesion_procesada_antes_de_b3_lo_dice(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    (entorno["cfg"].dir_sesion(sid) / "sesion.json").write_text(json.dumps({"estado": "listo", "identidad": {}}))
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, sid).estado = "listo"
        db.commit()
    r = cliente.get(f"/jobs/{sid}/tramos")
    assert r.status_code == 409 and "antes de la revision de tramos" in r.json()["detail"]


def test_otro_usuario_no_ve_ni_asigna_tramos(cliente, lista_con_tramos):
    r = cliente.post("/auth/registro", json={"email": "otro@b.com", "clave": "otraclave1"})
    cliente.headers["Authorization"] = f"Bearer {r.json()['token']}"
    assert cliente.get(f"/jobs/{lista_con_tramos}/tramos").status_code == 404
    assert cliente.put(f"/jobs/{lista_con_tramos}/tramos",
                       json={"asignaciones": [{"track": 26, "rol": "A"}]}).status_code == 404


def test_re_sembrar_saca_las_asignaciones_manuales_del_disco(cliente, lista_con_tramos, entorno):
    cliente.put(f"/jobs/{lista_con_tramos}/tramos", json={"asignaciones": [{"track": 26, "rol": "A"}]})
    from boxtwin_api.modelos import AsignacionManual, Sesion

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, lista_con_tramos).estado = "listo"
        db.commit()
    cliente.post(f"/jobs/{lista_con_tramos}/siembra", json={"track_a": 7, "track_b": 3})
    disco = json.loads((entorno["cfg"].dir_sesion(lista_con_tramos) / "sesion.json").read_text())
    assert "asignaciones_manuales" not in disco["identidad"]
    with entorno["db"].hacer_sesion() as db:
        assert db.query(AsignacionManual).count() == 1, "en la base quedan, para auditoria"


def test_con_un_trabajo_en_cola_el_disco_no_hace_retroceder_el_estado(cliente, registrado, entorno):
    # Encontrado en el e2e: recien sembrada, la sesion esta en_cola y sesion.json todavia dice
    # espera_siembra porque el worker no la tomo. GET /jobs/{id} no puede devolverla atras.
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps({"estado": "espera_siembra", "candidatos": [
        {"track": 3, "recorte": None}, {"track": 7, "recorte": None}]}))
    from boxtwin_api.modelos import Sesion, Trabajo

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, sid).estado = "espera_siembra"
        for t in db.query(Trabajo).filter(Trabajo.sesion_id == sid):
            t.estado = "listo"
        db.commit()
    assert cliente.get(f"/jobs/{sid}").json()["estado"] == "espera_siembra"
    cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3, "track_b": 7})
    assert cliente.get(f"/jobs/{sid}").json()["estado"] == "en_cola"
    assert cliente.get(f"/jobs/{sid}").json()["estado"] == "en_cola", "y no cambia al pedir de nuevo"
