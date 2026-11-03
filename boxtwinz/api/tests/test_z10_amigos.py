"""Z10: un amigo ve el perfil 1 y sus sesiones, y nunca los cajones 2 a 4 (CZ4)."""

import json

from conftest import golpes_cada, registrar, sesion_lista


def _amigos(a, b):
    a.post("/api/amigos/solicitud", json={"email": "b@x.ar"})
    sol = b.get("/api/amigos").json()["recibidas"][0]
    assert b.post(f"/api/amigos/{sol['id']}/aceptar").json()["estado"] == "aceptada"
    return sol


def test_amigo_ve_perfil_1_y_nunca_cajones(nuevo_cliente, db):
    a, b = nuevo_cliente(), nuevo_cliente()
    ya = registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    a.post("/api/yo/premium")
    cajon = a.get("/api/perfiles").json()[1]
    a.patch(f"/api/perfiles/{cajon['id']}", json={"customizado": True, "nombre": "Sparring Secreto"})
    sesion_lista(db, "a@x.ar", "sparring", golpes_cada(1.0, 240) + golpes_cada(2.0, 240, "rival"), rival_perfil_id=cajon["id"])
    sesion_lista(db, "a@x.ar", "bolsa", golpes_cada(1.0, 200))
    _amigos(a, b)
    d = b.get(f"/api/amigos/{ya['id']}/perfil").json()
    assert d["perfil_id"] == a.get("/api/perfiles").json()[0]["id"]
    assert {s["modalidad"] for s in d["sesiones"]} == {"sparring", "bolsa"}
    texto = json.dumps(d)
    assert cajon["id"] not in texto and "Sparring Secreto" not in texto
    for p in a.get("/api/perfiles").json()[1:]:
        assert p["id"] not in texto


def test_no_amigo_403(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    ya = registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    assert b.get(f"/api/amigos/{ya['id']}/perfil").status_code == 403
    a.post("/api/amigos/solicitud", json={"email": "b@x.ar"})          # pendiente tampoco alcanza
    assert b.get(f"/api/amigos/{ya['id']}/perfil").status_code == 403


def test_solo_el_destinatario_responde(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    a.post("/api/amigos/solicitud", json={"email": "b@x.ar"})
    sid = b.get("/api/amigos").json()["recibidas"][0]["id"]
    assert a.post(f"/api/amigos/{sid}/aceptar").status_code == 404
    assert b.post(f"/api/amigos/{sid}/rechazar").json()["estado"] == "rechazada"
    assert b.post(f"/api/amigos/{sid}/aceptar").status_code == 409


def test_solicitud_no_revela_si_existe_el_email(cliente):
    registrar(cliente)
    r = cliente.post("/api/amigos/solicitud", json={"email": "nadie@x.ar"})
    assert r.status_code == 201 and r.json() == {"enviada": True}


def test_solicitudes_cruzadas_quedan_aceptadas(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    a.post("/api/amigos/solicitud", json={"email": "b@x.ar"})
    b.post("/api/amigos/solicitud", json={"email": "a@x.ar"})
    assert len(a.get("/api/amigos").json()["amigos"]) == 1
