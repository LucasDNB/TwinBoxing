"""Z12: denuncias sobre perfiles visibles y su notificacion al admin."""

from sqlalchemy import select

from boxtwinz_api.models import Denuncia, NotificacionAdmin
from conftest import registrar


def test_denuncia_a_perfil_publico(nuevo_cliente, db):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]["id"]
    a.patch(f"/api/perfiles/{p1}", json={"publico": True})
    r = b.post("/api/denuncias", json={"perfil_id": p1, "motivo": "menor_en_video"})
    assert r.status_code == 201 and r.json()["estado"] == "abierta"
    assert db.scalar(select(Denuncia)).motivo == "menor_en_video"
    assert db.scalar(select(NotificacionAdmin)).tipo == "denuncia_nueva"


def test_no_se_denuncia_lo_que_no_se_ve(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    ps = a.get("/api/perfiles").json()
    assert b.post("/api/denuncias", json={"perfil_id": ps[0]["id"], "motivo": "otro", "detalle": "x"}).status_code == 404
    assert b.post("/api/denuncias", json={"perfil_id": ps[1]["id"], "motivo": "video_ajeno"}).status_code == 404
    assert a.post("/api/denuncias", json={"perfil_id": ps[0]["id"], "motivo": "video_ajeno"}).status_code == 404


def test_motivo_otro_pide_detalle_y_motivo_valido(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]["id"]
    a.patch(f"/api/perfiles/{p1}", json={"publico": True})
    assert b.post("/api/denuncias", json={"perfil_id": p1, "motivo": "otro"}).status_code == 422
    assert b.post("/api/denuncias", json={"perfil_id": p1, "motivo": "no_me_gusta"}).status_code == 422
