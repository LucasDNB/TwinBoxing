"""Z18: el sembrado de demo deja un recorrido completo por la API."""

import shutil

import pytest

from boxtwinz_api import demo

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")


def _ingresar(c, email):
    assert c.post("/api/auth/ingreso", json={"email": email, "clave": "clave-de-demo"}).status_code == 200


def test_sembrar_es_idempotente_y_recorre(nuevo_cliente):
    assert demo.sembrar("clave-de-demo")
    assert not demo.sembrar("clave-de-demo")
    f, p = nuevo_cliente(), nuevo_cliente()
    _ingresar(f, "demo-free@boxtwinz.ar")
    _ingresar(p, "demo-premium@boxtwinz.ar")
    ses = f.get("/api/sesiones").json()
    assert sorted({s["modalidad"] for s in ses}) == ["bolsa", "sombra", "sparring"] and len(ses) == 8
    r = f.get(f"/api/sesiones/{ses[0]['id']}/resultado").json()
    assert r["golpes"] and not r["valoracion_obligatoria"]
    assert f.get(f"/api/sesiones/{ses[0]['id']}/video", headers={"Range": "bytes=0-9"}).status_code == 206
    rk = f.get("/api/ranking", params={"modalidad": "bolsa"}).json()
    assert rk["total"] == 2 and rk["propia"] is not None
    amigo = f.get("/api/amigos").json()["amigos"][0]
    assert f.get(f"/api/amigos/{amigo['usuario_id']}/perfil").status_code == 200
    cajon = next(x for x in p.get("/api/perfiles").json() if x["customizado"])
    st = p.get(f"/api/perfiles/{cajon['id']}/stats", params={"modalidad": "sparring"}).json()
    assert st["sesiones"] == 2
    p1 = p.get("/api/perfiles").json()[0]["id"]
    assert p.get("/api/rival", params={"perfil_id": p1, "modalidad": "sparring", "rival_tipo": "cajon",
                                       "rival_id": cajon["id"]}).status_code == 200


def test_reiniciar(nuevo_cliente):
    demo.main(["sembrar", "--clave", "clave-de-demo"])
    demo.main(["reiniciar", "--clave", "otra-clave-larga"])
    c = nuevo_cliente()
    assert c.post("/api/auth/ingreso", json={"email": "demo-free@boxtwinz.ar", "clave": "otra-clave-larga"}).status_code == 200
    assert len(c.get("/api/sesiones").json()) == 8
