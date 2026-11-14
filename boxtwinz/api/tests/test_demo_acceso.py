"""Login de demo: clave, selector de usuarios, crear usuario rapido y amistad entre todos."""

import shutil

import pytest
from sqlalchemy import inspect, text

from boxtwinz_api import db as dbmod
from boxtwinz_api.config import cfg

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")


@pytest.fixture(autouse=True)
def clave(monkeypatch):
    monkeypatch.setattr(cfg, "demo", True)
    monkeypatch.setattr(cfg, "demo_clave", "USAL@2026")


def _entrar_demo(c):
    r = c.post("/api/demo/acceso", json={"clave": "USAL@2026"})
    assert r.status_code == 204, r.text


def test_clave_incorrecta_y_sin_cookie(cliente):
    assert cliente.get("/api/demo").json() == {"habilitada": True}
    r = cliente.post("/api/demo/acceso", json={"clave": "usal@2026"})
    assert r.status_code == 401 and r.json()["detail"]["codigo"] == "clave_demo_invalida"
    assert cliente.get("/api/demo/usuarios").status_code == 401
    assert cliente.post("/api/demo/usuarios", json={"nombre": "X", "guardia": "zurda", "peso_kg": 70}).status_code == 401


def test_jwt_de_usuario_no_abre_la_demo(cliente):
    from conftest import registrar

    registrar(cliente)
    cliente.cookies.set("boxtwinz_demo", cliente.cookies.get("boxtwinz_sesion"))
    assert cliente.get("/api/demo/usuarios").status_code == 401


def test_demo_apagada(cliente, monkeypatch):
    monkeypatch.setattr(cfg, "demo", False)
    assert cliente.post("/api/demo/acceso", json={"clave": "USAL@2026"}).status_code == 404


def test_estandar_crear_ingresar_y_amigos(nuevo_cliente):
    c = nuevo_cliente()
    _entrar_demo(c)
    _entrar_demo(c)   # idempotente: no duplica los estandar
    us = c.get("/api/demo/usuarios").json()
    assert [(u["nombre"], u["estilo"], u["guardia"]) for u in us] == [
        ("Fajador", "fajador", "ortodoxa"), ("Estilista", "estilista", "ortodoxa"), ("Zurdo", None, "zurda")]
    assert all(u["estandar"] and u["peso_kg"] for u in us)

    r = c.post("/api/demo/usuarios", json={"nombre": " Lucas ", "estilo": None, "guardia": "zurda", "peso_kg": 72.25})
    assert r.status_code == 201
    nuevo = r.json()
    assert nuevo == {**nuevo, "nombre": "Lucas", "estilo": None, "guardia": "zurda", "peso_kg": 72.2, "estandar": False}
    assert c.post("/api/demo/usuarios", json={"nombre": "Y", "estilo": "otro", "guardia": "zurda", "peso_kg": 70}).status_code == 422
    assert c.post("/api/demo/usuarios", json={"nombre": "Y", "guardia": "zurda", "peso_kg": 10}).status_code == 422
    assert [u["nombre"] for u in c.get("/api/demo/usuarios").json()][-1] == "Lucas"

    # Ingresa como el nuevo: es amigo de los tres estandar y ve sus perfiles.
    yo = c.post(f"/api/demo/usuarios/{nuevo['id']}/ingresar").json()
    assert yo["demo"] is True and yo["nombre"] == "Lucas"
    amigos = c.get("/api/amigos").json()["amigos"]
    assert sorted(a["nombre"] for a in amigos) == ["Estilista", "Fajador", "Zurdo"]
    for a in amigos:
        assert c.get(f"/api/amigos/{a['usuario_id']}/perfil").status_code == 200

    # Un segundo creado tambien es amigo del primero.
    otro = c.post("/api/demo/usuarios", json={"nombre": "Ana", "estilo": "estilista", "guardia": "ortodoxa", "peso_kg": 60}).json()
    c.post(f"/api/demo/usuarios/{otro['id']}/ingresar")
    assert "Lucas" in {a["nombre"] for a in c.get("/api/amigos").json()["amigos"]}

    # Los estandar salen en el ranking de las tres modalidades.
    for m in ("sombra", "bolsa", "sparring"):
        rk = c.get("/api/ranking", params={"modalidad": m}).json()
        assert {"Fajador", "Estilista", "Zurdo"} <= {x["nombre"] for x in rk["top"]}
    # El fajador lanza mas que el estilista en bolsa (los parametros sinteticos lo dicen).
    rk = {x["nombre"]: x["valor"] for x in c.get("/api/ranking", params={"modalidad": "bolsa"}).json()["top"]}
    assert rk["Fajador"] > rk["Estilista"]


def test_no_ingresa_como_usuario_comun(nuevo_cliente):
    from conftest import registrar

    comun = registrar(nuevo_cliente(), email="real@boxtwinz.ar")
    c = nuevo_cliente()
    _entrar_demo(c)
    assert c.post(f"/api/demo/usuarios/{comun['id']}/ingresar").status_code == 404
    assert "real" not in {u["nombre"] for u in c.get("/api/demo/usuarios").json()}


def test_columnas_nuevas_en_base_existente():
    # Una base de antes de este cambio, con filas: las tablas sin las columnas nuevas.
    from conftest import registrar
    from fastapi.testclient import TestClient

    from boxtwinz_api.main import app

    with TestClient(app) as c:
        registrar(c, email="vieja@boxtwinz.ar")
    with dbmod.engine.begin() as con:
        for t, col in (("usuarios", "demo"), ("perfiles", "estilo_declarado"), ("perfiles", "peso_kg")):
            con.execute(text(f"ALTER TABLE {t} DROP COLUMN {col}"))
    dbmod.engine.dispose()   # como al reiniciar el servicio: conexiones nuevas, esquema nuevo
    agregadas = dbmod._columnas_faltantes()
    assert {"usuarios.demo", "perfiles.estilo_declarado", "perfiles.peso_kg"} <= set(agregadas)
    assert "demo" in {c["name"] for c in inspect(dbmod.engine).get_columns("usuarios")}
    with TestClient(app) as c:   # la cuenta vieja sigue entrando y no es de demo
        r = c.post("/api/auth/ingreso", json={"email": "vieja@boxtwinz.ar", "clave": "una-clave-larga"})
        assert r.status_code == 200 and r.json()["demo"] is False
