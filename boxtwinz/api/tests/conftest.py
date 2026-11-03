"""Fixtures: base sqlite por test, datos en tmp, cliente HTTP y fabricas de usuarios."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from boxtwinz_api import db as dbmod
from boxtwinz_api.config import cfg


@pytest.fixture(autouse=True)
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "datos", tmp_path / "datos")
    monkeypatch.setattr(cfg, "lock_gpu", tmp_path / "gpu.lock")
    monkeypatch.setattr(cfg, "admins", ("admin@boxtwinz.ar",))
    monkeypatch.setattr(cfg, "secreto", "secreto-de-test")
    dbmod.configurar(f"sqlite:///{tmp_path / 'z.db'}")
    dbmod.crear_tablas()
    yield
    dbmod.engine.dispose()


@pytest.fixture
def db():
    with dbmod.hacer_sesion() as s:
        yield s


@pytest.fixture
def app():
    from boxtwinz_api.main import app as a

    return a


def registrar(cliente, email="ana@boxtwinz.ar", guardia="ortodoxa", reent=False, **extra):
    r = cliente.post("/api/auth/registro", json={
        "email": email, "clave": "una-clave-larga", "nombre": email.split("@")[0],
        "guardia": guardia, "consentimiento_uso": True, "consentimiento_reentrenamiento": reent, **extra})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def cliente(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def nuevo_cliente(app):
    """Varios clientes, cada uno con su cookie: para amigos, ranking, denuncias."""
    abiertos = []

    def fabrica():
        c = TestClient(app)
        c.__enter__()
        abiertos.append(c)
        return c

    yield fabrica
    for c in abiertos:
        c.__exit__(None, None, None)
