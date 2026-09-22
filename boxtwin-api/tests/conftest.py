"""
Fixtures de la API: una base y un directorio de datos por test.

La configuracion se toca ANTES de importar la app, porque el engine se crea al importar
db.py. Hacerlo despues dejaria los tests corriendo contra la base de desarrollo, que es la
clase de error que se descubre cuando ya se borro algo.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1] / "src"
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Recarga los modulos con una base y un directorio nuevos."""
    monkeypatch.setenv("BOXTWIN_DB", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("BOXTWIN_DATOS", str(tmp_path / "datos"))
    monkeypatch.setenv("BOXTWIN_SECRETO", "secreto-de-prueba-que-no-cambia")
    monkeypatch.setenv("BOXTWIN_CMD", "echo")
    # El registro esta CERRADO por defecto, asi que los tests lo abren a proposito. Que
    # haya que escribirlo es parte de lo que se quiere: una instancia con el registro
    # abierto tiene que ser una decision visible.
    monkeypatch.setenv("BOXTWIN_INVITACION", "abierto")
    monkeypatch.delenv("BOXTWIN_WEB", raising=False)
    monkeypatch.setenv("BOXTWIN_INTENTOS", "2")

    import boxtwin_api.config as config

    importlib.reload(config)
    for nombre in ("boxtwin_api.db", "boxtwin_api.cola", "boxtwin_api.app",
                   "boxtwin_api.worker"):
        if nombre in sys.modules:
            importlib.reload(sys.modules[nombre])
    import boxtwin_api.app as app_mod
    import boxtwin_api.cola as cola_mod
    import boxtwin_api.db as db_mod
    import boxtwin_api.worker as worker_mod

    importlib.reload(db_mod)
    importlib.reload(cola_mod)
    importlib.reload(app_mod)
    importlib.reload(worker_mod)
    db_mod.crear_tablas()
    return {"app": app_mod, "db": db_mod, "cola": cola_mod, "worker": worker_mod,
            "cfg": config.cfg, "tmp": tmp_path}


@pytest.fixture
def cliente(entorno):
    from fastapi.testclient import TestClient

    with TestClient(entorno["app"].app) as c:
        yield c


@pytest.fixture
def registrado(cliente):
    """Un usuario con su token puesto en el cliente."""
    r = cliente.post("/auth/registro",
                     json={"email": "lucas@usal.edu.ar", "clave": "sparring2026"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    cliente.headers["Authorization"] = f"Bearer {token}"
    return token


def subir_video(cliente, nombre="spar.mp4", **datos):
    return cliente.post(
        "/videos",
        files={"archivo": (nombre, b"no es un video de verdad, nadie lo decodifica aca",
                           "video/mp4")},
        data={"nombre": "sparring del martes", **datos},
    )
