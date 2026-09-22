"""
BoxTwin API - Conexion y sesion de base de datos.

POR QUE EXISTE
  Para que la cola pueda escribirse una sola vez y correr sobre PostgreSQL en produccion y
  sobre sqlite en los tests. La diferencia que importa no es el dialecto sino el bloqueo:
  SKIP LOCKED existe en uno y no en el otro, y eso esta resuelto en `cola.py`, no aca.

QUE HACE
  Crea el engine, expone una sesion por request y arma las tablas.

USO
  from boxtwin_api.db import obtener_sesion, crear_tablas
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from boxtwin_api.config import cfg
from boxtwin_api.modelos import Base

__all__ = ["crear_tablas", "engine", "hacer_sesion", "obtener_sesion", "es_postgres"]


def _crear_engine(url: str):
    if url.startswith("sqlite"):
        # check_same_thread=False porque el worker y la API comparten proceso en
        # desarrollo. En produccion son procesos distintos contra PostgreSQL.
        e = create_engine(url, future=True, connect_args={"check_same_thread": False})

        @event.listens_for(e, "connect")
        def _pragmas(dbapi, _record):  # pragma: no cover - configuracion del driver
            cur = dbapi.cursor()
            # WAL para que el worker pueda escribir mientras la API lee, que es
            # exactamente lo que pasa cuando el usuario refresca la pantalla de estado.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        return e
    return create_engine(url, future=True, pool_pre_ping=True)


engine = _crear_engine(cfg.base_de_datos)
hacer_sesion = sessionmaker(engine, expire_on_commit=False, class_=Session)


def es_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def crear_tablas() -> None:
    if cfg.base_de_datos.startswith("sqlite:///"):
        ruta = cfg.base_de_datos.replace("sqlite:///", "", 1)
        if ruta and ruta != ":memory:":
            Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def obtener_sesion() -> Iterator[Session]:
    with hacer_sesion() as s:
        yield s
