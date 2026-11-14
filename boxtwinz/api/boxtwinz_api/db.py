"""
BoxTwinZ API - Conexion y sesion de base de datos.

POR QUE EXISTE
  Igual que en boxtwin-api: la cola tiene que correr sobre PostgreSQL con SKIP LOCKED y sobre
  sqlite en los tests y en la estacion, que hoy no tiene PostgreSQL (desvio D2 en TAREAS.md).

QUE HACE
  Crea el engine (WAL y busy_timeout en sqlite, como A), expone una sesion por request, arma
  las tablas y carga los arquetipos del seed. El engine se puede reconfigurar para los tests.

USO
  from boxtwinz_api.db import obtener_sesion, crear_tablas, hacer_sesion
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from boxtwinz_api.config import cfg
from boxtwinz_api.models import Base

__all__ = ["configurar", "crear_tablas", "es_postgres", "hacer_sesion", "obtener_sesion"]


def _crear_engine(url: str):
    if url.startswith("sqlite"):
        e = create_engine(url, future=True, connect_args={"check_same_thread": False})

        @event.listens_for(e, "connect")
        def _pragmas(dbapi, _record):  # pragma: no cover - configuracion del driver
            cur = dbapi.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        return e
    return create_engine(url, future=True, pool_pre_ping=True)


engine = _crear_engine(cfg.base_de_datos)
hacer_sesion = sessionmaker(engine, expire_on_commit=False, class_=Session)


def configurar(url: str) -> None:
    """Cambia la base. Lo usan los tests; en produccion se fija por BOXTWINZ_DB."""
    global engine
    engine.dispose()
    engine = _crear_engine(url)
    hacer_sesion.configure(bind=engine)


def es_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def _columnas_faltantes() -> list[str]:
    """
    ADD COLUMN de lo que el modelo tiene y la tabla no (copiado de boxtwin-api/db.py, desvio
    D1: sin Alembic). `create_all` no altera tablas existentes, y la base desplegada ya tiene
    filas. No borra, no renombra ni cambia tipos; una columna obligatoria sin default frena.
    """
    from sqlalchemy import inspect, text

    agregadas = []
    insp = inspect(engine)
    existentes = set(insp.get_table_names())
    with engine.begin() as con:
        for tabla in Base.metadata.sorted_tables:
            if tabla.name not in existentes:
                continue
            hay = {c["name"] for c in insp.get_columns(tabla.name)}
            for col in tabla.columns:
                if col.name in hay:
                    continue
                if not col.nullable and col.server_default is None:
                    raise RuntimeError(f"{tabla.name}.{col.name} es obligatoria: hay que migrarla a mano")
                con.execute(text(f"ALTER TABLE {tabla.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"))
                agregadas.append(f"{tabla.name}.{col.name}")
    return agregadas


def crear_tablas() -> None:
    url = str(engine.url)
    if url.startswith("sqlite:///"):
        ruta = url.replace("sqlite:///", "", 1)
        if ruta and ruta != ":memory:":
            Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    _columnas_faltantes()
    Base.metadata.create_all(engine)
    from boxtwinz_api.services.archetypes import cargar_seed

    with hacer_sesion() as s:
        cargar_seed(s)
        s.commit()


def obtener_sesion() -> Iterator[Session]:
    with hacer_sesion() as s:
        yield s
