"""
BoxTwinZ API - La cola de trabajos, sobre una tabla. Copiada de boxtwin-api/cola.py.

POR QUE EXISTE
  Mismas razones que en A: un trabajo dura minutos y la GPU es una. En PostgreSQL la exclusion
  es SELECT ... FOR UPDATE SKIP LOCKED; en sqlite, un UPDATE condicional. Se agrega lo que la
  spec pide (inicio_gpu, fin_gpu, log) y la devolucion del cupo cuando el fallo es definitivo.

USO
  from boxtwinz_api.cola import encolar, reclamar, terminar, fallar
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.db import es_postgres
from boxtwinz_api.models import ETAPAS, NotificacionAdmin, Sesion, Trabajo, ahora

__all__ = ["encolar", "fallar", "liberar_abandonados", "reclamar", "terminar"]


def encolar(db: Session, sesion_id: str, etapa: str, **parametros) -> Trabajo:
    """Pone una etapa en la cola. No commitea: el que llama decide la transaccion."""
    if etapa not in ETAPAS:
        raise ValueError(f"etapa desconocida: {etapa!r}, se esperaba una de {ETAPAS}")
    previos = db.scalars(select(Trabajo).where(Trabajo.sesion_id == sesion_id, Trabajo.etapa == etapa)).all()
    t = Trabajo(sesion_id=sesion_id, etapa=etapa, parametros=parametros, intento=len(previos) + 1)
    db.add(t)
    return t


def reclamar(db: Session, worker: str) -> Trabajo | None:
    """Toma el trabajo mas viejo de la cola, o None. Commitea (ver cola.py de A)."""
    liberar_abandonados(db)
    if es_postgres():
        fila = db.execute(
            select(Trabajo).where(Trabajo.estado == "en_cola").order_by(Trabajo.creado)
            .limit(1).with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if fila is None:
            db.rollback()
            return None
        fila.estado, fila.tomado_por, fila.tomado_en = "tomado", worker, ahora()
        db.commit()
        return fila

    fila = db.execute(
        select(Trabajo).where(Trabajo.estado == "en_cola").order_by(Trabajo.creado).limit(1)
    ).scalar_one_or_none()
    if fila is None:
        db.rollback()
        return None
    r = db.execute(
        update(Trabajo).where(Trabajo.id == fila.id, Trabajo.estado == "en_cola")
        .values(estado="tomado", tomado_por=worker, tomado_en=ahora())
    )
    db.commit()
    if r.rowcount != 1:
        return None
    db.refresh(fila)
    return fila


def terminar(db: Session, trabajo: Trabajo, estado_sesion: str | None = None) -> None:
    trabajo.estado = "listo"
    trabajo.terminado_en = ahora()
    if estado_sesion:
        ses = db.get(Sesion, trabajo.sesion_id)
        if ses:
            ses.estado = estado_sesion
            ses.error = None
            ses.error_etapa = None
    db.commit()


def fallar(db: Session, trabajo: Trabajo, error: str) -> Trabajo | None:
    """
    Marca el fallo y reintenta si quedan intentos. En el fallo definitivo la sesion pasa a
    error con su etapa, se devuelve el cupo (spec 9: un error del sistema no lo consume) y se
    avisa al admin (fallo_trabajo).
    """
    from boxtwinz_api.services.quota import devolver

    trabajo.estado = "fallo"
    trabajo.terminado_en = ahora()
    trabajo.log = ((trabajo.log or "") + "\n" + error)[-8000:]
    ses = db.get(Sesion, trabajo.sesion_id)
    nuevo = None
    if trabajo.intento < cfg.intentos_maximos:
        nuevo = encolar(db, trabajo.sesion_id, trabajo.etapa, **(trabajo.parametros or {}))
        if ses:
            ses.error = f"reintentando: {error[:500]}"
    elif ses:
        ses.estado = "error"
        ses.error_etapa = trabajo.etapa
        ses.error = error[:4000]
        devolver(db, ses)
        db.add(NotificacionAdmin(tipo="fallo_trabajo", payload={
            "sesion_id": ses.id, "etapa": trabajo.etapa, "error": error[:500]}))
    db.commit()
    return nuevo


def liberar_abandonados(db: Session) -> int:
    """Devuelve a la cola los trabajos de un worker muerto. Commitea siempre (ver A, 22-09)."""
    limite = ahora() - timedelta(minutes=cfg.minutos_para_reclamar)
    r = db.execute(
        update(Trabajo).where(Trabajo.estado == "tomado", Trabajo.tomado_en < limite)
        .values(estado="en_cola", tomado_por=None, tomado_en=None)
    )
    db.commit()
    return int(r.rowcount or 0)
