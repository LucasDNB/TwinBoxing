"""
BoxTwin API - La cola de trabajos, sobre una tabla.

POR QUE EXISTE
  Un trabajo dura entre minutos y horas y corre en una GPU que hay una sola. La API no
  puede esperarlo (RF1: devolver el identificador sin esperar el procesamiento) y dos
  workers no pueden tomar el mismo trabajo.

  La exclusion se resuelve con SELECT ... FOR UPDATE SKIP LOCKED, que es una linea de SQL y
  no una pieza de infraestructura. SKIP LOCKED es lo que hace que el segundo worker se
  lleve el SIGUIENTE trabajo en vez de esperar al primero, que es la diferencia entre una
  cola y una fila de un solo carril.

  sqlite no tiene SKIP LOCKED y ahi la exclusion sale de otro lado: la transaccion
  inmediata serializa a los escritores, que con un solo proceso de worker -que es el caso
  de desarrollo y el de los tests- alcanza. El codigo distingue los dos casos en un solo
  lugar y lo dice, en vez de fingir que es lo mismo.

QUE HACE
  Encolar, reclamar con exclusion, terminar, fallar con reintento, y devolver a la cola los
  trabajos de un worker que se murio.

USO
  from boxtwin_api.cola import encolar, reclamar, terminar, fallar
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from boxtwin_api.config import cfg
from boxtwin_api.db import es_postgres
from boxtwin_api.modelos import ETAPAS, Sesion, Trabajo, ahora

__all__ = ["encolar", "fallar", "liberar_abandonados", "reclamar", "terminar"]


def encolar(db: Session, sesion_id: str, etapa: str, **parametros) -> Trabajo:
    """
    Pone una etapa en la cola. No commitea: el que llama decide la transaccion.

    Que no commitee es lo que permite crear la sesion y encolar su primer trabajo en el
    mismo commit. Si fueran dos, existiria el estado intermedio "hay sesion y no hay
    trabajo", que nadie limpia porque nadie lo mira.
    """
    if etapa not in ETAPAS:
        raise ValueError(f"etapa desconocida: {etapa!r}, se esperaba una de {ETAPAS}")
    previos = db.scalars(
        select(Trabajo).where(Trabajo.sesion_id == sesion_id, Trabajo.etapa == etapa)
    ).all()
    t = Trabajo(
        sesion_id=sesion_id, etapa=etapa, parametros=parametros,
        intento=len(previos) + 1,
    )
    db.add(t)
    return t


def reclamar(db: Session, worker: str) -> Trabajo | None:
    """
    Toma el trabajo mas viejo de la cola, o None si no hay.

    Commitea: cuando esta funcion vuelve, el trabajo ya es de este worker y ningun otro lo
    puede tomar. Dejarlo sin commitear abriria la ventana en la que dos workers creen
    tenerlo.
    """
    liberar_abandonados(db)

    if es_postgres():
        fila = db.execute(
            select(Trabajo)
            .where(Trabajo.estado == "en_cola")
            .order_by(Trabajo.creado)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if fila is None:
            return None
        fila.estado = "tomado"
        fila.tomado_por = worker
        fila.tomado_en = ahora()
        db.commit()
        return fila

    # sqlite: el UPDATE condicional es la exclusion. Si otro se lo llevo entre el SELECT y
    # el UPDATE, el rowcount da 0 y este worker se va con las manos vacias en vez de
    # trabajar sobre un trabajo ajeno.
    fila = db.execute(
        select(Trabajo).where(Trabajo.estado == "en_cola").order_by(Trabajo.creado).limit(1)
    ).scalar_one_or_none()
    if fila is None:
        return None
    r = db.execute(
        update(Trabajo)
        .where(Trabajo.id == fila.id, Trabajo.estado == "en_cola")
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
    trabajo.error = None
    if estado_sesion:
        ses = db.get(Sesion, trabajo.sesion_id)
        if ses:
            ses.estado = estado_sesion
            ses.error = None
    db.commit()


def fallar(db: Session, trabajo: Trabajo, error: str) -> Trabajo | None:
    """
    Marca el trabajo como fallido y, si quedan intentos, encola otro.

    El reintento es para el fallo transitorio -la GPU ocupada, un archivo todavia
    escribiendose- y por eso son pocos: un video que rompe el preproceso lo va a romper
    igual la tercera vez, y reintentar para siempre esconde el error atras de una cola que
    nunca se vacia.
    """
    trabajo.estado = "fallo"
    trabajo.terminado_en = ahora()
    trabajo.error = error[:4000]
    ses = db.get(Sesion, trabajo.sesion_id)

    nuevo = None
    if trabajo.intento < cfg.intentos_maximos:
        nuevo = encolar(db, trabajo.sesion_id, trabajo.etapa, **(trabajo.parametros or {}))
        if ses:
            ses.estado = "en_cola"
            ses.error = f"reintentando: {error[:500]}"
    elif ses:
        ses.estado = "fallo"
        ses.error = error[:4000]
    db.commit()
    return nuevo


def liberar_abandonados(db: Session) -> int:
    """
    Devuelve a la cola los trabajos de un worker que se murio sin avisar.

    El corte es por tiempo porque no hay latido: un worker que corre pose sobre 30 minutos
    de video no contesta nada durante una hora, asi que el umbral tiene que ser mas largo
    que el trabajo mas largo esperable. Tres horas por defecto.
    """
    limite = ahora() - timedelta(minutes=cfg.minutos_para_reclamar)
    r = db.execute(
        update(Trabajo)
        .where(Trabajo.estado == "tomado", Trabajo.tomado_en < limite)
        .values(estado="en_cola", tomado_por=None, tomado_en=None)
    )
    if r.rowcount:
        db.commit()
    return int(r.rowcount or 0)
