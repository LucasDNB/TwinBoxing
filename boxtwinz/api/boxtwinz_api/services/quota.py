"""
BoxTwinZ - Cupo de minutos de video por mes (spec 9).

POR QUE EXISTE
  El cupo se descuenta al encolar y se devuelve si el trabajo falla por un error del sistema.
  Lo descontado se guarda en la sesion para devolver exactamente eso y no recalcularlo.

USO
  from boxtwinz_api.services.quota import cupo, descontar, devolver
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.models import Sesion, Usuario, UsoMinutos, ahora

__all__ = ["mes_actual", "cupo", "descontar", "devolver"]


def mes_actual(t: datetime | None = None) -> str:
    return (t or ahora()).strftime("%Y-%m")


def _fila(db: Session, usuario_id: str, mes: str) -> UsoMinutos:
    f = db.scalar(select(UsoMinutos).where(UsoMinutos.usuario_id == usuario_id, UsoMinutos.mes == mes))
    if f is None:
        f = UsoMinutos(usuario_id=usuario_id, mes=mes, segundos_procesados=0.0)
        db.add(f)
        db.flush()
    return f


def cupo(db: Session, u: Usuario) -> dict:
    limite_s = cfg.cupo_minutos.get(u.plan, cfg.cupo_minutos["free"]) * 60
    f = db.scalar(select(UsoMinutos).where(UsoMinutos.usuario_id == u.id, UsoMinutos.mes == mes_actual()))
    usados = f.segundos_procesados if f else 0.0
    return {"mes": mes_actual(), "limite_min": limite_s / 60, "usados_min": usados / 60,
            "restantes_min": max(0.0, limite_s - usados) / 60}


def descontar(db: Session, u: Usuario, ses: Sesion, segundos: float) -> None:
    f = _fila(db, u.id, mes_actual())
    f.segundos_procesados += segundos
    ses.segundos_cupo = segundos


def devolver(db: Session, ses: Sesion) -> None:
    if not ses.segundos_cupo:
        return
    f = _fila(db, ses.usuario_id, mes_actual(ses.creado))
    f.segundos_procesados = max(0.0, f.segundos_procesados - ses.segundos_cupo)
    ses.segundos_cupo = 0.0
