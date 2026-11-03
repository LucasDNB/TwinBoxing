"""
BoxTwinZ - Borrado de sesiones con todo lo que cuelga de ellas.

POR QUE EXISTE
  Las FK llevan ON DELETE CASCADE, pero sqlite solo las respeta con PRAGMA foreign_keys y
  una base vieja o una conexion sin el pragma deja huerfanos. Se borra explicito, en orden.
"""

from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.orm import Session

from boxtwinz_api.models import (
    Correccion, ExposicionSesion, Golpe, MetricaSesion, Sesion, Trabajo, Valoracion,
)


def borrar_sesiones(db: Session, ids: list[str]) -> None:
    if not ids:
        return
    db.execute(delete(Correccion).where(Correccion.sesion_id.in_(ids)))
    db.execute(delete(Golpe).where(Golpe.sesion_id.in_(ids)))
    db.execute(delete(MetricaSesion).where(MetricaSesion.sesion_id.in_(ids)))
    db.execute(delete(ExposicionSesion).where(ExposicionSesion.sesion_id.in_(ids)))
    db.execute(delete(Trabajo).where(Trabajo.sesion_id.in_(ids)))
    db.execute(delete(Valoracion).where(Valoracion.sesion_id.in_(ids)))
    db.execute(delete(Sesion).where(Sesion.id.in_(ids)))
