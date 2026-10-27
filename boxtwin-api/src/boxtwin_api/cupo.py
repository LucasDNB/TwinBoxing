"""
BoxTwin API - Horas de video del mes de una cuenta (versiones 26-10, B20; 02 §6).

POR QUE EXISTE
  Las horas incluidas son el limite de cada version (4, 15 y 50, Tabla 5.7 del Cap 5). La
  unidad es la hora de video subida, que es la "hora de video analizada" del Cap 5, y tiene
  que salir de la duracion real del video y no de lo que alguien declara, salvo mientras la
  real no existe.

  El mes se corta en hora de Buenos Aires: una sesion subida el 31 a las 23:30 en el
  gimnasio es de ese mes, aunque en UTC ya sea el 1.

QUE HACE
  Suma la duracion de las sesiones de la cuenta creadas en el mes calendario, sin las que
  fallaron. Los reprocesos no suben video nuevo y no crean sesion, asi que no cuentan.

USO
  from boxtwin_api.cupo import horas_del_mes, modo_de
  horas_del_mes(cuenta, db)
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwin_api import config
from boxtwin_api.modelos import Cuenta, Sesion
from boxtwin_api.planes import PLANES

__all__ = ["horas_del_mes", "inicio_de_mes", "modo_de", "texto_horas"]

ZONA = ZoneInfo("America/Argentina/Buenos_Aires")


def modo_de(cuenta: Cuenta) -> str:
    """En una cuenta de prueba el cupo bloquea siempre (RNF-D3), sin mirar BOXTWIN_CUPO."""
    return "bloquear" if cuenta.demo else config.cfg.cupo


def _utc(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def inicio_de_mes(ahora: datetime) -> tuple[datetime, datetime, str]:
    """Inicio del mes y del siguiente, en UTC, cortados en Buenos Aires; y el mes 'AAAA-MM'."""
    local = _utc(ahora).astimezone(ZONA)
    desde = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    hasta = desde.replace(year=desde.year + 1, month=1) if desde.month == 12 \
        else desde.replace(month=desde.month + 1)
    return desde.astimezone(timezone.utc), hasta.astimezone(timezone.utc), f"{desde:%Y-%m}"


def horas_del_mes(cuenta: Cuenta, db: Session, ahora: datetime | None = None) -> dict:
    desde, hasta, mes = inicio_de_mes(ahora or datetime.now(timezone.utc))
    usadas_s = 0.0
    sin_duracion = 0
    for s in db.scalars(select(Sesion).where(Sesion.cuenta_id == cuenta.id)).all():
        if s.estado == "fallo" or s.creada is None or not (desde <= _utc(s.creada) < hasta):
            continue
        dur = s.duracion_s if s.duracion_s is not None else s.duracion_declarada_s
        if dur is None:
            sin_duracion += 1
            continue
        usadas_s += float(dur)
    incluidas = float(PLANES.get(cuenta.version, PLANES["entrenador"]).horas_mes)
    usadas = usadas_s / 3600
    return {
        "mes": mes,
        "incluidas": incluidas,
        "usadas": round(usadas, 2),
        "restantes": round(max(incluidas - usadas, 0.0), 2),
        "sin_duracion": sin_duracion,
        "modo": modo_de(cuenta),
    }


def texto_horas(h: float) -> str:
    """0.8 -> '0,8'. El mensaje de rechazo va en castellano, con coma decimal."""
    return f"{h:.1f}".replace(".", ",")
