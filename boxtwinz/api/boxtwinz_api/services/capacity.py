"""
BoxTwinZ - Capacidad de la GPU y disparadores (spec 10).

POR QUE EXISTE
  El Cap 5 necesita saber cuanto de la estacion se usa y cuando conviene pasar a un servidor
  externo. La utilizacion diaria es segundos de GPU ocupada / 86.400, y dos disparadores
  avisan al admin una vez por episodio: >70% 15 dias seguidos y >90% 5 dias seguidos
  (umbrales y dias [PROVISIONAL], en boxtwinz.toml).

  "Una vez por episodio": el aviso sale el dia en que la racha alcanza el largo pedido y no
  vuelve a salir mientras la racha siga. Si la racha se corta, el proximo cruce es otro
  episodio.

USO
  from boxtwinz_api.services.capacity import sumar_segundos, cerrar_dia, evaluar_disparadores
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.models import NotificacionAdmin, UsoGpuDiario

__all__ = ["sumar_segundos", "cerrar_dia", "racha", "disparadores", "evaluar_disparadores", "TEXTO_PASE"]

DIA_S = 86400.0
TEXTO_PASE = "Evaluar pase a servidor externo"


def _fila(db: Session, d: date) -> UsoGpuDiario:
    f = db.scalar(select(UsoGpuDiario).where(UsoGpuDiario.fecha == d))
    if f is None:
        f = UsoGpuDiario(fecha=d, segundos_ocupados=0.0)
        db.add(f)
        db.flush()
    return f


def sumar_segundos(db: Session, inicio: datetime, fin: datetime) -> float:
    """Reparte [inicio, fin) entre los dias locales que toca. Devuelve el total. No commitea."""
    if fin <= inicio:
        return 0.0
    a, b = inicio.astimezone(), fin.astimezone()
    total = (b - a).total_seconds()
    while a < b:
        medianoche = (a + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        corte = min(b, medianoche)
        _fila(db, a.date()).segundos_ocupados += (corte - a).total_seconds()
        a = corte
    return total


def cerrar_dia(db: Session, d: date) -> list[NotificacionAdmin]:
    f = _fila(db, d)
    f.utilizacion = min(1.0, f.segundos_ocupados / DIA_S)
    db.flush()
    return evaluar_disparadores(db, d)


def racha(serie: dict[date, float], hasta: date, umbral: float) -> int:
    """Dias consecutivos terminando en `hasta` con utilizacion > umbral. Un dia sin dato corta."""
    n, d = 0, hasta
    while d in serie and serie[d] is not None and serie[d] > umbral:
        n += 1
        d -= timedelta(days=1)
    return n


def disparadores(serie: dict[date, float], hoy: date) -> dict[str, bool]:
    """
    Que disparadores CRUZAN hoy (racha exactamente igual al largo pedido). Funcion pura: la
    usan los tests con series sinteticas y evaluar_disparadores con la base.
    """
    return {
        "capacidad_70_15": racha(serie, hoy, cfg.capacidad_umbral_alto) == cfg.capacidad_dias_alto,
        "capacidad_90_5": racha(serie, hoy, cfg.capacidad_umbral_critico) == cfg.capacidad_dias_critico,
    }


def estado_disparadores(serie: dict[date, float], hoy: date) -> dict[str, dict]:
    return {
        "capacidad_70_15": {"racha": racha(serie, hoy, cfg.capacidad_umbral_alto),
                            "dias": cfg.capacidad_dias_alto, "umbral": cfg.capacidad_umbral_alto},
        "capacidad_90_5": {"racha": racha(serie, hoy, cfg.capacidad_umbral_critico),
                           "dias": cfg.capacidad_dias_critico, "umbral": cfg.capacidad_umbral_critico},
    }


def serie_db(db: Session, desde: date | None = None) -> dict[date, float]:
    q = select(UsoGpuDiario).where(UsoGpuDiario.utilizacion.is_not(None))
    if desde:
        q = q.where(UsoGpuDiario.fecha >= desde)
    return {f.fecha: f.utilizacion for f in db.scalars(q)}


def evaluar_disparadores(db: Session, hoy: date) -> list[NotificacionAdmin]:
    """Crea las notificaciones que cruzan hoy. No commitea."""
    serie = serie_db(db, hoy - timedelta(days=max(cfg.capacidad_dias_alto, cfg.capacidad_dias_critico) + 1))
    nuevas = []
    ya = {(n.tipo, (n.payload or {}).get("fecha")) for n in db.scalars(
        select(NotificacionAdmin).where(NotificacionAdmin.tipo.in_(("capacidad_70_15", "capacidad_90_5"))))}
    for tipo, cruza in disparadores(serie, hoy).items():
        # Cerrar el mismo dia dos veces (un reinicio del worker) no duplica el aviso.
        if cruza and (tipo, hoy.isoformat()) not in ya:
            n = NotificacionAdmin(tipo=tipo, payload={"fecha": hoy.isoformat(), "texto": TEXTO_PASE})
            db.add(n)
            nuevas.append(n)
    return nuevas
