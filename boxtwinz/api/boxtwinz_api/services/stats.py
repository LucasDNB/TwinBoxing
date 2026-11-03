"""
BoxTwinZ - Stat general por perfil y modalidad (spec 6.3).

POR QUE EXISTE
  El perfil muestra el promedio de las ultimas N sesiones y si viene subiendo, estable o
  bajando. N y la banda son [PROVISIONAL] (boxtwinz.toml). Un cajon premium se mira desde el
  rival: sus numeros salen de las sesiones donde ese cajon fue el rival sembrado.

USO
  from boxtwinz_api.services.stats import stat_general
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.config import cfg
from boxtwinz_api.models import Arquetipo, MetricaSesion, Perfil, Sesion
from boxtwinz_api.services.archetypes import metricas_arquetipo
from boxtwinz_api.services.metrics import tendencia

__all__ = ["METRICAS", "stat_general", "sesiones_del_perfil"]

METRICAS = ("volumen", "pct_izquierda", "pct_mano_adelantada", "combinaciones_min",
            "pct_golpes_en_combinacion", "sosten_ritmo", "iniciativa")


def sesiones_del_perfil(db: Session, p: Perfil, modalidad: str | None = None) -> tuple[list[Sesion], str]:
    """Las sesiones listas del perfil, de la mas vieja a la mas nueva, y desde que lado se miran."""
    if p.ranura == 1:
        q = select(Sesion).where(Sesion.perfil_id == p.id)
        quien = "usuario"
    else:
        q = select(Sesion).where(Sesion.rival_perfil_id == p.id)
        quien = "rival"
    q = q.where(Sesion.estado == "lista")
    if modalidad:
        q = q.where(Sesion.modalidad == modalidad)
    return list(db.scalars(q.order_by(Sesion.creado))), quien


def stat_general(db: Session, p: Perfil, modalidad: str) -> dict:
    if p.ranura > 1 and not p.customizado:
        a = db.get(Arquetipo, p.arquetipo_id) if p.arquetipo_id else None
        return {"perfil_id": p.id, "modalidad": modalidad, "simulado": True,
                "rotulo": textos.PERFIL_SIMULADO, "sesiones": 0,
                "promedio": metricas_arquetipo(a, modalidad) if a else {}, "tendencia": {}}

    sesiones, quien = sesiones_del_perfil(db, p, modalidad)
    ultimas = sesiones[-cfg.tendencia_n:]
    ids = [s.id for s in ultimas]
    filas = {m.sesion_id: m for m in db.scalars(select(MetricaSesion).where(
        MetricaSesion.sesion_id.in_(ids), MetricaSesion.quien == quien, MetricaSesion.segmento.is_(None)))}
    serie = [filas[i] for i in ids if i in filas]
    promedio, tend = {}, {}
    for m in METRICAS:
        valores = [getattr(f, m) for f in serie if getattr(f, m) is not None]
        promedio[m] = sum(valores) / len(valores) if valores else None
        tend[m] = tendencia(valores, cfg.tendencia_banda)
    return {
        "perfil_id": p.id, "modalidad": modalidad, "simulado": False, "sesiones": len(serie),
        "ventana": cfg.tendencia_n, "promedio": promedio, "tendencia": tend,
        "guardia_supuesta": p.ranura > 1,
        "rotulo_rounds": textos.ROUNDS_SIN_VALIDAR,
        "rotulo": textos.EN_EVALUACION if modalidad in ("sombra", "bolsa") else None,
    }
