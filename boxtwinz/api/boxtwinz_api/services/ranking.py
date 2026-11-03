"""
BoxTwinZ - Ranking de actividad (spec 5 y 6.4, Z9).

POR QUE EXISTE
  El ranking cuenta golpes DETECTADOS, no golpes que conectaron, y solo entre sesiones aptas:
  perfil 1 publico, 180 s o mas, calidad de pose suficiente, cuenta no suspendida y perfil no
  sacado del ranking. La aptitud se recalcula al consultar: el usuario puede volver privado
  su perfil o un admin sacarlo, y eso tiene que verse en el momento, no al reprocesar.

USO
  from boxtwinz_api.services.ranking import ranking
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api.models import MetricaSesion, Perfil, Sesion, Usuario
from boxtwinz_api.services.ingest import es_apta_ranking

__all__ = ["METRICAS_RANKING", "ranking"]

METRICAS_RANKING = ("volumen", "combinaciones_min", "sosten_ritmo", "pct_golpes_en_combinacion")


def ranking(db: Session, modalidad: str, metrica: str, usuario_id: str | None = None, top: int = 100) -> dict:
    filas = db.execute(
        select(Sesion, Perfil, Usuario, MetricaSesion)
        .join(Perfil, Perfil.id == Sesion.perfil_id)
        .join(Usuario, Usuario.id == Sesion.usuario_id)
        .join(MetricaSesion, (MetricaSesion.sesion_id == Sesion.id) & (MetricaSesion.quien == "usuario")
              & MetricaSesion.segmento.is_(None))
        .where(Sesion.estado == "lista", Sesion.modalidad == modalidad, Perfil.ranura == 1,
               Perfil.publico.is_(True))
    ).all()
    por_perfil: dict[str, dict] = {}
    for s, p, u, m in filas:
        v = getattr(m, metrica)
        if v is None or not es_apta_ranking(s, p, u):
            continue
        e = por_perfil.setdefault(p.id, {"perfil_id": p.id, "usuario_id": u.id, "nombre": p.nombre, "valores": []})
        e["valores"].append(v)
    tabla = [{"perfil_id": e["perfil_id"], "usuario_id": e["usuario_id"], "nombre": e["nombre"],
              "valor": sum(e["valores"]) / len(e["valores"]), "sesiones": len(e["valores"])}
             for e in por_perfil.values()]
    tabla.sort(key=lambda x: (-x["valor"], x["nombre"]))
    for i, x in enumerate(tabla, start=1):
        x["posicion"] = i
    propia = next((x for x in tabla if x["usuario_id"] == usuario_id), None)
    for x in tabla:
        x.pop("usuario_id")
    if propia is not None:
        propia = dict(propia)
        propia.pop("usuario_id", None)
    return {"modalidad": modalidad, "metrica": metrica, "total": len(tabla), "top": tabla[:top], "propia": propia,
            "pie": "Ranking de actividad: cuenta golpes detectados, no golpes que conectaron"}
