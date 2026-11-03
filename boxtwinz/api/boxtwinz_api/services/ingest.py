"""
BoxTwinZ - Ingesta de la salida del motor: golpes, metricas y exposicion de una sesion.

POR QUE EXISTE
  Cuando `completar` deja la Fight-Card, BoxTwinZ la traduce una sola vez a sus tablas. Las
  pantallas leen de la base y no del disco, asi un cambio de contrato del motor rompe aca y
  no en diez rutas.

USO
  from boxtwinz_api.services.ingest import ingerir
  ingerir(db, sesion, cfg.dir_trabajo(sesion.id))
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.models import ExposicionSesion, Golpe, MetricaSesion, Perfil, Sesion, Usuario
from boxtwinz_api.services.engine_adapter import leer_salida
from boxtwinz_api.services.metrics import GolpeT, calcular_sesion, exposicion, iniciativa

__all__ = ["ingerir", "recalcular_metricas", "es_apta_ranking"]


def es_apta_ranking(s: Sesion, p: Perfil, u: Usuario) -> bool:
    """Spec 6.4. Los umbrales son [PROVISIONAL] y salen de la config."""
    return bool(
        p.ranura == 1 and p.publico and not p.fuera_de_ranking and not u.suspendido
        and (s.video_duracion_s or 0) >= cfg.duracion_ranking_s
        and s.calidad_pose is not None and s.calidad_pose >= cfg.calidad_pose_min
    )


def recalcular_metricas(db: Session, s: Sesion) -> None:
    """Borra y vuelve a escribir metricas y exposicion desde los golpes de la base. No commitea."""
    db.execute(delete(MetricaSesion).where(MetricaSesion.sesion_id == s.id))
    db.execute(delete(ExposicionSesion).where(ExposicionSesion.sesion_id == s.id))
    golpes = list(db.query(Golpe).filter(Golpe.sesion_id == s.id))
    p = db.get(Perfil, s.perfil_id)
    guardia_usuario = p.guardia if p and p.ranura == 1 else "ortodoxa"
    dur = s.video_duracion_s or 0.0
    por_quien = {q: [GolpeT(g.t_inicio, g.mano) for g in golpes if g.quien == q] for q in ("usuario", "rival")}
    quienes = ["usuario"] + (["rival"] if s.modalidad == "sparring" else [])
    ini = iniciativa(len(por_quien["usuario"]), len(por_quien["rival"])) if s.modalidad == "sparring" else None
    for q in quienes:
        # El rival se mira con guardia supuesta ortodoxa (spec 6.2).
        m = calcular_sesion(por_quien[q], dur, guardia_usuario if q == "usuario" else "ortodoxa",
                            s.rounds_declarados, cfg.gap_combinacion_s)
        t = m["total"]
        db.add(MetricaSesion(
            sesion_id=s.id, quien=q, segmento=None, volumen=t["volumen"], golpes=t["golpes"],
            pct_izquierda=t["pct_izquierda"], pct_mano_adelantada=t["pct_mano_adelantada"],
            combinaciones_min=t["combinaciones_min"], pct_golpes_en_combinacion=t["pct_golpes_en_combinacion"],
            sosten_ritmo=t["sosten_ritmo"], curva=t["curva"],
            iniciativa=(ini if q == "usuario" else (1 - ini if ini is not None else None)),
        ))
        for i, sg in enumerate(m["segmentos"], start=1):
            db.add(MetricaSesion(
                sesion_id=s.id, quien=q, segmento=i, volumen=sg["volumen"], golpes=sg["golpes"],
                pct_izquierda=sg["pct_izquierda"], pct_mano_adelantada=sg["pct_mano_adelantada"],
                combinaciones_min=sg["combinaciones_min"],
                pct_golpes_en_combinacion=sg["pct_golpes_en_combinacion"],
                sosten_ritmo=sg["sosten_ritmo"], curva=[sg["inicio_s"], sg["fin_s"]],
            ))
    if s.modalidad == "sparring":
        db.add(ExposicionSesion(sesion_id=s.id, **exposicion(por_quien["rival"])))


def ingerir(db: Session, s: Sesion, directorio: Path) -> None:
    """Lee la Fight-Card y escribe todo. No commitea: lo hace el worker con el estado."""
    salida = leer_salida(directorio)
    db.execute(delete(Golpe).where(Golpe.sesion_id == s.id))
    for g in salida.golpes:
        if s.modalidad != "sparring" and g.quien != "usuario":
            continue
        db.add(Golpe(sesion_id=s.id, quien=g.quien, mano=g.mano, t_inicio=g.t_inicio, t_fin=g.t_fin,
                     confianza=g.confianza, altura_estimada=g.altura_estimada, id_motor=g.id_motor))
    if salida.duracion_s:
        s.video_duracion_s = float(salida.duracion_s)
    if salida.fps:
        s.video_fps = float(salida.fps)
    s.calidad_pose = salida.calidad_pose
    s.fraccion_sin_identidad = salida.fraccion_sin_identidad
    db.flush()
    recalcular_metricas(db, s)
    p = db.get(Perfil, s.perfil_id)
    u = db.get(Usuario, s.usuario_id)
    s.apta_ranking = es_apta_ranking(s, p, u) if p and u else False
