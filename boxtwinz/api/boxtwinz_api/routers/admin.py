"""
BoxTwinZ API - Panel de administracion (spec 10 y 11, Z13).

POR QUE EXISTE
  Es de donde salen los numeros de BoxTwinZ para la tesis: costo de GPU por modalidad,
  capacidad de la estacion y su proyeccion (Cap 5), valoraciones y correcciones (Cap 7). Cada
  bloque devuelve sus tablas con la misma forma, asi el CSV es uno solo para todos.

  Ningun bloque muestra emails ni videos: lo que se agrega son conteos. La moderacion es la
  excepcion y muestra el perfil denunciado, que es lo que hay que mirar para decidir.

USO
  app.include_router(router, prefix="/api")
  GET /api/admin/exportar/capacidad.csv
"""

from __future__ import annotations

import csv
import io
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import admin_actual, error
from boxtwinz_api.models import (
    MODALIDADES, AccionModeracion, Correccion, Denuncia, EventoPublicidad, NotificacionAdmin,
    Perfil, Sesion, TandaModeracion, Trabajo, Usuario, UsoMinutos, Valoracion, ahora,
)
from boxtwinz_api.services.capacity import DIA_S, TEXTO_PASE, estado_disparadores, serie_db
from boxtwinz_api.services.quota import mes_actual

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(admin_actual)])


def percentil(valores: list[float], p: float) -> float | None:
    """Percentil por interpolacion lineal (el de numpy por defecto). None sin datos."""
    if not valores:
        return None
    v = sorted(valores)
    k = (len(v) - 1) * p
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def _dia(dt) -> str:
    return _aware(dt).astimezone().date().isoformat()


def resumen(db: Session) -> dict:
    usuarios = list(db.scalars(select(Usuario)))
    planes = Counter(u.plan for u in usuarios)
    desde = ahora() - timedelta(days=30)
    altas = Counter(_dia(u.creado) for u in usuarios if u.creado and _aware(u.creado) >= desde)
    ses = Counter(m for (m,) in db.execute(select(Sesion.modalidad)))
    minutos = defaultdict(float)
    for plan, seg in db.execute(select(Usuario.plan, UsoMinutos.segundos_procesados)
                                .join(UsoMinutos, UsoMinutos.usuario_id == Usuario.id).where(UsoMinutos.mes == mes_actual())):
        minutos[plan] += seg / 60
    return {"tablas": {
        "usuarios_por_plan": [{"plan": p, "usuarios": planes.get(p, 0)} for p in ("free", "premium")],
        "altas_por_dia": [{"dia": d, "altas": n} for d, n in sorted(altas.items())],
        "sesiones_por_modalidad": [{"modalidad": m, "sesiones": ses.get(m, 0)} for m in MODALIDADES],
        "minutos_por_plan_mes": [{"mes": mes_actual(), "plan": p, "minutos": round(minutos.get(p, 0.0), 2)} for p in ("free", "premium")],
    }}


def capacidad(db: Session, hoy: date | None = None) -> dict:
    hoy = hoy or (ahora().astimezone().date() - timedelta(days=1))
    serie = serie_db(db, hoy - timedelta(days=29))
    cola = dict(db.execute(select(Trabajo.estado, func.count()).group_by(Trabajo.estado)).all())
    desde = ahora() - timedelta(days=7)
    esperas = [(_aware(t.tomado_en) - _aware(t.creado)).total_seconds() for t in db.scalars(
        select(Trabajo).where(Trabajo.etapa == "procesar", Trabajo.tomado_en.is_not(None)))
        if t.creado and _aware(t.creado) >= desde]
    # Proyeccion (spec 10): horas de video procesadas por hora de GPU ocupada, en las sesiones
    # que terminaron, extrapolado a un mes con la GPU al 70% y al 100%.
    hechas = list(db.scalars(select(Sesion).where(Sesion.estado == "lista", Sesion.segundos_gpu > 0)))
    video_h = sum(s.video_duracion_s or 0 for s in hechas) / 3600
    gpu_h = sum(s.segundos_gpu for s in hechas) / 3600
    ratio = video_h / gpu_h if gpu_h else None
    mes_h = 24 * 30
    disp = estado_disparadores(serie, hoy)
    return {
        "umbrales": [cfg.capacidad_umbral_alto, cfg.capacidad_umbral_critico],
        "cola": {"en_cola": cola.get("en_cola", 0), "tomados": cola.get("tomado", 0)},
        "espera_s": {"p50": percentil(esperas, 0.5), "p95": percentil(esperas, 0.95), "trabajos": len(esperas)},
        "disparadores": {k: {**v, "activo": v["racha"] >= v["dias"], "texto": TEXTO_PASE} for k, v in disp.items()},
        "proyeccion": {"horas_video_por_hora_gpu": ratio, "sesiones_medidas": len(hechas),
                       "horas_video_mes_70": ratio * mes_h * 0.7 if ratio else None,
                       "horas_video_mes_100": ratio * mes_h if ratio else None},
        "tablas": {"utilizacion_diaria": [
            {"fecha": (hoy - timedelta(days=i)).isoformat(), "utilizacion": serie.get(hoy - timedelta(days=i))}
            for i in range(29, -1, -1)]},
    }


def _aware(dt):
    from datetime import timezone

    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def rendimiento(db: Session) -> dict:
    por_mod: dict[str, list[float]] = defaultdict(list)
    sin_id: list[float] = []
    for s in db.scalars(select(Sesion).where(Sesion.estado == "lista")):
        if s.segundos_gpu and s.video_duracion_s:
            por_mod[s.modalidad].append(s.segundos_gpu / s.video_duracion_s)
        if s.modalidad == "sparring" and s.fraccion_sin_identidad is not None:
            sin_id.append(s.fraccion_sin_identidad)
    fallos = Counter(e for (e,) in db.execute(select(Sesion.error_etapa).where(Sesion.estado == "error")))
    return {
        "sparring_sin_identidad": {
            "sesiones": len(sin_id),
            "fraccion_media": statistics.fmean(sin_id) if sin_id else None,
            "pct_sesiones_con_cuadros_sin_identidad": (sum(1 for x in sin_id if x > 0) / len(sin_id)) if sin_id else None,
            "nota": "Fracción de cuadros sin pose confiable del peleador (desvío D6).",
        },
        "tablas": {
            "gpu_sobre_video_por_modalidad": [
                {"modalidad": m, "sesiones": len(por_mod[m]), "mediana": percentil(por_mod[m], 0.5), "p95": percentil(por_mod[m], 0.95)}
                for m in MODALIDADES],
            "fallos_por_etapa": [{"etapa": e, "fallos": fallos.get(e, 0)} for e in ("procesar", "completar")],
        },
    }


def calidad(db: Session) -> dict:
    vals = list(db.scalars(select(Valoracion)))
    por_mod: dict[str, list[int]] = defaultdict(list)
    por_semana: dict[tuple[str, str], list[int]] = defaultdict(list)
    for v in vals:
        mod = v.modalidad or "desconocida"
        por_mod[mod].append(v.estrellas)
        a, w, _ = v.creado.isocalendar()
        por_semana[(f"{a}-W{w:02d}", mod)].append(v.estrellas)
    estrellas = Counter(v.estrellas for v in vals)
    corr = Counter(t for (t,) in db.execute(select(Correccion.tipo)))
    con_consent = db.scalar(select(func.count()).select_from(Sesion).join(Usuario, Usuario.id == Sesion.usuario_id)
                            .where(Usuario.consentimiento_reent_at.is_not(None))) or 0
    return {
        "sesiones_con_consentimiento_reentrenamiento": con_consent,
        "tablas": {
            "valoracion_por_modalidad": [{"modalidad": m, "valoraciones": len(por_mod.get(m, [])),
                                          "promedio": statistics.fmean(por_mod[m]) if por_mod.get(m) else None} for m in MODALIDADES],
            "valoracion_por_semana": [{"semana": s, "modalidad": m, "valoraciones": len(x), "promedio": statistics.fmean(x)}
                                      for (s, m), x in sorted(por_semana.items())],
            "distribucion_estrellas": [{"estrellas": e, "cantidad": estrellas.get(e, 0)} for e in range(1, 6)],
            "correcciones_por_tipo": [{"tipo": t, "cantidad": corr.get(t, 0)} for t in ("no_fue_golpe", "falto_golpe", "otra_mano")],
        },
    }


def ingresos(db: Session) -> dict:
    premium = db.scalar(select(func.count()).select_from(Usuario).where(Usuario.plan == "premium")) or 0
    imp = Counter(_dia(e.creado) for e in db.scalars(select(EventoPublicidad).where(EventoPublicidad.tipo == "impresion")))
    return {"premium_simulado": premium, "aviso": "Pago simulado — versión de demostración",
            "tablas": {"impresiones_por_dia": [{"dia": d, "impresiones": n} for d, n in sorted(imp.items())]}}


BLOQUES = {"resumen": resumen, "capacidad": capacidad, "rendimiento": rendimiento, "calidad": calidad, "ingresos": ingresos}


@router.get("/resumen")
def ver_resumen(db: Session = Depends(obtener_sesion)) -> dict:
    return resumen(db)


@router.get("/capacidad")
def ver_capacidad(db: Session = Depends(obtener_sesion)) -> dict:
    return capacidad(db)


@router.get("/rendimiento")
def ver_rendimiento(db: Session = Depends(obtener_sesion)) -> dict:
    return rendimiento(db)


@router.get("/calidad")
def ver_calidad(db: Session = Depends(obtener_sesion)) -> dict:
    return calidad(db)


@router.get("/ingresos")
def ver_ingresos(db: Session = Depends(obtener_sesion)) -> dict:
    return ingresos(db)


@router.get("/exportar/{bloque}.csv")
def exportar(bloque: str, db: Session = Depends(obtener_sesion)) -> Response:
    if bloque not in BLOQUES:
        raise error(404, "no_encontrado", f"Los bloques son {', '.join(BLOQUES)}")
    buf = io.StringIO()
    w = csv.writer(buf)
    for nombre, filas in BLOQUES[bloque](db)["tablas"].items():
        w.writerow([f"# {nombre}"])
        if filas:
            w.writerow(list(filas[0]))
            for f in filas:
                w.writerow(["" if v is None else v for v in f.values()])
        w.writerow([])
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="boxtwinz-{bloque}-{date.today().isoformat()}.csv"'})


def _denuncia_dict(db: Session, d: Denuncia) -> dict:
    p = db.get(Perfil, d.perfil_id)
    u = db.get(Usuario, p.usuario_id) if p else None
    return {"id": d.id, "motivo": d.motivo, "detalle": d.detalle, "estado": d.estado, "tanda_id": d.tanda_id,
            "creado": d.creado.isoformat(),
            "perfil": {"id": p.id, "nombre": p.nombre, "publico": p.publico, "fuera_de_ranking": p.fuera_de_ranking} if p else None,
            "cuenta_suspendida": u.suspendido if u else None}


@router.get("/denuncias")
def denuncias(db: Session = Depends(obtener_sesion)) -> dict:
    abiertas = db.scalars(select(Denuncia).where(Denuncia.estado == "abierta").order_by(Denuncia.creado))
    tandas = db.scalars(select(TandaModeracion).order_by(TandaModeracion.inicio.desc()).limit(50))
    return {"abiertas": [_denuncia_dict(db, d) for d in abiertas],
            "tandas": [{"id": t.id, "inicio": t.inicio.isoformat(), "fin": t.fin.isoformat() if t.fin else None,
                        "cantidad_revisada": t.cantidad_revisada} for t in tandas]}


@router.post("/tandas", status_code=201)
def empezar_tanda(admin: Usuario = Depends(admin_actual), db: Session = Depends(obtener_sesion)) -> dict:
    """Agrupa las denuncias abiertas que no estan en otra tanda."""
    sueltas = list(db.scalars(select(Denuncia).where(Denuncia.estado == "abierta", Denuncia.tanda_id.is_(None))))
    if not sueltas:
        raise error(409, "sin_denuncias", "No hay denuncias abiertas sin tanda")
    t = TandaModeracion(admin_id=admin.id, inicio=ahora())
    db.add(t)
    db.flush()
    for d in sueltas:
        d.tanda_id = t.id
    db.commit()
    return {"id": t.id, "denuncias": len(sueltas)}


class PedidoAccion(BaseModel):
    accion: Literal["desestimar", "sacar_ranking", "suspender"]
    nota: str | None = Field(default=None, max_length=2000)


@router.post("/denuncias/{denuncia_id}/accion")
def accionar(denuncia_id: str, datos: PedidoAccion, admin: Usuario = Depends(admin_actual),
             db: Session = Depends(obtener_sesion)) -> dict:
    d = db.get(Denuncia, denuncia_id)
    if d is None:
        raise error(404, "no_encontrada", "No existe esa denuncia")
    if d.estado != "abierta":
        raise error(409, "ya_resuelta", "La denuncia ya fue resuelta")
    p = db.get(Perfil, d.perfil_id)
    if datos.accion == "sacar_ranking" and p:
        p.fuera_de_ranking = True
    elif datos.accion == "suspender" and p:
        u = db.get(Usuario, p.usuario_id)
        if u:
            u.suspendido = True
    d.estado = "resuelta"
    db.add(AccionModeracion(denuncia_id=d.id, admin_id=admin.id, accion=datos.accion, nota=datos.nota))
    if d.tanda_id:
        t = db.get(TandaModeracion, d.tanda_id)
        t.cantidad_revisada += 1
        db.flush()
        quedan = db.scalar(select(func.count()).select_from(Denuncia).where(Denuncia.tanda_id == t.id, Denuncia.estado == "abierta"))
        if not quedan:
            t.fin = ahora()
    db.commit()
    return _denuncia_dict(db, d)


@router.get("/notificaciones")
def notificaciones(db: Session = Depends(obtener_sesion)) -> dict:
    ns = list(db.scalars(select(NotificacionAdmin).order_by(NotificacionAdmin.creado.desc()).limit(100)))
    return {"no_leidas": sum(1 for n in ns if not n.leida),
            "notificaciones": [{"id": n.id, "tipo": n.tipo, "payload": n.payload, "leida": n.leida,
                                "creado": n.creado.isoformat()} for n in ns]}


@router.post("/notificaciones/{notificacion_id}/leida")
def marcar_leida(notificacion_id: str, db: Session = Depends(obtener_sesion)) -> dict:
    n = db.get(NotificacionAdmin, notificacion_id)
    if n is None:
        raise error(404, "no_encontrada", "No existe esa notificación")
    n.leida = True
    db.commit()
    return {"id": n.id, "leida": True}


__all__ = ["router", "percentil", "capacidad", "DIA_S"]
