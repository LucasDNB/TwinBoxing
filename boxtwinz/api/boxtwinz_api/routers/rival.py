"""
BoxTwinZ API - Rival fantasma (spec 8, Z11).

POR QUE EXISTE
  Compara el perfil propio contra un rival que no estuvo en la sesion: un arquetipo, un
  perfil publico, un amigo o, en premium, un cajon propio. Es una comparacion de numeros de
  golpes detectados en la misma modalidad, nunca una prediccion de quien gana.

  El mapa de exposicion sale de los sparrings reales del perfil contra ese estilo (cajones de
  ese estilo) o contra ese cajon. Contra un arquetipo sin sparrings de su estilo, el mapa es el
  simulado del arquetipo, con rotulo, y no dispara el consejo de exposicion: decirle a alguien
  "te entran por la derecha" con un numero inventado seria mentirle.

USO
  GET /api/rival?perfil_id=&modalidad=&rival_tipo=arquetipo|publico|amigo|cajon&rival_id=
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import error, premium_requerido, usuario_actual
from boxtwinz_api.models import MODALIDADES, Arquetipo, ExposicionSesion, Golpe, MetricaSesion, Perfil, Sesion, Usuario
from boxtwinz_api.routers.social import amistad_aceptada, perfil_publico
from boxtwinz_api.services.archetypes import metricas_arquetipo
from boxtwinz_api.services.rules import Entrada, consejos
from boxtwinz_api.services.stats import sesiones_del_perfil, stat_general

router = APIRouter(tags=["rival"])


def curva_promedio(db: Session, p: Perfil, modalidad: str, n: int = 10) -> list[float]:
    """Promedio bloque a bloque de las curvas de las ultimas n sesiones (hasta el largo menor)."""
    sesiones, quien = sesiones_del_perfil(db, p, modalidad)
    ids = [s.id for s in sesiones[-n:]]
    curvas = [m.curva for m in db.scalars(select(MetricaSesion).where(
        MetricaSesion.sesion_id.in_(ids), MetricaSesion.quien == quien, MetricaSesion.segmento.is_(None))) if m.curva]
    if not curvas:
        return []
    largo = min(len(c) for c in curvas)
    return [round(sum(c[i] for c in curvas) / len(curvas), 3) for i in range(largo)]


def exposicion_real(db: Session, propio: Perfil, cajones: list[str]) -> dict | None:
    if not cajones:
        return None
    ids = [s.id for s in db.scalars(select(Sesion).where(
        Sesion.perfil_id == propio.id, Sesion.estado == "lista", Sesion.modalidad == "sparring",
        Sesion.rival_perfil_id.in_(cajones)))]
    if not ids:
        return None
    filas = list(db.scalars(select(ExposicionSesion).where(ExposicionSesion.sesion_id.in_(ids))))
    izq = sum(f.lado_izquierdo_usuario for f in filas)
    der = sum(f.lado_derecho_usuario for f in filas)
    return {"lado_izquierdo_usuario": izq, "lado_derecho_usuario": der, "sesiones": len(ids), "simulado": False,
            "aclaracion": "Golpes dirigidos del rival por lado en tus sparrings; no dice cuáles llegaron."}


def altura(db: Session, propio: Perfil, modalidad: str) -> dict | None:
    ids = [s.id for s in db.scalars(select(Sesion).where(Sesion.perfil_id == propio.id, Sesion.modalidad == modalidad,
                                                         Sesion.estado == "lista"))]
    alts = [g.altura_estimada for g in db.scalars(select(Golpe).where(
        Golpe.sesion_id.in_(ids), Golpe.quien == "usuario", Golpe.altura_estimada.is_not(None)))] if ids else []
    if not alts:
        return None
    return {"rotulo": "estimación", "cabeza": alts.count("cabeza"), "cuerpo": alts.count("cuerpo")}


@router.get("/rival")
def rival(perfil_id: str = Query(...), modalidad: str = Query(...), rival_tipo: str = Query(...),
          rival_id: str = Query(...), u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    if modalidad not in MODALIDADES:
        raise error(422, "modalidad_invalida", "La modalidad es sombra, bolsa o sparring")
    propio = db.get(Perfil, perfil_id)
    if propio is None or propio.usuario_id != u.id or propio.ranura != 1:
        raise error(404, "no_encontrado", "El rival fantasma se compara con tu perfil")

    estilo: str | None = None
    cajones_expo: list[str] = []
    expo_simulada = None
    if rival_tipo == "arquetipo":
        a = db.get(Arquetipo, rival_id)
        visible = a is not None and (a.usuario_id in (None, u.id)) and (not a.solo_premium or u.plan == "premium")
        if not visible:
            raise error(404, "no_encontrado", "No existe ese arquetipo")
        estilo = a.estilo
        r_metricas = metricas_arquetipo(a, modalidad)
        r_curva = list(a.curva_ritmo or [])
        r_nombre, r_rotulo = a.nombre, textos.PERFIL_SIMULADO
        cajones_expo = [p.id for p in db.scalars(select(Perfil).where(
            Perfil.usuario_id == u.id, Perfil.ranura > 1, Perfil.customizado.is_(True), Perfil.estilo == estilo))]
        expo_simulada = {"izquierda": (a.exposicion or {}).get("izquierda"), "derecha": (a.exposicion or {}).get("derecha"),
                         "simulado": True, "rotulo": textos.PERFIL_SIMULADO}
    elif rival_tipo in ("publico", "amigo"):
        if rival_tipo == "publico":
            otro = perfil_publico(db, rival_id)
        else:
            if not amistad_aceptada(db, u.id, rival_id):
                raise error(403, "no_es_amigo", "Solo podés usar como rival a tus amigos")
            otro = db.scalar(select(Perfil).where(Perfil.usuario_id == rival_id, Perfil.ranura == 1))
            if otro is None:
                raise error(404, "no_encontrado", "No existe ese perfil")
        st = stat_general(db, otro, modalidad)
        r_metricas, r_curva, r_nombre, r_rotulo = st["promedio"], curva_promedio(db, otro, modalidad), otro.nombre, st.get("rotulo")
    elif rival_tipo == "cajon":
        premium_requerido(u)
        c = db.get(Perfil, rival_id)
        if c is None or c.usuario_id != u.id or c.ranura == 1 or not c.customizado:
            raise error(404, "no_encontrado", "No existe ese cajón")
        estilo = c.estilo
        st = stat_general(db, c, modalidad)
        r_metricas, r_curva, r_nombre, r_rotulo = st["promedio"], curva_promedio(db, c, modalidad), c.nombre, textos.GUARDIA_SUPUESTA
        cajones_expo = [c.id]
    else:
        raise error(422, "rival_tipo_invalido", "El rival es arquetipo, publico, amigo o cajon")

    yo = stat_general(db, propio, modalidad)
    m = yo["promedio"]
    expo = exposicion_real(db, propio, cajones_expo)
    entrada = Entrada(
        sesion_id=f"{propio.id}:{rival_tipo}:{rival_id}:{modalidad}",
        golpes=int(round(m.get("golpes") or 0)),   # promedio por sesion, contra el minimo de la regla
        pct_mano_adelantada=m.get("pct_mano_adelantada"), sosten_ritmo=m.get("sosten_ritmo"),
        pct_golpes_en_combinacion=m.get("pct_golpes_en_combinacion"), guardia=propio.guardia,
        lado_izquierdo=expo["lado_izquierdo_usuario"] if expo else 0,
        lado_derecho=expo["lado_derecho_usuario"] if expo else 0,
        estilo_rival=estilo, volumen=m.get("volumen"), volumen_rival=(r_metricas or {}).get("volumen"),
    )
    return {
        "modalidad": modalidad,
        "yo": {"nombre": propio.nombre, "metricas": m, "curva": curva_promedio(db, propio, modalidad), "sesiones": yo["sesiones"]},
        "rival": {"tipo": rival_tipo, "nombre": r_nombre, "estilo": estilo, "metricas": r_metricas, "curva": r_curva,
                  "rotulo": r_rotulo},
        "exposicion": expo or expo_simulada,
        "altura": altura(db, propio, modalidad),
        "consejos": consejos(entrada),
        "aclaracion": "Comparación de golpes detectados en la misma modalidad; no predice un resultado.",
    }
