"""
BoxTwinZ API - Perfiles y arquetipos (spec 5 y 7, Z3).

POR QUE EXISTE
  Las ranuras 2 a 4 son de solo lectura en free (403 plan_requerido) y editables en premium.
  La regla vive en el servidor: la web solo esconde los botones.

USO
  app.include_router(router, prefix="/api")
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import error, premium_requerido, usuario_actual
from boxtwinz_api.models import MODALIDADES, Arquetipo, Perfil, Usuario
from boxtwinz_api.services.profiles import perfil_dict, perfiles_de
from boxtwinz_api.services.stats import stat_general

router = APIRouter(tags=["perfiles"])

Estilo = Literal["fajador", "estilista", "hibrido"]


class CambioPerfil(BaseModel):
    nombre: str | None = Field(default=None, min_length=1, max_length=120)
    guardia: Literal["ortodoxa", "zurda"] | None = None
    publico: bool | None = None
    estilo: Estilo | None = None
    customizado: bool | None = None


class PedidoCustom(BaseModel):
    arquetipo_id: str
    ranura: int = Field(ge=2, le=4)
    nombre: str | None = Field(default=None, min_length=1, max_length=120)
    # Medias por modalidad a pisar: {"sparring": {"volumen": 30}}. Lo demas se copia.
    medias: dict[str, dict[str, float]] = Field(default_factory=dict)


def _perfil_propio_de(db: Session, perfil_id: str, u: Usuario) -> Perfil:
    p = db.get(Perfil, perfil_id)
    if p is None or p.usuario_id != u.id:
        raise error(404, "no_encontrado", "No existe ese perfil")
    return p


@router.get("/perfiles")
def listar(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> list[dict]:
    return [perfil_dict(p, db) for p in perfiles_de(db, u.id)]


@router.patch("/perfiles/{perfil_id}")
def cambiar(perfil_id: str, datos: CambioPerfil, u: Usuario = Depends(usuario_actual),
            db: Session = Depends(obtener_sesion)) -> dict:
    p = _perfil_propio_de(db, perfil_id, u)
    if p.ranura == 1:
        if datos.estilo is not None or datos.customizado is not None:
            raise error(422, "campo_no_editable", "El perfil propio no tiene estilo ni cajón")
        if datos.nombre is not None:
            p.nombre = datos.nombre.strip()
        if datos.guardia is not None:
            p.guardia = datos.guardia
        if datos.publico is not None:
            p.publico = datos.publico
    else:
        premium_requerido(u)
        if datos.guardia is not None or datos.publico is not None:
            raise error(422, "campo_no_editable",
                        "Los cajones no tienen guardia ni son públicos: solo el perfil propio")
        if datos.nombre is not None:
            p.nombre = datos.nombre.strip()
        if datos.estilo is not None:
            p.estilo = datos.estilo
        if datos.customizado is not None:
            p.customizado = datos.customizado
    db.commit()
    return perfil_dict(p, db)


@router.get("/perfiles/{perfil_id}/stats")
def stats(perfil_id: str, modalidad: str = Query(...), u: Usuario = Depends(usuario_actual),
          db: Session = Depends(obtener_sesion)) -> dict:
    if modalidad not in MODALIDADES:
        raise error(422, "modalidad_invalida", "La modalidad es sombra, bolsa o sparring")
    return stat_general(db, _perfil_propio_de(db, perfil_id, u), modalidad)


def arquetipo_dict(a: Arquetipo) -> dict:
    return {"id": a.id, "clave": a.clave, "nombre": a.nombre, "estilo": a.estilo,
            "variante": a.variante, "solo_premium": a.solo_premium, "custom": a.usuario_id is not None,
            "metricas": a.metricas, "exposicion": a.exposicion, "curva_ritmo": a.curva_ritmo,
            "rotulo": textos.PERFIL_SIMULADO, "guardia_supuesta": True}


def arquetipos_visibles(db: Session, u: Usuario) -> list[Arquetipo]:
    q = select(Arquetipo).where(or_(Arquetipo.usuario_id.is_(None), Arquetipo.usuario_id == u.id))
    if u.plan != "premium":
        q = q.where(Arquetipo.solo_premium.is_(False), Arquetipo.usuario_id.is_(None))
    return list(db.scalars(q.order_by(Arquetipo.solo_premium, Arquetipo.clave)))


@router.get("/arquetipos")
def arquetipos(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> list[dict]:
    return [arquetipo_dict(a) for a in arquetipos_visibles(db, u)]


@router.post("/arquetipos/custom", status_code=201)
def custom(datos: PedidoCustom, u: Usuario = Depends(usuario_actual),
           db: Session = Depends(obtener_sesion)) -> dict:
    premium_requerido(u)
    base = db.get(Arquetipo, datos.arquetipo_id)
    if base is None or (base.usuario_id is not None and base.usuario_id != u.id):
        raise error(404, "no_encontrado", "No existe ese arquetipo")
    p = db.scalar(select(Perfil).where(Perfil.usuario_id == u.id, Perfil.ranura == datos.ranura))
    if p is None:
        raise error(404, "no_encontrado", "No existe esa ranura")
    metricas = {m: {"medias": dict(v.get("medias", {})), "desvios": dict(v.get("desvios", {}))}
                for m, v in (base.metricas or {}).items()}
    for mod, medias in datos.medias.items():
        if mod not in MODALIDADES:
            raise error(422, "modalidad_invalida", "La modalidad es sombra, bolsa o sparring")
        metricas.setdefault(mod, {"medias": {}, "desvios": {}})["medias"].update(medias)
    copia = Arquetipo(
        clave=f"custom-{uuid.uuid4().hex[:12]}", nombre=(datos.nombre or f"{base.nombre} (mío)").strip(),
        estilo=base.estilo, variante="custom", solo_premium=True, metricas=metricas,
        exposicion=dict(base.exposicion or {}), curva_ritmo=list(base.curva_ritmo or []),
        usuario_id=u.id,
    )
    db.add(copia)
    db.flush()
    p.arquetipo_id = copia.id
    p.nombre = copia.nombre
    p.estilo = copia.estilo
    p.customizado = False
    db.commit()
    return {"arquetipo": arquetipo_dict(copia), "perfil": perfil_dict(p, db)}
