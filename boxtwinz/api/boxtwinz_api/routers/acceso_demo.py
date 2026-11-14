"""
BoxTwinZ API - Acceso de demo desde el login: clave compartida y selector de usuarios.

POR QUE EXISTE
  Para la presentacion y para quien prueba la app hace falta entrar sin registrarse. La
  clave de demo (BOXTWINZ_DEMO_CLAVE) abre una pantalla con usuarios estandar (Fajador,
  Estilista, Zurdo) y los creados ahi mismo; todos son amigos entre si para compararse y
  tienen el perfil publico para salir en el ranking.

QUE HACE
  GET  /demo                          si la demo esta encendida (BOXTWINZ_DEMO)
  POST /demo/acceso                   valida la clave, siembra los estandar y deja la cookie de demo
  GET  /demo/usuarios                 estandar primero y despues los creados
  POST /demo/usuarios                 crear usuario rapido: nombre, estilo, guardia y peso
  POST /demo/usuarios/{id}/ingresar   deja la cookie de sesion de ese usuario

  La cookie de demo es un JWT con sub "demo", separada de la de sesion: un JWT de usuario
  no la reemplaza ni al reves.

USO
  app.include_router(router, prefix="/api")
"""

from __future__ import annotations

import hmac
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api import demo
from boxtwinz_api.config import cfg
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import error
from boxtwinz_api.models import Perfil, Usuario
from boxtwinz_api.routers.cuentas import _poner_cookie, yo_dict
from boxtwinz_api.seguridad import emitir_jwt, leer_jwt

router = APIRouter(tags=["demo"])

COOKIE_DEMO = "boxtwinz_demo"


class Acceso(BaseModel):
    clave: str = Field(max_length=200)


class NuevoUsuario(BaseModel):
    nombre: str = Field(min_length=1, max_length=60)
    estilo: Literal["fajador", "estilista", "hibrido"] | None = None
    guardia: Literal["ortodoxa", "zurda"]
    peso_kg: float = Field(ge=30, le=200)


def demo_validada(request: Request) -> None:
    if not cfg.demo:
        raise error(404, "demo_apagada", "La demo no está disponible")
    token = request.cookies.get(COOKIE_DEMO)
    if not token or leer_jwt(token, cfg.secreto) != "demo":
        raise error(401, "demo_requerida", "Ingresá la contraseña de la demo")


def _usuario_dict(db: Session, u: Usuario) -> dict:
    p = db.scalar(select(Perfil).where(Perfil.usuario_id == u.id, Perfil.ranura == 1))
    estandar = u.email in {demo.email_estandar(k) for k in demo.ESTANDAR}
    return {"id": u.id, "nombre": u.nombre, "estandar": estandar,
            "estilo": p.estilo_declarado if p else None, "guardia": p.guardia if p else None,
            "peso_kg": p.peso_kg if p else None}


@router.get("/demo")
def estado() -> dict:
    return {"habilitada": cfg.demo}


@router.post("/demo/acceso", status_code=204)
def acceso(datos: Acceso, response: Response, db: Session = Depends(obtener_sesion)) -> Response:
    if not cfg.demo:
        raise error(404, "demo_apagada", "La demo no está disponible")
    if not hmac.compare_digest(datos.clave.encode(), cfg.demo_clave.encode()):
        raise error(401, "clave_demo_invalida", "La contraseña de la demo no es correcta")
    demo.asegurar_estandar(db)
    response.set_cookie(COOKIE_DEMO, emitir_jwt("demo", cfg.secreto, cfg.demo_horas / 24),
                        max_age=cfg.demo_horas * 3600, httponly=True, samesite="lax",
                        secure=cfg.cookie_segura, path="/")
    response.status_code = 204
    return response


@router.get("/demo/usuarios", dependencies=[Depends(demo_validada)])
def usuarios(db: Session = Depends(obtener_sesion)) -> list[dict]:
    us = db.scalars(select(Usuario).where(Usuario.demo.is_(True), Usuario.suspendido.is_(False))
                    .order_by(Usuario.creado)).all()
    orden = {demo.email_estandar(k): i for i, k in enumerate(demo.ESTANDAR)}
    # Estandar en su orden fijo y despues los creados, del mas viejo al mas nuevo (sort estable).
    return [_usuario_dict(db, u) for u in sorted(us, key=lambda u: orden.get(u.email, len(orden)))]


@router.post("/demo/usuarios", status_code=201, dependencies=[Depends(demo_validada)])
def crear(datos: NuevoUsuario, db: Session = Depends(obtener_sesion)) -> dict:
    u = demo.crear_usuario_demo(db, datos.nombre, datos.estilo, datos.guardia, round(datos.peso_kg, 1))
    db.commit()
    return _usuario_dict(db, u)


@router.post("/demo/usuarios/{usuario_id}/ingresar", dependencies=[Depends(demo_validada)])
def ingresar(usuario_id: str, response: Response, db: Session = Depends(obtener_sesion)) -> dict:
    u = db.get(Usuario, usuario_id)
    if u is None or not u.demo or u.suspendido:
        raise error(404, "no_encontrado", "No existe ese usuario de la demo")
    _poner_cookie(response, u.id)
    return yo_dict(u, db)
