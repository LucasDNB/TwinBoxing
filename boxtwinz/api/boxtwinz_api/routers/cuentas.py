"""
BoxTwinZ API - Cuentas (spec 3 y 5, Z2): registro con los dos checks, ingreso, plan y baja.

POR QUE EXISTE
  El registro es donde se juega el consentimiento: el de uso es obligatorio y se guarda con
  fecha, version del texto e IP; el de reentrenamiento es opcional y revocable. Al registrarse
  se crean las 4 ranuras de perfil (spec 7), en la misma transaccion que el usuario.

USO
  app.include_router(router, prefix="/api")
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.config import cfg
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import COOKIE, error, usuario_actual
from boxtwinz_api.models import (
    Amistad, Arquetipo, Denuncia, EventoPublicidad, Perfil, Sesion, Usuario, UsoMinutos,
    Valoracion, ahora,
)
from boxtwinz_api.seguridad import emitir_jwt, hashear, verificar
from boxtwinz_api.services.borrado import borrar_sesiones
from boxtwinz_api.services.profiles import crear_ranuras, perfil_propio

router = APIRouter(tags=["cuentas"])


class Registro(BaseModel):
    email: EmailStr
    clave: str = Field(min_length=8, max_length=200)
    nombre: str = Field(min_length=1, max_length=120)
    guardia: Literal["ortodoxa", "zurda"]
    consentimiento_uso: bool = False
    consentimiento_reentrenamiento: bool = False


class Ingreso(BaseModel):
    email: EmailStr
    clave: str


class CambioYo(BaseModel):
    nombre: str | None = Field(default=None, min_length=1, max_length=120)
    guardia: Literal["ortodoxa", "zurda"] | None = None
    consentimiento_reentrenamiento: bool | None = None


def yo_dict(u: Usuario, db: Session) -> dict:
    p1 = perfil_propio(db, u.id)
    return {
        "id": u.id, "email": u.email, "nombre": u.nombre, "plan": u.plan, "rol": u.rol,
        "guardia": p1.guardia if p1 else None,
        "consentimiento_uso_at": u.consentimiento_uso_at.isoformat(),
        "consentimiento_uso_version": u.consentimiento_uso_version,
        "consentimiento_reentrenamiento": u.consentimiento_reent_at is not None,
        "suspendido": u.suspendido,
        "pago_simulado": textos.PAGO_SIMULADO if u.plan == "premium" else None,
    }


def _poner_cookie(resp: Response, uid: str) -> None:
    resp.set_cookie(
        COOKIE, emitir_jwt(uid, cfg.secreto, cfg.dias_de_sesion),
        max_age=cfg.dias_de_sesion * 86400, httponly=True, samesite="lax",
        secure=cfg.cookie_segura, path="/",
    )


@router.post("/auth/registro", status_code=201)
def registro(datos: Registro, request: Request, response: Response,
             db: Session = Depends(obtener_sesion)) -> dict:
    if not datos.consentimiento_uso:
        raise error(422, "consentimiento_requerido",
                    "Para usar BoxTwinZ tenés que aceptar el consentimiento de uso")
    email = datos.email.lower()
    if db.scalar(select(Usuario).where(Usuario.email == email)):
        raise error(409, "email_registrado", "Ya hay una cuenta con ese email")
    t = ahora()
    u = Usuario(
        email=email, hash=hashear(datos.clave), nombre=datos.nombre.strip(),
        rol="admin" if email in cfg.admins else "usuario", plan="free",
        consentimiento_uso_at=t, consentimiento_uso_version=cfg.version_consentimiento_uso,
        consentimiento_uso_ip=request.client.host if request.client else None,
        consentimiento_reent_at=t if datos.consentimiento_reentrenamiento else None,
        consentimiento_reent_version=(cfg.version_consentimiento_reent
                                      if datos.consentimiento_reentrenamiento else None),
    )
    db.add(u)
    db.flush()
    crear_ranuras(db, u, datos.guardia)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise error(409, "email_registrado", "Ya hay una cuenta con ese email") from None
    _poner_cookie(response, u.id)
    return yo_dict(u, db)


@router.post("/auth/ingreso")
def ingreso(datos: Ingreso, response: Response, db: Session = Depends(obtener_sesion)) -> dict:
    u = db.scalar(select(Usuario).where(Usuario.email == datos.email.lower()))
    if u is None or not verificar(datos.clave, u.hash):
        raise error(401, "credenciales_invalidas", "Email o contraseña incorrectos")
    if u.suspendido:
        raise error(403, "cuenta_suspendida", "La cuenta está suspendida")
    # El rol sigue a BOXTWINZ_ADMINS: sacar un email de la lista le quita el admin al ingresar.
    rol = "admin" if u.email in cfg.admins else "usuario"
    if u.rol != rol:
        u.rol = rol
        db.commit()
    _poner_cookie(response, u.id)
    return yo_dict(u, db)


@router.post("/auth/salida", status_code=204)
def salida(response: Response) -> Response:
    response.delete_cookie(COOKIE, path="/")
    response.status_code = 204
    return response


@router.get("/yo")
def yo(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return yo_dict(u, db)


@router.patch("/yo")
def cambiar_yo(datos: CambioYo, u: Usuario = Depends(usuario_actual),
               db: Session = Depends(obtener_sesion)) -> dict:
    if datos.nombre is not None:
        u.nombre = datos.nombre.strip()
    if datos.guardia is not None:
        p1 = perfil_propio(db, u.id)
        if p1:
            p1.guardia = datos.guardia
    if datos.consentimiento_reentrenamiento is not None:
        if datos.consentimiento_reentrenamiento:
            u.consentimiento_reent_at = ahora()
            u.consentimiento_reent_version = cfg.version_consentimiento_reent
        else:
            # Revocar: lo que ya se uso en un modelo entrenado no se retira (texto del check).
            u.consentimiento_reent_at = None
            u.consentimiento_reent_version = None
    db.commit()
    return yo_dict(u, db)


@router.post("/yo/premium")
def premium(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    """Pago simulado: no cobra. Registra la intencion y cambia el plan."""
    u.plan = "premium"
    u.premium_intencion_at = ahora()
    db.commit()
    return {**yo_dict(u, db), "aviso": textos.PAGO_SIMULADO}


@router.delete("/yo", status_code=204)
def baja(response: Response, u: Usuario = Depends(usuario_actual),
         db: Session = Depends(obtener_sesion)) -> Response:
    """
    Borra videos, perfiles y sesiones. Las valoraciones ya usadas en un modelo quedan
    anonimizadas (sin sesion ni comentario, con la modalidad para el panel); el resto se borra.
    """
    sesiones = db.scalars(select(Sesion).where(Sesion.usuario_id == u.id)).all()
    ids = [s.id for s in sesiones]
    for s in sesiones:
        Path(s.video_ruta).unlink(missing_ok=True)
        shutil.rmtree(cfg.dir_trabajo(s.id), ignore_errors=True)
    if ids:
        db.execute(update(Valoracion).where(Valoracion.sesion_id.in_(ids), Valoracion.usada_en_modelo)
                   .values(sesion_id=None, comentario=None))
    # El orden importa en sqlite: las sesiones apuntan a perfiles y los perfiles al usuario.
    borrar_sesiones(db, ids)
    perfiles = [p.id for p in db.scalars(select(Perfil).where(Perfil.usuario_id == u.id))]
    if perfiles:
        db.execute(delete(Denuncia).where(Denuncia.perfil_id.in_(perfiles)))
    db.execute(update(Sesion).where(Sesion.rival_perfil_id.in_(perfiles)).values(rival_perfil_id=None))
    db.execute(delete(Perfil).where(Perfil.usuario_id == u.id))
    db.execute(delete(Arquetipo).where(Arquetipo.usuario_id == u.id))
    db.execute(delete(Amistad).where(or_(Amistad.solicitante_id == u.id, Amistad.destinatario_id == u.id)))
    db.execute(update(Denuncia).where(Denuncia.denunciante_id == u.id).values(denunciante_id=None))
    db.execute(update(EventoPublicidad).where(EventoPublicidad.usuario_id == u.id).values(usuario_id=None))
    db.execute(delete(UsoMinutos).where(UsoMinutos.usuario_id == u.id))
    db.delete(u)
    db.commit()
    response.delete_cookie(COOKIE, path="/")
    response.status_code = 204
    return response
