"""
BoxTwinZ API - Dependencias de FastAPI: usuario actual, admin y errores con codigo.

POR QUE EXISTE
  La spec nombra los errores por codigo (402 cupo_agotado, 403 plan_requerido). La web los
  distingue por ese codigo y no por el texto, que puede cambiar.

USO
  def ruta(u: Usuario = Depends(usuario_actual)): ...
  raise error(403, "plan_requerido", "Esto es solo premium")
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.models import Usuario
from boxtwinz_api.seguridad import leer_jwt

__all__ = ["COOKIE", "error", "usuario_actual", "admin_actual", "premium_requerido"]

COOKIE = "boxtwinz_sesion"


def error(status: int, codigo: str, mensaje: str) -> HTTPException:
    return HTTPException(status, {"codigo": codigo, "mensaje": mensaje})


def usuario_actual(request: Request, db: Session = Depends(obtener_sesion)) -> Usuario:
    token = request.cookies.get(COOKIE)
    uid = leer_jwt(token, cfg.secreto) if token else None
    u = db.get(Usuario, uid) if uid else None
    if u is None:
        raise error(401, "no_autenticado", "Tenés que ingresar")
    if u.suspendido:
        raise error(403, "cuenta_suspendida", "La cuenta está suspendida")
    return u


def admin_actual(u: Usuario = Depends(usuario_actual)) -> Usuario:
    if u.rol != "admin":
        raise error(403, "solo_admin", "Solo para administración")
    return u


def premium_requerido(u: Usuario) -> None:
    if u.plan != "premium":
        raise error(403, "plan_requerido", "Esta función es del plan premium")
