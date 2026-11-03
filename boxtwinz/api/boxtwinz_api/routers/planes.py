"""
BoxTwinZ API - Planes: impresiones del espacio publicitario (spec 9).

POR QUE EXISTE
  No hay red de anuncios: el espacio es un recuadro rotulado y lo unico que se registra es
  la impresion, para el bloque de Ingresos del panel. En premium no se registra nada.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import usuario_actual
from boxtwinz_api.models import EventoPublicidad, Usuario

router = APIRouter(tags=["planes"])


class Impresion(BaseModel):
    ubicacion: str = Field(min_length=1, max_length=40)


@router.post("/publicidad/impresion", status_code=201)
def impresion(datos: Impresion, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    if u.plan != "free":
        return {"registrada": False}
    db.add(EventoPublicidad(usuario_id=u.id, ubicacion=datos.ubicacion, tipo="impresion"))
    db.commit()
    return {"registrada": True}
