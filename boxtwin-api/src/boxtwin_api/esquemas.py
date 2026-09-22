"""
BoxTwin API - Los cuerpos de entrada y salida.

POR QUE EXISTE
  Para que el contrato de la API este escrito en un solo lugar y no repartido entre las
  funciones de ruta. Lo que se valida aca no es formato sino reglas del producto: un round
  no puede durar cero, un tipo de golpe es uno de cuatro, y una semilla es un track.

QUE HACE
  Modelos de pydantic 2 para request y response.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

__all__ = [
    "Credenciales",
    "EntradaCorreccion",
    "Siembra",
    "Token",
    "sesion_a_dict",
]

TIPOS = ("jab", "cross", "hook", "uppercut")


class Credenciales(BaseModel):
    email: EmailStr
    clave: str = Field(min_length=8, max_length=256)
    # Solo lo mira el registro. El login lo ignora, asi que cerrar el registro no deja
    # afuera a los que ya tienen cuenta.
    invitacion: str = Field(default="", max_length=256)


class Token(BaseModel):
    token: str
    tipo: str = "bearer"
    usuario: str


class Siembra(BaseModel):
    """La respuesta del paso 4: cual de los dos es cual."""

    track_a: int
    track_b: int

    @field_validator("track_b")
    @classmethod
    def distintos(cls, v: int, info):
        if info.data.get("track_a") == v:
            raise ValueError("las dos semillas son el mismo track")
        return v


class EntradaBoxeador(BaseModel):
    """Un boxeador con nombre. La guardia es opcional: se puede completar despues."""

    nombre: str = Field(min_length=1, max_length=120)
    guardia: Literal["ortodoxa", "zurda"] | None = None
    notas: str | None = None


class AsignarBoxeadores(BaseModel):
    """
    A quien corresponde cada lado de una sesion.

    Los dos son opcionales y se pueden mandar de a uno: sparring contra alguien que no esta
    cargado es el caso normal, y obligar a nombrar a los dos convertiria una anotacion util
    en un tramite.
    """

    boxeador_a: str | None = None
    boxeador_b: str | None = None

    @field_validator("boxeador_b")
    @classmethod
    def distintos(cls, v, info):
        if v is not None and info.data.get("boxeador_a") == v:
            raise ValueError("los dos lados no pueden ser el mismo boxeador")
        return v


class EntradaCorreccion(BaseModel):
    tipo: Literal["jab", "cross", "hook", "uppercut"]


def sesion_a_dict(s, extra: dict | None = None) -> dict:
    """La sesion como la ve el frontend. La Fight-Card va aparte, que es pesada."""
    d = {
        "id": s.id,
        "nombre": s.nombre,
        "estado": s.estado,
        "error": s.error,
        "video": s.video_nombre,
        "duracion_s": s.duracion_s,
        "fps": s.fps,
        "rounds": {"round_s": s.round_s, "descanso_s": s.descanso_s},
        "siembra": (
            {"track_a": s.semilla_a, "track_b": s.semilla_b}
            if s.semilla_a is not None else None
        ),
        "creada": s.creada.isoformat() if s.creada else None,
        "actualizada": s.actualizada.isoformat() if s.actualizada else None,
    }
    if extra:
        d.update(extra)
    return d
