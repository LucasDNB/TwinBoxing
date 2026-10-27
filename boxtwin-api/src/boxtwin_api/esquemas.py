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

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

__all__ = [
    "Credenciales",
    "EntradaCorreccion",
    "Siembra",
    "Token",
    "sesion_a_dict",
]

TIPOS = ("jab", "cross", "hook", "uppercut")
CONTEXTOS = ("bolsa", "sombra", "sparring", "combate")
# Los de una sola persona (B8): una semilla, sin rival.
CONTEXTOS_INDIVIDUALES = ("bolsa", "sombra")
PRACTICAS = ("tecnica_libre", "combinaciones", "defensa")


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
    # B8: nulo en sesiones de un solo peleador (bolsa y sombra). Para sparring y combate lo
    # exige la ruta, que es la que sabe el contexto de la sesion.
    track_b: int | None = None

    @field_validator("track_b")
    @classmethod
    def distintos(cls, v: int | None, info):
        if v is not None and info.data.get("track_a") == v:
            raise ValueError("las dos semillas son el mismo track")
        return v


class EntradaBoxeador(BaseModel):
    """Un boxeador con nombre. La guardia es opcional: se puede completar despues."""

    nombre: str = Field(min_length=1, max_length=120)
    guardia: Literal["ortodoxa", "zurda"] | None = None
    notas: str | None = None
    # B6: todo opcional. Los rangos dejan afuera el error de unidad (175 en vez de 1,75).
    categoria: str | None = Field(default=None, max_length=40)
    altura_m: float | None = Field(default=None, ge=1.20, le=2.20)
    envergadura_m: float | None = Field(default=None, ge=1.20, le=2.40)
    # B17 (cambios-14-10): cargados a mano.
    distancia: Literal["larga", "media", "corta"] | None = None
    estilo: Literal["fajador", "estilista"] | None = None


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


class AsignacionTramo(BaseModel):
    track: int
    rol: Literal["A", "B", "ninguno"]


class EntradaTramos(BaseModel):
    """Las decisiones de la revision de tramos (B3). Al menos una."""

    asignaciones: list[AsignacionTramo] = Field(min_length=1, max_length=500)


class EntradaCorreccion(BaseModel):
    # "no_golpe" es la etiqueta de falso positivo (B4): el golpe sigue contando como
    # detectado y se suma a los descartados por el entrenador.
    tipo: Literal["jab", "cross", "hook", "uppercut", "no_golpe"] | None = None
    # B11 (ampliacion 03 §5): la altura. Una correccion es de tipo o de altura, no de las dos.
    objetivo: Literal["cabeza", "cuerpo"] | None = None

    @model_validator(mode="after")
    def _uno_solo(self):
        if (self.tipo is None) == (self.objetivo is None):
            raise ValueError("una correccion lleva tipo u objetivo, uno solo")
        return self


def _boxeador_corto(b) -> dict | None:
    return {"id": b.id, "nombre": b.nombre} if b is not None else None


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
        "rounds": {"round_s": s.round_s, "descanso_s": s.descanso_s, "cantidad": s.rounds},
        "contexto": s.contexto or "sparring",
        "practica": s.practica,
        "consentimiento": s.consentimiento_en is not None,
        # Los nombres van en la lista para que la web no pida los peleadores uno por uno.
        "boxeadores": {
            "A": _boxeador_corto(s.boxeador_a),
            "B": _boxeador_corto(s.boxeador_b),
        },
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
