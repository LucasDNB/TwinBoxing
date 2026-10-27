"""
BoxTwin API - Que tiene cada version (prototipo/specs/versiones-26-10, 01 §4 y 02 §5).

POR QUE EXISTE
  Una version es el plan de la cuenta, no otra aplicacion: una web, una API, una base. Lo
  que distingue a Individual de Entrenador y de Gimnasio tiene que estar escrito UNA vez,
  aca, para que la API rechace lo que la version no tiene (RF-V2) y la web lo reciba de
  GET /cuenta en vez de copiarlo.

  Las horas salen de la Tabla 5.7 del Cap 5. Si el Cap 5 cambia, cambian aca y en ningun
  otro lado.

QUE HACE
  La tabla de planes y los cuatro perfiles de la version de prueba (B29).

USO
  from boxtwin_api.planes import PLANES, PERFILES_DEMO
  PLANES["individual"].rival   # False
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["PERFILES_DEMO", "PLANES", "VERSIONES", "Plan", "PerfilDemo", "email_demo"]


@dataclass(frozen=True)
class Plan:
    version: str
    horas_mes: float            # Cap 5, Tabla 5.7
    miembros_max: int | None
    peleadores: str             # "propio" | "libre"
    rival: bool
    invitaciones: bool
    plantel_compartido: bool
    consentimiento: str         # "sesion" | "atleta"
    en_vivo: bool               # False en las tres: el Cap 5 lo preve para Gimnasio y no existe

    def funciones(self) -> dict:
        d = asdict(self)
        d.pop("version")
        return d


# Tabla 5.7 del Cap 5: 4, 15 y 50 horas de video por mes. Federacion queda fuera (R-08).
PLANES: dict[str, Plan] = {
    "individual": Plan("individual", 4, 1, "propio", False, False, False, "sesion", False),
    "entrenador": Plan("entrenador", 15, 1, "libre", True, False, False, "sesion", False),
    "gimnasio": Plan("gimnasio", 50, None, "libre", True, True, True, "atleta", False),
}
VERSIONES = tuple(PLANES)

NOMBRE_VERSION = {"individual": "Individual", "entrenador": "Entrenador", "gimnasio": "Gimnasio"}


@dataclass(frozen=True)
class PerfilDemo:
    perfil: str
    nombre: str
    version: str
    rol: str
    descripcion: str
    # Los dos de Gimnasio se pueden apagar aca si la tanda 2 no llega a desplegarse: la web
    # los muestra en "En construccion" y la ruta contesta 409 (04 B29).
    disponible: bool = True


# El orden es el de los botones del ingreso (03 §2). La web no los fija.
PERFILES_DEMO: dict[str, PerfilDemo] = {
    p.perfil: p
    for p in (
        PerfilDemo("individual", "Individual", "individual", "titular",
                   "Un boxeador con su bolsa, su sombra y sus sparrings."),
        PerfilDemo("entrenador", "Entrenador", "entrenador", "titular",
                   "Un entrenador con sus peleadores."),
        PerfilDemo("gimnasio_titular", "Gimnasio · titular", "gimnasio", "titular",
                   "Administra el gimnasio: plantel, equipo y horas."),
        PerfilDemo("gimnasio_entrenador", "Gimnasio · entrenador", "gimnasio", "entrenador",
                   "Un entrenador del mismo gimnasio."),
    )
}

DOMINIO_DEMO = "prueba.boxtwin"


def email_demo(perfil: str) -> str:
    """El email interno del usuario de un perfil. No se muestra ni sirve para el login."""
    return f"{perfil}@{DOMINIO_DEMO}"


def perfil_de_email(email: str) -> str | None:
    if not email.endswith("@" + DOMINIO_DEMO):
        return None
    p = email.split("@", 1)[0]
    return p if p in PERFILES_DEMO else None
