"""
BoxTwinZ - Las 4 ranuras de perfil y sus reglas free/premium (spec 7, Z3).

POR QUE EXISTE
  La ranura 1 es el boxeador propio; las 2 a 4 nacen vinculadas a los arquetipos base y en
  premium se pueden volver cajones que reciben sesiones de sparring. Esa regla se consulta
  desde varias rutas (perfiles, siembra, rival, amigos) y tiene que decir lo mismo en todas.

USO
  from boxtwinz_api.services.profiles import crear_ranuras, perfil_propio, puede_recibir_sesiones
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.models import Arquetipo, Perfil, Usuario
from boxtwinz_api.services.archetypes import BASE

__all__ = ["crear_ranuras", "perfil_propio", "perfiles_de", "puede_recibir_sesiones", "perfil_dict"]


def crear_ranuras(db: Session, u: Usuario, guardia: str) -> list[Perfil]:
    """Ranura 1 propia y 2 a 4 con fajador, estilista e hibrido. No commitea."""
    arqs = {a.clave: a for a in db.scalars(select(Arquetipo).where(Arquetipo.clave.in_(BASE)))}
    perfiles = [Perfil(usuario_id=u.id, ranura=1, nombre=u.nombre, estilo="propio",
                       guardia=guardia, publico=False)]
    for i, clave in enumerate(BASE, start=2):
        a = arqs.get(clave)
        perfiles.append(Perfil(
            usuario_id=u.id, ranura=i, nombre=a.nombre if a else clave.capitalize(),
            estilo=clave, arquetipo_id=a.id if a else None, customizado=False,
        ))
    db.add_all(perfiles)
    db.flush()
    return perfiles


def perfil_propio(db: Session, usuario_id: str) -> Perfil | None:
    return db.scalar(select(Perfil).where(Perfil.usuario_id == usuario_id, Perfil.ranura == 1))


def perfiles_de(db: Session, usuario_id: str) -> list[Perfil]:
    return list(db.scalars(select(Perfil).where(Perfil.usuario_id == usuario_id).order_by(Perfil.ranura)))


def puede_recibir_sesiones(p: Perfil, u: Usuario) -> bool:
    """La ranura 1 siempre; un cajon 2 a 4 solo si es premium y esta customizado."""
    return p.ranura == 1 or (u.plan == "premium" and p.customizado)


def perfil_dict(p: Perfil, db: Session) -> dict:
    a = db.get(Arquetipo, p.arquetipo_id) if p.arquetipo_id else None
    simulado = p.ranura > 1 and not p.customizado
    return {
        "id": p.id, "ranura": p.ranura, "nombre": p.nombre, "estilo": p.estilo,
        "guardia": p.guardia if p.ranura == 1 else None,
        "publico": p.publico if p.ranura == 1 else False,
        "customizado": p.customizado,
        "arquetipo": ({"id": a.id, "clave": a.clave, "nombre": a.nombre} if a else None),
        "rotulo": textos.PERFIL_SIMULADO if simulado else None,
        "guardia_supuesta": p.ranura > 1,
    }
