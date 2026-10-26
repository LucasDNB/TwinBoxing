"""
BoxTwin API - La cuenta, sus miembros y los consentimientos (versiones 26-10, B19, B24, B25).

POR QUE EXISTE
  La web nunca decide que version es ni que funciones tiene: lo pregunta a GET /cuenta y
  dibuja, igual que con GET /producto. Y en Gimnasio hay dos cosas que son de la cuenta y no
  de una sesion: quienes trabajan en ella y el consentimiento de cada atleta del plantel.

  El consentimiento por atleta existe porque en un gimnasio se filma a los mismos todas las
  semanas, y el Cap 5 §5.2 pide el de los atletas y no solo el de quien contrata (DV-5).
  El texto que acepta el atleta NO lo escribe este codigo: vive en textos/ y lo carga Lucas.
  Si falta, la API lo dice y no inventa uno.

QUE HACE
  GET /cuenta; GET /cuenta/miembros (titular de Gimnasio, solo lectura: las bajas son de
  B23, postergada); el texto vigente; registrar, ver y revocar el consentimiento de un
  atleta; y el registro de consentimientos en CSV para el titular.

USO
  app.include_router(cuentas.router)
"""

from __future__ import annotations

import csv
import io
import re

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwin_api import config
from boxtwin_api.acceso import (
    Acceso,
    acceso_actual,
    boxeador_de_la_cuenta,
    exigir_funcion,
    exigir_titular,
)
from boxtwin_api.cupo import horas_del_mes
from boxtwin_api.db import obtener_sesion
from boxtwin_api.esquemas import EntradaConsentimiento
from boxtwin_api.modelos import Boxeador, Consentimiento, Miembro, Usuario, ahora
from boxtwin_api.planes import PERFILES_DEMO, perfil_de_email

__all__ = ["estado_consentimiento", "router", "texto_vigente"]

router = APIRouter()


def _nombre(u: Usuario | None) -> str | None:
    return (u.nombre or u.email) if u is not None else None


@router.get("/cuenta")
def cuenta(acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion)) -> dict:
    """Version, rol, funciones y horas del mes. Es lo primero que pide la web al entrar."""
    c, u = acceso.cuenta, acceso.usuario
    propio = db.get(Boxeador, c.boxeador_propio_id) if c.boxeador_propio_id else None
    perfil = perfil_de_email(u.email) if c.demo else None
    return {
        "id": c.id,
        "nombre": c.nombre,
        "version": acceso.plan.version,
        "rol": acceso.rol,
        # En una cuenta de prueba el email es interno (planes.email_demo): no se muestra.
        "usuario": {"id": u.id, "email": None if c.demo else u.email, "nombre": u.nombre},
        "boxeador_propio": ({"id": propio.id, "nombre": propio.nombre}
                            if propio is not None and propio.cuenta_id == c.id else None),
        "funciones": acceso.plan.funciones(),
        "horas": horas_del_mes(c, db),
        "demo": bool(c.demo),
        "perfil_demo": ({"perfil": perfil, "nombre": PERFILES_DEMO[perfil].nombre}
                        if perfil else None),
    }


@router.get("/cuenta/miembros")
def miembros(acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion)) -> list:
    """B24: los miembros, para la pantalla Equipo. Solo el titular de un Gimnasio."""
    exigir_funcion(acceso, acceso.plan.plantel_compartido)
    exigir_titular(acceso)
    filas = db.scalars(
        select(Miembro).where(Miembro.cuenta_id == acceso.cuenta.id).order_by(Miembro.desde)
    ).all()
    usuarios = {u.id: u for u in db.scalars(
        select(Usuario).where(Usuario.id.in_({m.usuario_id for m in filas}))).all()}
    out = []
    for m in filas:
        u = usuarios.get(m.usuario_id)
        d = {
            "usuario_id": m.usuario_id,
            "nombre": _nombre(u),
            "rol": m.rol,
            "desde": m.desde.isoformat() if m.desde else None,
            "baja_en": m.baja_en.isoformat() if m.baja_en else None,
        }
        # En una cuenta de prueba el email es interno y no se manda (04 B24).
        if not acceso.cuenta.demo:
            d["email"] = u.email if u is not None else None
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Consentimiento por atleta (B25)
# ---------------------------------------------------------------------------

_PATRON_TEXTO = re.compile(r"^consentimiento-(?P<version>[A-Za-z0-9._-]+)\.md$")


def texto_vigente() -> dict | None:
    """El texto con la version mas nueva de textos/, o None si Lucas todavia no lo cargo."""
    carpeta = config.cfg.textos
    if not carpeta.is_dir():
        return None
    versiones = []
    for f in carpeta.iterdir():
        m = _PATRON_TEXTO.match(f.name)
        if m and f.is_file():
            versiones.append((m.group("version"), f))
    if not versiones:
        return None
    version, f = max(versiones)
    return {"version": version, "texto": f.read_text(encoding="utf-8")}


def _exigir_texto() -> dict:
    t = texto_vigente()
    if t is None:
        raise HTTPException(
            409, "falta el texto del consentimiento: cargalo en "
                 "boxtwin_api/textos/consentimiento-<version>.md (DV-5)"
        )
    return t


def estado_consentimiento(b: Boxeador, db: Session) -> dict:
    """vigente si hay uno sin revocar; revocado si todos estan revocados; falta si no hay."""
    filas = db.scalars(
        select(Consentimiento).where(Consentimiento.boxeador_id == b.id)
        .order_by(Consentimiento.registrado_en)
    ).all()
    vigente = next((c for c in reversed(filas) if c.revocado_en is None), None)
    if vigente is not None:
        return {"estado": "vigente", "desde": vigente.registrado_en.isoformat()}
    if filas:
        ultimo = filas[-1]
        return {"estado": "revocado",
                "desde": ultimo.revocado_en.isoformat() if ultimo.revocado_en else None}
    return {"estado": "falta", "desde": None}


def _historia(b: Boxeador, db: Session) -> list[dict]:
    filas = db.scalars(
        select(Consentimiento).where(Consentimiento.boxeador_id == b.id)
        .order_by(Consentimiento.registrado_en)
    ).all()
    ids = {c.registrado_por for c in filas} | {c.revocado_por for c in filas if c.revocado_por}
    usuarios = {u.id: u for u in db.scalars(select(Usuario).where(Usuario.id.in_(ids))).all()}
    return [
        {
            "id": c.id,
            "texto_version": c.texto_version,
            "mayor_de_edad": c.mayor_de_edad,
            "nota": c.nota,
            "registrado_por": _nombre(usuarios.get(c.registrado_por)),
            "registrado_en": c.registrado_en.isoformat(),
            "revocado_por": _nombre(usuarios.get(c.revocado_por)) if c.revocado_por else None,
            "revocado_en": c.revocado_en.isoformat() if c.revocado_en else None,
            "semilla": bool(c.semilla),
        }
        for c in filas
    ]


def _exigir_gimnasio(acceso: Acceso) -> None:
    exigir_funcion(acceso, acceso.plan.consentimiento == "atleta")


@router.get("/cuenta/consentimiento")
def consentimiento_vigente(acceso: Acceso = Depends(acceso_actual)) -> dict:
    """El texto vigente, entero, para mostrarlo antes de registrar."""
    _exigir_gimnasio(acceso)
    return _exigir_texto()


@router.get("/boxeadores/{boxeador_id}/consentimiento")
def consentimiento_de(
    boxeador_id: str, acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """Estado e historia: quien lo registro y quien lo revoco, con fecha (G-08)."""
    b = boxeador_de_la_cuenta(boxeador_id, acceso, db)
    _exigir_gimnasio(acceso)
    return {**estado_consentimiento(b, db), "historia": _historia(b, db)}


@router.post("/boxeadores/{boxeador_id}/consentimiento", status_code=201)
def registrar_consentimiento(
    boxeador_id: str, entrada: EntradaConsentimiento,
    acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion),
) -> dict:
    """Cualquier miembro lo registra: es quien esta en el gimnasio con el atleta (02 §4)."""
    b = boxeador_de_la_cuenta(boxeador_id, acceso, db)
    _exigir_gimnasio(acceso)
    t = _exigir_texto()
    if entrada.mayor_de_edad is not True:
        raise HTTPException(422, "el atleta tiene que tener 18 años o más (R-09)")
    if entrada.texto_version != t["version"]:
        raise HTTPException(422, f"la versión vigente del texto es {t['version']}")
    if estado_consentimiento(b, db)["estado"] == "vigente":
        raise HTTPException(409, f"{b.nombre} ya tiene un consentimiento vigente")
    db.add(Consentimiento(
        cuenta_id=acceso.cuenta.id, boxeador_id=b.id, texto_version=entrada.texto_version,
        mayor_de_edad=True, nota=(entrada.nota or None), registrado_por=acceso.usuario.id,
    ))
    db.commit()
    return {**estado_consentimiento(b, db), "historia": _historia(b, db)}


@router.delete("/boxeadores/{boxeador_id}/consentimiento", status_code=204)
def revocar_consentimiento(
    boxeador_id: str, acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> Response:
    """Revocar frena las asignaciones nuevas. No borra la fila ni las sesiones anteriores."""
    b = boxeador_de_la_cuenta(boxeador_id, acceso, db)
    _exigir_gimnasio(acceso)
    vigentes = db.scalars(select(Consentimiento).where(
        Consentimiento.boxeador_id == b.id, Consentimiento.revocado_en.is_(None))).all()
    if not vigentes:
        raise HTTPException(409, f"{b.nombre} no tiene un consentimiento vigente")
    if acceso.cuenta.demo and any(c.semilla for c in vigentes):
        raise HTTPException(403, "en la versión de prueba este consentimiento viene cargado")
    for c in vigentes:
        c.revocado_en = ahora()
        c.revocado_por = acceso.usuario.id
    db.commit()
    return Response(status_code=204)


@router.get("/cuenta/consentimientos.csv")
def consentimientos_csv(
    acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion)
) -> Response:
    """El registro entero, para mostrar si alguien pregunta. Solo el titular."""
    _exigir_gimnasio(acceso)
    exigir_titular(acceso)
    filas = db.scalars(
        select(Consentimiento).where(Consentimiento.cuenta_id == acceso.cuenta.id)
        .order_by(Consentimiento.registrado_en)
    ).all()
    atletas = {b.id: b.nombre for b in db.scalars(
        select(Boxeador).where(Boxeador.id.in_({c.boxeador_id for c in filas}))).all()}
    ids = {c.registrado_por for c in filas} | {c.revocado_por for c in filas if c.revocado_por}
    usuarios = {u.id: u for u in db.scalars(select(Usuario).where(Usuario.id.in_(ids))).all()}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["atleta", "estado", "texto_version", "registrado_por", "registrado_en",
                "revocado_por", "revocado_en", "nota"])
    for c in filas:
        w.writerow([
            atletas.get(c.boxeador_id, ""),
            "revocado" if c.revocado_en else "vigente",
            c.texto_version,
            _nombre(usuarios.get(c.registrado_por)) or "",
            c.registrado_en.isoformat(),
            _nombre(usuarios.get(c.revocado_por)) if c.revocado_por else "",
            c.revocado_en.isoformat() if c.revocado_en else "",
            c.nota or "",
        ])
    return Response(
        buf.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="consentimientos.csv"'},
    )
