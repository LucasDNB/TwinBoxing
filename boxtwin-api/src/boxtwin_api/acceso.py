"""
BoxTwin API - De que cuenta es cada pedido (versiones 26-10, B19; 02 §4).

POR QUE EXISTE
  Hasta el 26-10 cada sesion tenia un dueno y cada ruta comparaba `usuario_id`. Con las
  versiones el dueno es la cuenta: en Gimnasio un entrenador ve lo que subio otro. Si cada
  ruta siguiera decidiendo por su cuenta, alcanza con una que se olvide para que una sesion
  quede a la vista de quien no debe, y ese error no se ve como un error sino como una sesion
  que alguien no deberia ver (riesgo 1 de 01 §8).

  Por eso hay UNA dependencia que resuelve usuario, cuenta, rol y plan, y dos funciones que
  resuelven recursos de la cuenta. RNF-V2 lo verifica una prueba estructural.

  La membresia se lee en cada pedido, junto con el usuario: un miembro dado de baja queda
  afuera en el pedido siguiente aunque su token no haya vencido (RF-G5).

QUE HACE
  Acceso y acceso_actual; el lector de tickets que ademas verifica la cuenta; recursos de
  la cuenta con 404 para lo ajeno; chequeos de version y de rol; y el relleno idempotente
  que convierte cada usuario anterior en una cuenta Entrenador (02 §8).

USO
  from boxtwin_api.acceso import Acceso, acceso_actual, sesion_de_la_cuenta
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from boxtwin_api import config
from boxtwin_api.db import obtener_sesion
from boxtwin_api.modelos import Boxeador, Cuenta, Miembro, Sesion, Usuario
from boxtwin_api.planes import NOMBRE_VERSION, PLANES, Plan
from boxtwin_api.seguridad import leer_ticket, leer_token

__all__ = [
    "Acceso",
    "acceso_actual",
    "acceso_de_usuario",
    "acceso_por_ticket",
    "boxeador_de_la_cuenta",
    "completar_cuentas",
    "exigir_funcion",
    "es_de_prueba",
    "exigir_titular",
    "miembro_activo",
    "sesion_de_la_cuenta",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Acceso:
    usuario: Usuario
    cuenta: Cuenta
    rol: str          # "titular" | "entrenador"
    plan: Plan


def acceso_de_usuario(u: Usuario, db: Session) -> Acceso:
    """La cuenta y el rol de un usuario, creando la cuenta Entrenador si es de antes (02 §8)."""
    m = db.scalar(select(Miembro).where(Miembro.usuario_id == u.id))
    if m is None:
        # Un usuario sin miembro: de una base anterior a las cuentas, de la base de pruebas
        # o creado a mano. Un solo camino para los tres.
        completar_cuentas(db, u)
        db.commit()
        m = db.scalar(select(Miembro).where(Miembro.usuario_id == u.id))
    cuenta = db.get(Cuenta, m.cuenta_id) if m is not None else None
    if m is None or cuenta is None:   # pragma: no cover - completar_cuentas lo acaba de crear
        raise HTTPException(401, "la cuenta ya no existe")
    if m.baja_en is not None:
        # Un entrenador quitado no recibe una cuenta nueva sin que nadie lo decida.
        raise HTTPException(401, f"tu acceso a {cuenta.nombre} fue dado de baja")
    plan = PLANES.get(cuenta.version) or PLANES["entrenador"]
    return Acceso(usuario=u, cuenta=cuenta, rol=m.rol, plan=plan)


def _usuario_de_id(uid: str | None, db: Session) -> Usuario:
    if not uid:
        raise HTTPException(401, "hace falta iniciar sesion")
    u = db.get(Usuario, uid)
    if u is None:
        raise HTTPException(401, "la cuenta ya no existe")
    return u


def _token_del_header(authorization: str) -> str:
    return authorization[7:] if authorization.lower().startswith("bearer ") else ""


def acceso_actual(
    authorization: str = Header(default=""), db: Session = Depends(obtener_sesion)
) -> Acceso:
    """La dependencia de toda ruta que toca datos. Reemplaza a usuario_actual."""
    token = _token_del_header(authorization)
    uid = leer_token(token, config.cfg.secreto) if token else None
    return acceso_de_usuario(_usuario_de_id(uid, db), db)


def acceso_por_ticket(
    sesion_id: str,
    t: str = "",
    authorization: str = Header(default=""),
    db: Session = Depends(obtener_sesion),
) -> Acceso:
    """
    El acceso, entrando por el header o por un ticket de esta sesion.

    El <video> y el <img> no mandan headers, asi que llevan un ticket de media hora para una
    sola sesion. Al leerlo se resuelve tambien la cuenta del usuario: la ruta despues pide la
    sesion con sesion_de_la_cuenta, y un ticket emitido para un usuario que ya no es miembro
    de la cuenta duena no sirve.
    """
    uid = leer_ticket(t, sesion_id, config.cfg.secreto) if t else None
    if uid is None:
        token = _token_del_header(authorization)
        uid = leer_token(token, config.cfg.secreto) if token else None
    return acceso_de_usuario(_usuario_de_id(uid, db), db)


def sesion_de_la_cuenta(sesion_id: str, acceso: Acceso, db: Session) -> Sesion:
    """
    La sesion, si es de esta cuenta. RNF-V1 pasa por aca y por ningun otro lado.

    404 y no 403 cuando es de otra cuenta: contestar "existe pero no es tuya" le dice a
    cualquiera que adivine un id que ese id existe.
    """
    s = db.get(Sesion, sesion_id)
    if s is None or s.cuenta_id != acceso.cuenta.id:
        raise HTTPException(404, "no hay una sesion con ese id")
    return s


def boxeador_de_la_cuenta(bid: str, acceso: Acceso, db: Session) -> Boxeador:
    b = db.get(Boxeador, bid)
    # El mismo 404 para inexistente y ajeno, igual que en el login.
    if b is None or b.cuenta_id != acceso.cuenta.id:
        raise HTTPException(404, "no existe ese boxeador")
    return b


def miembro_activo(db: Session, cuenta_id: str, usuario_id: str) -> Miembro | None:
    """El miembro activo de esa cuenta con ese usuario, o None."""
    m = db.scalar(select(Miembro).where(Miembro.usuario_id == usuario_id))
    if m is None or m.cuenta_id != cuenta_id or m.baja_en is not None:
        return None
    return m


def es_de_prueba(u: Usuario, db: Session) -> bool:
    """Los usuarios de la version de prueba entran solo por su boton (02 §11), nunca por clave."""
    m = db.scalar(select(Miembro).where(Miembro.usuario_id == u.id))
    c = db.get(Cuenta, m.cuenta_id) if m is not None else None
    return bool(c is not None and c.demo)


def exigir_funcion(acceso: Acceso, tiene: bool) -> None:
    """403 si la version no tiene la funcion. Va despues del 404 del recurso (02 §4)."""
    if not tiene:
        raise HTTPException(
            403, f"esta función no está en la versión {NOMBRE_VERSION.get(acceso.plan.version)}"
        )


def exigir_titular(acceso: Acceso) -> None:
    if acceso.rol != "titular":
        raise HTTPException(403, "solo el titular de la cuenta puede hacer esto")


# ---------------------------------------------------------------------------
# Relleno (02 §8)
# ---------------------------------------------------------------------------


def _duracion_en_disco(sesion_id: str) -> tuple[float | None, float | None]:
    ruta = config.cfg.dir_sesion(sesion_id) / "sesion.json"
    try:
        v = (json.loads(ruta.read_text()).get("video") or {})
    except (OSError, ValueError, AttributeError):
        return None, None
    dur, fps = v.get("duracion_s"), v.get("fps")
    return (float(dur) if isinstance(dur, (int, float)) else None,
            float(fps) if isinstance(fps, (int, float)) else None)


def completar_cuentas(db: Session, usuario: Usuario | None = None) -> dict:
    """
    Idempotente: a cada usuario sin miembro le crea una cuenta Entrenador con su email como
    nombre y un miembro titular, y pone esa cuenta en sus sesiones y peleadores. A cada sesion
    sin duracion le copia la de su sesion.json (B20). No commitea; devuelve lo que toco.

    Con `usuario` solo mira a ese, que es el camino de acceso_actual.
    """
    tocado = {"cuentas": 0, "sesiones": 0, "boxeadores": 0, "duraciones": 0}
    con_miembro = select(Miembro.usuario_id)
    q = select(Usuario).where(Usuario.id.not_in(con_miembro))
    if usuario is not None:
        q = q.where(Usuario.id == usuario.id)
    for u in db.scalars(q).all():
        c = Cuenta(nombre=u.email[:120], version="entrenador")
        db.add(c)
        db.flush()
        db.add(Miembro(cuenta_id=c.id, usuario_id=u.id, rol="titular"))
        tocado["cuentas"] += 1
    db.flush()

    # Sesiones y peleadores sin cuenta toman la de quien los subio o cargo. Cubre tambien lo
    # que subio el codigo anterior si alguien volvio de commit y despues avanzo otra vez.
    cuenta_de = {m.usuario_id: m.cuenta_id for m in db.scalars(select(Miembro)).all()}
    for modelo, clave in ((Sesion, "sesiones"), (Boxeador, "boxeadores")):
        q = select(modelo).where(modelo.cuenta_id.is_(None))
        if usuario is not None:
            q = q.where(modelo.usuario_id == usuario.id)
        for fila in db.scalars(q).all():
            cid = cuenta_de.get(fila.usuario_id)
            if cid is not None:
                fila.cuenta_id = cid
                tocado[clave] += 1

    if usuario is None:
        for s in db.scalars(select(Sesion).where(Sesion.duracion_s.is_(None))).all():
            dur, fps = _duracion_en_disco(s.id)
            if dur is not None:
                s.duracion_s = dur
                if s.fps is None:
                    s.fps = fps
                tocado["duraciones"] += 1
    db.flush()
    return tocado


def crear_indices(db: Session) -> None:
    """El indice unico por cuenta, en una base anterior (create_all no toca tablas existentes)."""
    db.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_boxeador_cuenta_nombre "
        "ON boxeadores (cuenta_id, nombre)"
    ))


def migrar(db: Session) -> dict:
    """Lo que corre crear_tablas: indice y relleno, en un commit. Registra cuanto toco."""
    crear_indices(db)
    tocado = completar_cuentas(db)
    db.commit()
    if any(tocado.values()):
        log.info("relleno de cuentas: %s", tocado)
    return tocado
