"""
BoxTwinZ API - Ranking, perfiles publicos, amigos y denuncias (spec 5, Z9, Z10, Z12).

POR QUE EXISTE
  Lo social nunca expone los cajones 2 a 4 (CZ4): son sparrings de otra gente que el usuario
  guardo para si. Toda vista de otro usuario sale de su perfil 1 y nada mas, y eso se hace
  en un solo lugar (`vista_publica`).

USO
  app.include_router(router, prefix="/api")
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import error, usuario_actual
from boxtwinz_api.models import MODALIDADES, Amistad, Denuncia, NotificacionAdmin, Perfil, Sesion, Usuario
from boxtwinz_api.services.ranking import METRICAS_RANKING, ranking
from boxtwinz_api.services.stats import stat_general

router = APIRouter(tags=["social"])


def vista_publica(db: Session, p: Perfil, con_sesiones: bool = False) -> dict:
    """La vista de un perfil 1 ajeno: nombre, guardia y stats por modalidad. Nunca cajones."""
    assert p.ranura == 1, "solo el perfil propio se muestra a otros"
    out = {"perfil_id": p.id, "usuario_id": p.usuario_id, "nombre": p.nombre, "guardia": p.guardia,
           "stats": {m: stat_general(db, p, m) for m in MODALIDADES}}
    if con_sesiones:
        ses = db.scalars(select(Sesion).where(Sesion.perfil_id == p.id, Sesion.estado == "lista")
                         .order_by(Sesion.creado.desc()).limit(60))
        out["sesiones"] = [{"id": s.id, "modalidad": s.modalidad, "creado": s.creado.isoformat(),
                            "duracion_s": s.video_duracion_s} for s in ses]
    return out


def amistad_aceptada(db: Session, a: str, b: str) -> bool:
    return db.scalar(select(Amistad).where(
        Amistad.estado == "aceptada",
        or_((Amistad.solicitante_id == a) & (Amistad.destinatario_id == b),
            (Amistad.solicitante_id == b) & (Amistad.destinatario_id == a)))) is not None


def perfil_publico(db: Session, perfil_id: str) -> Perfil:
    p = db.get(Perfil, perfil_id)
    u = db.get(Usuario, p.usuario_id) if p else None
    if p is None or p.ranura != 1 or not p.publico or u is None or u.suspendido:
        raise error(404, "no_encontrado", "No existe ese perfil público")
    return p


@router.get("/ranking")
def ver_ranking(modalidad: str = Query(...), metrica: str = Query("volumen"),
                u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    if modalidad not in MODALIDADES:
        raise error(422, "modalidad_invalida", "La modalidad es sombra, bolsa o sparring")
    if metrica not in METRICAS_RANKING:
        raise error(422, "metrica_invalida", f"La métrica es una de {', '.join(METRICAS_RANKING)}")
    return ranking(db, modalidad, metrica, u.id)


@router.get("/publicos")
def publicos(q: str = "", u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> list[dict]:
    consulta = (select(Perfil).join(Usuario, Usuario.id == Perfil.usuario_id)
                .where(Perfil.ranura == 1, Perfil.publico.is_(True), Usuario.suspendido.is_(False)))
    if q.strip():
        consulta = consulta.where(Perfil.nombre.ilike(f"%{q.strip()}%"))
    return [{"perfil_id": p.id, "nombre": p.nombre, "guardia": p.guardia}
            for p in db.scalars(consulta.order_by(Perfil.nombre).limit(50))]


@router.get("/publicos/{perfil_id}")
def ver_publico(perfil_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return vista_publica(db, perfil_publico(db, perfil_id))


class Solicitud(BaseModel):
    email: EmailStr


@router.post("/amigos/solicitud", status_code=201)
def solicitar(datos: Solicitud, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    otro = db.scalar(select(Usuario).where(Usuario.email == datos.email.lower()))
    # Misma respuesta exista o no la cuenta: la solicitud no sirve para averiguar emails.
    if otro is None or otro.id == u.id:
        if otro is not None:
            raise error(422, "solicitud_invalida", "No te podés agregar a vos")
        return {"enviada": True}
    previa = db.scalar(select(Amistad).where(
        or_((Amistad.solicitante_id == u.id) & (Amistad.destinatario_id == otro.id),
            (Amistad.solicitante_id == otro.id) & (Amistad.destinatario_id == u.id))))
    if previa is None:
        db.add(Amistad(solicitante_id=u.id, destinatario_id=otro.id, estado="pendiente"))
    elif previa.estado == "rechazada" and previa.solicitante_id == u.id:
        previa.estado = "pendiente"
    elif previa.estado == "pendiente" and previa.destinatario_id == u.id:
        previa.estado = "aceptada"   # los dos se pidieron: queda aceptada
    db.commit()
    return {"enviada": True}


def _responder(db: Session, u: Usuario, amistad_id: str, estado: str) -> dict:
    a = db.get(Amistad, amistad_id)
    if a is None or a.destinatario_id != u.id:
        raise error(404, "no_encontrada", "No existe esa solicitud")
    if a.estado != "pendiente":
        raise error(409, "ya_respondida", "La solicitud ya fue respondida")
    a.estado = estado
    db.commit()
    return {"id": a.id, "estado": a.estado}


@router.post("/amigos/{amistad_id}/aceptar")
def aceptar(amistad_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return _responder(db, u, amistad_id, "aceptada")


@router.post("/amigos/{amistad_id}/rechazar")
def rechazar(amistad_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return _responder(db, u, amistad_id, "rechazada")


@router.get("/amigos")
def amigos(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    filas = db.scalars(select(Amistad).where(or_(Amistad.solicitante_id == u.id, Amistad.destinatario_id == u.id)))
    out: dict[str, list] = {"amigos": [], "recibidas": [], "enviadas": []}
    for a in filas:
        otro_id = a.destinatario_id if a.solicitante_id == u.id else a.solicitante_id
        otro = db.get(Usuario, otro_id)
        if otro is None:
            continue
        item = {"id": a.id, "usuario_id": otro.id, "nombre": otro.nombre}
        if a.estado == "aceptada":
            out["amigos"].append(item)
        elif a.estado == "pendiente":
            out["recibidas" if a.destinatario_id == u.id else "enviadas"].append(
                {**item, **({"email": otro.email} if a.destinatario_id == u.id else {})})
    return out


@router.get("/amigos/{usuario_id}/perfil")
def perfil_amigo(usuario_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    if not amistad_aceptada(db, u.id, usuario_id):
        raise error(403, "no_es_amigo", "Solo ves el perfil de tus amigos")
    p = db.scalar(select(Perfil).where(Perfil.usuario_id == usuario_id, Perfil.ranura == 1))
    if p is None:
        raise error(404, "no_encontrado", "No existe ese perfil")
    return vista_publica(db, p, con_sesiones=True)


class PedidoDenuncia(BaseModel):
    perfil_id: str
    motivo: Literal["no_es_la_persona", "video_ajeno", "contenido_inapropiado", "menor_en_video", "otro"]
    detalle: str | None = Field(default=None, max_length=2000)


@router.post("/denuncias", status_code=201)
def denunciar(datos: PedidoDenuncia, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    p = db.get(Perfil, datos.perfil_id)
    visible = p is not None and p.ranura == 1 and p.usuario_id != u.id and (
        p.publico or amistad_aceptada(db, u.id, p.usuario_id))
    if not visible:
        raise error(404, "no_encontrado", "No existe ese perfil")
    if datos.motivo == "otro" and not (datos.detalle or "").strip():
        raise error(422, "detalle_requerido", "Contá qué pasa en el detalle")
    d = Denuncia(denunciante_id=u.id, perfil_id=p.id, motivo=datos.motivo, detalle=datos.detalle, estado="abierta")
    db.add(d)
    db.flush()
    db.add(NotificacionAdmin(tipo="denuncia_nueva", payload={"denuncia_id": d.id, "motivo": d.motivo}))
    db.commit()
    return {"id": d.id, "estado": d.estado}
