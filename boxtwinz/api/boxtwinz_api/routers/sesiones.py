"""
BoxTwinZ API - Sesiones: subida, siembra, resultado, video, valoracion y correcciones (spec 5).

POR QUE EXISTE
  Es el circuito que no se recorta (spec 13): subir -> procesar -> resultado -> valoracion.
  La duracion del video se mide con ffprobe ANTES de encolar, porque el cupo se descuenta al
  encolar (spec 9) y la duracion declarada por la web no es evidencia.

USO
  app.include_router(router, prefix="/api")
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api import textos
from boxtwinz_api.cola import encolar
from boxtwinz_api.config import cfg
from boxtwinz_api.db import obtener_sesion
from boxtwinz_api.deps import error, usuario_actual
from boxtwinz_api.models import (
    MODALIDADES, Correccion, ExposicionSesion, Golpe, MetricaSesion, Perfil, Sesion, Usuario,
    Valoracion,
)
from boxtwinz_api.services.borrado import borrar_sesiones
from boxtwinz_api.services.engine_adapter import leer_sesion_json
from boxtwinz_api.services.profiles import perfil_propio
from boxtwinz_api.services.quota import cupo, descontar, devolver
from boxtwinz_api.services.rules import Entrada, consejos

router = APIRouter(tags=["sesiones"])


def medir_video(ruta: Path) -> dict | None:
    """Duracion, fps y resolucion con ffprobe. None si no es un video legible."""
    try:
        r = subprocess.run(  # noqa: S603,S607 - argumentos fijos, la ruta la arma el servidor
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(ruta)],
            capture_output=True, text=True, timeout=60,
        )
        d = json.loads(r.stdout or "{}")
        st = (d.get("streams") or [{}])[0]
        num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else None
        dur = float((d.get("format") or {}).get("duration"))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    if dur <= 0:
        return None
    return {"duracion_s": dur, "fps": fps, "ancho": st.get("width"), "alto": st.get("height")}


def sesion_de(db: Session, sesion_id: str, u: Usuario) -> Sesion:
    s = db.get(Sesion, sesion_id)
    if s is None or s.usuario_id != u.id:
        raise error(404, "no_encontrada", "No existe esa sesión")
    return s


def _cajon_valido(db: Session, u: Usuario, perfil_id: str | None) -> Perfil | None:
    """El cajon del rival: solo premium, solo customizado, solo propio. Si no, se ignora."""
    if not perfil_id or u.plan != "premium":
        return None
    p = db.get(Perfil, perfil_id)
    if p is None or p.usuario_id != u.id or p.ranura == 1 or not p.customizado:
        return None
    return p


def sesion_dict(s: Sesion) -> dict:
    return {
        "id": s.id, "perfil_id": s.perfil_id, "modalidad": s.modalidad, "estado": s.estado,
        "error_etapa": s.error_etapa, "error": s.error if s.estado == "error" else None,
        "rival_perfil_id": s.rival_perfil_id, "rounds_declarados": s.rounds_declarados,
        "duracion_s": s.video_duracion_s, "fps": s.video_fps,
        "resolucion": [s.video_ancho, s.video_alto] if s.video_ancho else None,
        "calidad_pose": s.calidad_pose, "apta_ranking": s.apta_ranking,
        "creado": s.creado.isoformat(), "vista_resultado": s.vista_resultado,
        "rotulo": textos.EN_EVALUACION if s.modalidad in ("sombra", "bolsa") else None,
    }


@router.post("/sesiones", status_code=202)
def subir(
    video: UploadFile = File(...),
    modalidad: str = Form(...),
    perfil_id: str = Form(...),
    rounds_declarados: str = Form(default=""),
    rival_perfil_id: str = Form(default=""),
    u: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    if modalidad not in MODALIDADES:
        raise error(422, "modalidad_invalida", "La modalidad es sombra, bolsa o sparring")
    p = db.get(Perfil, perfil_id)
    if p is None or p.usuario_id != u.id:
        raise error(404, "no_encontrado", "No existe ese perfil")
    if p.ranura != 1:
        # Las sesiones se suben con el perfil propio; un cajon premium recibe sparrings como
        # rival en la siembra (desvio D7). En free los cajones son de solo lectura.
        if u.plan != "premium":
            raise error(403, "plan_requerido", "En el plan free las sesiones son del perfil propio")
        raise error(422, "perfil_invalido", "Subí la sesión con tu perfil y elegí el cajón como rival")
    rounds = None
    if rounds_declarados.strip():
        try:
            rounds = int(rounds_declarados)
        except ValueError:
            raise error(422, "rounds_invalido", "Los rounds tienen que ser un número entero") from None
        if not 1 <= rounds <= 30:
            raise error(422, "rounds_invalido", "Los rounds van de 1 a 30")
    ext = Path(video.filename or "").suffix.lower()
    if ext not in cfg.formatos:
        raise error(415, "formato_no_soportado", f"Se aceptan {', '.join(cfg.formatos)}")

    s = Sesion(usuario_id=u.id, perfil_id=p.id, modalidad=modalidad, video_ruta="",
               rounds_declarados=rounds, estado="subida")
    cajon = _cajon_valido(db, u, rival_perfil_id) if modalidad == "sparring" else None
    s.rival_perfil_id = cajon.id if cajon else None
    db.add(s)
    db.flush()
    destino = cfg.dir_videos() / f"{s.id}{ext}"
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritos = 0
    try:
        with destino.open("wb") as f:
            while chunk := video.file.read(1024 * 1024):
                escritos += len(chunk)
                if escritos > cfg.max_bytes:
                    raise error(413, "video_grande", "El video supera el tamaño máximo")
                f.write(chunk)
        meta = medir_video(destino)
        if meta is None:
            raise error(422, "video_ilegible", "No se pudo leer el video")
        if meta["duracion_s"] > cfg.max_minutos * 60:
            raise error(422, "video_largo", f"El video dura más de {cfg.max_minutos:g} minutos")
        c = cupo(db, u)
        if meta["duracion_s"] / 60 > c["restantes_min"]:
            raise error(402, "cupo_agotado",
                        f"Este video usa {meta['duracion_s'] / 60:.1f} min y te quedan "
                        f"{c['restantes_min']:.1f} min este mes")
    except Exception:
        destino.unlink(missing_ok=True)
        db.rollback()
        raise
    s.video_ruta = str(destino)
    s.video_duracion_s, s.video_fps = meta["duracion_s"], meta["fps"]
    s.video_ancho, s.video_alto = meta["ancho"], meta["alto"]
    descontar(db, u, s, meta["duracion_s"])
    encolar(db, s.id, "procesar")
    db.commit()
    return {**sesion_dict(s), "bytes": escritos, "cupo": cupo(db, u)}


@router.get("/sesiones")
def listar(perfil_id: str | None = None, modalidad: str | None = None,
           u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> list[dict]:
    q = select(Sesion).where(Sesion.usuario_id == u.id)
    if perfil_id:
        q = q.where((Sesion.perfil_id == perfil_id) | (Sesion.rival_perfil_id == perfil_id))
    if modalidad:
        q = q.where(Sesion.modalidad == modalidad)
    return [sesion_dict(s) for s in db.scalars(q.order_by(Sesion.creado.desc()))]


@router.get("/sesiones/{sesion_id}")
def ver(sesion_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return sesion_dict(sesion_de(db, sesion_id, u))


@router.get("/sesiones/{sesion_id}/candidatos")
def candidatos(sesion_id: str, u: Usuario = Depends(usuario_actual),
               db: Session = Depends(obtener_sesion)) -> dict:
    s = sesion_de(db, sesion_id, u)
    if s.modalidad != "sparring" or s.estado != "esperando_siembra":
        raise error(409, "sin_candidatos", "La sesión no está esperando la siembra")
    disco = leer_sesion_json(cfg.dir_trabajo(s.id))
    lista = [{
        "track": c["track"], "sugerido": bool(c.get("sugerido")),
        "color_guante": (c.get("color_guante") or {}).get("nombre"),
        "t_desde": c.get("t_desde"), "t_hasta": c.get("t_hasta"),
        "recorte_url": f"/api/sesiones/{s.id}/candidatos/{int(c['track'])}.jpg",
    } for c in disco.get("candidatos", [])]
    cajones = []
    if u.plan == "premium":
        cajones = [{"id": p.id, "ranura": p.ranura, "nombre": p.nombre, "estilo": p.estilo}
                   for p in db.scalars(select(Perfil).where(Perfil.usuario_id == u.id, Perfil.ranura > 1,
                                                            Perfil.customizado.is_(True)))]
    return {"candidatos": lista, "cajones": cajones}


@router.get("/sesiones/{sesion_id}/candidatos/{track}.jpg")
def recorte(sesion_id: str, track: int, u: Usuario = Depends(usuario_actual),
            db: Session = Depends(obtener_sesion)) -> FileResponse:
    s = sesion_de(db, sesion_id, u)
    ruta = cfg.dir_trabajo(s.id) / "candidatos" / f"track_{int(track)}.jpg"
    if not ruta.is_file():
        raise error(404, "no_encontrado", "No hay recorte para ese candidato")
    return FileResponse(ruta, media_type="image/jpeg")


class Siembra(BaseModel):
    usuario: int = Field(ge=0)
    rival: int = Field(ge=0)
    rival_cajon_perfil_id: str | None = None


@router.post("/sesiones/{sesion_id}/siembra", status_code=202)
def sembrar(sesion_id: str, datos: Siembra, u: Usuario = Depends(usuario_actual),
            db: Session = Depends(obtener_sesion)) -> dict:
    s = sesion_de(db, sesion_id, u)
    if s.modalidad != "sparring":
        raise error(422, "sin_siembra", "En sombra y bolsa no hay siembra")
    if s.estado not in ("esperando_siembra", "lista", "error"):
        raise error(409, "sin_candidatos", f"La sesión está en '{s.estado}'")
    if datos.usuario == datos.rival:
        raise error(422, "siembra_invalida", "Elegí dos personas distintas")
    tracks = {c["track"] for c in leer_sesion_json(cfg.dir_trabajo(s.id)).get("candidatos", [])}
    if tracks and not {datos.usuario, datos.rival} <= tracks:
        # A deja elegir un track que el filtro descarto; aca la pantalla solo muestra
        # candidatos, asi que uno que no esta es un pedido armado a mano.
        raise error(422, "siembra_invalida", "Esa persona no está entre los candidatos")
    cajon = _cajon_valido(db, u, datos.rival_cajon_perfil_id)
    if cajon:
        s.rival_perfil_id = cajon.id
    s.semilla_usuario, s.semilla_rival = datos.usuario, datos.rival
    s.estado = "completando"
    encolar(db, s.id, "completar", track_a=datos.usuario, track_b=datos.rival)
    db.commit()
    return sesion_dict(s)


def _metricas(db: Session, s: Sesion) -> dict:
    out: dict = {}
    for m in db.scalars(select(MetricaSesion).where(MetricaSesion.sesion_id == s.id)
                        .order_by(MetricaSesion.quien, MetricaSesion.segmento)):
        d = {k: getattr(m, k) for k in ("golpes", "volumen", "pct_izquierda", "pct_mano_adelantada",
                                        "combinaciones_min", "pct_golpes_en_combinacion",
                                        "sosten_ritmo", "iniciativa")}
        bloque = out.setdefault(m.quien, {"total": None, "segmentos": []})
        if m.segmento is None:
            bloque["total"] = {**d, "curva": m.curva}
        else:
            bloque["segmentos"].append({**d, "segmento": m.segmento, "inicio_s": m.curva[0], "fin_s": m.curva[1]})
    return out


def resultado_dict(db: Session, s: Sesion) -> dict:
    golpes = list(db.scalars(select(Golpe).where(Golpe.sesion_id == s.id).order_by(Golpe.t_inicio)))
    metricas = _metricas(db, s)
    expo = db.scalar(select(ExposicionSesion).where(ExposicionSesion.sesion_id == s.id))
    p = db.get(Perfil, s.perfil_id)
    cajon = db.get(Perfil, s.rival_perfil_id) if s.rival_perfil_id else None
    tot = (metricas.get("usuario") or {}).get("total") or {}
    entrada = Entrada(
        sesion_id=s.id, golpes=tot.get("golpes") or 0,
        pct_mano_adelantada=tot.get("pct_mano_adelantada"), sosten_ritmo=tot.get("sosten_ritmo"),
        pct_golpes_en_combinacion=tot.get("pct_golpes_en_combinacion"),
        guardia=p.guardia if p else "ortodoxa",
        lado_izquierdo=expo.lado_izquierdo_usuario if expo else 0,
        lado_derecho=expo.lado_derecho_usuario if expo else 0,
        estilo_rival=cajon.estilo if cajon else None,
    )
    alturas = [g.altura_estimada for g in golpes if g.altura_estimada]
    valorada = db.scalar(select(Valoracion).where(Valoracion.sesion_id == s.id)) is not None
    return {
        "sesion": sesion_dict(s),
        "golpes": [{"id": g.id, "quien": g.quien, "mano": g.mano, "t_inicio": g.t_inicio, "t_fin": g.t_fin,
                    "confianza": g.confianza, "altura_estimada": g.altura_estimada} for g in golpes],
        "metricas": metricas,
        "exposicion": ({"rival_izquierda": expo.rival_izquierda, "rival_derecha": expo.rival_derecha,
                        "lado_izquierdo_usuario": expo.lado_izquierdo_usuario,
                        "lado_derecho_usuario": expo.lado_derecho_usuario,
                        "aclaracion": "Golpes dirigidos del rival por lado; no dice cuáles llegaron."}
                       if expo else None),
        "altura": ({"rotulo": "estimación", "cabeza": alturas.count("cabeza"),
                    "cuerpo": alturas.count("cuerpo")} if alturas else None),
        "consejos": consejos(entrada),
        "valorada": valorada,
        "valoracion_obligatoria": not valorada,
        "rotulos": {
            "golpes": "golpes detectados",
            "evaluacion": textos.EN_EVALUACION if s.modalidad in ("sombra", "bolsa") else None,
            "rounds": textos.ROUNDS_SIN_VALIDAR,
            "guardia_rival": textos.GUARDIA_SUPUESTA if s.modalidad == "sparring" else None,
        },
    }


@router.get("/sesiones/{sesion_id}/resultado")
def resultado(sesion_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    s = sesion_de(db, sesion_id, u)
    if s.estado != "lista":
        raise error(409, "no_lista", f"La sesión está en '{s.estado}'")
    return resultado_dict(db, s)


_RANGO = re.compile(r"bytes=(\d*)-(\d*)")


@router.get("/sesiones/{sesion_id}/video")
def video(sesion_id: str, request: Request, u: Usuario = Depends(usuario_actual),
          db: Session = Depends(obtener_sesion)) -> Response:
    s = sesion_de(db, sesion_id, u)
    ruta = Path(s.video_ruta)
    if not ruta.is_file():
        raise error(404, "no_encontrado", "El video ya no está")
    tam = ruta.stat().st_size
    tipo = "video/quicktime" if ruta.suffix == ".mov" else "video/mp4"
    m = _RANGO.fullmatch(request.headers.get("range", "").strip())
    if not m:
        return FileResponse(ruta, media_type=tipo, headers={"Accept-Ranges": "bytes"})
    a, b = m.groups()
    if a == "":
        inicio, fin = max(0, tam - int(b or 0)), tam - 1
    else:
        inicio, fin = int(a), min(int(b) if b else tam - 1, tam - 1)
    if inicio >= tam or inicio > fin:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{tam}"})

    def trozos():
        with ruta.open("rb") as f:
            f.seek(inicio)
            falta = fin - inicio + 1
            while falta > 0:
                buf = f.read(min(1024 * 1024, falta))
                if not buf:
                    break
                falta -= len(buf)
                yield buf

    return StreamingResponse(trozos(), status_code=206, media_type=tipo, headers={
        "Content-Range": f"bytes {inicio}-{fin}/{tam}", "Accept-Ranges": "bytes",
        "Content-Length": str(fin - inicio + 1)})


class PedidoValoracion(BaseModel):
    estrellas: int = Field(ge=1, le=5)
    comentario: str | None = Field(default=None, max_length=2000)


@router.post("/sesiones/{sesion_id}/valoracion", status_code=201)
def valorar(sesion_id: str, datos: PedidoValoracion, u: Usuario = Depends(usuario_actual),
            db: Session = Depends(obtener_sesion)) -> dict:
    s = sesion_de(db, sesion_id, u)
    if s.estado != "lista":
        raise error(409, "no_lista", "Se valora una sesión lista")
    v = db.scalar(select(Valoracion).where(Valoracion.sesion_id == s.id))
    if v is None:
        v = Valoracion(sesion_id=s.id, modalidad=s.modalidad, estrellas=datos.estrellas)
        db.add(v)
    v.estrellas, v.comentario = datos.estrellas, (datos.comentario or None)
    s.vista_resultado = True
    db.commit()
    return {"estrellas": v.estrellas, "comentario": v.comentario}


class PedidoCorreccion(BaseModel):
    tipo: Literal["no_fue_golpe", "falto_golpe", "otra_mano"]
    golpe_id: str | None = None
    t: float | None = Field(default=None, ge=0)


@router.post("/sesiones/{sesion_id}/correcciones", status_code=201)
def corregir(sesion_id: str, datos: PedidoCorreccion, u: Usuario = Depends(usuario_actual),
             db: Session = Depends(obtener_sesion)) -> dict:
    """
    Se registra para reentrenar; no cambia los golpes ni las metricas mostradas. Las metricas
    salen del detector y corregirlas a mano mezclaria dos fuentes en un mismo numero.
    """
    s = sesion_de(db, sesion_id, u)
    if datos.tipo == "falto_golpe":
        if datos.t is None:
            raise error(422, "correccion_invalida", "Falta el instante del golpe que faltó")
        golpe_id = None
    else:
        g = db.get(Golpe, datos.golpe_id) if datos.golpe_id else None
        if g is None or g.sesion_id != s.id:
            raise error(422, "correccion_invalida", "Falta el golpe a corregir")
        golpe_id = g.id
    c = Correccion(sesion_id=s.id, golpe_id=golpe_id, tipo=datos.tipo, t=datos.t)
    db.add(c)
    db.commit()
    return {"id": c.id, "tipo": c.tipo, "golpe_id": c.golpe_id, "t": c.t}


@router.delete("/sesiones/{sesion_id}", status_code=204)
def borrar(sesion_id: str, u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> Response:
    s = sesion_de(db, sesion_id, u)
    if s.estado in ("subida", "procesando", "completando"):
        # Hay un trabajo en cola o corriendo; borrar ahora deja al worker sin sesion. El cupo
        # se devuelve porque no se proceso.
        devolver(db, s)
    Path(s.video_ruta).unlink(missing_ok=True)
    shutil.rmtree(cfg.dir_trabajo(s.id), ignore_errors=True)
    borrar_sesiones(db, [s.id])
    db.commit()
    return Response(status_code=204)


@router.get("/comparar")
def comparar(a: str = Query(...), b: str = Query(...), u: Usuario = Depends(usuario_actual),
             db: Session = Depends(obtener_sesion)) -> dict:
    sa, sb = sesion_de(db, a, u), sesion_de(db, b, u)
    if sa.modalidad != sb.modalidad:
        raise error(422, "modalidades_distintas", "Se comparan sesiones de la misma modalidad")
    if sa.estado != "lista" or sb.estado != "lista":
        raise error(409, "no_lista", "Las dos sesiones tienen que estar listas")
    return {"a": {"sesion": sesion_dict(sa), "metricas": _metricas(db, sa)},
            "b": {"sesion": sesion_dict(sb), "metricas": _metricas(db, sb)},
            "rotulo_rounds": textos.ROUNDS_SIN_VALIDAR}


@router.get("/cupo")
def ver_cupo(u: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)) -> dict:
    return cupo(db, u)


__all__ = ["router", "medir_video", "resultado_dict", "sesion_de", "perfil_propio"]
