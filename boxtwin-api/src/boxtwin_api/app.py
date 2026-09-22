"""
BoxTwin API - El servidor.

POR QUE EXISTE
  El flujo del producto tiene una espera larga en el medio y una pregunta humana adentro de
  esa espera. Eso no entra en un request: subir devuelve un identificador (RF1), el estado
  se consulta, y cuando el sistema llega al punto de siembra el usuario contesta y el
  trabajo sigue.

  Todo lo que sirve un archivo filtra por usuario (RNF3). No hay una capa de permisos
  porque no hace falta una: hay un dueno por sesion y se compara en cada ruta. Lo que si
  hay es una sola funcion que resuelve la sesion del usuario, para que no exista una ruta
  que se olvide de comparar.

QUE HACE
  Registro y login, subida de video, estado, siembra, Fight-Card, correccion de tipo,
  export y streaming del video con rangos para que la linea de tiempo pueda saltar.

USO
  uvicorn boxtwin_api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hmac
import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwin_api.cola import encolar
from boxtwin_api.config import cfg
from boxtwin_api.db import crear_tablas, obtener_sesion
from boxtwin_api.esquemas import Credenciales, EntradaCorreccion, Siembra, Token, sesion_a_dict
from boxtwin_api.exportar import a_csv, a_html
from boxtwin_api.modelos import Correccion, Sesion, Trabajo, Usuario
from boxtwin_api.seguridad import (
    emitir_ticket,
    emitir_token,
    hashear,
    leer_ticket,
    leer_token,
    verificar,
)

__all__ = ["app"]


@asynccontextmanager
async def _ciclo(_app: FastAPI):
    crear_tablas()
    cfg.datos.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    lifespan=_ciclo,
    title="BoxTwin",
    version="0.1",
    docs_url="/docs" if cfg.docs else None,
    redoc_url="/redoc" if cfg.docs else None,
    openapi_url="/openapi.json" if cfg.docs else None,
    description=(
        "Analisis tactico de sparring filmado. Reporta golpes DETECTADOS, con la precision "
        "y el recall medidos del detector. No reporta conexion, puntuacion ni veredicto: "
        "un sistema monocular no establece contacto fisico."
    ),
)

# El frontend corre en otro origen durante el desarrollo (vite en 5173) y detras del mismo
# tunel en produccion. Se listan los origenes en vez de abrir a todos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Autenticacion
# ---------------------------------------------------------------------------


def usuario_actual(
    authorization: str = Header(default=""), db: Session = Depends(obtener_sesion)
) -> Usuario:
    token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    uid = leer_token(token, cfg.secreto) if token else None
    if not uid:
        raise HTTPException(401, "hace falta iniciar sesion")
    u = db.get(Usuario, uid)
    if u is None:
        raise HTTPException(401, "la cuenta ya no existe")
    return u


def sesion_del_usuario(sesion_id: str, usuario: Usuario, db: Session) -> Sesion:
    """
    La sesion, si es de este usuario. RNF3 pasa por aca y por ningun otro lado.

    Devuelve 404 y no 403 cuando es de otro: contestar "existe pero no es tuya" le dice a
    cualquiera que adivine un id que ese id existe.
    """
    s = db.get(Sesion, sesion_id)
    if s is None or s.usuario_id != usuario.id:
        raise HTTPException(404, "no hay una sesion con ese id")
    return s


def _puede_registrarse(codigo: str) -> None:
    """
    Quien se puede crear una cuenta.

    Cerrado por defecto y a proposito: apenas esto sale por un tunel, una instancia con el
    registro abierto es una GPU ajena gratis para cualquiera que tenga la URL, y la GPU es
    una sola y esta abajo del escritorio. Abrirlo tiene que ser una decision explicita.

    La comparacion va en tiempo constante porque el codigo es un secreto compartido y
    compararlo con == filtra por cuanto tarda en fallar.
    """
    modo = cfg.modo_registro
    if modo == "abierto":
        return
    if modo == "cerrado":
        raise HTTPException(
            403,
            "el registro esta cerrado en esta instancia. Para abrirlo, BOXTWIN_INVITACION "
            "con un codigo, o con 'abierto' si no hace falta ninguno",
        )
    if not hmac.compare_digest(codigo or "", cfg.invitacion):
        raise HTTPException(403, "el codigo de invitacion no es correcto")


@app.post("/auth/registro", response_model=Token)
def registro(c: Credenciales, db: Session = Depends(obtener_sesion)) -> Token:
    _puede_registrarse(c.invitacion)
    email = c.email.lower().strip()
    if db.scalar(select(Usuario).where(Usuario.email == email)):
        raise HTTPException(409, "ya hay una cuenta con ese email")
    u = Usuario(email=email, hash_clave=hashear(c.clave))
    db.add(u)
    db.commit()
    return Token(token=emitir_token(u.id, cfg.secreto, cfg.horas_de_sesion), usuario=u.email)


@app.post("/auth/login", response_model=Token)
def login(c: Credenciales, db: Session = Depends(obtener_sesion)) -> Token:
    u = db.scalar(select(Usuario).where(Usuario.email == c.email.lower().strip()))
    # El mismo mensaje para email inexistente y clave equivocada: distinguirlos le dice a
    # cualquiera cuales emails tienen cuenta.
    if u is None or not verificar(c.clave, u.hash_clave):
        raise HTTPException(401, "email o clave incorrectos")
    return Token(token=emitir_token(u.id, cfg.secreto, cfg.horas_de_sesion), usuario=u.email)


# ---------------------------------------------------------------------------
# Subida y estado
# ---------------------------------------------------------------------------


@app.post("/videos", status_code=202)
def subir(
    archivo: UploadFile = File(...),
    nombre: str = Form(default=""),
    round_s: float | None = Form(default=None),
    descanso_s: float = Form(default=60.0),
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    RF1: encola y devuelve el identificador sin esperar el procesamiento.

    El archivo se escribe en disco a medida que llega y no en memoria: 30 minutos de 1080p
    desde un celular son varios gigabytes y un servidor de un nodo no los aguanta en RAM.
    """
    ext = Path(archivo.filename or "").suffix.lower()
    if ext not in cfg.formatos:
        raise HTTPException(
            415, f"formato no soportado: {ext or 'sin extension'}. "
                 f"Se aceptan {', '.join(cfg.formatos)}"
        )
    if round_s is not None and round_s <= 0:
        raise HTTPException(422, "la duracion del round tiene que ser mayor que cero")

    ses = Sesion(
        usuario_id=usuario.id,
        nombre=(nombre or Path(archivo.filename or "sesion").stem)[:255],
        video_nombre=_nombre_seguro(archivo.filename or "video.mp4"),
        round_s=round_s,
        descanso_s=descanso_s,
        estado="en_cola",
    )
    db.add(ses)
    db.flush()

    destino = cfg.dir_sesion(ses.id) / "videos" / ses.video_nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritos = 0
    try:
        with destino.open("wb") as f:
            while chunk := archivo.file.read(1024 * 1024):
                escritos += len(chunk)
                if escritos > cfg.max_bytes:
                    raise HTTPException(413, "el video supera el tamano maximo")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(cfg.dir_sesion(ses.id), ignore_errors=True)
        db.rollback()
        raise

    # La sesion y su primer trabajo, en el mismo commit: no existe el estado intermedio.
    encolar(db, ses.id, "procesar", round_s=round_s, descanso_s=descanso_s)
    db.commit()
    return {"job_id": ses.id, "estado": ses.estado, "bytes": escritos}


def _nombre_seguro(nombre: str) -> str:
    """El nombre que subio el usuario no elige rutas del servidor."""
    base = Path(nombre).name
    limpio = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return limpio[:120] or "video.mp4"


@app.get("/jobs/{sesion_id}")
def estado(
    sesion_id: str,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    El estado, y cuando corresponde los candidatos de siembra con sus recortes (RF3).

    Lo que se lee del directorio se lee del archivo que escribe el worker, que es la unica
    fuente de verdad de lo que paso adentro del proceso. La base guarda el estado para
    poder listarlo sin tocar disco, pero cuando los dos difieren manda el archivo.
    """
    s = sesion_del_usuario(sesion_id, usuario, db)
    d = _sesion_en_disco(s.id)
    extra: dict = {"candidatos": [], "avisos": [], "etapas": []}
    if d:
        extra["candidatos"] = [
            {**c, "recorte_url": (f"/jobs/{s.id}/candidatos/{c['track']}"
                                  if c.get("recorte") else None)}
            for c in d.get("candidatos", [])
        ]
        extra["avisos"] = d.get("avisos", [])
        extra["etapas"] = d.get("etapas", [])
        extra["pareja_sugerida"] = (d.get("identidad") or {}).get("pareja_sugerida")
        if d.get("estado") and d["estado"] != s.estado and s.estado != "fallo":
            s.estado = d["estado"]
            db.commit()
    trabajos = db.scalars(
        select(Trabajo).where(Trabajo.sesion_id == s.id).order_by(Trabajo.creado)
    ).all()
    extra["trabajos"] = [
        {"etapa": t.etapa, "estado": t.estado, "intento": t.intento, "error": t.error}
        for t in trabajos
    ]
    return sesion_a_dict(s, extra)


@app.get("/jobs")
def listar(
    usuario: Usuario = Depends(usuario_actual), db: Session = Depends(obtener_sesion)
) -> list[dict]:
    """El historial del usuario (F8). Solo el suyo."""
    filas = db.scalars(
        select(Sesion).where(Sesion.usuario_id == usuario.id).order_by(Sesion.creada.desc())
    ).all()
    return [sesion_a_dict(s) for s in filas]


@app.get("/jobs/{sesion_id}/candidatos/{track}")
def recorte(
    track: int,
    sesion_id: str,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> FileResponse:
    s = sesion_del_usuario(sesion_id, usuario, db)
    ruta = cfg.dir_sesion(s.id) / "candidatos" / f"track_{int(track)}.jpg"
    if not ruta.is_file():
        raise HTTPException(404, "no hay recorte para ese track")
    return FileResponse(ruta, media_type="image/jpeg")


@app.post("/jobs/{sesion_id}/siembra", status_code=202)
def siembra(
    sesion_id: str,
    s_in: Siembra,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    RF4: el usuario contesta cual es cual y el trabajo sigue.

    Se acepta tambien sobre una sesion ya lista, que es re-sembrar: si el entrenador ve que
    el sistema le puso los nombres al reves, cambiar la respuesta y volver a correr la
    segunda etapa cuesta minutos y no requiere volver a subir el video.
    """
    s = sesion_del_usuario(sesion_id, usuario, db)
    if s.estado not in ("espera_siembra", "listo", "fallo"):
        raise HTTPException(
            409, f"la sesion esta en '{s.estado}' y todavia no hay candidatos que elegir"
        )
    disco = _sesion_en_disco(s.id) or {}
    tracks = {c["track"] for c in disco.get("candidatos", [])}
    # Se permite un track que no este entre los propuestos -el usuario puede haber visto al
    # peleador en un track que los filtros descartaron- pero no uno que no existe.
    if tracks and not {s_in.track_a, s_in.track_b} <= tracks:
        ajenos = sorted({s_in.track_a, s_in.track_b} - tracks)
        if any(t < 0 for t in ajenos):
            raise HTTPException(422, f"track invalido: {ajenos}")

    s.semilla_a, s.semilla_b = s_in.track_a, s_in.track_b
    s.estado = "en_cola"
    s.error = None
    encolar(db, s.id, "completar", track_a=s_in.track_a, track_b=s_in.track_b)
    db.commit()
    return {"job_id": s.id, "estado": s.estado}


# ---------------------------------------------------------------------------
# Fight-Card
# ---------------------------------------------------------------------------


def _sesion_en_disco(sesion_id: str) -> dict | None:
    ruta = cfg.dir_sesion(sesion_id) / "sesion.json"
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text())
    except json.JSONDecodeError:  # pragma: no cover - el worker escribe atomico
        return None


def _fightcard(sesion_id: str) -> dict:
    ruta = cfg.dir_sesion(sesion_id) / "fightcard.json"
    if not ruta.is_file():
        raise HTTPException(409, "la Fight-Card todavia no esta lista")
    return json.loads(ruta.read_text())


@app.get("/fightcards/{sesion_id}")
def fightcard(
    sesion_id: str,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> JSONResponse:
    s = sesion_del_usuario(sesion_id, usuario, db)
    return JSONResponse(_fightcard(s.id))


@app.get("/fightcards/{sesion_id}/export")
def exportar(
    sesion_id: str,
    formato: str = "csv",
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
):
    s = sesion_del_usuario(sesion_id, usuario, db)
    fc = _fightcard(s.id)
    if formato == "csv":
        return Response(
            a_csv(fc),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{s.id}.csv"'},
        )
    if formato in ("html", "pdf"):
        # pdf devuelve la misma pagina: se imprime desde el navegador. Esta dicho en la
        # pagina y en el README, no es un fallback silencioso.
        return HTMLResponse(a_html(fc))
    raise HTTPException(422, "formato tiene que ser csv o html")


@app.patch("/fightcards/{sesion_id}/golpes/{golpe}")
def corregir(
    sesion_id: str,
    golpe: str,
    entrada: EntradaCorreccion,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    F14 y RF10: el entrenador corrige el tipo y la correccion se guarda como etiqueta.

    La ruta esta anidada bajo la sesion y no es `/golpes/{id}` como propone el spec, porque
    el id de un golpe sale de su carril y su cuadro y solo es unico adentro de su sesion.
    Un id global pediria un contador, y un contador se rompe al reclasificar.
    """
    s = sesion_del_usuario(sesion_id, usuario, db)
    try:
        from boxtwin.mvp.orquesta import registrar_correccion
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise HTTPException(
            501, "esta instancia no tiene instalado boxtwin-annotator, que es donde vive "
                 "el contrato de la Fight-Card"
        ) from e

    try:
        fc = registrar_correccion(cfg.dir_sesion(s.id), golpe, entrada.tipo, por=usuario.email)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    ev = next(
        (g for p in fc["peleadores"].values() for g in p["golpes"] if g["id"] == golpe), None
    )
    c = (ev or {}).get("corregido") or {}
    db.add(
        Correccion(
            sesion_id=s.id, golpe=golpe, tipo=entrada.tipo,
            tipo_original=c.get("tipo_original"),
            checkpoint=fc.get("clasificador", {}).get("checkpoint"),
            datos={"t_inicio": (ev or {}).get("t_inicio"), "por": usuario.email},
        )
    )
    db.commit()
    return ev or {}


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

_TROZO = 1024 * 1024


@app.get("/videos/{sesion_id}/ticket")
def ticket(
    sesion_id: str,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """Un permiso corto para reproducir ESTE video, que es lo que el <video> puede llevar."""
    s = sesion_del_usuario(sesion_id, usuario, db)
    return {"ticket": emitir_ticket(usuario.id, s.id, cfg.secreto), "minutos": 30}


@app.get("/videos/{sesion_id}/stream")
def stream(
    sesion_id: str,
    t: str = "",
    range: str = Header(default=""),
    authorization: str = Header(default=""),
    db: Session = Depends(obtener_sesion),
):
    """
    El video con soporte de rangos, que es lo que hace posible saltar a un evento.

    Sin 206, el navegador tiene que bajar el archivo entero antes de poder posicionarse en
    el minuto 12, y la linea de tiempo -que es la funcion que convierte un numero dudoso en
    una herramienta de revision- deja de servir.
    """
    # Dos formas de entrar: el header, para fetch, y el ticket, para el <video>, que no
    # puede mandar headers. El ticket vale para esta sesion y para ninguna otra.
    uid = leer_ticket(t, sesion_id, cfg.secreto) if t else None
    if uid is None:
        token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        uid = leer_token(token, cfg.secreto) if token else None
    if uid is None:
        raise HTTPException(401, "hace falta iniciar sesion")
    u = db.get(Usuario, uid)
    if u is None:
        raise HTTPException(401, "la cuenta ya no existe")

    s = sesion_del_usuario(sesion_id, u, db)
    ruta = cfg.dir_sesion(s.id) / "videos" / s.video_nombre
    if not ruta.is_file():
        raise HTTPException(404, "no esta el video de esta sesion")
    total = ruta.stat().st_size

    inicio, fin = 0, total - 1
    m = re.match(r"bytes=(\d*)-(\d*)", range or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            inicio = int(m.group(1))
            if m.group(2):
                fin = min(int(m.group(2)), total - 1)
        else:
            inicio = max(total - int(m.group(2)), 0)
        if inicio >= total:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})

    def leer():
        with ruta.open("rb") as f:
            f.seek(inicio)
            restan = fin - inicio + 1
            while restan > 0:
                datos = f.read(min(_TROZO, restan))
                if not datos:
                    break
                restan -= len(datos)
                yield datos

    cabeceras = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(fin - inicio + 1),
        "Content-Range": f"bytes {inicio}-{fin}/{total}",
    }
    return StreamingResponse(
        leer(), status_code=206 if m else 200, media_type="video/mp4", headers=cabeceras
    )


@app.get("/salud")
def salud() -> dict:
    return {
        "ok": True,
        "version": app.version,
        # Las dos cosas que hay que poder mirar desde afuera antes de abrir el tunel.
        "secreto_efimero": cfg.secreto_efimero,
        "registro": cfg.modo_registro,
        "docs": cfg.docs,
        "frontend": bool(cfg.web and (cfg.web / "index.html").is_file()),
    }


# ---------------------------------------------------------------------------
# El frontend, servido por la misma API
# ---------------------------------------------------------------------------
#
# UN SOLO ORIGEN. Con el frontend en otro puerto hacen falta dos tuneles y CORS en
# produccion; sirviendolo desde aca alcanza con exponer este puerto y nada mas.
#
# Va ULTIMO en el archivo y no es cosmetico: Starlette prueba las rutas en orden de
# registro y este mount matchea todo, asi que cualquier ruta declarada despues quedaria
# tapada por el index.html.
if cfg.web and (cfg.web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(cfg.web), html=True), name="web")
