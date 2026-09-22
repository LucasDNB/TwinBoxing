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
from boxtwin_api.esquemas import (
    AsignarBoxeadores,
    Credenciales,
    EntradaBoxeador,
    EntradaCorreccion,
    Siembra,
    Token,
    sesion_a_dict,
)
from boxtwin_api.exportar import a_csv, a_html
from boxtwin_api.modelos import Boxeador, Correccion, Sesion, Trabajo, Usuario
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


def usuario_por_ticket_o_header(
    sesion_id: str, t: str, authorization: str, db: Session
) -> Usuario:
    """
    El usuario, entrando por el header o por un ticket de esta sesion.

    Existe porque hay dos elementos del navegador que NO pueden mandar headers, y los dos
    son centrales en este producto: el <video> de la linea de tiempo y el <img> de los
    recortes de siembra. El ticket es lo unico que pueden llevar, va en la URL, dura media
    hora y sirve para una sola sesion.
    """
    uid = leer_ticket(t, sesion_id, cfg.secreto) if t else None
    if uid is None:
        token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        uid = leer_token(token, cfg.secreto) if token else None
    if uid is None:
        raise HTTPException(401, "hace falta iniciar sesion")
    u = db.get(Usuario, uid)
    if u is None:
        raise HTTPException(401, "la cuenta ya no existe")
    return u


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
    extra: dict = {
        "candidatos": [], "avisos": [], "etapas": [],
        # Lo escribe la etapa que esta corriendo, en su propio archivo, porque sesion.json
        # recien se escribe cuando termina. Sobre 30 minutos de video la pose son 45
        # minutos, y sin esto la pantalla de espera es una rueda quieta.
        "progreso": _progreso_en_disco(s.id),
    }
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
    t: str = "",
    authorization: str = Header(default=""),
    db: Session = Depends(obtener_sesion),
) -> FileResponse:
    """
    El recorte de un candidato. Entra por header o por ticket, igual que el video.

    Un <img src> no manda headers, asi que sin el ticket el navegador pide la imagen sin
    token, recibe 401 y dibuja el icono roto: la pantalla de siembra queda inusable
    justo donde el usuario tiene que mirar dos fotos y elegir.
    """
    usuario = usuario_por_ticket_o_header(sesion_id, t, authorization, db)
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


def _progreso_en_disco(sesion_id: str) -> dict | None:
    ruta = cfg.dir_sesion(sesion_id) / "progreso.json"
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text())
    except json.JSONDecodeError:
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


def _servir_con_rangos(ruta: Path, range: str):
    """
    Sirve un archivo con soporte de rangos, que es lo que hace posible saltar a un evento.

    Sin 206 el navegador tiene que bajar el archivo entero antes de poder posicionarse en el
    minuto 12, y la linea de tiempo -que es la funcion que convierte un numero dudoso en una
    herramienta de revision- deja de servir.

    Vive aparte porque hay dos videos que servir, el original y el procesado, y la logica de
    rangos es la misma: tenerla dos veces es tenerla mal una de las dos.
    """
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


@app.get("/videos/{sesion_id}/stream")
def stream(
    sesion_id: str,
    t: str = "",
    range: str = Header(default=""),
    authorization: str = Header(default=""),
    db: Session = Depends(obtener_sesion),
):
    """El video original de la sesion."""
    u = usuario_por_ticket_o_header(sesion_id, t, authorization, db)
    s = sesion_del_usuario(sesion_id, u, db)
    ruta = cfg.dir_sesion(s.id) / "videos" / s.video_nombre
    if not ruta.is_file():
        raise HTTPException(404, "no esta el video de esta sesion")
    return _servir_con_rangos(ruta, range)


# GET y HEAD: la interfaz pregunta con HEAD si el video ya esta, porque solo le interesa
# la existencia y el archivo pesa. FastAPI NO enruta HEAD solo por declarar GET -devuelve
# 405- asi que hay que pedirlo.
@app.api_route("/videos/{sesion_id}/procesado", methods=["GET", "HEAD"])
def procesado(
    sesion_id: str,
    t: str = "",
    range: str = Header(default=""),
    authorization: str = Header(default=""),
    db: Session = Depends(obtener_sesion),
):
    """
    El video con los dos peleadores marcados y una consola por peleador.

    Es una etapa aparte y la ultima, asi que puede no existir todavia cuando la Fight-Card ya
    esta lista. El 404 es entonces un estado normal y no un error: la interfaz muestra el
    original mientras tanto.
    """
    u = usuario_por_ticket_o_header(sesion_id, t, authorization, db)
    s = sesion_del_usuario(sesion_id, u, db)
    ruta = cfg.dir_sesion(s.id) / "procesado.mp4"
    if not ruta.is_file():
        raise HTTPException(404, "el video procesado todavia no esta")
    return _servir_con_rangos(ruta, range)


# ---------------------------------------------------------------------------
# Boxeadores y perfiles


def _boxeador_del_usuario(bid: str, usuario, db) -> "Boxeador":
    b = db.get(Boxeador, bid)
    # El mismo 404 para inexistente y ajeno: distinguirlos le dice a un desconocido cuales
    # ids existen, igual que en el login.
    if b is None or b.usuario_id != usuario.id:
        raise HTTPException(404, "no existe ese boxeador")
    return b


def _boxeador_a_dict(b) -> dict:
    return {"id": b.id, "nombre": b.nombre, "guardia": b.guardia, "notas": b.notas,
            "creado": b.creado.isoformat()}


@app.post("/boxeadores", status_code=201)
def crear_boxeador(
    entrada: EntradaBoxeador,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """Da de alta un boxeador. El nombre es unico por usuario, no global."""
    nombre = entrada.nombre.strip()
    ya = (
        db.query(Boxeador)
        .filter(Boxeador.usuario_id == usuario.id, Boxeador.nombre == nombre)
        .first()
    )
    if ya is not None:
        raise HTTPException(409, f"ya tenes un boxeador que se llama {nombre}")
    b = Boxeador(usuario_id=usuario.id, nombre=nombre, guardia=entrada.guardia,
                 notas=entrada.notas)
    db.add(b)
    db.commit()
    return _boxeador_a_dict(b)


@app.get("/boxeadores")
def listar_boxeadores(
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> list[dict]:
    bs = (
        db.query(Boxeador)
        .filter(Boxeador.usuario_id == usuario.id)
        .order_by(Boxeador.nombre)
        .all()
    )
    return [_boxeador_a_dict(b) for b in bs]


@app.patch("/boxeadores/{boxeador_id}")
def editar_boxeador(
    boxeador_id: str,
    entrada: EntradaBoxeador,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    b = _boxeador_del_usuario(boxeador_id, usuario, db)
    b.nombre = entrada.nombre.strip()
    b.guardia = entrada.guardia
    b.notas = entrada.notas
    db.commit()
    return _boxeador_a_dict(b)


@app.put("/sesiones/{sesion_id}/boxeadores")
def asignar_boxeadores(
    sesion_id: str,
    entrada: AsignarBoxeadores,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Dice a quien corresponde cada lado de una sesion.

    Es la unica forma de que el peleador A de hoy y el de la semana pasada sean la misma
    persona: A y B se asignan por posicion en pantalla y no significan nada entre videos.
    Se puede cambiar despues, y cambiarlo solo reescribe esta relacion: ni la Fight-Card ni
    los golpes se tocan.
    """
    s = sesion_del_usuario(sesion_id, usuario, db)
    for campo, valor in (("boxeador_a_id", entrada.boxeador_a),
                         ("boxeador_b_id", entrada.boxeador_b)):
        if valor is not None:
            _boxeador_del_usuario(valor, usuario, db)
        setattr(s, campo, valor)
    db.commit()
    return {"sesion": s.id, "boxeador_a": s.boxeador_a_id, "boxeador_b": s.boxeador_b_id}


@app.get("/boxeadores/{boxeador_id}/perfil")
def perfil_de_boxeador(
    boxeador_id: str,
    usuario: Usuario = Depends(usuario_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    El perfil: las sesiones de este boxeador puestas una al lado de la otra.

    Se arma leyendo las Fight-Cards de disco en cada pedido y no se cachea. Son pocas por
    boxeador y el entrenador corrige tipos: un perfil cacheado mostraria la version anterior
    a su correccion, que es justo cuando mira.
    """
    from boxtwin_api.perfil import construir_perfil

    b = _boxeador_del_usuario(boxeador_id, usuario, db)
    sesiones = []
    for s in db.query(Sesion).filter(Sesion.usuario_id == usuario.id).all():
        if s.boxeador_a_id == b.id:
            sesiones.append((s, "A"))
        if s.boxeador_b_id == b.id:
            sesiones.append((s, "B"))
    return construir_perfil(b, sesiones, cfg.dir_sesion)


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
