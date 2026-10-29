"""
BoxTwin API - El servidor.

POR QUE EXISTE
  El flujo del producto tiene una espera larga en el medio y una pregunta humana adentro de
  esa espera. Eso no entra en un request: subir devuelve un identificador (RF1), el estado
  se consulta, y cuando el sistema llega al punto de siembra el usuario contesta y el
  trabajo sigue.

  Todo lo que sirve un archivo filtra por cuenta (RNF-V1, versiones 26-10). El dueno de una
  sesion es la cuenta, y en Gimnasio la ven todos sus miembros. Hay una sola dependencia que
  resuelve de que cuenta es el pedido (`acceso.py`) y una sola funcion que resuelve la sesion
  de la cuenta, para que no exista una ruta que se olvide de comparar. Una prueba
  estructural lo verifica (tests/test_acceso_estructural.py).

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
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwin_api.acceso import (
    Acceso,
    acceso_actual,
    acceso_de_usuario,
    acceso_por_ticket,
    boxeador_de_la_cuenta,
    es_de_prueba,
    exigir_funcion,
    miembro_activo,
    sesion_de_la_cuenta,
)
from boxtwin_api.cola import encolar, posicion_en_cola
from boxtwin_api.config import cfg
from boxtwin_api.cupo import horas_del_mes, texto_horas
from boxtwin_api.db import crear_tablas, obtener_sesion
from boxtwin_api.esquemas import (
    CONTEXTOS,
    PRACTICAS,
    AsignarBoxeadores,
    EntradaDemo,
    EntradaTramos,
    Credenciales,
    EntradaBoxeador,
    EntradaCorreccion,
    Siembra,
    Token,
    sesion_a_dict,
)
from boxtwin_api.exportar import a_csv, a_html
from boxtwin_api.modelos import AsignacionManual, Boxeador, Correccion, Sesion, Trabajo, Usuario
from boxtwin_api.perfil import regla_de_combinacion
from boxtwin_api.planes import PERFILES_DEMO, PLANES, email_demo
from boxtwin_api.seguridad import (
    emitir_ticket,
    emitir_token,
    hashear,
    verificar,
)

__all__ = ["app"]

# Cuando arranco ESTE proceso, y sobre que commit. Las dos cosas se miran desde afuera con
# GET /salud, y existen porque ya costaron dos rondas de diagnostico: un servicio que quedo
# corriendo codigo viejo contesta 405 a las rutas que todavia no tiene -las atiende el
# montaje del frontend, que solo acepta GET- y eso se lee como un bug del codigo nuevo.
ARRANCADO = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _commit(checkout: Path | None = None) -> str | None:
    """El commit del checkout donde vive el paquete, leido del archivo, sin subprocesos.

    En un worktree `.git` es un archivo con `gitdir:`; HEAD se lee ahi y las referencias en el
    directorio comun que indica `commondir` (docs/specs/2026-10-25-dos-instancias, T1).
    """
    try:
        raiz = (checkout or Path(__file__).resolve().parents[3]) / ".git"
        comun = raiz
        if raiz.is_file():
            texto = raiz.read_text().strip()
            if not texto.startswith("gitdir: "):
                return None
            raiz = Path(texto[8:])
            comun = raiz
            if (raiz / "commondir").is_file():
                comun = raiz / (raiz / "commondir").read_text().strip()
        cabeza = (raiz / "HEAD").read_text().strip()
        if cabeza.startswith("ref: "):
            ref = comun / cabeza[5:]
            if not ref.is_file():          # empaquetado en packed-refs
                nombre = cabeza[5:]
                for linea in (comun / "packed-refs").read_text().splitlines():
                    if linea.endswith(" " + nombre):
                        return linea.split()[0][:7]
                return None
            return ref.read_text().strip()[:7]
        return cabeza[:7]
    except OSError:
        return None


COMMIT = _commit()


def _modelo_pose(ruta: Path) -> dict | None:
    """Nombre y los 12 primeros del sha256 del checkpoint de pose; None si no esta.

    Una vez al arrancar, como el commit: en la estacion conviven A con yolov8l-pose y C con
    yolo26l-pose, y /salud tiene que decir cual (docs/specs/2026-10-27-gemelo-yolo26, T2).
    """
    import hashlib

    try:
        h = hashlib.sha256()
        with Path(ruta).open("rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
    except OSError:
        return None
    return {"archivo": Path(ruta).name, "sha256": h.hexdigest()[:12]}


MODELO_POSE = _modelo_pose(cfg.modelo_pose)


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


# El acceso de cada pedido (usuario, cuenta, rol y plan) lo resuelve acceso.py: una sola
# dependencia para todas las rutas con datos, y otra para las que entran con ticket.


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
    # Los usuarios de la version de prueba entran solo por su boton (02 §11), nunca por clave.
    if u is None or not verificar(c.clave, u.hash_clave) or es_de_prueba(u, db):
        raise HTTPException(401, "email o clave incorrectos")
    return Token(token=emitir_token(u.id, cfg.secreto, cfg.horas_de_sesion), usuario=u.email)


# ---------------------------------------------------------------------------
# Version de prueba (versiones 26-10, B29; 02 §11)
# ---------------------------------------------------------------------------
#
# Con BOXTWIN_DEMO apagada, que es el valor por defecto, la ruta ni se registra: responde el
# 404 de una ruta que no existe. Encender o apagar es cambiar el env y reiniciar.


def _token_de_prueba(entrada: EntradaDemo, db: Session) -> Token:
    """
    El token de un perfil de prueba. Busca el usuario por el perfil, que es un nombre fijo:
    no recibe ids ni emails, asi que no hay nada que cambiar en el pedido para llegar a otra
    cuenta. Y antes de emitir verifica que la cuenta este marcada como de prueba.
    """
    import logging

    perfil = PERFILES_DEMO.get(entrada.perfil)
    if perfil is None:
        raise HTTPException(422, f"perfil tiene que ser uno de {', '.join(PERFILES_DEMO)}")
    if not perfil.disponible:
        raise HTTPException(409, "en construcción")
    u = db.scalar(select(Usuario).where(Usuario.email == email_demo(perfil.perfil)))
    if u is None:
        raise HTTPException(
            409, "faltan las cuentas de prueba: correr `python -m boxtwin_api.demo sembrar`"
        )
    acceso = acceso_de_usuario(u, db)
    cuenta = acceso.cuenta
    if not cuenta.demo or acceso.rol != perfil.rol or cuenta.version != perfil.version:
        # Un error de integridad, no algo que el pedido pueda provocar: el usuario del perfil
        # quedo en una cuenta que no es de prueba. No se emite nada.
        logging.getLogger(__name__).error(
            "ingreso de prueba: el perfil %s apunta a la cuenta %s, que no es de prueba o no "
            "es de su version y rol", perfil.perfil, cuenta.id)
        raise HTTPException(500, "las cuentas de prueba no estan bien cargadas")
    return Token(token=emitir_token(u.id, cfg.secreto, HORAS_DEMO), usuario=perfil.nombre,
                 perfil=perfil.perfil)


HORAS_DEMO = 8

if cfg.demo:
    @app.post("/auth/demo", response_model=Token)
    def ingreso_de_prueba(entrada: EntradaDemo, db: Session = Depends(obtener_sesion)) -> Token:
        return _token_de_prueba(entrada, db)


# ---------------------------------------------------------------------------
# Subida y estado
# ---------------------------------------------------------------------------


@app.post("/videos", status_code=202)
def subir(
    archivo: UploadFile = File(...),
    nombre: str = Form(default=""),
    round_s: float | None = Form(default=None),
    descanso_s: float = Form(default=60.0),
    contexto: str = Form(default="sparring"),
    practica: str = Form(default=""),
    rounds: str = Form(default=""),
    consentimiento: str = Form(default=""),
    duracion_declarada_s: str = Form(default=""),
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    RF1: encola y devuelve el identificador sin esperar el procesamiento.

    El archivo se escribe en disco a medida que llega y no en memoria: 30 minutos de 1080p
    desde un celular son varios gigabytes y un servidor de un nodo no los aguanta en RAM.

    Los campos de contexto y el consentimiento son de B1 (web entrenador). El
    consentimiento se valida ANTES de leer el archivo: no tiene sentido recibir un gigabyte
    para rechazarlo despues. Lo mismo el cupo de horas (B20): con `bloquear`, una subida que
    pasa lo que queda se rechaza con 409 antes de recibir un byte del video.
    """
    usuario, cuenta = acceso.usuario, acceso.cuenta
    if cuenta.demo and not cfg.demo_subir:
        raise HTTPException(403, "la subida está apagada en la versión de prueba")
    if consentimiento.strip().lower() != "true":
        raise HTTPException(422, "falta el consentimiento de las personas filmadas")
    if contexto not in CONTEXTOS:
        raise HTTPException(422, f"contexto tiene que ser uno de {', '.join(CONTEXTOS)}")
    practica = practica.strip() or None
    if practica is not None and contexto != "sombra":
        raise HTTPException(422, "lo que se practico solo se declara en una sesion de sombra")
    if practica is not None and practica not in PRACTICAS:
        raise HTTPException(422, f"practica tiene que ser una de {', '.join(PRACTICAS)}")
    cantidad = None
    if rounds.strip():
        try:
            cantidad = int(rounds)
        except ValueError:
            raise HTTPException(422, "rounds tiene que ser un entero") from None
        if cantidad < 1:
            raise HTTPException(422, "rounds tiene que ser 1 o mas, o vacio para sin rounds")

    ext = Path(archivo.filename or "").suffix.lower()
    if ext not in cfg.formatos:
        raise HTTPException(
            415, f"formato no soportado: {ext or 'sin extension'}. "
                 f"Se aceptan {', '.join(cfg.formatos)}"
        )
    if round_s is not None and round_s <= 0:
        raise HTTPException(422, "la duracion del round tiene que ser mayor que cero")
    declarada = None
    if duracion_declarada_s.strip():
        try:
            declarada = float(duracion_declarada_s)
        except ValueError:
            raise HTTPException(422, "duracion_declarada_s tiene que ser un numero") from None
        if declarada < 0:
            raise HTTPException(422, "duracion_declarada_s no puede ser negativa")

    horas = horas_del_mes(cuenta, db)
    pedido_h = (declarada or 0.0) / 3600
    excedido = declarada is not None and pedido_h > horas["restantes"]
    if excedido and horas["modo"] == "bloquear":
        raise HTTPException(
            409, f"esta sesión usaría {texto_horas(pedido_h)} h y quedan "
                 f"{texto_horas(horas['restantes'])} h en el mes"
        )

    ses = Sesion(
        usuario_id=usuario.id,
        cuenta_id=cuenta.id,
        duracion_declarada_s=declarada,
        # B22: en Individual la sesion es del peleador propio, sin elegirlo (RF-I2).
        boxeador_a_id=(cuenta.boxeador_propio_id if acceso.plan.peleadores == "propio"
                       else None),
        nombre=(nombre or Path(archivo.filename or "sesion").stem)[:255],
        video_nombre=_nombre_seguro(archivo.filename or "video.mp4"),
        round_s=round_s,
        descanso_s=descanso_s,
        estado="en_cola",
        contexto=contexto,
        practica=practica,
        rounds=cantidad,
        consentimiento_en=datetime.now(timezone.utc),
        consentimiento_por=usuario.email,
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
    return {"job_id": ses.id, "estado": ses.estado, "bytes": escritos,
            "cupo": {"excedido": excedido, "restantes_h": horas["restantes"]}}


def _nombre_seguro(nombre: str) -> str:
    """El nombre que subio el usuario no elige rutas del servidor."""
    base = Path(nombre).name
    limpio = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return limpio[:120] or "video.mp4"


@app.get("/jobs/{sesion_id}")
def estado(
    sesion_id: str,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    El estado, y cuando corresponde los candidatos de siembra con sus recortes (RF3).

    Lo que se lee del directorio se lee del archivo que escribe el worker, que es la unica
    fuente de verdad de lo que paso adentro del proceso. La base guarda el estado para
    poder listarlo sin tocar disco, pero cuando los dos difieren manda el archivo.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    d = _sesion_en_disco(s.id)
    trabajos = db.scalars(
        select(Trabajo).where(Trabajo.sesion_id == s.id).order_by(Trabajo.creado)
    ).all()
    # Con un `completar` esperando en la cola, el archivo todavia describe el ciclo ANTERIOR
    # (espera_siembra, o listo si se re-sembro o se asignaron tramos): el worker no lo tomo.
    # Hacerle caso hacia retroceder una sesion recien sembrada a espera_siembra, y la web,
    # que deja de sondear ahi, se quedaba clavada. Encontrado en el e2e de la web entrenador.
    # Vale tambien con el trabajo ya tomado: entre que el worker lo reclama y la etapa escribe
    # su primer estado hay una ventana en la que el disco sigue diciendo espera_siembra. El CI,
    # con un worker mas rapido que la estacion, cayo ahi.
    completar_pendiente = any(t.estado in ("en_cola", "tomado") and t.etapa == "completar"
                              for t in trabajos)
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
        viejo = completar_pendiente and d.get("estado") in ("espera_siembra", "listo")
        if d.get("estado") and d["estado"] != s.estado and s.estado != "fallo" and not viejo:
            s.estado = d["estado"]
            db.commit()
    extra["trabajos"] = [
        {"etapa": t.etapa, "estado": t.estado, "intento": t.intento, "error": t.error}
        for t in trabajos
    ]
    # B27: cuantos trabajos de todas las cuentas van antes, sin decir de quien son.
    if s.estado == "en_cola":
        extra["cola"] = posicion_en_cola(db, s.id, cfg.factor_min, cfg.factor_max)
    extra["subida_por"] = _subida_por(s, db)
    return sesion_a_dict(s, extra)


def _nombre_de(u: Usuario | None) -> str | None:
    if u is None:
        return None
    return u.nombre or u.email


def _subida_por(s: Sesion, db: Session) -> dict | None:
    """B24: quien subio la sesion. En Gimnasio es lo que distingue el trabajo de cada uno."""
    u = db.get(Usuario, s.usuario_id)
    return {"id": u.id, "nombre": _nombre_de(u)} if u is not None else None


@app.get("/jobs")
def listar(
    acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion)
) -> list[dict]:
    """El historial de la cuenta (F8). Solo el suyo; en Gimnasio, el de todos sus miembros."""
    filas = db.scalars(
        select(Sesion).where(Sesion.cuenta_id == acceso.cuenta.id).order_by(Sesion.creada.desc())
    ).all()
    nombres = {u.id: _nombre_de(u) for u in db.scalars(
        select(Usuario).where(Usuario.id.in_({s.usuario_id for s in filas}))).all()}
    # B5: el progreso de las que estan en curso, para que la lista muestre "pose · 64%"
    # sin un pedido por sesion. Solo esas: leer disco para las listas no aporta nada.
    from boxtwin_api.panel import EN_CURSO

    return [
        sesion_a_dict(s, {
            **({"progreso": _progreso_en_disco(s.id)} if s.estado in EN_CURSO else {}),
            "subida_por": {"id": s.usuario_id, "nombre": nombres.get(s.usuario_id)},
        })
        for s in filas
    ]


@app.get("/panel")
def panel(
    acceso: Acceso = Depends(acceso_actual), db: Session = Depends(obtener_sesion)
) -> dict:
    """Los indicadores y lo pendiente de la pantalla 01 (B5). Solo de la cuenta."""
    from boxtwin_api.panel import construir_panel

    d = construir_panel(acceso, db, cfg.dir_sesion, cfg.perfil_preliminar_sesiones)
    d["horas"] = horas_del_mes(acceso.cuenta, db)   # B20
    return d


@app.get("/jobs/{sesion_id}/candidatos/{track}")
def recorte(
    track: int,
    sesion_id: str,
    acceso: Acceso = Depends(acceso_por_ticket),
    db: Session = Depends(obtener_sesion),
) -> FileResponse:
    """
    El recorte de un candidato. Entra por header o por ticket, igual que el video.

    Un <img src> no manda headers, asi que sin el ticket el navegador pide la imagen sin
    token, recibe 401 y dibuja el icono roto: la pantalla de siembra queda inusable
    justo donde el usuario tiene que mirar dos fotos y elegir.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    ruta = cfg.dir_sesion(s.id) / "candidatos" / f"track_{int(track)}.jpg"
    if not ruta.is_file():
        raise HTTPException(404, "no hay recorte para ese track")
    return FileResponse(ruta, media_type="image/jpeg")


@app.post("/jobs/{sesion_id}/siembra", status_code=202)
def siembra(
    sesion_id: str,
    s_in: Siembra,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    RF4: el usuario contesta cual es cual y el trabajo sigue.

    Se acepta tambien sobre una sesion ya lista, que es re-sembrar: si el entrenador ve que
    el sistema le puso los nombres al reves, cambiar la respuesta y volver a correr la
    segunda etapa cuesta minutos y no requiere volver a subir el video.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    if s.estado not in ("espera_siembra", "listo", "fallo"):
        raise HTTPException(
            409, f"la sesion esta en '{s.estado}' y todavia no hay candidatos que elegir"
        )
    individual = _es_individual(s)
    if s_in.track_b is None and not individual:
        raise HTTPException(422, "en sparring y combate hay que marcar a los dos peleadores")
    if s_in.track_b is not None and individual:
        raise HTTPException(422, "en bolsa y sombra se marca a una sola persona")
    disco = _sesion_en_disco(s.id) or {}
    tracks = {c["track"] for c in disco.get("candidatos", [])}
    elegidos = {s_in.track_a} | ({s_in.track_b} if s_in.track_b is not None else set())
    # Se permite un track que no este entre los propuestos -el usuario puede haber visto al
    # peleador en un track que los filtros descartaron- pero no uno que no existe.
    if tracks and not elegidos <= tracks:
        ajenos = sorted(elegidos - tracks)
        if any(t < 0 for t in ajenos):
            raise HTTPException(422, f"track invalido: {ajenos}")

    s.semilla_a, s.semilla_b = s_in.track_a, s_in.track_b
    s.estado = "en_cola"
    s.error = None
    # Una semilla nueva redefine que es A y que es B: las asignaciones de tramos hechas con la
    # anterior dejan de significar lo mismo. Quedan en la base, para auditoria; se sacan de
    # sesion.json para que el proximo completar manual no las aplique al reves.
    if disco.get("identidad", {}).get("asignaciones_manuales"):
        disco["identidad"].pop("asignaciones_manuales")
        _escribir_sesion_en_disco(s.id, disco)
    encolar(db, s.id, "completar", track_a=s_in.track_a, track_b=s_in.track_b)
    db.commit()
    return {"job_id": s.id, "estado": s.estado}


# ---------------------------------------------------------------------------
# Tramos sin asignar (B3 de la web entrenador)
# ---------------------------------------------------------------------------


def _es_individual(s: Sesion) -> bool:
    return (s.contexto or "sparring") in ("bolsa", "sombra")


def _tramos_en_disco(s: Sesion) -> tuple[dict, list[dict]]:
    if s.estado != "listo":
        raise HTTPException(
            409, f"la sesion esta en '{s.estado}': los tramos se revisan sobre una sesion lista"
        )
    disco = _sesion_en_disco(s.id) or {}
    tramos = (disco.get("identidad") or {}).get("tramos")
    if tramos is None:
        raise HTTPException(
            409, "esta sesion se proceso antes de la revision de tramos; reprocesala para "
                 "revisarlos"
        )
    return disco, tramos


@app.get("/jobs/{sesion_id}/tramos")
def tramos(
    sesion_id: str,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Lo que la identidad no pudo asignar, para que una persona lo termine (plan B de C1).

    Cada tramo viaja con su motivo y, si ya lo asigno alguien, con esa asignacion: la
    lista no se achica al asignar, para que una decision se pueda cambiar.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    disco, lista = _tramos_en_disco(s)
    manuales = (disco.get("identidad") or {}).get("asignaciones_manuales") or {}
    ident = _fightcard(s.id).get("identidad") or {}
    cobertura = {"A": ident.get("cobertura_A", 0.0)}
    # B8b: en bolsa y sombra hay una sola persona; un B en cero se leeria como "no lo vio".
    if not _es_individual(s):
        cobertura["B"] = ident.get("cobertura_B", 0.0)
    cobertura["sin_asignar"] = ident.get("sin_asignar", 0.0)
    return {
        "cobertura": cobertura,
        "separacion_color": ident.get("separacion_color", ident.get("separacion_de_perfiles")),
        "tramos": [
            {
                "track": t["track"],
                "t_desde": t["t_desde"],
                "t_hasta": t["t_hasta"],
                "motivo": t["motivo"],
                "recorte_url": (f"/jobs/{s.id}/tramos/{t['track']}/recorte"
                                if t.get("recorte") else None),
                "asignado": manuales.get(str(t["track"]), t.get("asignado")),
            }
            for t in lista
        ],
    }


@app.get("/jobs/{sesion_id}/tramos/{track}/recorte")
def recorte_de_tramo(
    track: int,
    sesion_id: str,
    acceso: Acceso = Depends(acceso_por_ticket),
    db: Session = Depends(obtener_sesion),
) -> FileResponse:
    """El cuadro central de un tramo. Entra con ticket, igual que los recortes de siembra."""
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    ruta = cfg.dir_sesion(s.id) / "tramos" / f"track_{int(track)}.jpg"
    if not ruta.is_file():
        raise HTTPException(404, "no hay recorte para ese tramo")
    return FileResponse(ruta, media_type="image/jpeg")


@app.put("/jobs/{sesion_id}/tramos", status_code=202)
def asignar_tramos(
    sesion_id: str,
    entrada: EntradaTramos,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Guarda las asignaciones de la persona y vuelve a correr la segunda etapa con ellas.

    La segunda etapa reutiliza la cache de pose y tracking: aplica la asignacion por color,
    fuerza las manuales encima y vuelve a correr detector, guardia y clasificacion. Cuesta
    segundos, no los minutos de la pose.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    disco, lista = _tramos_en_disco(s)
    if _es_individual(s) and any(a.rol == "B" for a in entrada.asignaciones):
        raise HTTPException(422, "en bolsa y sombra hay una sola persona: un tramo es A o ninguno")
    validos = {int(t["track"]) for t in lista}
    ajenos = sorted({a.track for a in entrada.asignaciones} - validos)
    if ajenos:
        raise HTTPException(422, f"esos tracks no son tramos de esta sesion: {ajenos}")
    if s.semilla_a is None:
        raise HTTPException(409, "la sesion no tiene siembra: primero hay que marcar a A y B")

    ident = disco.setdefault("identidad", {})
    manuales = dict(ident.get("asignaciones_manuales") or {})
    for a in entrada.asignaciones:
        manuales[str(a.track)] = a.rol
        db.add(AsignacionManual(sesion_id=s.id, track=a.track, rol=a.rol,
                                usuario_id=acceso.usuario.id))
    ident["asignaciones_manuales"] = manuales
    _escribir_sesion_en_disco(s.id, disco)

    s.estado = "en_cola"
    s.error = None
    encolar(db, s.id, "completar", track_a=s.semilla_a, track_b=s.semilla_b, manual=True)
    db.commit()
    return {"job_id": s.id, "estado": s.estado, "asignaciones": len(manuales)}


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


def _escribir_sesion_en_disco(sesion_id: str, datos: dict) -> None:
    """Escritura atomica de sesion.json, para que el worker nunca lea uno a medias."""
    ruta = cfg.dir_sesion(sesion_id) / "sesion.json"
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(ruta)


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
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> JSONResponse:
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    fc = _fightcard(s.id)
    # 05 §4: lo que sale de la base y puede cambiar despues del procesamiento va al
    # responder y no se guarda en el archivo. Asignar un peleador no reescribe la Fight-Card.
    fc["sesion"] = {
        "contexto": s.contexto or "sparring",
        "practica": s.practica,
        "boxeadores": {"A": _boxeador_de_sesion(s.boxeador_a), "B": _boxeador_de_sesion(s.boxeador_b)},
        # B24: quien subio y quien corrigio cada golpe. El contrato 0.3 no cambia: esto no
        # se guarda en el archivo.
        "autoria": _autoria(s, db),
    }
    return JSONResponse(fc)


def _autoria(s: Sesion, db: Session) -> dict:
    subida = _subida_por(s, db)
    correcciones: dict[str, str | None] = {}
    filas = db.scalars(
        select(Correccion).where(Correccion.sesion_id == s.id).order_by(Correccion.creada)
    ).all()
    quienes = {u.id: _nombre_de(u) for u in db.scalars(select(Usuario).where(
        Usuario.id.in_({c.usuario_id for c in filas if c.usuario_id}))).all()}
    for c in filas:
        # La ultima correccion de cada golpe; las anteriores a las cuentas quedan sin nombre.
        correcciones[c.golpe] = quienes.get(c.usuario_id) if c.usuario_id else None
    return {"subida_por": subida["nombre"] if subida else None, "correcciones": correcciones}


def _boxeador_de_sesion(b) -> dict | None:
    if b is None:
        return None
    return {"id": b.id, "nombre": b.nombre, "guardia": b.guardia, "categoria": b.categoria}


@app.get("/fightcards/{sesion_id}/export")
def exportar(
    sesion_id: str,
    formato: str = "csv",
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
):
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
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
        return HTMLResponse(a_html(fc, _nombres_para_export(s, acceso)))
    raise HTTPException(422, "formato tiene que ser csv o html")


def _nombres_para_export(s: Sesion, acceso: Acceso) -> dict:
    """En Individual, el peleador propio y "Compañero" (CA-I05.1); si no, los asignados."""
    if acceso.plan.peleadores == "propio":
        return {"A": s.boxeador_a.nombre if s.boxeador_a else "Vos", "B": "Compañero"}
    return {k: b.nombre for k, b in (("A", s.boxeador_a), ("B", s.boxeador_b)) if b is not None}


@app.patch("/fightcards/{sesion_id}/golpes/{golpe}")
def corregir(
    sesion_id: str,
    golpe: str,
    entrada: EntradaCorreccion,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    F14 y RF10: el entrenador corrige el tipo y la correccion se guarda como etiqueta. B11
    suma la altura: `{"objetivo": "cabeza" | "cuerpo"}`, que queda con `campo = objetivo`.

    La ruta esta anidada bajo la sesion y no es `/golpes/{id}` como propone el spec, porque
    el id de un golpe sale de su carril y su cuadro y solo es unico adentro de su sesion.
    Un id global pediria un contador, y un contador se rompe al reclasificar.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    try:
        from boxtwin.mvp.orquesta import registrar_correccion
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise HTTPException(
            501, "esta instancia no tiene instalado boxtwin-annotator, que es donde vive "
                 "el contrato de la Fight-Card"
        ) from e

    try:
        fc = registrar_correccion(cfg.dir_sesion(s.id), golpe, entrada.tipo,
                                  por=acceso.usuario.email, objetivo=entrada.objetivo)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    ev = next(
        (g for p in fc["peleadores"].values() for g in p["golpes"] if g["id"] == golpe), None
    )
    if entrada.objetivo is not None:
        campo, valor = "objetivo", entrada.objetivo
        original = ((ev or {}).get("objetivo_corregido") or {}).get("objetivo_original")
    else:
        campo, valor = "tipo", entrada.tipo
        original = ((ev or {}).get("corregido") or {}).get("tipo_original")
    db.add(
        Correccion(
            sesion_id=s.id, golpe=golpe, tipo=valor, campo=campo,
            tipo_original=original,
            checkpoint=fc.get("clasificador", {}).get("checkpoint"),
            datos={"t_inicio": (ev or {}).get("t_inicio"), "por": acceso.usuario.email},
            usuario_id=acceso.usuario.id,
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
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """Un permiso corto para reproducir ESTE video, que es lo que el <video> puede llevar."""
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    return {"ticket": emitir_ticket(acceso.usuario.id, s.id, cfg.secreto), "minutos": 30}


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
    range: str = Header(default=""),
    acceso: Acceso = Depends(acceso_por_ticket),
    db: Session = Depends(obtener_sesion),
):
    """El video original de la sesion."""
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
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
    range: str = Header(default=""),
    acceso: Acceso = Depends(acceso_por_ticket),
    db: Session = Depends(obtener_sesion),
):
    """
    El video con los dos peleadores marcados y una consola por peleador.

    Es una etapa aparte y la ultima, asi que puede no existir todavia cuando la Fight-Card ya
    esta lista. El 404 es entonces un estado normal y no un error: la interfaz muestra el
    original mientras tanto.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    ruta = cfg.dir_sesion(s.id) / "procesado.mp4"
    if not ruta.is_file():
        raise HTTPException(404, "el video procesado todavia no esta")
    return _servir_con_rangos(ruta, range)


@app.post("/jobs/{sesion_id}/render", status_code=202)
def pedir_render(
    sesion_id: str,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Encola el armado del video con las marcas.

    Existe porque el render se encadena despues de completar, y eso deja afuera a toda
    sesion que se proceso antes de que la etapa existiera: no hay ningun trabajo que lo
    arme y la interfaz se queda diciendo "se esta armando" para siempre. Tambien es el
    camino para reintentarlo cuando fallo.

    Se niega si ya hay uno esperando, que es lo que evita que tocar dos veces el boton
    encole dos renders del mismo video sobre la unica GPU que hay.
    """
    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    if s.estado != "listo":
        raise HTTPException(
            409, f"la sesion esta en '{s.estado}': el video con marcas se arma sobre la "
                 "Fight-Card, asi que tiene que estar lista"
        )
    pendiente = db.scalar(
        select(Trabajo).where(
            Trabajo.sesion_id == s.id,
            Trabajo.etapa == "render",
            Trabajo.estado.in_(("en_cola", "tomado")),
        )
    )
    if pendiente is not None:
        return {"job_id": s.id, "estado": pendiente.estado, "ya_estaba": True}
    encolar(db, s.id, "render")
    db.commit()
    return {"job_id": s.id, "estado": "en_cola", "ya_estaba": False}


# ---------------------------------------------------------------------------
# Boxeadores y perfiles


def _boxeador_a_dict(b, uso: dict | None = None, extra: dict | None = None) -> dict:
    d = {"id": b.id, "nombre": b.nombre, "guardia": b.guardia, "notas": b.notas,
         "creado": b.creado.isoformat(),
         # B6
         "categoria": b.categoria,
         "altura_m": b.altura_m,
         "envergadura_m": b.envergadura_m,
         # B17
         "distancia": b.distancia,
         "estilo": b.estilo}
    if uso is not None:
        n, ultima = uso.get(b.id, (0, None))
        d["sesiones"] = n
        d["ultima_sesion"] = ultima.isoformat() if ultima else None
    if extra is not None:
        d.update(extra.get(b.id, {}))
    return d


def _plantel(acceso: Acceso, bs, db: Session) -> dict:
    """B24 y B25: entrenador a cargo y consentimiento de cada atleta. Nulos fuera de Gimnasio."""
    from boxtwin_api.cuentas import estado_consentimiento

    gimnasio = acceso.plan.consentimiento == "atleta"
    entrenadores = {u.id: u for u in db.scalars(select(Usuario).where(
        Usuario.id.in_({b.entrenador_id for b in bs if b.entrenador_id}))).all()}
    out = {}
    for b in bs:
        u = entrenadores.get(b.entrenador_id) if b.entrenador_id else None
        out[b.id] = {
            "entrenador": {"id": u.id, "nombre": _nombre_de(u)} if u is not None else None,
            "consentimiento": estado_consentimiento(b, db) if gimnasio else None,
        }
    return out


def _uso_de_boxeadores(acceso: Acceso, db) -> dict:
    """
    Cuantas sesiones tiene cada boxeador, como A o como B, y la ultima (B5b, B6).

    Dos consultas agrupadas sobre `sesiones`, sin leer Fight-Cards: es lo que la lista y el
    panel muestran debajo de cada nombre, y no puede costar un archivo por sesion.
    """
    from sqlalchemy import func

    uso: dict[str, tuple[int, object]] = {}
    for col in (Sesion.boxeador_a_id, Sesion.boxeador_b_id):
        filas = db.execute(
            select(col, func.count(Sesion.id), func.max(Sesion.creada))
            .where(Sesion.cuenta_id == acceso.cuenta.id, col.is_not(None))
            .group_by(col)
        ).all()
        for bid, n, ultima in filas:
            n0, u0 = uso.get(bid, (0, None))
            uso[bid] = (n0 + n, max(x for x in (u0, ultima) if x is not None))
    return uso


def _entrenador_valido(acceso: Acceso, uid: str | None, db: Session) -> str | None:
    """B24: el entrenador a cargo tiene que ser un miembro activo de la cuenta, y solo en Gimnasio."""
    if uid is None:
        return None
    exigir_funcion(acceso, acceso.plan.plantel_compartido)
    if miembro_activo(db, acceso.cuenta.id, uid) is None:
        raise HTTPException(422, "el entrenador a cargo tiene que ser un miembro activo del gimnasio")
    return uid


@app.post("/boxeadores", status_code=201)
def crear_boxeador(
    entrada: EntradaBoxeador,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """Da de alta un boxeador. El nombre es unico por cuenta, no global."""
    # B22: en Individual la cuenta tiene un solo peleador, el propio.
    exigir_funcion(acceso, acceso.plan.peleadores == "libre")
    nombre = entrada.nombre.strip()
    ya = (
        db.query(Boxeador)
        .filter(Boxeador.cuenta_id == acceso.cuenta.id, Boxeador.nombre == nombre)
        .first()
    )
    if ya is not None:
        raise HTTPException(409, f"ya tenes un boxeador que se llama {nombre}")
    entrenador = _entrenador_valido(acceso, entrada.entrenador_id, db)
    b = Boxeador(usuario_id=acceso.usuario.id, cuenta_id=acceso.cuenta.id, nombre=nombre,
                 guardia=entrada.guardia,
                 notas=entrada.notas, categoria=entrada.categoria,
                 altura_m=entrada.altura_m, envergadura_m=entrada.envergadura_m,
                 distancia=entrada.distancia, estilo=entrada.estilo, entrenador_id=entrenador)
    db.add(b)
    db.commit()
    return _boxeador_a_dict(b, {}, _plantel(acceso, [b], db))


@app.get("/boxeadores")
def listar_boxeadores(
    a_cargo: str = "",
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> list[dict]:
    q = db.query(Boxeador).filter(Boxeador.cuenta_id == acceso.cuenta.id)
    if a_cargo == "yo":
        # B24: el filtro "a mi cargo" del plantel.
        q = q.filter(Boxeador.entrenador_id == acceso.usuario.id)
    bs = q.order_by(Boxeador.nombre).all()
    uso = _uso_de_boxeadores(acceso, db)
    extra = _plantel(acceso, bs, db)
    return [_boxeador_a_dict(b, uso, extra) for b in bs]


@app.patch("/boxeadores/{boxeador_id}")
def editar_boxeador(
    boxeador_id: str,
    entrada: EntradaBoxeador,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    b = boxeador_de_la_cuenta(boxeador_id, acceso, db)
    guardia_previa = b.guardia
    b.nombre = entrada.nombre.strip()
    b.guardia = entrada.guardia
    if b.guardia != guardia_previa:
        # RT8 del tipo v2: jab y cross salen de la guardia, asi que cambiarla reclasifica.
        _reclasificar(db, db.scalars(select(Sesion).where(
            (Sesion.boxeador_a_id == b.id) | (Sesion.boxeador_b_id == b.id))).all())
    b.notas = entrada.notas
    b.categoria = entrada.categoria
    b.altura_m = entrada.altura_m
    b.envergadura_m = entrada.envergadura_m
    b.distancia = entrada.distancia
    b.estilo = entrada.estilo
    # B24: solo si viene en el pedido. La hoja de edicion de las otras versiones no lo manda
    # y no lo tiene que borrar.
    if "entrenador_id" in entrada.model_fields_set:
        b.entrenador_id = _entrenador_valido(acceso, entrada.entrenador_id, db)
    db.commit()
    return _boxeador_a_dict(b, _uso_de_boxeadores(acceso, db), _plantel(acceso, [b], db))


def _reclasificar(db: Session, sesiones) -> list[str]:
    """
    Encola `clasificar` en las sesiones listas, solo con el tipo v2 encendido (RT8). Con v1
    la guardia no interviene en el tipo y no hay nada que recalcular. No commitea.
    """
    if not cfg.tipo_v2:
        return []
    encoladas = []
    for s in sesiones:
        if s.estado != "listo":
            continue
        ya = db.scalars(select(Trabajo).where(
            Trabajo.sesion_id == s.id, Trabajo.etapa == "clasificar", Trabajo.estado == "en_cola")).first()
        if ya is None:
            encolar(db, s.id, "clasificar")
            encoladas.append(s.id)
    return encoladas


@app.put("/sesiones/{sesion_id}/boxeadores")
def asignar_boxeadores(
    sesion_id: str,
    entrada: AsignarBoxeadores,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Dice a quien corresponde cada lado de una sesion.

    Es la unica forma de que el peleador A de hoy y el de la semana pasada sean la misma
    persona: A y B se asignan por posicion en pantalla y no significan nada entre videos.
    Se puede cambiar despues, y cambiarlo solo reescribe esta relacion: ni la Fight-Card ni
    los golpes se tocan.

    Versiones: en Individual el compañero no tiene perfil (B22, DV-4); en Gimnasio no se
    asigna un atleta sin consentimiento vigente (B25). Desasignar siempre se puede.
    """
    from boxtwin_api.cuentas import estado_consentimiento

    s = sesion_de_la_cuenta(sesion_id, acceso, db)
    previa = (s.boxeador_a_id, s.boxeador_b_id)
    elegidos = {}
    for campo, valor in (("boxeador_a_id", entrada.boxeador_a),
                         ("boxeador_b_id", entrada.boxeador_b)):
        if valor is not None:
            elegidos[campo] = boxeador_de_la_cuenta(valor, acceso, db)
    if acceso.plan.peleadores == "propio" and entrada.boxeador_b is not None:
        raise HTTPException(403, "en la versión Individual el compañero no tiene perfil")
    if acceso.plan.consentimiento == "atleta":
        for campo, b in elegidos.items():
            if getattr(s, campo) != b.id and estado_consentimiento(b, db)["estado"] != "vigente":
                raise HTTPException(422, f"Falta el consentimiento de {b.nombre}")
    for campo, valor in (("boxeador_a_id", entrada.boxeador_a),
                         ("boxeador_b_id", entrada.boxeador_b)):
        setattr(s, campo, valor)
    if (s.boxeador_a_id, s.boxeador_b_id) != previa:
        # RT8 del tipo v2: con otro boxeador cambia la guardia de la que salen jab y cross.
        _reclasificar(db, [s])
    db.commit()
    return {"sesion": s.id, "boxeador_a": s.boxeador_a_id, "boxeador_b": s.boxeador_b_id}


@app.get("/boxeadores/{boxeador_id}/perfil")
def perfil_de_boxeador(
    boxeador_id: str,
    contexto: str = "todas",
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    El perfil: las sesiones de este boxeador puestas una al lado de la otra. En Gimnasio,
    todas las de la cuenta, sin importar quien las subio (CA-G04.1).

    Se arma leyendo las Fight-Cards de disco en cada pedido y no se cachea. Son pocas por
    boxeador y el entrenador corrige tipos: un perfil cacheado mostraria la version anterior
    a su correccion, que es justo cuando mira.
    """
    if contexto != "todas" and contexto not in CONTEXTOS:
        raise HTTPException(422, f"contexto tiene que ser todas o uno de {', '.join(CONTEXTOS)}")
    return _perfil(boxeador_de_la_cuenta(boxeador_id, acceso, db), acceso, db, contexto)


def _perfil(b, acceso: Acceso, db, contexto: str = "todas", solo_pelea: bool = False,
            excluir_rival: str | None = None) -> dict:
    from boxtwin_api.perfil import construir_perfil

    sesiones = []
    for s in db.query(Sesion).filter(Sesion.cuenta_id == acceso.cuenta.id).all():
        if s.boxeador_a_id == b.id:
            sesiones.append((s, "A"))
        if s.boxeador_b_id == b.id:
            sesiones.append((s, "B"))
    banderas = {"comparacion_rounds": _comparacion_rounds(), "guardia": cfg.c3 == "aprobado",
                "objetivo": _altura_visible()}
    return construir_perfil(b, sesiones, cfg.dir_sesion, contexto=contexto, banderas=banderas,
                            solo_pelea=solo_pelea, excluir_rival=excluir_rival)


@app.get("/rival")
def rival(
    a: str,
    b: str,
    acceso: Acceso = Depends(acceso_actual),
    db: Session = Depends(obtener_sesion),
) -> dict:
    """
    Ataque contra exposicion entre dos perfiles (cambios-14-10, B18). Solo sparring y combate
    de cada uno. La regla vive en boxtwin_api.rival; la web no la copia.
    """
    from boxtwin_api.rival import cruzar

    # B22: la funcion entera no esta en Individual; no hay recurso que filtrar antes.
    exigir_funcion(acceso, acceso.plan.rival)
    if a == b:
        raise HTTPException(422, "elegi dos peleadores distintos")
    # DC-10: la exposicion de cada uno, sin las sesiones contra el otro.
    pa = _perfil(boxeador_de_la_cuenta(a, acceso, db), acceso, db, solo_pelea=True, excluir_rival=b)
    pb = _perfil(boxeador_de_la_cuenta(b, acceso, db), acceso, db, solo_pelea=True, excluir_rival=a)
    return cruzar(pa, pb)


@app.get("/producto")
def producto() -> dict:
    """
    Lo que la web necesita saber de esta instancia antes de dibujar nada (05 B5).

    Sin autenticacion: no expone datos de nadie. La web nunca decide sola si un bloque
    condicionado se muestra: las banderas salen de aca, y aca salen del estado de los
    criterios pre-registrados que Lucas carga por entorno cuando mide.
    """
    return {
        "registro": cfg.modo_registro,
        "max_bytes": cfg.max_bytes,
        "max_minutos": cfg.max_minutos,
        "formatos": list(cfg.formatos),
        "factor_procesamiento": {
            "min": cfg.factor_min,
            "mediana": cfg.factor_mediana,
            "max": cfg.factor_max,
            "medido_sobre": "campaña de costo del Cap 5, estación de desarrollo",
        },
        "criterios": {
            "C2": {"estado": cfg.c2, "fecha": cfg.c2_fecha},
            "C3": {"estado": cfg.c3, "fecha": cfg.c3_fecha},
            # Ampliacion del 13-10 (08 §2 y §3)
            "C8": {"estado": cfg.c8, "fecha": cfg.c8_fecha},
            "C9": {"estado": cfg.c9, "fecha": cfg.c9_fecha},
            "C10": {"estado": cfg.c10, "fecha": cfg.c10_fecha},
        },
        "banderas": {
            # C2 aprobado, o decision de producto rotulada (cambios-14-10, DC-1).
            "comparacion_rounds": _comparacion_rounds(),
            "guardia": cfg.c3 == "aprobado",
            "sesiones_individuales": cfg.sesiones_individuales,
            "postura_2d": False,
            # La altura se muestra con C8 aprobado o por decision de producto (14-10), siempre
            # como estimacion. La exposicion, con C8 aprobado y C9 medido (08 §3) o con la
            # misma decision de producto.
            "objetivo": _altura_visible(),
            # B16 (cambios-14-10): el mapa sale de la misma altura, asi que va con ella. La
            # correccion por sesgo del detector (C9) se aplica con el recall medido el 14-10.
            "exposicion": (cfg.c8 == "aprobado" and cfg.c9 != "pendiente") or cfg.objetivo_estimado,
            "avatar_3d": cfg.c10 == "aprobado",
        },
        "dominio": {
            # La regla vive en el anotador (B10): la API la publica, no la define.
            "combinacion": regla_de_combinacion(),
            "perfil_preliminar_sesiones": cfg.perfil_preliminar_sesiones,
        },
        # Versiones 26-10 (B20, B29). Nada de esto es de nadie: modos, perfiles fijos y horas.
        "cupo": cfg.cupo,
        "demo": _demo_en_producto(),
        "versiones": {v: {"horas_mes": p.horas_mes} for v, p in PLANES.items()},
    }


def _demo_en_producto() -> dict:
    if not cfg.demo:
        return {"activo": False}
    return {
        "activo": True,
        "subir": cfg.demo_subir,
        "perfiles": [
            {"perfil": p.perfil, "nombre": p.nombre, "version": p.version, "rol": p.rol,
             "descripcion": p.descripcion, "disponible": p.disponible}
            for p in PERFILES_DEMO.values()
        ],
    }


def _altura_visible() -> bool:
    return cfg.c8 == "aprobado" or cfg.objetivo_estimado


def _comparacion_rounds() -> bool:
    return cfg.c2 == "aprobado" or cfg.rounds_estimado


@app.get("/salud")
def salud() -> dict:
    return {
        "ok": True,
        "version": app.version,
        # Las dos cosas que hay que poder mirar desde afuera antes de abrir el tunel.
        "secreto_efimero": cfg.secreto_efimero,
        "registro": cfg.modo_registro,
        "docs": cfg.docs,
        # Para poder contestar desde afuera "esto corre el codigo que creo que corre".
        "arrancado": ARRANCADO,
        "commit": COMMIT,
        "modelo_pose": MODELO_POSE,
        "frontend": bool(cfg.web and (cfg.web / "index.html").is_file()),
        # Tipo de golpe v2: que metodo clasifica (BOXTWIN_CLASIFICADOR).
        "clasificador": cfg.clasificador,
        # Bolsa y sombra (B8): si la bandera esta encendida y con que modelo se clasifican.
        "sesiones_individuales": cfg.sesiones_individuales,
        "modelo_individual": (cfg.modelo_clasificador_geo_individual / "meta.json").is_file()
        if cfg.tipo_v2 else None,
        # Modo academico E7 (25-10): si esta el modelo de 42 features. Solo con e7.
        "modelo_e7": (cfg.modelo_clasificador_e7 / "meta.json").is_file()
        if cfg.clasificador == "e7" else None,
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
#
# Las cabeceras (05 B0): CSP para todo lo estatico, cache larga e inmutable para lo que Next
# nombra con hash (/_next/static/) y no-cache para los .html, que son los que cambian de un
# despliegue a otro sin cambiar de nombre. 'unsafe-inline' en estilos porque Motion escribe
# estilos en linea; las fuentes las sirve el build, asi que font-src queda en 'self'.
CSP_WEB = (
    "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; font-src 'self'; script-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
)


def _cabeceras_web(ruta: str, respuesta) -> None:
    respuesta.headers["Content-Security-Policy"] = CSP_WEB
    respuesta.headers["X-Content-Type-Options"] = "nosniff"
    if ruta.startswith("_next/static/"):
        respuesta.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif respuesta.headers.get("content-type", "").startswith("text/html"):
        respuesta.headers["Cache-Control"] = "no-cache"


# Versiones 26-10: /cuenta y el consentimiento por atleta. Antes del montaje de la web, que
# tapa todo lo que se registre despues.
from boxtwin_api.cuentas import router as _router_cuentas  # noqa: E402

app.include_router(_router_cuentas)

if cfg.web and (cfg.web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    class _Web(StaticFiles):
        async def get_response(self, path, scope):
            r = await super().get_response(path, scope)
            _cabeceras_web(path, r)
            return r

    app.mount("/", _Web(directory=str(cfg.web), html=True), name="web")
