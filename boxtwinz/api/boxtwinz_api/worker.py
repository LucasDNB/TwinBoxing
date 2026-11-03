"""
BoxTwinZ API - El worker: toma trabajos de la cola y corre el motor por subproceso.

POR QUE EXISTE
  Igual que el worker de A (boxtwin-api/worker.py), del que se copia la invocacion: el motor
  vive en twinboxing_env y se llama por subproceso con un JSON en disco entre etapas. Una etapa
  que se cae por memoria de GPU se lleva su proceso, no el servidor.

  Diferencias con A, todas de la spec:
  - Sin clasificar ni render: BoxTwinZ no tiene tipo de golpe y no muestra video procesado.
  - Sombra y bolsa se siembran solas con el candidato del motor (desvio D3).
  - Lock de archivo sobre la GPU (`/tmp/boxtwin-gpu.lock`) mientras corre el motor. A no lo
    conoce, asi que solo serializa a BoxTwinZ consigo mismo y con lo que respete el lock: el
    riesgo de que A y BoxTwinZ coincidan en la GPU sigue (TAREAS.md, riesgo R1).
  - Registra inicio_gpu y fin_gpu por trabajo y los suma en uso_gpu_diario.

QUE HACE
  Un bucle: reclama, corre, ingiere, marca. De a un trabajo por vez. Ademas, una vez por dia
  despues de las 00:05, calcula la utilizacion del dia anterior y evalua los disparadores.

USO
  python -m boxtwinz_api.worker
  python -m boxtwinz_api.worker --una-vuelta
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boxtwinz_api.cola import encolar, fallar, reclamar, terminar
from boxtwinz_api.config import cfg
from boxtwinz_api.db import crear_tablas, hacer_sesion
from boxtwinz_api.models import Sesion, Trabajo, ahora
from boxtwinz_api.services.engine_adapter import candidato_individual, leer_sesion_json

__all__ = ["comando_de", "ejecutar_trabajo", "correr", "lock_gpu"]

log = logging.getLogger("boxtwinz.worker")
_seguir = True


def _parar(_signo, _marco):  # pragma: no cover
    global _seguir
    _seguir = False


def comando_de(trabajo: Trabajo, sesion: Sesion) -> list[str]:
    """
    La linea de comando de una etapa, armada igual que comando_de() en el worker de A.

    procesar sin --round: los rounds declarados de BoxTwinZ son una cantidad, no una duracion,
    y los segmentos los calcula metrics.py (spec 6.2). completar sin --semilla-b es la sesion
    de un solo peleador (B8 de A).
    """
    directorio = cfg.dir_trabajo(sesion.id)
    p = trabajo.parametros or {}
    if trabajo.etapa == "procesar":
        return [
            *cfg.cmd_boxtwin, "procesar", sesion.video_ruta,
            "--out", str(directorio),
            "--modelo-guantes", str(cfg.modelo_guantes),
            "--modelo-pose", str(cfg.modelo_pose),
            "--device", cfg.device,
        ]
    if trabajo.etapa == "completar":
        return [
            *cfg.cmd_boxtwin, "completar", str(directorio),
            "--semilla-a", str(p["track_a"]),
            *(["--semilla-b", str(p["track_b"])] if p.get("track_b") is not None else []),
            "--detector", str(cfg.modelo_detector),
            "--umbral", str(cfg.umbral_detector),
            *(["--cortar-mixtos"] if cfg.corte_mixtos else []),
            *(["--estricta"] if cfg.identidad_estricta else []),
            *(["--apariencia"] if cfg.identidad_apariencia else []),
        ]
    raise RuntimeError(f"etapa desconocida: {trabajo.etapa}")


@contextlib.contextmanager
def lock_gpu(ruta: Path | None = None):
    """Lock exclusivo de archivo mientras el motor usa la GPU. Bloquea hasta conseguirlo."""
    ruta = ruta or cfg.lock_gpu
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _sumar_gpu(db, inicio: datetime, fin: datetime) -> float:
    """Suma los segundos al dia (o los dias, si cruza medianoche) en uso_gpu_diario."""
    from boxtwinz_api.services.capacity import sumar_segundos

    return sumar_segundos(db, inicio, fin)


def _correr_motor(cmd: list[str], timeout: float | None) -> subprocess.CompletedProcess:
    log.info("corriendo: %s", " ".join(cmd))
    return subprocess.run(  # noqa: S603 - la linea la arma comando_de, no el usuario
        cmd, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def ejecutar_trabajo(db, trabajo: Trabajo, timeout: float | None = None, correr_motor=None) -> str:
    """
    Corre la etapa y devuelve el estado en que queda la sesion. Como en A, el estado del motor
    se lee de sesion.json y no del codigo de salida. `correr_motor` existe para los tests.
    """
    correr_motor = correr_motor or _correr_motor
    ses = db.get(Sesion, trabajo.sesion_id)
    if ses is None:
        raise RuntimeError("el trabajo apunta a una sesion que ya no existe")
    ses.estado = "procesando" if trabajo.etapa == "procesar" else "completando"
    db.commit()

    cmd = comando_de(trabajo, ses)
    cfg.dir_trabajo(ses.id).mkdir(parents=True, exist_ok=True)
    with lock_gpu():
        trabajo.inicio_gpu = ahora()
        db.commit()
        try:
            r = correr_motor(cmd, timeout)
        finally:
            trabajo.fin_gpu = ahora()
            segundos = _sumar_gpu(db, trabajo.inicio_gpu, trabajo.fin_gpu)
            ses.segundos_gpu = (ses.segundos_gpu or 0.0) + segundos
            db.commit()
    trabajo.log = ((r.stdout or "")[-3000:] + "\n" + (r.stderr or "")[-3000:]).strip()
    if r.returncode != 0:
        cola = (r.stderr or r.stdout or "").strip().splitlines()[-25:]
        raise RuntimeError(f"{trabajo.etapa} fallo con codigo {r.returncode}:\n" + "\n".join(cola))

    directorio = cfg.dir_trabajo(ses.id)
    disco = leer_sesion_json(directorio)
    if trabajo.etapa == "procesar":
        v = disco.get("video") or {}
        ses.video_duracion_s = v.get("duracion_s", ses.video_duracion_s)
        ses.video_fps = v.get("fps", ses.video_fps)
        ses.video_ancho = v.get("ancho", ses.video_ancho)
        ses.video_alto = v.get("alto", ses.video_alto)
        if disco.get("estado") not in ("espera_siembra", "listo"):
            raise RuntimeError(f"procesar termino con la sesion en {disco.get('estado')!r}: "
                               f"{disco.get('error') or 'sin detalle'}")
        if ses.modalidad == "sparring":
            return "esperando_siembra"
        semilla = candidato_individual(disco)
        if semilla is None:
            raise RuntimeError("el motor no encontro ninguna persona en el video")
        ses.semilla_usuario = semilla
        encolar(db, ses.id, "completar", track_a=semilla, track_b=None)
        return "completando"

    if disco.get("estado") != "listo":
        raise RuntimeError(f"completar termino con la sesion en {disco.get('estado')!r}: "
                           f"{disco.get('error') or 'sin detalle'}")
    from boxtwinz_api.services.ingest import ingerir

    ingerir(db, ses, directorio)
    return "lista"


def _tarea_diaria(ultimo: datetime | None) -> datetime | None:
    """Despues de las 00:05 locales, una vez por dia: utilizacion de ayer y disparadores."""
    hoy = datetime.now().astimezone()
    if hoy.hour == 0 and hoy.minute < 5:
        return ultimo
    if ultimo is not None and ultimo.date() == hoy.date():
        return ultimo
    from boxtwinz_api.services.capacity import cerrar_dia

    with hacer_sesion() as db:
        cerrar_dia(db, (hoy - timedelta(days=1)).date())
        db.commit()
    return hoy


def correr(una_vuelta: bool = False, worker: str | None = None) -> int:
    crear_tablas()
    nombre = worker or f"{socket.gethostname()}:{os.getpid()}"
    hechos = 0
    ultimo_dia: datetime | None = None
    while _seguir:
        try:
            ultimo_dia = _tarea_diaria(ultimo_dia)
        except Exception:  # noqa: BLE001 - el calculo diario no tira el worker
            log.exception("fallo el calculo diario de capacidad")
        with hacer_sesion() as db:
            trabajo = reclamar(db, nombre)
            if trabajo is not None:
                try:
                    estado = ejecutar_trabajo(db, trabajo)
                except Exception as e:  # noqa: BLE001 - no se cae por un video
                    log.exception("el trabajo %s fallo", trabajo.id)
                    db.rollback()
                    fallar(db, trabajo, str(e))
                else:
                    terminar(db, trabajo, estado_sesion=estado)
                hechos += 1
        if una_vuelta:
            return hechos
        if trabajo is None:
            time.sleep(cfg.espera_s)
    return hechos


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--una-vuelta", action="store_true", dest="una_vuelta")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _parar)
    signal.signal(signal.SIGINT, _parar)
    log.info("%d trabajo(s) procesado(s)", correr(una_vuelta=args.una_vuelta))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
