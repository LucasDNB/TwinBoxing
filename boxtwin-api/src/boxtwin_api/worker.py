"""
BoxTwin API - El worker: toma trabajos de la cola y corre las etapas.

POR QUE LLAMA A SUBPROCESOS Y NO IMPORTA EL PIPELINE
  Por los dos entornos conda, que no se mezclan. El detector vive en twinboxing_env con
  torch 2.6 y el clasificador en boxtwin_mmaction con torch 2.1 y mmcv pinneado contra
  CUDA 11.8. Un proceso no puede tener los dos. Entre ellos va un JSON, que es como ya
  funciona el pipeline medido.

  El efecto colateral es bueno: una etapa que se cae por memoria de GPU se lleva su
  proceso y no el servidor, y la GPU queda liberada de verdad, que con torch adentro del
  proceso de la API no siempre pasa.

QUE HACE
  Un bucle: reclama, corre, marca. Corre de a un trabajo por vez a proposito, porque hay
  una sola GPU y dos procesos de pose compitiendo por 8 GB de VRAM terminan los dos mas
  lento que uno solo.

USO
  python -m boxtwin_api.worker
  python -m boxtwin_api.worker --una-vuelta      # para probar sin quedarse colgado
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from boxtwin_api.cola import fallar, reclamar, terminar
from boxtwin_api.config import cfg
from boxtwin_api.db import crear_tablas, hacer_sesion
from boxtwin_api.modelos import Sesion, Trabajo

__all__ = ["correr", "ejecutar_trabajo", "comando_de"]

log = logging.getLogger("boxtwin.worker")

_seguir = True


def _parar(_signo, _marco):  # pragma: no cover - se prueba a mano
    global _seguir
    _seguir = False
    log.info("recibida la senal de parada: termino el trabajo en curso y salgo")


def comando_de(trabajo: Trabajo, sesion: Sesion) -> list[list[str]]:
    """
    Los comandos de una etapa. Devuelve una lista porque clasificar son dos.

    Vive aparte del bucle para poder verificarlo sin correr nada: que la etapa de
    clasificacion arme mal su linea de comando es un error que en produccion aparece
    recien despues de media hora de pose.
    """
    directorio = cfg.dir_sesion(sesion.id)
    p = trabajo.parametros or {}

    if trabajo.etapa == "procesar":
        cmd = [
            *cfg.cmd_boxtwin, "procesar",
            str(directorio / "videos" / sesion.video_nombre),
            "--out", str(directorio),
            "--modelo-guantes", str(cfg.modelo_guantes),
            "--modelo-pose", str(cfg.modelo_pose),
            "--device", cfg.device,
        ]
        if p.get("round_s"):
            cmd += ["--round", str(p["round_s"]), "--descanso", str(p.get("descanso_s", 60))]
        return [cmd]

    if trabajo.etapa == "completar":
        return [[
            *cfg.cmd_boxtwin, "completar", str(directorio),
            "--semilla-a", str(p["track_a"]),
            "--semilla-b", str(p["track_b"]),
            "--detector", str(cfg.modelo_detector),
            "--umbral", str(cfg.umbral_detector),
        ]]

    if trabajo.etapa == "clasificar":
        if not cfg.cmd_clasificador:
            raise RuntimeError(
                "no hay comando de clasificacion configurado (BOXTWIN_CMD_CLASIFICADOR). "
                "El tipo de golpe queda sin estimar y el resto de la Fight-Card no depende "
                "de el"
            )
        return [
            [
                *cfg.cmd_clasificador, str(directorio),
                "--config", str(cfg.config_clasificador),
                "--checkpoint", str(cfg.checkpoint_clasificador),
            ],
            [*cfg.cmd_boxtwin, "tipos", str(directorio)],
        ]

    raise RuntimeError(f"etapa desconocida: {trabajo.etapa}")


def ejecutar_trabajo(db, trabajo: Trabajo, timeout: float | None = None) -> str:
    """
    Corre la etapa y devuelve el estado en que queda la sesion.

    El estado no se adivina desde el codigo de salida: lo escribe el comando en sesion.json
    y se lee de ahi. Un proceso que termina en 0 y dejo la sesion a medias es un caso real
    -por ejemplo, evidencia reutilizada y candidatos vacios- y creerle al codigo de salida
    lo taparia.
    """
    ses = db.get(Sesion, trabajo.sesion_id)
    if ses is None:
        raise RuntimeError("el trabajo apunta a una sesion que ya no existe")

    ses.estado = {"procesar": "procesando", "completar": "completando"}.get(
        trabajo.etapa, ses.estado
    )
    db.commit()

    for cmd in comando_de(trabajo, ses):
        log.info("corriendo: %s", " ".join(cmd))
        r = subprocess.run(  # noqa: S603 - la linea la arma comando_de, no el usuario
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if r.returncode != 0:
            cola = (r.stderr or r.stdout or "").strip().splitlines()[-25:]
            raise RuntimeError(
                f"{trabajo.etapa} fallo con codigo {r.returncode}:\n" + "\n".join(cola)
            )
        log.debug("%s", (r.stdout or "")[-2000:])

    return _estado_en_disco(cfg.dir_sesion(ses.id)) or "listo"


def _estado_en_disco(directorio: Path) -> str | None:
    import json

    ruta = directorio / "sesion.json"
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text()).get("estado")
    except json.JSONDecodeError:
        return None


def correr(una_vuelta: bool = False, worker: str | None = None) -> int:
    """
    El bucle. Devuelve cuantos trabajos proceso.

    Con `una_vuelta` toma como mucho uno y sale, que es como se lo prueba y como se lo
    corre desde un cron si alguna vez hace falta.
    """
    crear_tablas()
    nombre = worker or f"{socket.gethostname()}:{os.getpid()}"
    hechos = 0
    while _seguir:
        with hacer_sesion() as db:
            trabajo = reclamar(db, nombre)
            if trabajo is None:
                if una_vuelta:
                    return hechos
                time.sleep(cfg.espera_s)
                continue
            try:
                estado = ejecutar_trabajo(db, trabajo)
            except Exception as e:  # noqa: BLE001 - el worker no se puede caer por un video
                log.exception("el trabajo %s fallo", trabajo.id)
                fallar(db, trabajo, str(e))
            else:
                terminar(db, trabajo, estado_sesion=estado)
                # Encadenar la clasificacion aca y no en la API: recien cuando existe
                # segmentos.json hay algo que clasificar, y eso lo sabe el worker.
                if trabajo.etapa == "completar" and estado == "listo" and cfg.cmd_clasificador:
                    from boxtwin_api.cola import encolar

                    encolar(db, trabajo.sesion_id, "clasificar")
                    db.commit()
            hechos += 1
        if una_vuelta:
            return hechos
    return hechos


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--una-vuelta", action="store_true", dest="una_vuelta")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _parar)
    signal.signal(signal.SIGINT, _parar)
    n = correr(una_vuelta=args.una_vuelta)
    log.info("%d trabajo(s) procesado(s)", n)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
