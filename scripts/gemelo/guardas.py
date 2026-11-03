"""
BoxTwin - Guardas de ruta y candado de GPU del gemelo YOLO26 (27-10-2026).

POR QUE EXISTE
  El preproceso ya piso una anotacion dos veces el 19-08 armando pruebas aisladas: el camino
  parecia aislado y un enlace llevaba al proyecto real. La fase 1 de E8 escribe catorce
  caches, siete anotaciones del puente y los exports de las dos condiciones, todo al lado de
  material de A que solo se puede leer (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md
  §4). Y comparte la 2080 con A: una etapa larga no puede arrancar con trabajos de A en cola
  (04-convivencia.md, regla 3). Si cada script chequea esto a su manera, el dia que uno se lo
  olvida no se entera nadie.

QUE HACE
  - exigir_bajo_gemelo: la ruta, sin seguir enlaces y siguiendolos, cae bajo ~/boxtwin-gemelo.
  - exigir_fuera_del_repo: la ruta, siguiendo enlaces, no cae en el checkout ni en la copia
    principal de A.
  - exigir_out: un export sin --out, o con --out en el repo, se niega.
  - exigir_annot_del_puente: el destino no tiene un annot.json que no escribio el puente.
  - esperar_candado: espera mientras la base de A tenga trabajos en_cola o tomado (lectura
    sola) o nvidia-smi muestre otro proceso, y deja cada espera y cada arranque en el log.

USO
  from guardas import exigir_bajo_gemelo, esperar_candado
  python scripts/gemelo/guardas.py candado "S7 recache"     # lo llama e8.sh antes de cada etapa
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPO_A = Path(os.environ.get("BOXTWIN_REPO_A", "~/Proyectos/TwinBoxing")).expanduser()
GEMELO = Path(os.environ.get("BOXTWIN_GEMELO", "~/boxtwin-gemelo")).expanduser()
DB_A = Path(os.environ.get("BOXTWIN_DB_A", "~/boxtwin/datos/boxtwin.db")).expanduser()
CANDADO_LOG = GEMELO / "e8" / "candado.log"
ANOTADOR_PUENTE = "puente-e8"
ESPERA_S = 60


class GuardaError(RuntimeError):
    """Una escritura que podria caer sobre A, o sobre algo que no escribio el gemelo."""


def _bajo(ruta: Path, raiz: Path) -> bool:
    return Path(ruta).is_relative_to(raiz)


def exigir_bajo_gemelo(ruta: Path) -> Path:
    """La ruta cae bajo el gemelo tal como esta escrita y tambien despues de seguir enlaces."""
    ruta = Path(os.path.abspath(Path(ruta).expanduser()))
    raiz = Path(os.path.abspath(GEMELO))
    if not _bajo(ruta, raiz) or not _bajo(Path(os.path.realpath(ruta)), Path(os.path.realpath(raiz))):
        raise GuardaError(f"{ruta} no esta bajo {GEMELO} (o un enlace la saca de ahi)")
    return ruta


def exigir_fuera_del_repo(ruta: Path) -> Path:
    ruta = Path(os.path.abspath(Path(ruta).expanduser()))
    real = Path(os.path.realpath(ruta))
    for repo in (REPO, REPO_A):
        for r in (Path(os.path.abspath(repo)), Path(os.path.realpath(repo))):
            if _bajo(ruta, r) or _bajo(real, r):
                raise GuardaError(f"{ruta} cae dentro de {repo}: el gemelo no escribe en el repo")
    return ruta


def exigir_out(out: Path | None) -> Path:
    """Un export del gemelo va siempre a un --out explicito, bajo el gemelo y fuera del repo."""
    if out is None:
        raise GuardaError("export sin --out: escribiria en exports/ del proyecto que infiere")
    return exigir_fuera_del_repo(exigir_bajo_gemelo(out))


def exigir_annot_del_puente(destino: Path) -> None:
    """Si ya hay un annot.json en el destino, tiene que ser del puente; si no, no se pisa."""
    destino = Path(destino)
    if not destino.is_file():
        return
    try:
        asg = json.loads(destino.read_text())["identity"]["assignments"]
    except (ValueError, KeyError, TypeError) as e:
        raise GuardaError(f"{destino} existe y no se puede leer como anotacion: {e}") from e
    if not asg or any(a.get("origin", {}).get("annotator") != ANOTADOR_PUENTE for a in asg):
        raise GuardaError(f"{destino} ya tiene una anotacion que no escribio el puente")


# ---------------------------------------------------------------------------
# Candado
# ---------------------------------------------------------------------------


def trabajos_de_a(db: Path = DB_A) -> int:
    """Trabajos de A en_cola o tomado. Abre la base en solo lectura, como la huella."""
    con = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True)
    try:
        return int(con.execute(
            "select count(*) from trabajos where estado in ('en_cola', 'tomado')").fetchone()[0])
    finally:
        con.close()


def salida_nvidia_smi() -> str:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    return r.stdout


def pids_en_gpu(salida: str) -> list[int]:
    return [int(x) for x in salida.split() if x.strip().isdigit()]


def _anotar(log: Path, linea: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {linea}\n")


def esperar_candado(etapa: str, db: Path = DB_A, log: Path = CANDADO_LOG, gpu=salida_nvidia_smi,
                    dormir=time.sleep, espera_s: int = ESPERA_S) -> int:
    """
    Vuelve cuando A no tiene trabajos pendientes y la GPU no tiene otros procesos.
    Devuelve cuantas veces espero. Cada espera, y el arranque, quedan en el log (CG8).
    """
    esperas = 0
    while True:
        n, pids = trabajos_de_a(db), pids_en_gpu(gpu())
        if n == 0 and not pids:
            _anotar(log, f"libre etapa='{etapa}' esperas={esperas}")
            return esperas
        _anotar(log, f"espera etapa='{etapa}' trabajos_a={n} gpu={pids}")
        esperas += 1
        dormir(espera_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candado", help="espera a que A y la GPU esten libres")
    c.add_argument("etapa")
    c.add_argument("--db", type=Path, default=DB_A)
    c.add_argument("--log", type=Path, default=CANDADO_LOG)
    args = ap.parse_args(argv)
    esperar_candado(args.etapa, args.db, exigir_bajo_gemelo(args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
