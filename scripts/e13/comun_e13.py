"""
BoxTwin - E13: rutas, colas y reglas sin GPU (3-11-2026).

POR QUE EXISTE
  La regla de P18a y el control H13-2 estan fijados en
  docs/specs/2026-10-30-mejoras-post-e8/06-e13-tracker.md §3 y §4. Separados del tracker se pueden
  probar, y un numero para la tesis no puede quedar sin test.

QUE HACE
  Rutas, lectura de colas de A y BoxTwinZ, la numeracion de ids a traves de cortes (igual que
  writer.cut_seam), la comparacion exacta de H13-2 y la regla de P18a.

USO
  import comun_e13 as c
  No corre solo.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

E13 = Path(os.environ.get("BOXTWIN_E13", "~/boxtwin-mejoras/e13")).expanduser()
DET = E13 / "det"
RAIZ_V8 = Path("~/boxtwin-gemelo/raiz-v8").expanduser()
DB_A = Path("~/boxtwin/datos/boxtwin.db").expanduser()
DB_Z = Path("~/boxtwinz-datos/boxtwinz.db").expanduser()
CANDADO = Path("/tmp/boxtwin-gpu.lock")
FUENTES = {"Sparring": "anotacion", "01-sparring": "anotacion-spar-01", "02-sparring": "anotacion-spar-02",
           "03-sparring": "anotacion-spar-03", "04-sparring": "anotacion-spar-04",
           "sparring-3-rounds": "anotacion-sparring-3", "pacquiao_margarito": "anotacion-pacquiao"}
ORDEN = list(FUENTES)                  # gimnasio primero; la transmision, la mas larga, al final
SEIS = [f for f in FUENTES if f != "pacquiao_margarito"]
BUFFER_S = 4.0
BAJA_MIXTOS = 0.30
SUBA_CAMBIOS = 0.20


def log(m: str) -> None:
    E13.mkdir(parents=True, exist_ok=True)
    linea = f"{datetime.now().isoformat(timespec='seconds')} {m}"
    print(linea, flush=True)
    with (E13 / "e13.log").open("a") as f:
        f.write(linea + "\n")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_de(fuente: str) -> Path:
    return RAIZ_V8 / FUENTES[fuente] / "cache" / f"{fuente}.pose.npz"


def meta_de(fuente: str) -> dict:
    return json.loads((RAIZ_V8 / FUENTES[fuente] / "cache" / f"{fuente}.meta.json").read_text())


def video_de(fuente: str) -> Path:
    return RAIZ_V8 / FUENTES[fuente] / "videos" / f"{fuente}.mp4"


def _en_curso(db: Path) -> int:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return int(con.execute("select count(*) from trabajos where estado in ('en_cola','tomado')").fetchone()[0])
    finally:
        con.close()


def esperar_colas() -> None:
    """Se llama con el candado suelto: BoxTwinZ puede tomarlo mientras se espera."""
    avisado = False
    while _en_curso(DB_A) or _en_curso(DB_Z):
        if not avisado:
            log("hay trabajos en cola: espero")
            avisado = True
        time.sleep(30)


class Ids:
    """Los ids como los escribe writer: en un corte, offset = mayor id visto y el tracker arranca de cero."""

    def __init__(self):
        self.offset = 0
        self.maximo = 0

    def corte(self) -> None:
        self.offset = self.maximo

    def global_(self, local: np.ndarray) -> np.ndarray:
        g = np.asarray(local, np.int64) + self.offset
        if g.size:
            self.maximo = max(self.maximo, int(g.max()))
        return g


def comparar_ids(off_a, ids_a, off_b, ids_b) -> dict:
    """H13-2: fraccion de cuadros con la misma lista de track_id, en el mismo orden."""
    n = min(len(off_a), len(off_b)) - 1
    distintos, primero = 0, None
    for f in range(n):
        if not np.array_equal(ids_a[off_a[f]:off_a[f + 1]], ids_b[off_b[f]:off_b[f + 1]]):
            distintos += 1
            primero = f if primero is None else primero
    return {"cuadros": n, "distintos": distintos, "primer_distinto": primero,
            "fraccion": (n - distintos) / n if n else None,
            "mismo_largo": len(off_a) == len(off_b)}


def cambios_por_minuto(total_cond: dict) -> float | None:
    """A y B juntos: cambios sobre minutos, ya sumados por fuente con su fps."""
    cambios = sum(total_cond[p]["cambios"] for p in ("A", "B"))
    minutos = sum(total_cond[p]["cambios"] / total_cond[p]["cambios_por_minuto"]
                  for p in ("A", "B") if total_cond[p]["cambios_por_minuto"])
    return cambios / minutos if minutos else None


def regla_p18a(mixtos_base: int, mixtos_t: int, cpm_base: float, cpm_t: float) -> dict:
    baja = (mixtos_base - mixtos_t) / mixtos_base if mixtos_base else None
    sube = (cpm_t - cpm_base) / cpm_base if cpm_base else None
    return {"baja_mixtos": baja, "sube_cambios": sube,
            "mejora": baja is not None and sube is not None and baja >= BAJA_MIXTOS and sube <= SUBA_CAMBIOS}


def _nvidia_pids() -> list[int]:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
