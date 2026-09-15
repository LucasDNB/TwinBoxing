#!/usr/bin/env python3
"""
BoxTwin - Muestreo de potencia de la GPU durante una corrida de medicion.

POR QUE EXISTE
  El Capitulo 5 declara 250 W por instancia (supuesto S-17) sin ninguna medicion detras.
  Para reemplazar ese numero hace falta la potencia real de la placa durante el
  procesamiento, muestreada a intervalo fijo y con marca de tiempo alineable con las
  marcas por etapa de medicion_cadena.py.

  Se muestrea en un proceso aparte a proposito: meter el muestreo adentro del bucle de
  inferencia agregaria una llamada a nvidia-smi por cuadro, que cuesta decenas de ms y
  contaminaria justamente el tiempo que se quiere medir.

  ADVERTENCIA MEDIDA EL 15-09-2026: en esta maquina nvidia-smi no arranca. El modulo del
  kernel quedo en 595.84 y la biblioteca NVML en 595.91 despues de una actualizacion de
  driver sin reiniciar. CUDA computa igual -torch ve la placa y corre-, pero NVML es la
  unica via de telemetria y esta caida. Este script lo detecta y aborta con el motivo, en
  vez de escribir un CSV vacio que despues parezca una medicion.

QUE HACE
  Llama a nvidia-smi a intervalo fijo y vuelca potencia, utilizacion y memoria a un CSV con
  marca de tiempo en el mismo reloj que usa el arnes de etapas (time.time(), epoch UTC).
  Termina con SIGTERM o SIGINT, cerrando el archivo.

USO
  python ../scripts/medicion_potencia.py --out mediciones/potencia.csv --intervalo 0.2
  python ../scripts/medicion_potencia.py --verificar        # solo chequea que NVML responda
"""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from pathlib import Path

CAMPOS = ["power.draw", "utilization.gpu", "memory.used"]
QUERY = ",".join(CAMPOS)

_seguir = True


def _parar(signum, frame):  # noqa: ARG001
    global _seguir
    _seguir = False


def disponible() -> tuple[bool, str]:
    """Dice si nvidia-smi responde, y si no, por que. No lanza excepcion."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        return False, "no hay nvidia-smi en el PATH"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi no respondio en 10 s"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout) else "codigo != 0"
    return True, r.stdout.strip()


def muestrear(out: Path, intervalo: float, etiqueta: str) -> int:
    ok, detalle = disponible()
    if not ok:
        print(f"error: la telemetria de la GPU no responde -> {detalle}", file=sys.stderr)
        print("  S-17 (potencia) no se puede medir en este estado de la maquina.", file=sys.stderr)
        return 2

    signal.signal(signal.SIGTERM, _parar)
    signal.signal(signal.SIGINT, _parar)

    out.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not out.is_file()
    # append: la campana son muchas corridas y todas caen en el mismo CSV crudo
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["epoch_s", "iso", "etiqueta", "potencia_w", "util_gpu_pct", "mem_usada_mib"])
        n = 0
        while _seguir:
            t = time.time()
            try:
                r = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode != 0:
                    time.sleep(intervalo)
                    continue
                pot, util, mem = [x.strip() for x in r.stdout.strip().split(",")]
            except (subprocess.TimeoutExpired, ValueError):
                time.sleep(intervalo)
                continue
            w.writerow([f"{t:.3f}", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)),
                        etiqueta, pot, util, mem])
            fh.flush()
            n += 1
            # se descuenta lo que tardo la propia llamada para no derivar el intervalo
            resto = intervalo - (time.time() - t)
            if resto > 0:
                time.sleep(resto)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--out", type=Path, default=Path("mediciones/potencia.csv"))
    p.add_argument("--intervalo", type=float, default=0.2, help="segundos entre muestras")
    p.add_argument("--etiqueta", default="", help="que corrida es, para poder filtrar despues")
    p.add_argument("--verificar", action="store_true", help="solo chequear que NVML responda")
    a = p.parse_args()

    if a.verificar:
        ok, detalle = disponible()
        print(("OK: " if ok else "CAIDO: ") + detalle)
        return 0 if ok else 2

    return muestrear(a.out, a.intervalo, a.etiqueta)


if __name__ == "__main__":
    raise SystemExit(main())
