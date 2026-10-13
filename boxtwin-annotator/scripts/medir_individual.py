"""
BoxTwin - Medicion del criterio de B8: una persona asignada en bolsa y sombra.

POR QUE EXISTE
  El criterio de B8 decia "mirado a ojo por Lucas sobre el video procesado", que no es
  repetible. La ampliacion del 13-10 (02 §2) fija el procedimiento ANTES de mirar: se recorre
  el video procesado en saltos de 2 s y en cada salto se anota si la persona esta en cuadro y,
  si esta, si tiene la marca A. El porcentaje es saltos con marca sobre saltos con la persona
  en cuadro. Umbral: >= 95% en cada uno de los dos videos (uno de bolsa y uno de sombra, de al
  menos 3 minutos). Este script arma la planilla y saca el numero; mirar sigue siendo de Lucas.

QUE HACE
  planilla    escribe un CSV con un renglon cada 2 s (t, en_cuadro, marca_A) para llenar
  resultado   lee una o mas planillas llenas y da, por video, saltos, porcentaje e intervalo
              de Wilson al 95%, y si pasa el umbral. Un renglon sin llenar es un error: una
              planilla a medias no es una medicion.

USO
  python scripts/medir_individual.py planilla ~/boxtwin/datos/sesiones/<id> -o bolsa.csv
  python scripts/medir_individual.py resultado bolsa.csv sombra.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PASO_S = 2.0
UMBRAL = 0.95
DURACION_MINIMA_S = 180.0
SI = {"1", "si", "s", "x"}
NO = {"0", "no", "n", ""}


def saltos(duracion_s: float, paso: float = PASO_S) -> list[float]:
    n = int(math.floor(duracion_s / paso)) + 1
    return [round(i * paso, 2) for i in range(n) if i * paso <= duracion_s]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - medio), min(1.0, centro + medio))


def _si(v: str, campo: str, fila: int) -> bool:
    v = (v or "").strip().lower()
    if v in SI:
        return True
    if v in NO:
        return False
    raise ValueError(f"renglon {fila}: {campo}={v!r} no es si/no")


def resultado(ruta: Path) -> dict:
    with ruta.open() as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError(f"{ruta}: planilla vacia")
    en_cuadro = marcados = 0
    for i, r in enumerate(filas, start=2):
        if (r.get("en_cuadro") or "").strip() == "":
            raise ValueError(f"{ruta}, renglon {i}: falta en_cuadro (una planilla a medias no mide)")
        if _si(r["en_cuadro"], "en_cuadro", i):
            en_cuadro += 1
            if (r.get("marca_A") or "").strip() == "":
                raise ValueError(f"{ruta}, renglon {i}: la persona esta en cuadro y falta marca_A")
            marcados += _si(r["marca_A"], "marca_A", i)
    duracion = float(filas[-1]["t"])
    frac = marcados / en_cuadro if en_cuadro else 0.0
    lo, hi = wilson(marcados, en_cuadro)
    return {
        "planilla": ruta.name, "saltos": len(filas), "en_cuadro": en_cuadro, "con_marca": marcados,
        "fraccion": round(frac, 4), "wilson95": [round(lo, 4), round(hi, 4)],
        "duracion_s": duracion, "dura_lo_minimo": duracion >= DURACION_MINIMA_S,
        "pasa": frac >= UMBRAL and duracion >= DURACION_MINIMA_S,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("planilla", help="CSV con un renglon cada 2 s para llenar")
    pl.add_argument("sesion", type=Path, help="directorio de la sesion procesada")
    pl.add_argument("-o", "--salida", type=Path, required=True)
    rs = sub.add_parser("resultado", help="porcentaje por planilla llena")
    rs.add_argument("planillas", type=Path, nargs="+")
    a = ap.parse_args()

    if a.cmd == "planilla":
        if a.salida.exists():
            raise SystemExit(f"{a.salida} ya existe: no se pisa una planilla que quizas esta llena")
        ses = json.loads((a.sesion / "sesion.json").read_text())
        with a.salida.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "en_cuadro", "marca_A"])
            for t in saltos(float(ses["video"]["duracion_s"])):
                w.writerow([t, "", ""])
        print(f"{a.salida}: llenar en_cuadro y marca_A (si/no) mirando procesado.mp4")
        return 0

    resultados = [resultado(p) for p in a.planillas]
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    return 0 if all(r["pasa"] for r in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
