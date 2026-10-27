"""
BoxTwin - Fija en un archivo chico los golpes anotados de las seis fuentes con identidad.

POR QUE EXISTE
  Las pruebas de B10 (combinaciones) y B11 (cabeza/cuerpo) tienen que poder correr sin las
  anotaciones originales, que pesan, viven fuera del paquete y no se tocan. Este script lee
  las seis `annot.json` una vez y escribe solo lo que esas pruebas necesitan: peleador, brazo,
  tiempos, altura anotada y si es finta o abortado. El archivo resultante se versiona y la
  prueba de concordancia (01 §3, prueba 6) corre contra el.

QUE HACE
  Para cada fuente: fps y la lista entera de golpes, con la marca de finta y de abortado.
  La tabla de combinaciones (01 §2) se calcula sin los abortados (962); la medicion de C8
  (03 §4) sobre todos (972), que es como quedaron pre-registradas. No escribe nada fuera
  de `--salida`.

USO
  python scripts/fijar_golpes_anotados.py --raiz ~/Proyectos/TwinBoxing \
      --salida tests/datos/golpes_anotados.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FUENTES = {
    "01-sparring": "anotacion-spar-01/annotations/01-sparring.annot.json",
    "02-sparring": "anotacion-spar-02/annotations/02-sparring.annot.json",
    "03-sparring": "anotacion-spar-03/annotations/03-sparring.annot.json",
    "04-sparring": "anotacion-spar-04/annotations/04-sparring.annot.json",
    "sparring-3": "anotacion-sparring-3/annotations/sparring-3-rounds.annot.json",
    "pacquiao_margarito": "anotacion-pacquiao/annotations/pacquiao_margarito.annot.json",
}


def fijar(raiz: Path) -> dict:
    salida = {}
    for nombre, rel in FUENTES.items():
        d = json.loads((raiz / rel).read_text())
        fps = float(d["video"]["fps"])
        golpes = []
        for e in sorted(d["events"], key=lambda e: (e["fighter"], e["start_frame"])):
            golpes.append({
                "id": e["id"],
                "peleador": "A" if e["fighter"] == "fighter_A" else "B",
                "brazo": e["side"],
                "cuadro_inicio": int(e["start_frame"]),
                "cuadro_fin": int(e["end_frame"]),
                "t_inicio": round(e["start_frame"] / fps, 4),
                "t_fin": round(e["end_frame"] / fps, 4),
                "objetivo": {"head": "cabeza", "body": "cuerpo"}.get(e.get("target")),
                "finta": e.get("completeness") == "feint",
                "abortado": e.get("completeness") == "aborted",
            })
        salida[nombre] = {"fps": fps, "golpes": golpes}
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--salida", type=Path,
                    default=Path(__file__).resolve().parents[1] / "tests/datos/golpes_anotados.json")
    a = ap.parse_args()
    datos = fijar(a.raiz)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{sum(len(f['golpes']) for f in datos.values())} golpes de {len(datos)} fuentes "
          f"-> {a.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
