"""
BoxTwin - K-reuso: la mejora apagada da los mismos tensores que E8 (T12, 30-10-2026).

POR QUE EXISTE
  E8b, E10 y E12 no re-entrenan su base: comparan contra la condicion base de E8, ya auditada
  (docs/specs/2026-10-30-mejoras-post-e8/00-README.md, decision 2). Eso solo vale si el codigo
  nuevo, con la mejora apagada, arma exactamente los mismos tensores que armo E8. Si no, la
  diferencia que se mida puede venir del codigo y no de la mejora.

QUE HACE
  Recorre los .det.npz y los golpes.npz de dos directorios y compara, arreglo por arreglo, el
  sha256 de (dtype, forma, bytes). No compara el archivo: el zip guarda fechas, y dos npz con
  el mismo contenido escritos en otro momento difieren en bytes. Un arreglo de mas, uno de
  menos o un archivo que falta tambien es una diferencia. Escribe un JSON y sale con codigo 2
  si hay alguna.

USO
  python scripts/mejoras/kreuso.py ~/boxtwin-gemelo/e8/base ~/boxtwin-mejoras/kreuso/base \
    --out ~/boxtwin-mejoras/kreuso/base.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PATRONES = ("**/*.det.npz", "**/golpes.npz")


def huella(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def huellas(raiz: Path) -> dict[str, dict[str, str]]:
    """{ruta relativa: {clave: sha256}} de los tensores bajo `raiz`."""
    raiz = Path(raiz)
    salida = {}
    for patron in PATRONES:
        for p in sorted(raiz.glob(patron)):
            with np.load(p, allow_pickle=False) as z:
                salida[str(p.relative_to(raiz))] = {k: huella(z[k]) for k in z.files}
    return salida


def comparar(esperado: Path, nuevo: Path) -> dict:
    a, b = huellas(esperado), huellas(nuevo)
    dif = []
    for archivo in sorted(set(a) | set(b)):
        if archivo not in a or archivo not in b:
            dif.append({"archivo": archivo, "clave": "*",
                        "motivo": "falta en el nuevo" if archivo not in b else "sobra en el nuevo"})
            continue
        for clave in sorted(set(a[archivo]) | set(b[archivo])):
            ha, hb = a[archivo].get(clave), b[archivo].get(clave)
            if ha != hb:
                dif.append({"archivo": archivo, "clave": clave, "esperado": ha, "nuevo": hb})
    return {"esperado": str(esperado), "nuevo": str(nuevo), "archivos": len(set(a) | set(b)),
            "diferencias": dif, "pasa": not dif and bool(a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("esperado", type=Path, help="los tensores de E8")
    ap.add_argument("nuevo", type=Path, help="los del codigo nuevo con la mejora apagada")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    r = comparar(args.esperado, args.nuevo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"K-reuso {'pasa' if r['pasa'] else 'FALLA'}: {r['archivos']} archivos, "
          f"{len(r['diferencias'])} diferencias -> {args.out}")
    return 0 if r["pasa"] else 2


if __name__ == "__main__":
    sys.exit(main())
