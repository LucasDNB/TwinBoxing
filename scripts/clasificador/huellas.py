"""
BoxTwin - Huellas sha256 de las anotaciones originales.

POR QUE EXISTE
  La regla del repo es que nadie toca las anotaciones originales, y en esta campana las leen
  scripts que corren solos durante horas. Las reglas de permisos de Claude Code no alcanzan
  a un script de Python que abre un archivo por su cuenta, asi que hace falta una prueba que
  si alcance: la huella de cada archivo antes de empezar y la comprobacion despues de cada
  fase. Ademas el cargador del anotador puede migrar un archivo viejo al abrirlo; si eso
  pasara, aca se ve.

QUE HACE
  `tomar` escribe un JSON con el sha256 de cada anotacion de las siete fuentes (y de las
  reanotaciones que haya). `verificar` lo compara con el disco y sale con codigo 1 si algo
  cambio o falta.

USO
  python scripts/clasificador/huellas.py tomar
  python scripts/clasificador/huellas.py verificar
  python scripts/clasificador/huellas.py verificar --archivo otra/ruta/huellas.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import REPO, TRABAJO  # noqa: E402


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def archivos(repo: Path = REPO) -> list[Path]:
    return sorted(repo.glob("anotacion*/annotations/*.json"))


def tomar(destino: Path, repo: Path = REPO) -> dict:
    if destino.exists():
        raise SystemExit(f"ya existen huellas en {destino}; no se pisan (borralas a mano si es a proposito)")
    datos = {
        "tomadas_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archivos": {str(p.relative_to(repo)): _sha(p) for p in archivos(repo)},
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, indent=2) + "\n")
    return datos


def verificar(origen: Path, repo: Path = REPO) -> list[str]:
    datos = json.loads(origen.read_text())
    problemas = []
    for rel, sha in datos["archivos"].items():
        p = repo / rel
        if not p.is_file():
            problemas.append(f"falta {rel}")
        elif _sha(p) != sha:
            problemas.append(f"cambio {rel}")
    nuevos = {str(p.relative_to(repo)) for p in archivos(repo)} - set(datos["archivos"])
    problemas += [f"nuevo {r} (no estaba al tomar las huellas)" for r in sorted(nuevos)]
    return problemas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("accion", choices=["tomar", "verificar"])
    ap.add_argument("--archivo", type=Path, default=TRABAJO / "huellas.json")
    args = ap.parse_args(argv)
    if args.accion == "tomar":
        d = tomar(args.archivo)
        print(f"{len(d['archivos'])} anotaciones -> {args.archivo}")
        return 0
    problemas = verificar(args.archivo)
    if problemas:
        print("ANOTACIONES ALTERADAS: frenar la campana y avisar a Lucas")
        for p in problemas:
            print("  " + p)
        return 1
    print("anotaciones intactas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
