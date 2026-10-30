"""
BoxTwin - Rutas comunes de la campana de mejoras posterior a E8 (30-10-2026).

POR QUE EXISTE
  Los scripts de scripts/mejoras/ leen los caches de E8 y escriben en ~/boxtwin-mejoras. Si
  cada uno fija sus rutas por su cuenta, el dia que difieren dos experimentos dejan de leer la
  misma base sin que nadie lo note (docs/specs/2026-10-30-mejoras-post-e8/00-README.md,
  "Donde vive cada cosa").

QUE HACE
  Rutas, el sha256 de un archivo y el acceso a scripts/gemelo y scripts/clasificador.

USO
  from rutas import MEJORAS, RAIZ_V8, RAIZ_Y26, sha256
  No corre solo.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
for _r in (REPO / "scripts" / "gemelo", REPO / "scripts" / "clasificador"):
    if str(_r) not in sys.path:
        sys.path.append(str(_r))

MEJORAS = Path(os.environ.get("BOXTWIN_MEJORAS", "~/boxtwin-mejoras")).expanduser()
GEMELO = Path(os.environ.get("BOXTWIN_GEMELO", "~/boxtwin-gemelo")).expanduser()
REPO_A = Path(os.environ.get("BOXTWIN_REPO_A", "~/Proyectos/TwinBoxing")).expanduser()
RAIZ_V8 = GEMELO / "raiz-v8"      # v8 sin ReID (D-H2), la base de E8. Solo lectura
RAIZ_Y26 = GEMELO / "raiz-y26"    # YOLO26 de E8. Solo lectura
E8 = GEMELO / "e8"
C1 = Path(os.environ.get("BOXTWIN_C1", "~/boxtwin/mediciones/c1")).expanduser()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()
