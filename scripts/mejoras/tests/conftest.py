"""
Pruebas de scripts/mejoras (docs/specs/2026-10-30-mejoras-post-e8). Corren sin GPU, sin videos y
sin las anotaciones reales: todo es sintetico y chico.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/mejoras/tests -q
"""

import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parents[1]
REPO = AQUI.parents[1]
for ruta in (AQUI, REPO / "scripts" / "gemelo", REPO / "boxtwin-annotator" / "src",
             REPO / "boxtwin-detector" / "src"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))
