"""
Pruebas de scripts/identidad (docs/specs/2026-11-02-identidad-estable). Sin GPU, sin videos y
sin anotaciones reales: todo sintetico y chico.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/identidad/tests -q
"""

import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parents[1]
REPO = AQUI.parents[1]
for ruta in (AQUI, REPO / "scripts" / "mejoras", REPO / "boxtwin-annotator" / "scripts",
             REPO / "boxtwin-annotator" / "src"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))
