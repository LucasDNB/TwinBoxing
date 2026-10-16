"""
BoxTwin - Piezas comunes de E4 a E6 (clasificador geometrico, 3D y rol por brazo, 16-10-2026).

POR QUE EXISTE
  Los scripts de scripts/clasificador_geo/ comparten el directorio de trabajo, el largo de la
  ventana, la cantidad de pasos y los nombres de las configuraciones. Si cada uno los fija por
  su cuenta, el dia que difieren E4 y E5 dejan de ser comparables sin que nadie lo note. Las
  fuentes, las clases y las metricas NO se definen aca: se importan de
  scripts/clasificador/comun.py, que es la copia unica de la campana anterior.

QUE HACE
  Rutas, constantes pre-registradas (01-preregistro.md, seccion 2) y la importacion de comun.

USO
  from geo import TRABAJO, PAD, PASOS, comun
  No corre solo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "clasificador"))
import comun  # noqa: E402

# Directorio de trabajo de esta campana. Nada pesado entra a git.
TRABAJO = Path(os.environ.get("BOXTWIN_CLASIFICADOR_GEO", "~/boxtwin/clasificador-geo-2026-10")).expanduser()
# Datos de la campana anterior (folds de E1, pred.pkl de E3, huellas). Solo lectura.
TRABAJO_E1E3 = comun.TRABAJO

# Pre-registro, seccion 2, punto 1: del inicio menos 8 al fin mas 8, remuestreado a 32 pasos.
PAD = 8
PASOS = 32

# Pre-registro, seccion 2, punto 4.
SEMILLAS = (0, 1, 2, 3, 4)

# Golpes completos por fuente (pre-registro, datos de base). Si no coinciden, D3.
GOLPES_POR_FUENTE = {
    "Sparring": 114, "01-sparring": 96, "02-sparring": 62, "03-sparring": 129,
    "04-sparring": 120, "sparring-3-rounds": 381, "pacquiao_margarito": 145,
}
TOTAL_GOLPES = 1047

TIPOS = comun.TIPOS
ALTURAS = comun.ALTURAS
ROLES = ("lead", "rear")
FUENTES = list(comun.FUENTES)
FUENTES_C8 = comun.FUENTES_C8
