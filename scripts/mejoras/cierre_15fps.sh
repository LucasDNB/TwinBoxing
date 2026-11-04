#!/usr/bin/env bash
# BoxTwin - E4 con la pose muestreada a 15 fps (cierre del 4-11-2026).
#
# POR QUE EXISTE
#   CLAUDE.md §6.2 pide la degradacion del clasificador con muestreo de pose. Regla en
#   docs/specs/2026-11-04-cierre/00-preregistro.md §2, escrita antes de medir.
#
# QUE HACE
#   Copia los datos de E4 de E8 base, arma golpes.npz con BOXTWIN_MUESTREO_FPS=15 y corre E4
#   completo como E10, con las colas vacias y el candado de GPU. No toca los datos de E8.
#
# USO
#   bash scripts/mejoras/cierre_15fps.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$HOME/miniforge3/envs/twinboxing_env/bin/python
E8=$HOME/boxtwin-gemelo/e8/base
RAIZ=$HOME/boxtwin-gemelo/raiz-v8
D=$HOME/boxtwin-mejoras/cierre/v8-15
GEO=$REPO/scripts/clasificador_geo
export PYTHONPATH=$REPO/boxtwin-annotator/src:$REPO/boxtwin-api/src:$REPO/boxtwin-detector/src
export BOXTWIN_CLASIFICADOR_GEO=$D/clasificador-geo BOXTWIN_RAIZ_FUENTES=$RAIZ
mkdir -p "$D/clasificador-geo"
cp -r --no-preserve=mode "$E8/clasificador-geo/datos" "$D/clasificador-geo/"
rm -f "$D/clasificador-geo/datos/golpes.npz" "$D/clasificador-geo/datos/golpes2d.meta.json"
BOXTWIN_MUESTREO_FPS=15 "$PY" "$GEO/features2d.py"
"$PY" -c "import sys; sys.path.insert(0, '$REPO/scripts/e13'); import comun_e13 as c; c.esperar_colas()"
flock /tmp/boxtwin-gpu.lock "$PY" "$GEO/entrenar.py" --exp E4 --dispositivo cuda
"$PY" "$GEO/evaluar.py" --exp E4
