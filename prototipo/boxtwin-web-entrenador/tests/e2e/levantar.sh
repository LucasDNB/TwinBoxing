#!/usr/bin/env bash
# POR QUE EXISTE: el servidor de las pruebas de punta a punta (06 §4). Levanta la API real
# con SQLite sirviendo el build exportado, y el worker real con boxtwin_falso.py en lugar
# del anotador. Todo en un directorio temporal que se borra al salir.
# USO: lo llama playwright.config.ts (webServer). A mano: bash tests/e2e/levantar.sh
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
WEB="$(cd "$AQUI/../.." && pwd)"
REPO="$(cd "$WEB/../.." && pwd)"
PY="${BOXTWIN_PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
TMP="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMP"' EXIT

ffmpeg -loglevel error -f lavfi -i testsrc=duration=8:size=640x360:rate=30 -pix_fmt yuv420p "$TMP/sparring.mp4"
mkdir -p "$AQUI/.e2e" && cp "$TMP/sparring.mp4" "$AQUI/.e2e/sparring.mp4"

export PYTHONPATH="$REPO/boxtwin-api/src:$REPO/boxtwin-annotator/src"
export BOXTWIN_DB="sqlite:///$TMP/e2e.db"
export BOXTWIN_DATOS="$TMP/datos"
export BOXTWIN_SECRETO="secreto-de-las-pruebas-e2e"
export BOXTWIN_INVITACION="abierto"
export BOXTWIN_WEB="$WEB/out"
export BOXTWIN_CMD="$PY -I $AQUI/boxtwin_falso.py"
export BOXTWIN_ESPERA="0.3"

"$PY" -m boxtwin_api.worker &
exec "$PY" -m uvicorn boxtwin_api.app:app --host 127.0.0.1 --port "${PUERTO:-8765}" --log-level warning
