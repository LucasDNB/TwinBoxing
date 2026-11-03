#!/usr/bin/env bash
# POR QUE EXISTE: el servidor de los e2e de BoxTwinZ, igual que el de la web entrenador. API
# real con SQLite sirviendo el build exportado y worker real con el motor falso. Todo en un
# directorio temporal que se borra al salir.
# USO: lo llama playwright.config.ts (webServer). A mano: bash tests/e2e/levantar.sh
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
WEB="$(cd "$AQUI/../.." && pwd)"
API="$(cd "$WEB/../api" && pwd)"
PY="${BOXTWINZ_PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
TMP="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMP"' EXIT

mkdir -p "$AQUI/.e2e"
[[ -f "$AQUI/.e2e/sesion.mp4" ]] || ffmpeg -loglevel error -f lavfi -i testsrc=duration=12:size=640x360:rate=30 -pix_fmt yuv420p "$AQUI/.e2e/sesion.mp4"

export PYTHONPATH="$API"
export BOXTWINZ_CONFIG="$TMP/no-existe.toml"
export BOXTWINZ_DB="sqlite:///$TMP/e2e.db"
export BOXTWINZ_DATOS="$TMP/datos"
export BOXTWINZ_SECRETO="secreto-de-los-e2e"
export BOXTWINZ_ADMINS="admin@boxtwinz.ar"
export BOXTWINZ_WEB="$WEB/out"
export BOXTWINZ_CMD="$PY -I $AQUI/boxtwinz_falso.py"
export BOXTWINZ_ESPERA="0.3"
export BOXTWINZ_LOCK_GPU="$TMP/gpu.lock"

"$PY" -m boxtwinz_api.worker &
exec "$PY" -m uvicorn boxtwinz_api.main:app --host 127.0.0.1 --port "${PUERTO:-8766}" --log-level warning
