#!/usr/bin/env bash
# CI local de BoxTwinZ (Z1): pytest de la API, tsc, lint y build exportado de la web.
# USO: bash boxtwinz/ci.sh   (desde cualquier directorio); CI_E2E=1 suma los e2e
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
PY="${BOXTWINZ_PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
echo "== pytest (api)"; (cd "$AQUI/api" && "$PY" -m pytest -q -p no:warnings)
echo "== tsc";          (cd "$AQUI/web" && npx tsc --noEmit)
echo "== lint";         (cd "$AQUI/web" && npx eslint .)
echo "== vitest";       (cd "$AQUI/web" && npx vitest run)
echo "== build";        (cd "$AQUI/web" && npm run build >/dev/null && test -f out/index.html)
if [[ "${CI_E2E:-}" == "1" ]]; then
  echo "== e2e (Playwright, Chromium y WebKit)"; (cd "$AQUI/web" && npx playwright test)
fi
echo "CI local en verde"
