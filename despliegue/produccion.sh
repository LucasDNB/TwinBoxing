#!/usr/bin/env bash
# POR QUE EXISTE
#   La rama `produccion` despliega dos instancias en la estacion: A (BoxTwin con las versiones
#   Individual, Entrenador y Gimnasio, publica por el Funnel) y B (la misma base con el modo
#   academico E7, solo en la tailnet). Los pasos sin sudo se repetian a mano desde tres specs
#   distintos y cada vez faltaba uno (el respaldo, el build, mirar el commit de /salud). Aca
#   estan escritos una vez. Lo que pide sudo o Tailscale no esta: lo hace Lucas, y
#   despliegue/PRODUCCION.md dice cuando.
#
# QUE HACE
#   respaldo              copia la base de A con la API de backup de sqlite y prueba que abre
#   web <checkout>        npm ci, build, presupuesto y sin-demo de la web en ese checkout
#   worktree-academico <commit>
#                         el worktree fijo y desacoplado de B (S3 de dos-instancias)
#   env-academico         ~/boxtwin-academico/{datos,boxtwin-academico.env} con secretos nuevos (S5)
#   previo                chequeo antes de arrancar B: nada en su puerto, A responde (S6)
#   huella [archivo]      la huella de A (academico/huella-A.sh, dos-instancias/01-produccion.md)
#   verificar a|b         /salud de la instancia contra lo que tiene que decir
#   demo sembrar|reiniciar [yaml]
#                         las cuentas de la version de prueba de A, con el entorno de A
#
# USO
#   bash despliegue/produccion.sh respaldo
#   bash despliegue/produccion.sh verificar a
#   Las rutas se pueden cambiar por entorno (REPO, REPO_B, DATOS, DIR_B, ENV_A, PY,
#   PUERTO_A, PUERTO_B), que es como se lo prueba fuera de la estacion.
set -euo pipefail

REPO="${REPO:-$HOME/Proyectos/TwinBoxing}"
REPO_B="${REPO_B:-$HOME/Proyectos/TwinBoxing-academico}"
DATOS="${DATOS:-$HOME/boxtwin/datos}"
DIR_B="${DIR_B:-$HOME/boxtwin-academico}"
ENV_A="${ENV_A:-/etc/boxtwin.env}"
PY="${PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
PUERTO_A="${PUERTO_A:-8000}"
PUERTO_B="${PUERTO_B:-8001}"
AQUI="$(cd "$(dirname "$0")" && pwd)"

fallar() { echo "ERROR: $*" >&2; exit 1; }
ok() { echo "ok   $*"; }

salud() { curl -fsS "http://127.0.0.1:$1/salud"; }

campo() {  # campo <json> <clave>
  "$PY" -c 'import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else str(v).lower() if isinstance(v,bool) else v)' "$1" "$2"
}

respaldo() {
  local base="$DATOS/boxtwin.db" destino
  [[ -f "$base" ]] || fallar "no esta $base"
  destino="$DATOS/boxtwin.db.antes-de-produccion-$(date +%Y%m%d-%H%M).bak"
  # La API de backup y no cp: la base esta en WAL y el worker puede estar escribiendo.
  "$PY" - "$base" "$destino" <<'PYEOF'
import sqlite3, sys
origen = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
copia = sqlite3.connect(sys.argv[2])
origen.backup(copia)
n = copia.execute("select count(*) from sesiones").fetchone()[0]
copia.close(); origen.close()
print(f"ok   respaldo en {sys.argv[2]} ({n} sesiones, abre)")
PYEOF
}

web() {
  local checkout="${1:?falta el checkout}" dir
  dir="$checkout/prototipo/boxtwin-web-entrenador"
  [[ -d "$dir" ]] || fallar "no esta $dir"
  (
    cd "$dir"
    npm ci --no-audit --no-fund
    # Next 16.4.0 corta el primer build con un aviso de seguridad y sigue en el segundo: se
    # reintenta una vez. Actualizar Next es una decision aparte (pedir el OK de Lucas).
    npm run build || npm run build
    npm run presupuesto
    npm run sin-demo
  )
  ok "web construida en $dir/out"
}

worktree_academico() {
  local commit="${1:?falta el commit (la punta de produccion)}"
  [[ -e "$REPO_B" ]] && fallar "$REPO_B ya existe: moverlo es repetir S3, S4, L2 y S7 (02-academica.md)"
  git -C "$REPO" worktree add --detach "$REPO_B" "$commit"
  ok "B en $REPO_B, desacoplado en $(git -C "$REPO_B" rev-parse --short HEAD). Anotarlo en dos-instancias/04-pasos.md (S3)"
}

env_academico() {
  local destino="$DIR_B/boxtwin-academico.env"
  [[ -e "$destino" ]] && fallar "$destino ya existe: no se pisan secretos"
  mkdir -p "$DIR_B/datos"
  umask 077
  "$PY" - "$AQUI/academico/boxtwin-academico.env.ejemplo" "$destino" "$ENV_A" <<'PYEOF'
import os, secrets, sys
from pathlib import Path
ejemplo, destino, env_a = sys.argv[1:4]
texto = Path(ejemplo).read_text()
nuevos = {"BOXTWIN_SECRETO": secrets.token_hex(32), "BOXTWIN_INVITACION": secrets.token_urlsafe(12)}
# Distintos de los de A, comparados sin mostrarlos.
try:
    de_a = dict(l.split("=", 1) for l in Path(env_a).read_text().splitlines() if "=" in l and not l.startswith("#"))
except OSError:
    de_a = {}
lineas = []
for linea in texto.splitlines():
    k = linea.split("=", 1)[0]
    if k in nuevos and not linea.startswith("#"):
        assert de_a.get(k) != nuevos[k]
        linea = f"{k}={nuevos.pop(k)}"
    lineas.append(linea)
assert not nuevos, f"el ejemplo no tiene {sorted(nuevos)}"
texto = "\n".join(lineas) + "\n"
assert "<nuev" not in texto
Path(destino).write_text(texto)
os.chmod(destino, 0o600)
PYEOF
  ok "$destino (600) y $DIR_B/datos. La invitacion de B esta en ese archivo. Lo instala Lucas en L1"
}

previo() {
  if ss -ltn 2>/dev/null | grep -q ":$PUERTO_B "; then fallar "ya hay algo escuchando en :$PUERTO_B"; fi
  ok "nada en :$PUERTO_B"
  if command -v tailscale >/dev/null 2>&1; then
    if tailscale serve status 2>/dev/null | grep -q ":8443"; then fallar "el 8443 ya esta publicado"; fi
    ok "el 8443 no esta publicado"
  fi
  salud "$PUERTO_A" >/dev/null || fallar "A no responde en :$PUERTO_A"
  ok "A responde en :$PUERTO_A"
}

huella() {
  # La de dos-instancias/01-produccion.md, que ya escribio S1 en despliegue/academico/.
  local salida="${1:-}"
  if [[ -n "$salida" ]]; then
    bash "$AQUI/academico/huella-A.sh" > "$salida"
    ok "huella en $salida"
  else
    bash "$AQUI/academico/huella-A.sh"
  fi
}

verificar() {
  local cual="${1:?a o b}" puerto checkout d
  case "$cual" in
    a) puerto="$PUERTO_A"; checkout="$REPO" ;;
    b) puerto="$PUERTO_B"; checkout="$REPO_B" ;;
    *) fallar "verificar a|b" ;;
  esac
  d="$(salud "$puerto")" || fallar "no responde :$puerto"
  local esperado; esperado="$(git -C "$checkout" rev-parse --short=7 HEAD)"
  [[ "$(campo "$d" commit)" == "$esperado" ]] || fallar "corre $(campo "$d" commit) y en disco esta $esperado: falta reiniciar"
  ok "commit $esperado"
  [[ "$(campo "$d" secreto_efimero)" == "false" ]] || fallar "secreto efimero: falta BOXTWIN_SECRETO"
  ok "secreto fijo"
  [[ "$(campo "$d" registro)" != "abierto" ]] || fallar "el registro esta abierto"
  ok "registro $(campo "$d" registro)"
  [[ "$(campo "$d" frontend)" == "true" ]] || fallar "no sirve la web: BOXTWIN_WEB o falta el build"
  ok "sirve la web"
  if [[ "$cual" == a ]]; then
    [[ "$(campo "$d" clasificador)" != "e7" ]] || fallar "A con el modo academico: SMPL no tiene licencia comercial"
    ok "clasificador $(campo "$d" clasificador)"
    local demo; demo="$(curl -fsS "http://127.0.0.1:$puerto/producto" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("demo",{}).get("activo"))')"
    if [[ "$demo" == "True" ]]; then echo "AVISO version de prueba ENCENDIDA: las cuentas reales no entran por la web (DV-10)"; else ok "version de prueba apagada"; fi
  else
    [[ "$(campo "$d" clasificador)" == "e7" ]] || fallar "B no esta en e7"
    ok "clasificador e7"
    [[ "$(campo "$d" modelo_e7)" == "true" ]] || fallar "falta el modelo de E7"
    ok "modelo_e7"
    [[ "$(campo "$d" modelo_individual)" == "true" ]] || fallar "falta el modelo individual (E4a)"
    ok "modelo_individual"
  fi
}

demo() {
  local accion="${1:?sembrar o reiniciar}" yaml="${2:-$HOME/boxtwin/demo.yaml}"
  # Con el entorno de A leido linea por linea (systemd no usa shell, asi que bash tampoco).
  "$PY" - "$ENV_A" "$REPO" "$accion" "$yaml" <<'PYEOF'
import os, sys
from pathlib import Path
env, repo, accion, yaml = sys.argv[1:5]
for linea in Path(env).read_text().splitlines():
    if "=" in linea and not linea.lstrip().startswith("#"):
        k, v = linea.split("=", 1)
        os.environ[k.strip()] = v
sys.path[:0] = [f"{repo}/boxtwin-api/src", f"{repo}/boxtwin-annotator/src"]
from boxtwin_api.demo import main
sys.exit(main([accion, "--yaml", yaml]))
PYEOF
}

case "${1:-}" in
  respaldo) respaldo ;;
  web) web "${2:-$REPO}" ;;
  worktree-academico) worktree_academico "${2:-}" ;;
  env-academico) env_academico ;;
  previo) previo ;;
  huella) huella "${2:-}" ;;
  verificar) verificar "${2:-}" ;;
  demo) demo "${2:-}" "${3:-}" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
