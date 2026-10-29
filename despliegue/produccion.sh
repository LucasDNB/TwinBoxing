#!/usr/bin/env bash
# POR QUE EXISTE
#   La rama `produccion` despliega dos instancias en la estacion: A (BoxTwin con las versiones
#   Individual, Entrenador y Gimnasio, publica por el Funnel) y B (la misma base con el modo
#   academico E7, solo en la tailnet). Desde el 27-10 hay una tercera, C (el gemelo con
#   YOLO26l-pose, solo en 127.0.0.1; docs/specs/2026-10-27-gemelo-yolo26). Los pasos sin sudo se repetian a mano desde tres specs
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
#   worktree-gemelo <commit>
#                         el worktree fijo y desacoplado de C (S5 de gemelo-yolo26)
#   env-gemelo            ~/boxtwin-gemelo/{datos,boxtwin-gemelo.env,systemd/} con secretos
#                         nuevos, distintos de los de A y de B (S21)
#   previo [b|c]          chequeo antes de arrancar B (por defecto) o C: nada en su puerto, A
#                         responde; para C ademas B apagada y nada publicado hacia su puerto
#   huella [a|b] [archivo]
#                         la huella de A (academico/huella-A.sh) o de B (gemelo/huella-B.sh);
#                         sin a|b, la de A como antes
#   verificar a|b|c       /salud de la instancia contra lo que tiene que decir
#   demo sembrar|reiniciar [yaml]
#                         las cuentas de la version de prueba de A, con el entorno de A
#
# USO
#   bash despliegue/produccion.sh respaldo
#   bash despliegue/produccion.sh verificar a
#   Las rutas se pueden cambiar por entorno (REPO, REPO_B, REPO_C, DATOS, DIR_B, DIR_C, ENV_A,
#   ENV_B, PY, PUERTO_A, PUERTO_B, PUERTO_C, MODELO_POSE_C), que es como se lo prueba fuera de
#   la estacion (despliegue/tests/test_produccion_sh.py).
set -euo pipefail

REPO="${REPO:-$HOME/Proyectos/TwinBoxing}"
REPO_B="${REPO_B:-$HOME/Proyectos/TwinBoxing-academico}"
DATOS="${DATOS:-$HOME/boxtwin/datos}"
DIR_B="${DIR_B:-$HOME/boxtwin-academico}"
ENV_A="${ENV_A:-/etc/boxtwin.env}"
PY="${PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
PUERTO_A="${PUERTO_A:-8000}"
PUERTO_B="${PUERTO_B:-8001}"
REPO_C="${REPO_C:-$HOME/Proyectos/TwinBoxing-gemelo}"
DIR_C="${DIR_C:-$HOME/boxtwin-gemelo}"
ENV_B="${ENV_B:-/etc/boxtwin-academico.env}"
PUERTO_C="${PUERTO_C:-8002}"
MODELO_POSE_C="${MODELO_POSE_C:-$DIR_C/modelos/yolo26l-pose.pt}"
AQUI="$(cd "$(dirname "$0")" && pwd)"

fallar() { echo "ERROR: $*" >&2; exit 1; }
ok() { echo "ok   $*"; }

salud() { curl -fsS "http://127.0.0.1:$1/salud"; }

campo() {  # campo <json> <clave>, con puntos para lo anidado (modelo_pose.archivo)
  "$PY" -c '
import json, sys
v = json.loads(sys.argv[1])
for k in sys.argv[2].split("."):
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else str(v).lower() if isinstance(v, bool) else v)' "$1" "$2"
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

env_nuevo() {  # env_nuevo <ejemplo> <destino> <entorno a comparar>...
  umask 077
  "$PY" - "$@" <<'PYEOF'
import os, secrets, sys
from pathlib import Path
ejemplo, destino, otros = sys.argv[1], sys.argv[2], sys.argv[3:]
texto = Path(ejemplo).read_text()
nuevos = {"BOXTWIN_SECRETO": secrets.token_hex(32), "BOXTWIN_INVITACION": secrets.token_urlsafe(12)}
# Distintos de los de las otras instancias, comparados sin mostrarlos.
previos = []
for env in otros:
    try:
        previos.append(dict(l.split("=", 1) for l in Path(env).read_text().splitlines() if "=" in l and not l.startswith("#")))
    except OSError:
        pass
lineas = []
for linea in texto.splitlines():
    k = linea.split("=", 1)[0]
    if k in nuevos and not linea.startswith("#"):
        assert all(p.get(k) != nuevos[k] for p in previos)
        linea = f"{k}={nuevos.pop(k)}"
    lineas.append(linea)
assert not nuevos, f"el ejemplo no tiene {sorted(nuevos)}"
texto = "\n".join(lineas) + "\n"
assert "<nuev" not in texto
Path(destino).write_text(texto)
os.chmod(destino, 0o600)
PYEOF
}

env_academico() {
  local destino="$DIR_B/boxtwin-academico.env"
  [[ -e "$destino" ]] && fallar "$destino ya existe: no se pisan secretos"
  mkdir -p "$DIR_B/datos"
  env_nuevo "$AQUI/academico/boxtwin-academico.env.ejemplo" "$destino" "$ENV_A"
  ok "$destino (600) y $DIR_B/datos. La invitacion de B esta en ese archivo. Lo instala Lucas en L1"
}

worktree_gemelo() {
  local commit="${1:?falta el commit (la punta de main despues de mergear la fase 0)}"
  [[ -e "$REPO_C" ]] && fallar "$REPO_C ya existe: moverlo es repetir S5, y despues de la fase 1 todo desde S6 (03-instancia-c.md)"
  git -C "$REPO" worktree add --detach "$REPO_C" "$commit"
  ok "C en $REPO_C, desacoplado en $(git -C "$REPO_C" rev-parse --short HEAD). Anotarlo en gemelo-yolo26/05-pasos.md (S5)"
}

env_gemelo() {
  local destino="$DIR_C/boxtwin-gemelo.env"
  [[ -e "$destino" ]] && fallar "$destino ya existe: no se pisan secretos"
  mkdir -p "$DIR_C/datos" "$DIR_C/systemd"
  env_nuevo "$AQUI/gemelo/boxtwin-gemelo.env.ejemplo" "$destino" "$ENV_A" "$ENV_B"
  cp "$AQUI"/gemelo/systemd/boxtwin-gemelo-*.service "$DIR_C/systemd/"
  ok "$destino (600), $DIR_C/datos y las unidades en $DIR_C/systemd. Lo instala Lucas en L2, sin enable"
}

previo() {
  local cual="${1:-b}"
  if [[ "$cual" == c ]]; then
    if ss -ltn 2>/dev/null | grep -q ":$PUERTO_C "; then fallar "ya hay algo escuchando en :$PUERTO_C"; fi
    ok "nada en :$PUERTO_C"
    # Regla 2 de convivencia: B y C nunca encendidas a la vez.
    if salud "$PUERTO_B" >/dev/null 2>&1; then fallar "B responde en :$PUERTO_B: apagarla antes de encender C"; fi
    ok "B apagada"
    if command -v tailscale >/dev/null 2>&1; then
      if tailscale serve status 2>/dev/null | grep -q "127.0.0.1:$PUERTO_C"; then fallar "hay algo publicado hacia :$PUERTO_C (CG3)"; fi
      ok "nada publicado hacia :$PUERTO_C"
    fi
    salud "$PUERTO_A" >/dev/null || fallar "A no responde en :$PUERTO_A"
    ok "A responde en :$PUERTO_A"
    return
  fi
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
  # La de A es la de dos-instancias/01-produccion.md (academico/huella-A.sh); la de B, la de
  # gemelo-yolo26 (gemelo/huella-B.sh). Sin a|b, la de A, como antes.
  local script="$AQUI/academico/huella-A.sh"
  case "${1:-}" in
    a) shift ;;
    b) script="$AQUI/gemelo/huella-B.sh"; shift ;;
  esac
  local salida="${1:-}"
  if [[ -n "$salida" ]]; then
    REPO_B="$REPO_B" DIR_B="$DIR_B" ENV_B="$ENV_B" PY="$PY" PUERTO_B="$PUERTO_B" bash "$script" > "$salida"
    ok "huella en $salida"
  else
    REPO_B="$REPO_B" DIR_B="$DIR_B" ENV_B="$ENV_B" PY="$PY" PUERTO_B="$PUERTO_B" bash "$script"
  fi
}

verificar() {
  local cual="${1:?a, b o c}" puerto checkout d
  case "$cual" in
    a) puerto="$PUERTO_A"; checkout="$REPO" ;;
    b) puerto="$PUERTO_B"; checkout="$REPO_B" ;;
    c) puerto="$PUERTO_C"; checkout="$REPO_C" ;;
    *) fallar "verificar a|b|c" ;;
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
  elif [[ "$cual" == c ]]; then
    # CG2 (gemelo-yolo26/03-instancia-c.md §8)
    [[ "$(campo "$d" modelo_pose.archivo)" == "yolo26l-pose.pt" ]] || fallar "C no corre yolo26l-pose.pt: $(campo "$d" modelo_pose.archivo)"
    local sha; sha="$(sha256sum "$MODELO_POSE_C" | cut -c1-12)" || fallar "no esta $MODELO_POSE_C"
    [[ "$(campo "$d" modelo_pose.sha256)" == "$sha" ]] || fallar "el sha256 del modelo de pose no es el de S4 ($sha)"
    ok "modelo_pose yolo26l-pose.pt ($sha)"
    [[ "$(campo "$d" clasificador)" == "geometrico" ]] || fallar "C tiene que estar en geometrico, nunca e7"
    ok "clasificador geometrico"
    [[ "$(campo "$d" modelo_individual)" == "true" ]] || fallar "falta el modelo individual de C (E4a-y26)"
    ok "modelo_individual"
    local demo; demo="$(curl -fsS "http://127.0.0.1:$puerto/producto" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("demo",{}).get("activo"))')"
    if [[ "$demo" == "True" ]]; then echo "AVISO version de prueba encendida: en C es lo esperado (decision 7)"; else ok "version de prueba apagada"; fi
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
  worktree-gemelo) worktree_gemelo "${2:-}" ;;
  env-gemelo) env_gemelo ;;
  previo) previo "${2:-}" ;;
  huella) huella "${2:-}" "${3:-}" ;;
  verificar) verificar "${2:-}" ;;
  demo) demo "${2:-}" "${3:-}" ;;
  *) sed -n '2,40p' "$0"; exit 2 ;;
esac
