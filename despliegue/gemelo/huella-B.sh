#!/usr/bin/env bash
# POR QUE EXISTE: C (gemelo YOLO26) no puede cambiar nada de B (instancia academica). Esta huella
# se toma antes de la fase 1 (S2) y al cierre (S24); las dos tienen que dar igual (CG1).
# QUE HACE: imprime, sin escribir nada en B, lo mismo que academico/huella-A.sh mira de A:
# entorno y unidades, HEAD del worktree, /salud si esta encendida, el bloque :8443 de tailscale,
# pip freeze, la fecha del build de la web y las sesiones de su base.
# USO: despliegue/gemelo/huella-B.sh > ~/boxtwin-gemelo/huella-B-antes.txt
#      (rutas por entorno: REPO_B, DIR_B, ENV_B, PY, PUERTO_B)
# Spec: docs/specs/2026-10-27-gemelo-yolo26/00-README.md, CG1
set -u
REPO_B="${REPO_B:-/home/lucasb/Proyectos/TwinBoxing-academico}"
DIR_B="${DIR_B:-/home/lucasb/boxtwin-academico}"
ENV_B="${ENV_B:-/etc/boxtwin-academico.env}"
PY="${PY:-/home/lucasb/miniforge3/envs/twinboxing_env/bin/python}"
PUERTO_B="${PUERTO_B:-8001}"
UNIDADES="${UNIDADES:-/etc/systemd/system}"
echo "## sha256 entorno y unidades"
sha256sum "$ENV_B" "$UNIDADES/boxtwin-academico-api.service" "$UNIDADES/boxtwin-academico-worker.service" 2>&1
echo "## HEAD del worktree de B"
git -C "$REPO_B" rev-parse HEAD
echo "## /salud de B (vacio si esta apagada; se compara apagada con apagada)"
curl -s "127.0.0.1:$PUERTO_B/salud" | "$PY" -c 'import json,sys
t=sys.stdin.read()
d=json.loads(t) if t.strip() else {}
d.pop("arrancado", None)   # cambia con cada reinicio, no es de B sino del proceso
print(json.dumps(d, sort_keys=True) if d else "apagada")'
echo "## tailscale serve status, bloque :8443"
if command -v tailscale >/dev/null 2>&1; then
  tailscale serve status 2>&1 | awk '/:8443/{dentro=1} /^$/{dentro=0} dentro'
fi
echo "## pip freeze de twinboxing_env (sha256)"
"$PY" -m pip freeze 2>/dev/null | sha256sum
echo "## mtime de out/index.html de B"
stat -c '%Y' "$REPO_B/prototipo/boxtwin-web-entrenador/out/index.html" 2>&1
echo "## sesiones de B"
"$PY" -c "import sqlite3,sys; print(sqlite3.connect(f'file:{sys.argv[1]}?mode=ro', uri=True).execute('select count(*) from sesiones').fetchone()[0])" "$DIR_B/datos/boxtwin.db" 2>&1
echo "## carpetas en $DIR_B/datos/sesiones"
ls "$DIR_B/datos/sesiones" 2>/dev/null | wc -l
