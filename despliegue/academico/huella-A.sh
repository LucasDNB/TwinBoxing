#!/usr/bin/env bash
# POR QUE EXISTE: B (instancia academica) no puede cambiar nada de A (produccion). Esta huella
# se toma antes de lanzar B y despues de cada demo; las dos tienen que dar igual (CA1).
# QUE HACE: imprime, sin escribir nada en A, lo que lista 01-produccion.md, "Como se comprueba".
# USO: despliegue/academico/huella-A.sh > ~/boxtwin-academico/huella-A-antes.txt
# Spec: docs/specs/2026-10-25-dos-instancias/01-produccion.md
set -u
REPO=/home/lucasb/Proyectos/TwinBoxing
PY=/home/lucasb/miniforge3/envs/twinboxing_env/bin/python
echo "## sha256 entorno y unidades"
sha256sum /etc/boxtwin.env /etc/systemd/system/boxtwin-api.service /etc/systemd/system/boxtwin-worker.service 2>&1
echo "## HEAD de la copia principal"
git -C "$REPO" rev-parse HEAD
echo "## /salud de A"
curl -s 127.0.0.1:8000/salud; echo
echo "## tailscale serve status"
tailscale serve status 2>&1
echo "## pip freeze de twinboxing_env (sha256)"
"$PY" -m pip freeze 2>/dev/null | sha256sum
echo "## mtime de out/index.html de A"
stat -c '%Y' "$REPO/prototipo/boxtwin-web-entrenador/out/index.html"
echo "## sesiones de A (puede crecer por el uso de A)"
"$PY" -c "import sqlite3; print(sqlite3.connect('file:/home/lucasb/boxtwin/datos/boxtwin.db?mode=ro', uri=True).execute('select count(*) from sesiones').fetchone()[0])"
echo "## carpetas en ~/boxtwin/datos/sesiones"
ls /home/lucasb/boxtwin/datos/sesiones | wc -l
