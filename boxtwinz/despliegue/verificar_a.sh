#!/usr/bin/env bash
# CZ2: A no cambia. Huella de /etc/boxtwin.env, de los servicios de A y del commit que sirve
# A, para comparar antes y despues del despliegue de BoxTwinZ. No muestra el contenido.
# USO: bash verificar_a.sh > antes.txt ; (desplegar) ; bash verificar_a.sh > despues.txt ; diff antes.txt despues.txt
set -uo pipefail
for f in /etc/boxtwin.env /etc/systemd/system/boxtwin-api.service /etc/systemd/system/boxtwin-worker.service; do
  if [[ -r "$f" ]]; then sha256sum "$f"; else echo "sin acceso: $f"; fi
done
echo "salud_A_commit $(curl -s localhost:8000/salud | python3 -c 'import json,sys;print(json.load(sys.stdin).get("commit"))')"
for s in boxtwin-api boxtwin-worker; do echo "$s $(systemctl is-active $s)"; done
