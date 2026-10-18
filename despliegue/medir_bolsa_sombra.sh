#!/usr/bin/env bash
# BoxTwin - Medicion de B8 (una sola persona en bolsa y sombra) para encender la bandera.
#
# POR QUE EXISTE
#   Bolsa y sombra estan construidas (B8, B8b, W19) pero la bandera
#   BOXTWIN_SESIONES_INDIVIDUALES sigue apagada: el criterio pre-registrado en
#   prototipo/specs/05-api-y-datos.md y ampliacion-13-10/02 §2 pide, sobre dos videos propios
#   (uno de bolsa y uno de sombra, de al menos 3 minutos), la persona asignada en >= 95% de los
#   saltos de 2 s en que esta en cuadro. Este script corre la misma cadena que el worker, con la
#   misma linea de comando, sin pasar por la web, y deja el video procesado y la planilla.
#   Mirar y llenar la planilla es de Lucas; el script no decide nada.
#
# QUE HACE
#   procesar <video> <nombre>   pose, tracking y candidatos (GPU). Lista los candidatos para
#                               elegir la semilla mirando los recortes.
#   completar <nombre> <track> [ortodoxa|zurda]
#                               identidad con una semilla, detector, tipo v2 con el modelo sin
#                               rival, video procesado y planilla de saltos de 2 s.
#   resultado                   lee las planillas llenas de todos los videos y da el veredicto.
#   encender                    solo si resultado paso en los dos: BOXTWIN_SESIONES_INDIVIDUALES=1
#                               y reinicio de la API.
#
# USO (en la estacion, como lucasb, desde la raiz del repo, sin trabajos en el worker)
#   bash despliegue/medir_bolsa_sombra.sh procesar ~/videos/bolsa-16-10.mp4 bolsa
#   bash despliegue/medir_bolsa_sombra.sh completar bolsa 4 ortodoxa
#   (abrir ~/boxtwin/mediciones/b8/bolsa/procesado.mp4, llenar bolsa.csv)
#   bash despliegue/medir_bolsa_sombra.sh procesar ~/videos/sombra-espejo.mp4 sombra
#   bash despliegue/medir_bolsa_sombra.sh completar sombra 2 ortodoxa
#   bash despliegue/medir_bolsa_sombra.sh resultado
#   bash despliegue/medir_bolsa_sombra.sh encender

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="$HOME/miniforge3/envs/twinboxing_env/bin"
PY="$ENVS/python"
BT="$ENVS/boxtwin-annotator"
ENVF="${ENVF:-/etc/boxtwin.env}"
BASE="${BASE:-$HOME/boxtwin/mediciones/b8}"
DURACION_MINIMA=180

falla() { printf 'FALLA: %s\n' "$*" >&2; exit 1; }

# Un valor de /etc/boxtwin.env, para usar los mismos modelos que el worker.
var() {
  local v
  v="$(sudo grep -E "^$1=" "$ENVF" | tail -1 | cut -d= -f2- || true)"
  [ -n "$v" ] && printf '%s' "$v" || printf '%s' "${2:-}"
}

cmd="${1:-}"; shift || true
cd "$REPO"

case "$cmd" in
procesar)
  video="${1:?falta el video}"; nombre="${2:?falta el nombre (bolsa o sombra, por ejemplo)}"
  [ -f "$video" ] || falla "no existe $video"
  dur="$("$PY" -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(c.get(cv2.CAP_PROP_FRAME_COUNT)/(c.get(cv2.CAP_PROP_FPS) or 1))" "$video")"
  "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= $DURACION_MINIMA else 1)" "$dur" \
    || falla "el video dura ${dur%.*} s y el criterio pide al menos $DURACION_MINIMA"
  if systemctl is-active --quiet boxtwin-worker && \
     journalctl -u boxtwin-worker --since "-2 min" -q 2>/dev/null | grep -q "reclam"; then
    echo "AVISO: el worker parece estar trabajando; dos pasadas de pose en una placa de 8 GB van las dos mas lento."
  fi
  out="$BASE/$nombre"
  mkdir -p "$out"
  "$BT" procesar "$video" --out "$out" \
    --modelo-guantes "$(var BOXTWIN_MODELO_GUANTES "$REPO/boxtwin-guantes/modelos/guantes-v2.pt")" \
    --modelo-pose "$(var BOXTWIN_MODELO_POSE "$REPO/yolov8l-pose.pt")" \
    --device "$(var BOXTWIN_DEVICE 0)"
  "$PY" - "$out" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
s = json.loads((d / "sesion.json").read_text())
print("\nCandidatos (el de mas tiempo en cuadro primero). Mirar los recortes y elegir UNO:")
for c in sorted(s.get("candidatos", []), key=lambda c: -((c.get("t_hasta") or 0) - (c.get("t_desde") or 0))):
    espejo = f"  ESPEJO? coexiste con {c['mismo_color_a_la_vez']}" if c.get("mismo_color_a_la_vez") else ""
    print(f"  track {c['track']:>4}  {c.get('t_desde') or 0:6.1f}-{c.get('t_hasta') or 0:6.1f} s  "
          f"guante {(c.get('color_guante') or {}).get('nombre', '?'):<10} {d / (c.get('recorte') or '')}{espejo}")
EOF
  echo -e "\nSiguiente: bash despliegue/medir_bolsa_sombra.sh completar $nombre <track> [ortodoxa|zurda]"
  ;;

completar)
  nombre="${1:?falta el nombre}"; track="${2:?falta el track elegido}"; guardia="${3:-desconocida}"
  out="$BASE/$nombre"
  [ -f "$out/sesion.json" ] || falla "primero: procesar <video> $nombre"
  modelo_ind="$(var BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL "$HOME/boxtwin/modelos/clasificador_geo_v1_sin_rival")"
  [ -f "$modelo_ind/meta.json" ] || falla "falta el modelo sin rival en $modelo_ind: correr antes despliegue/encender_v2.sh"
  "$BT" completar "$out" --semilla-a "$track" \
    --detector "$(var BOXTWIN_MODELO_DETECTOR "$REPO/modelos/detector_7fuentes.ens.pt")" \
    --umbral "$(var BOXTWIN_UMBRAL 0.80)"
  "$BT" clasificar "$out" --modelo "$modelo_ind" --guardia-a "$guardia" --guardia-b desconocida
  "$BT" tipos "$out"
  "$BT" render "$out"
  "$PY" boxtwin-annotator/scripts/medir_individual.py planilla "$out" -o "$BASE/$nombre.csv"
  "$PY" - "$out" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
fc = json.loads((d / "fightcard.json").read_text())
print(f"\nPeleadores en la Fight-Card: {sorted((fc.get('peleadores') or {}).keys())} (en una sesion individual, solo A)")
EOF
  cat <<EOF

Ahora, a mano y antes de mirar ningun numero de la otra sesion:
  1. Abrir $out/procesado.mp4
  2. Llenar $BASE/$nombre.csv: en cada salto de 2 s, en_cuadro (si/no) y marca_A (si/no)
Despues: bash despliegue/medir_bolsa_sombra.sh resultado
EOF
  ;;

resultado)
  shopt -s nullglob
  planillas=("$BASE"/*.csv)
  [ "${#planillas[@]}" -ge 2 ] || falla "hacen falta las dos planillas (bolsa y sombra) en $BASE"
  # Un veredicto anterior no se pisa: queda al lado con fecha, y si cambia hay que contarlo.
  if [ -f "$BASE/resultado.txt" ]; then
    previo="$BASE/resultado.$(date +%Y%m%d-%H%M%S).txt"
    mv "$BASE/resultado.txt" "$previo"
    echo "AVISO: ya habia un resultado; quedo en $previo. Si cambio, el documento lo dice."
  fi
  st=0
  "$PY" boxtwin-annotator/scripts/medir_individual.py resultado "${planillas[@]}" > "$BASE/resultado.txt" || st=$?
  cat "$BASE/resultado.txt"
  if [ "$st" = 0 ]; then echo "VEREDICTO: PASA" | tee -a "$BASE/resultado.txt"
  else echo "VEREDICTO: NO PASA" | tee -a "$BASE/resultado.txt"; fi
  echo -e "\nGuardado en $BASE/resultado.txt. Va a docs/experiments/AAAA-MM-DD-sesiones-individuales.md,"
  echo "pase o no pase. Si no pasa, la bandera queda apagada y se declara en el Cap 7."
  ;;

encender)
  [ -f "$BASE/resultado.txt" ] || falla "primero: resultado"
  grep -qx "VEREDICTO: PASA" "$BASE/resultado.txt" || falla "el criterio no paso (o falta el veredicto); la bandera queda apagada"
  sudo cp -p "$ENVF" "$ENVF.$(date +%Y%m%d-%H%M%S).bak"
  tmp="$(mktemp)"; sudo cat "$ENVF" > "$tmp"
  if grep -qE '^#?[[:space:]]*BOXTWIN_SESIONES_INDIVIDUALES=' "$tmp"; then
    sed -i -E 's|^#?[[:space:]]*BOXTWIN_SESIONES_INDIVIDUALES=.*|BOXTWIN_SESIONES_INDIVIDUALES=1|' "$tmp"
  else
    echo "BOXTWIN_SESIONES_INDIVIDUALES=1" >> "$tmp"
  fi
  sudo install -m 600 -o "$(sudo stat -c %U "$ENVF")" -g "$(sudo stat -c %G "$ENVF")" "$tmp" "$ENVF"; rm -f "$tmp"
  sudo systemctl restart boxtwin-api boxtwin-worker
  sleep 3
  curl -fsS http://127.0.0.1:8000/salud | grep -q '"sesiones_individuales": *true' \
    && echo "OK: bolsa y sombra habilitadas en la web" || falla "/salud no dice sesiones_individuales true"
  ;;

*)
  sed -n '2,32p' "$0"; exit 2 ;;
esac
