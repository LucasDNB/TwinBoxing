#!/usr/bin/env bash
# BoxTwin - Encender el tipo de golpe v2 en la estacion (16-10-2026).
#
# POR QUE EXISTE
#   v2 esta construida y medida (docs/experiments/2026-10-16-tipo-v2.md) pero detras de
#   BOXTWIN_CLASIFICADOR, apagada. Encenderla son cuatro pasos a mano en /etc/boxtwin.env y
#   systemd, y faltaba el modelo de bolsa y sombra (E4a, sin rival). Este script hace todo en
#   orden, se detiene en el primer problema y no toca nada que ya este bien.
#
# QUE HACE
#   1. Verifica el checkout, el entorno twinboxing_env y los datos de E4 (golpes.npz).
#   2. Entrena en CPU lo que falte: el modelo final de E4 (sparring y combate) y el de E4a
#      (bolsa y sombra). Si ya estan, no los toca. En CPU es reproducible bit a bit.
#   3. Corre las pruebas del clasificador v2 contra los datos reales (CA1, CA2).
#   4. Escribe en /etc/boxtwin.env (con sudo, copia de respaldo al lado) BOXTWIN_CLASIFICADOR,
#      BOXTWIN_MODELO_CLASIFICADOR_GEO y BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL.
#      NO enciende BOXTWIN_SESIONES_INDIVIDUALES: eso va despues de medir B8
#      (despliegue/medir_bolsa_sombra.sh), con el criterio pre-registrado.
#   5. Reinicia API y worker y verifica /salud.
#
# USO (en la estacion, como lucasb, desde la raiz del repo)
#   bash despliegue/encender_v2.sh            # enciende
#   bash despliegue/encender_v2.sh --apagar   # vuelve a PoseConv3D (o sin tipo)
#   bash despliegue/encender_v2.sh --sin-reiniciar   # todo menos systemd

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}"
ENVF="${ENVF:-/etc/boxtwin.env}"
MODELOS="${MODELOS:-$HOME/boxtwin/modelos}"
GEO="$MODELOS/clasificador_geo_v1"
GEO_IND="$MODELOS/clasificador_geo_v1_sin_rival"
NPZ="${BOXTWIN_CLASIFICADOR_GEO:-$HOME/boxtwin/clasificador-geo-2026-10}/datos/golpes.npz"
SALUD="${SALUD:-http://127.0.0.1:8000/salud}"

APAGAR=0; REINICIAR=1
for a in "$@"; do
  case "$a" in
    --apagar) APAGAR=1 ;;
    --sin-reiniciar) REINICIAR=0 ;;
    *) echo "argumento desconocido: $a" >&2; exit 2 ;;
  esac
done

paso() { printf '\n== %s\n' "$*"; }
falla() { printf 'FALLA: %s\n' "$*" >&2; exit 1; }

# Escribe CLAVE=valor en el archivo de entorno: reemplaza la linea (comentada o no) o la agrega.
# systemd no expande nada, asi que el valor va literal, sin comillas.
fijar() {
  local clave="$1" valor="$2" tmp
  tmp="$(mktemp)"
  sudo cat "$ENVF" > "$tmp"
  if grep -qE "^#?[[:space:]]*${clave}=" "$tmp"; then
    sed -i -E "s|^#?[[:space:]]*${clave}=.*|${clave}=${valor}|" "$tmp"
  else
    printf '%s=%s\n' "$clave" "$valor" >> "$tmp"
  fi
  sudo install -m 600 -o "$(sudo stat -c %U "$ENVF")" -g "$(sudo stat -c %G "$ENVF")" "$tmp" "$ENVF"
  rm -f "$tmp"
}

comentar() {
  local clave="$1" tmp
  tmp="$(mktemp)"
  sudo cat "$ENVF" > "$tmp"
  sed -i -E "s|^(${clave}=.*)|#\1|" "$tmp"
  sudo install -m 600 -o "$(sudo stat -c %U "$ENVF")" -g "$(sudo stat -c %G "$ENVF")" "$tmp" "$ENVF"
  rm -f "$tmp"
}

reiniciar_y_verificar() {
  local esperado="$1"
  [ "$REINICIAR" = 1 ] || { echo "(sin reiniciar: los servicios siguen con la configuracion anterior)"; return; }
  paso "Reinicio de API y worker"
  echo "El worker termina el trabajo en curso antes de salir (TimeoutStopSec=3600)."
  sudo systemctl restart boxtwin-worker boxtwin-api
  local s=""
  for _ in $(seq 1 30); do
    s="$(curl -fsS "$SALUD" 2>/dev/null || true)"
    [ -n "$s" ] && break
    sleep 1
  done
  [ -n "$s" ] || falla "la API no contesta en $SALUD. Mirar: journalctl -u boxtwin-api -n 50"
  echo "$s"
  echo "$s" | grep -q "\"clasificador\": *\"$esperado\"" || falla "/salud no dice clasificador $esperado"
  if [ "$esperado" = geometrico ]; then
    echo "$s" | grep -q '"modelo_individual": *true' \
      || falla "/salud no encuentra el modelo de bolsa y sombra en $GEO_IND"
  fi
  echo "OK: clasificador $esperado"
}

cd "$REPO"
[ -f "$ENVF" ] || falla "no existe $ENVF (ver despliegue/README.md)"

if [ "$APAGAR" = 1 ]; then
  paso "Apagar v2"
  sudo cp -p "$ENVF" "$ENVF.$(date +%Y%m%d-%H%M%S).bak"
  comentar BOXTWIN_CLASIFICADOR
  echo "Las sesiones ya clasificadas con v2 conservan su tipo hasta que se reclasifiquen."
  reiniciar_y_verificar poseconv3d
  exit 0
fi

paso "1. Checkout y entorno"
echo "rama: $(git rev-parse --abbrev-ref HEAD), commit: $(git rev-parse --short HEAD)"
[ -z "$(git status --porcelain -- boxtwin-annotator boxtwin-api scripts/clasificador_geo)" ] \
  || echo "AVISO: hay cambios sin commitear en el codigo del producto; /salud va a mostrar este commit igual."
[ -x "$PY" ] || falla "no encuentro $PY"
"$PY" - <<'EOF' || falla "el entorno no tiene la version del anotador con el modelo sin rival (pip install -e boxtwin-annotator)"
from boxtwin.mvp.clasificador_geo import N_BRAZO, ClasificadorGeo
assert N_BRAZO == 20 and hasattr(ClasificadorGeo, "sin_rival")
EOF
"$PY" -c "from boxtwin_api.config import Config; Config().modelo_clasificador_geo_individual" \
  || falla "el entorno no tiene la version de la API con el modelo individual (pip install -e boxtwin-api)"
[ -f "$NPZ" ] || falla "no encuentro los datos de E4 en $NPZ"

paso "2. Modelos finales (CPU, reproducibles)"
if [ -f "$GEO/meta.json" ]; then
  echo "E4 ya esta: $GEO"
else
  echo "Entrenando E4 (5 semillas, CPU)..."
  "$PY" scripts/clasificador_geo/entrenar_final.py --exp E4 --dispositivo cpu --salida "$GEO"
fi
if [ -f "$GEO_IND/meta.json" ]; then
  echo "E4a ya esta: $GEO_IND"
else
  echo "Entrenando E4a, sin rival, para bolsa y sombra (5 semillas, CPU)..."
  "$PY" scripts/clasificador_geo/entrenar_final.py --exp E4a --dispositivo cpu --salida "$GEO_IND"
fi
"$PY" - "$GEO" "$GEO_IND" <<'EOF' || falla "un modelo no carga o no declara lo que deberia"
import sys
from pathlib import Path
from boxtwin.mvp.clasificador_geo import ClasificadorGeo
e4, e4a = (ClasificadorGeo.cargar(Path(p)) for p in sys.argv[1:3])
assert e4.n_features == 26 and not e4.sin_rival, e4.meta.get("nombre")
assert e4a.n_features == 20 and e4a.sin_rival, e4a.meta.get("nombre")
for c in (e4, e4a):
    m = c.meta.get("metricas_e4") or {}
    print(f"  {c.meta['nombre']}: {len(c.modelos)} semillas, exp {c.meta.get('exp')}, "
          f"familia con una fuente afuera {m.get('familia_fuente_no_vista')}")
EOF

paso "3. Pruebas del clasificador contra los datos reales"
"$PY" -m pytest -q -p no:warnings scripts/clasificador_geo/tests/test_equivalencia_v2.py \
  scripts/clasificador_geo/tests/test_entrenar_final.py
(cd boxtwin-annotator && "$PY" -m pytest -q -p no:warnings tests/test_mvp_clasificador_geo.py tests/test_mvp_tipos_v2.py)

paso "4. /etc/boxtwin.env"
sudo cp -p "$ENVF" "$ENVF.$(date +%Y%m%d-%H%M%S).bak"
fijar BOXTWIN_CLASIFICADOR geometrico
fijar BOXTWIN_MODELO_CLASIFICADOR_GEO "$GEO"
fijar BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL "$GEO_IND"
sudo grep -E '^#?(BOXTWIN_CLASIFICADOR|BOXTWIN_MODELO_CLASIFICADOR_GEO|BOXTWIN_SESIONES_INDIVIDUALES)' "$ENVF" || true

reiniciar_y_verificar geometrico

cat <<EOF

Listo. Las sesiones nuevas salen con tipo v2. Las que ya estaban listas no se reclasifican
solas; para una a mano:
  $HOME/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator clasificar <sesion> --modelo $GEO --guardia-a <ortodoxa|zurda|desconocida> --guardia-b <...>
  $HOME/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator tipos <sesion>
Bolsa y sombra: siguen "En construccion" hasta medir B8 (despliegue/medir_bolsa_sombra.sh).
EOF
