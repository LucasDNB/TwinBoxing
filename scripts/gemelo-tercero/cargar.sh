#!/usr/bin/env bash
# BoxTwin - Carga el gemelo de tercero en A (docs/specs/2026-11-02-gemelo-tercero, S1 y S2).
#
# POR QUE EXISTE
#   Los pasos S1 y S2 del registro, en un solo comando y con las guardas antes de escribir en
#   la base de A: video de rounds netos, A en 1058d5e, cola vacia, respaldo de la base, ensayo
#   y confirmacion. Lo de antes queda en ~/Proyectos/twinboxing-finales/gemelo/ para CT1.
#
# QUE HACE
#   1. ffprobe del video: 12 rounds de 180 s = 2160 s (+-5).
#   2. /salud de A: commit 1058d5e; anota el clasificador (CT4 y CT5 dependen de el).
#   3. Cola de A vacia (nada en_cola ni tomado): la GPU es una sola.
#   4. Conteos de sesiones, peleadores y trabajos de A antes (CT1) y respaldo de la base.
#   5. Baja cargar_tercero.py de origin/main SIN checkout en la copia de A.
#   6. Ensayo, pregunta, y carga. La salida queda en gemelo/carga.json.
#
# USO
#   bash cargar.sh <tu email en A>
#   (FORZAR=1 saltea la guarda de la duracion; nunca la del commit ni la de la cola)

set -euo pipefail

EMAIL="${1:?uso: bash cargar.sh <tu email en A>}"
FIN="$HOME/Proyectos/twinboxing-finales"
VIDEO="$FIN/pac-neto.mp4"
OUT="$FIN/gemelo"
A_REPO="$HOME/Proyectos/TwinBoxing"
DATOS="$HOME/boxtwin/datos"
BASE="$DATOS/boxtwin.db"
API="http://127.0.0.1:8000"
COMMIT_A="1058d5e"

mkdir -p "$OUT" "$FIN/scripts"
# Los scripts de activacion de cuda-nvcc de conda leen variables sin definir: con set -u
# cortan (NVCC_PREPEND_FLAGS, CUDAARCHS_BACKUP). Se apaga solo para activar.
if [ "${CONDA_DEFAULT_ENV:-}" != "twinboxing_env" ]; then
  set +u
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate twinboxing_env
  set -u
fi

echo "== 1. Video"
[ -f "$VIDEO" ] || { echo "no existe $VIDEO"; exit 1; }
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,width,height \
  -show_entries format=duration -of json "$VIDEO" | tee "$OUT/video.json"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
if ! python -c "import sys; sys.exit(0 if abs(float('$DUR') - 2160) <= 5 else 1)"; then
  echo "AVISO: el video dura $DUR s y 12 rounds de 180 s son 2160 s"
  [ "${FORZAR:-0}" = "1" ] || { echo "corta (FORZAR=1 para seguir)"; exit 1; }
fi

echo "== 2. A"
curl -sf "$API/salud" | tee "$OUT/salud-antes.json"; echo
python - "$OUT/salud-antes.json" "$COMMIT_A" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
if not str(s.get("commit", "")).startswith(sys.argv[2]):
    sys.exit(f"A corre {s.get('commit')}, no {sys.argv[2]}: no se carga")
print("clasificador:", s.get("clasificador"),
      "(geometrico: CT4 y CT5 aplican; poseconv3d: CT4 no aplica y el Rival no tiene altura)")
EOF

echo "== 3 y 4. Cola, conteos y respaldo"
python - "$BASE" "$OUT/conteos-antes.json" <<'EOF'
import json, sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
q = lambda sql: con.execute(sql).fetchone()[0]
c = {"sesiones": q("select count(*) from sesiones"),
     "boxeadores": q("select count(*) from boxeadores"),
     "trabajos": q("select count(*) from trabajos"),
     "pendientes": q("select count(*) from trabajos where estado in ('en_cola','tomado')")}
json.dump(c, open(sys.argv[2], "w"), indent=2)
print(c)
if c["pendientes"]:
    sys.exit("hay trabajos en cola o corriendo en A: esperar a que terminen")
EOF
RESPALDO="$BASE.antes-del-tercero.bak"
[ -f "$RESPALDO" ] && echo "respaldo ya existe: $RESPALDO" || { cp -p "$BASE" "$RESPALDO"; echo "respaldo: $RESPALDO"; }

echo "== 5. Script"
git -C "$A_REPO" fetch -q origin main
git -C "$A_REPO" show origin/main:boxtwin-api/scripts/cargar_tercero.py > "$FIN/scripts/cargar_tercero.py"
git -C "$A_REPO" rev-parse --short origin/main | sed 's/^/cargar_tercero.py de main en /'
[ "$(git -C "$A_REPO" rev-parse --short HEAD)" = "$COMMIT_A" ] || { echo "la copia de A no esta en $COMMIT_A"; exit 1; }

echo "== 6. Ensayo"
export BOXTWIN_DB="sqlite:///$BASE" BOXTWIN_DATOS="$DATOS"
ARGS=(--email "$EMAIL" --video "$VIDEO" --descanso-s 0)
python "$FIN/scripts/cargar_tercero.py" "${ARGS[@]}" --ensayo | tee "$OUT/ensayo.json"
read -r -p "¿Cargar en A? [s/N] " r
[ "$r" = "s" ] || { echo "no se cargo nada"; exit 0; }

python "$FIN/scripts/cargar_tercero.py" "${ARGS[@]}" | tee "$OUT/carga.json"
ID=$(python -c "import json; print(json.load(open('$OUT/carga.json'))['sesion']['id'])")
echo
echo "Sesion $ID en cola. El worker de A la toma solo: seguila en el panel de la web."
echo "Cuando quede en espera_siembra: siembra en la web, A = el zurdo (Tercero Z)."
