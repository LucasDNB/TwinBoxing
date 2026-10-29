#!/usr/bin/env bash
# POR QUE EXISTE: la fase 1 de E8 son siete etapas largas sobre la GPU que comparte A
# (docs/specs/2026-10-27-gemelo-yolo26, 05-pasos.md S7 a S13). Corridas a mano, una se lanza
# con trabajos de A en cola o se repite sin querer. Este es el unico camino de la fase 1.
# QUE HACE: antes de cada etapa espera el candado (guardas.py: base de A sin trabajos en_cola o
# tomado, nvidia-smi sin otros procesos; cada espera queda en e8/candado.log) y despues la
# corre. Una etapa terminada deja e8/etapas/<id>.hecho y no se repite; adentro, cada script es
# reanudable. Las dos condiciones (base y y26) se exportan, se entrenan y se evaluan con este
# mismo commit. Nada escribe en A: los exports van con --out bajo ~/boxtwin-gemelo.
# Con E8_DH2=1 entra la rama D-H2: v8 y YOLO26 re-cacheados con botsort-sin-reid.yaml, y la base
# tambien pasa por el puente. Con E8_ENSAYO=1 imprime los comandos sin correr nada.
# USO: bash scripts/gemelo/e8.sh            # todas las etapas pendientes
#      bash scripts/gemelo/e8.sh S10 S11    # solo esas
#      E8_ENSAYO=1 bash scripts/gemelo/e8.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
GEMELO=${BOXTWIN_GEMELO:-$HOME/boxtwin-gemelo}
REPO_A=${BOXTWIN_REPO_A:-$HOME/Proyectos/TwinBoxing}
PY=${PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}
BIN=$(dirname "$PY")
G=$REPO/scripts/gemelo
GEO=$REPO/scripts/clasificador_geo
E8=$GEMELO/e8
RAIZ_Y26=$GEMELO/raiz-y26
MODELO=${E8_MODELO:-$GEMELO/modelos/yolo26l-pose.pt}
MODELO_V8=${E8_MODELO_V8:-$REPO_A/yolov8l-pose.pt}
GUANTES=${E8_GUANTES:-$REPO_A/boxtwin-guantes/modelos/guantes-v2.pt}
CONFIGS=$REPO/boxtwin-annotator/src/boxtwin/configs
TRACKER=$CONFIGS/botsort.yaml
RAIZ_BASE=$REPO_A                   # sin D-H2 la base son los caches y anotaciones originales
if [[ "${E8_DH2:-0}" == 1 ]]; then
  TRACKER=$CONFIGS/botsort-sin-reid.yaml
  RAIZ_BASE=$GEMELO/raiz-v8
fi

# El codigo de este commit, no el de A (03-instancia-c.md §1)
export PYTHONPATH=$REPO/boxtwin-annotator/src:$REPO/boxtwin-api/src:$REPO/boxtwin-detector/src${PYTHONPATH:+:$PYTHONPATH}
export BOXTWIN_GEMELO=$GEMELO BOXTWIN_REPO_A=$REPO_A BOXTWIN_CMD=$BIN/boxtwin-annotator

ensayo() { [[ "${E8_ENSAYO:-0}" == 1 ]]; }
correr() {
  echo "+ $*"
  ensayo || env "$@"
}

# fuente dir video_original, una por linea, en el orden de comun.FUENTES
fuentes() {
  "$PY" - "$REPO" "$REPO_A" <<'EOF'
import json, sys
from pathlib import Path
repo, repo_a = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "scripts" / "clasificador"))
import comun
for fuente, d in comun.FUENTES.items():
    ann = repo_a / d / "annotations" / f"{fuente}.annot.json"
    video = json.loads(ann.read_text())["video"]["path"] if ann.is_file() else f"{fuente}.mp4"
    print(fuente, d, video)
EOF
}

etapa() {
  local id=$1 que=$2
  shift 2
  if [[ -f "$E8/etapas/$id.hecho" ]]; then
    echo "$id ya hecha ($E8/etapas/$id.hecho), se saltea"
    return 0
  fi
  echo "+ candado $id $que"
  ensayo || "$PY" "$G/guardas.py" candado "$id $que"
  local t0=$SECONDS
  "$@"
  if ! ensayo; then
    mkdir -p "$E8/etapas"
    echo "$(date -Iseconds) $id $que $((SECONDS - t0)) s" >> "$E8/etapas.log"
    touch "$E8/etapas/$id.hecho"
  fi
}

s7() {
  correr "$PY" "$G/recache.py" --modelo "$MODELO" --tracker "$TRACKER" --raiz-y26 "$RAIZ_Y26" \
    --registro "$E8/recache.json"
  if [[ "${E8_DH2:-0}" == 1 ]]; then
    correr "$PY" "$G/recache.py" --modelo "$MODELO_V8" --tracker "$TRACKER" --raiz-y26 "$RAIZ_BASE" \
      --registro "$E8/recache-v8.json"
  fi
}

s8() {
  correr "$PY" "$G/puente_identidad.py" --k1 --raiz-y26 "$RAIZ_Y26"
  correr "$PY" "$G/puente_identidad.py" --raiz-y26 "$RAIZ_Y26"
  if [[ "${E8_DH2:-0}" == 1 ]]; then
    correr "$PY" "$G/puente_identidad.py" --raiz-y26 "$RAIZ_BASE"
  fi
}

s9() {
  local cond raiz fuente dir video npz
  for cond in base y26; do
    raiz=$RAIZ_BASE; [[ $cond == y26 ]] && raiz=$RAIZ_Y26
    while read -r fuente dir video; do
      # el video del proyecto de la condicion: en A el original, en el gemelo el enlace
      [[ $raiz == "$REPO_A" ]] || video=$raiz/$dir/videos/$(basename "$video")
      correr "$BIN/boxtwin-annotator" export "$video" --format sequence --classes 14 \
        --label-space side --out "$E8/$cond/exports"
    done < <(fuentes)
    npz=("$E8/$cond/exports/"*.sequence.npz)   # en ensayo queda el patron, sin expandir
    correr "$BIN/boxtwin-detector" build "${npz[@]}" --out "$E8/$cond/det"
  done
}

s10() {
  local cond det
  for cond in base y26; do
    det=("$E8/$cond/det/"*.det.npz)
    correr "$BIN/boxtwin-detector" folds "${det[@]}" --semillas 42 1 2 3 4 \
      --out "$E8/$cond/folds.json"
  done
}

s11() {
  local cond raiz trabajo
  for cond in base y26; do
    raiz=$RAIZ_BASE; [[ $cond == y26 ]] && raiz=$RAIZ_Y26
    trabajo=$E8/$cond/clasificador-geo
    correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$trabajo" "$PY" "$GEO/datos.py"
    correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$trabajo" "$PY" "$GEO/features2d.py"
    correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$trabajo" "$PY" "$GEO/entrenar.py" \
      --exp E4 E4a --dispositivo cuda
    correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$trabajo" "$PY" "$GEO/evaluar.py" \
      --exp E4 E4a
  done
}

s12() {
  if [[ "${E8_DH2:-0}" == 1 ]]; then
    correr "$PY" "$G/medir_trackeo.py" --raiz-y26 "$RAIZ_Y26" --raiz-base "$RAIZ_BASE" --out "$E8/trackeo.json"
  else
    correr "$PY" "$G/medir_trackeo.py" --raiz-y26 "$RAIZ_Y26" --out "$E8/trackeo.json"
  fi
}

s13() {
  correr "$PY" "$REPO/boxtwin-annotator/scripts/medir_c1.py" medir --raiz "$RAIZ_Y26" \
    --modelo "$GUANTES" --device 0 -o "$E8/y26/c1"
}

declare -A QUE=([S7]="recache" [S8]="puente" [S9]="exports y tensores" [S10]="folds del detector"
                [S11]="E4 y E4a" [S12]="trackeo" [S13]="C1")
pedidas=("$@")
[[ ${#pedidas[@]} -gt 0 ]] || pedidas=(S7 S8 S9 S10 S11 S12 S13)
for id in "${pedidas[@]}"; do
  [[ -n "${QUE[$id]:-}" ]] || { echo "etapa desconocida: $id" >&2; exit 2; }
  etapa "$id" "${QUE[$id]}" "${id,,}"
done
