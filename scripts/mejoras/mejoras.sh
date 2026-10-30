#!/usr/bin/env bash
# POR QUE EXISTE: la campana posterior a E8 son trece etapas, varias de una noche sobre la GPU que
# comparte A (docs/specs/2026-10-30-mejoras-post-e8, 10-pasos.md S3 a S15). Corridas a mano, una
# se lanza con trabajos de A en cola o se repite sin querer. Este es el unico camino.
# QUE HACE: antes de cada etapa espera el candado de scripts/gemelo/guardas.py (A sin trabajos
# en_cola o tomado, nvidia-smi sin otros procesos; cada espera queda en candado.log) y despues la
# corre. Una etapa terminada deja etapas/<id>.hecho y no se repite; adentro, cada script es
# reanudable. Nada escribe en A, en B, en C ni en los caches de E8: lo nuevo va a ~/boxtwin-mejoras,
# y los datos de E4 de E8 se copian antes de rearmar golpes.npz. Con MEJORAS_ENSAYO=1 imprime los
# comandos sin correr nada. S11 necesita MEJORAS_E11_VIDEOS (los videos de L3, separados por
# espacios). S8c (0,20 s) y P16e (E4 con guante) no estan: son informativas y quedan para si sobra.
# Las etapas se piden por nombre: entre ellas hay tareas de Lucas (L2 antes de S14, L3 antes de S11).
# USO: bash scripts/mejoras/mejoras.sh S3          # una etapa
#      bash scripts/mejoras/mejoras.sh S4 S5 S6    # varias, en ese orden
#      MEJORAS_ENSAYO=1 bash scripts/mejoras/mejoras.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MEJORAS=${BOXTWIN_MEJORAS:-$HOME/boxtwin-mejoras}
GEMELO=${BOXTWIN_GEMELO:-$HOME/boxtwin-gemelo}
REPO_A=${BOXTWIN_REPO_A:-$HOME/Proyectos/TwinBoxing}
PY=${PY:-$HOME/miniforge3/envs/twinboxing_env/bin/python}
BIN=$(dirname "$PY")
M=$REPO/scripts/mejoras
GEO=$REPO/scripts/clasificador_geo
E8=$GEMELO/e8
RAIZ_V8=$GEMELO/raiz-v8
RAIZ_Y26=$GEMELO/raiz-y26
GUANTES=${MEJORAS_GUANTES:-$REPO_A/boxtwin-guantes/modelos/guantes-v2.pt}
UMBRALES=$MEJORAS/e8b/umbrales.json

# El codigo de este checkout (el worktree fijo de S2), no el de A
export PYTHONPATH=$REPO/boxtwin-annotator/src:$REPO/boxtwin-api/src:$REPO/boxtwin-detector/src${PYTHONPATH:+:$PYTHONPATH}
export BOXTWIN_MEJORAS=$MEJORAS BOXTWIN_GEMELO=$GEMELO BOXTWIN_REPO_A=$REPO_A

ensayo() { [[ "${MEJORAS_ENSAYO:-0}" == 1 ]]; }
correr() {
  echo "+ $*"
  ensayo || env "$@"
}

etapa() {
  local id=$1 que=$2
  shift 2
  if [[ -f "$MEJORAS/etapas/$id.hecho" ]]; then
    echo "$id ya hecha ($MEJORAS/etapas/$id.hecho), se saltea"
    return 0
  fi
  echo "+ candado $id $que"
  # el candado de E8 tal cual: con BOXTWIN_GEMELO apuntando aca, su log cae bajo ~/boxtwin-mejoras
  ensayo || BOXTWIN_GEMELO=$MEJORAS "$PY" "$REPO/scripts/gemelo/guardas.py" candado "$id $que" \
    --log "$MEJORAS/candado.log"
  local t0=$SECONDS
  "$@"
  if ! ensayo; then
    mkdir -p "$MEJORAS/etapas"
    echo "$(date -Iseconds) $id $que $((SECONDS - t0)) s $(git -C "$REPO" rev-parse --short HEAD)" >> "$MEJORAS/etapas.log"
    touch "$MEJORAS/etapas/$id.hecho"
  fi
}

# build del detector sobre los exports de E8 de una condicion: cond (base|y26), destino, opciones
build() {
  local cond=$1 destino=$2
  shift 2
  local npz=("$E8/$cond/exports/"*.sequence.npz)   # en ensayo queda el patron, sin expandir
  correr "$BIN/boxtwin-detector" build "${npz[@]}" --out "$destino/det" "$@"
}

# golpes.npz de E4 de una condicion, sobre una copia de los datos de E8: cond, destino, env extra
e4_datos() {
  local cond=$1 destino=$2
  shift 2
  correr mkdir -p "$destino/clasificador-geo"
  correr cp -r --no-preserve=mode "$E8/$cond/clasificador-geo/datos" "$destino/clasificador-geo/"
  # el golpes.npz copiado es el de E8: si quedara y features2d fallara, K-reuso compararia E8
  # contra si mismo y pasaria
  correr rm -f "$destino/clasificador-geo/datos/golpes.npz" "$destino/clasificador-geo/datos/golpes2d.meta.json"
  correr BOXTWIN_CLASIFICADOR_GEO="$destino/clasificador-geo" "$@" "$PY" "$GEO/features2d.py"
}

# folds del detector y E4 completo de una condicion ya armada: destino, raiz de fuentes
entrenar() {
  local destino=$1 raiz=$2
  local det=("$destino/det/"*.det.npz)
  correr "$BIN/boxtwin-detector" folds "${det[@]}" --semillas 42 1 2 3 4 --out "$destino/folds.json"
  correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$destino/clasificador-geo" \
    "$PY" "$GEO/entrenar.py" --exp E4 --dispositivo cuda
  correr BOXTWIN_RAIZ_FUENTES="$raiz" BOXTWIN_CLASIFICADOR_GEO="$destino/clasificador-geo" \
    "$PY" "$GEO/evaluar.py" --exp E4
}

s3() {
  local cond
  for cond in base y26; do
    build "$cond" "$MEJORAS/kreuso/$cond"
    e4_datos "$cond" "$MEJORAS/kreuso/$cond"
    correr "$PY" "$M/kreuso.py" "$E8/$cond" "$MEJORAS/kreuso/$cond" --out "$MEJORAS/kreuso/$cond.json"
  done
}

s4() {
  correr "$PY" "$M/calibrar_umbrales.py" --out "$UMBRALES"
  build y26 "$MEJORAS/e8b/y26cal" --umbrales "$UMBRALES"
  e4_datos y26 "$MEJORAS/e8b/y26cal" BOXTWIN_UMBRALES="$UMBRALES"
}

s5() { entrenar "$MEJORAS/e8b/y26cal" "$RAIZ_Y26"; }
s6() { correr "$PY" "$M/lectura.py" e8b; }

s7() {
  correr "$PY" "$M/huecos.py" "$E8/base/exports/"*.sequence.npz --out "$MEJORAS/e10/huecos-v8.json"
  correr "$PY" "$M/huecos.py" "$E8/y26/exports/"*.sequence.npz --out "$MEJORAS/e10/huecos-y26.json"
  if ensayo || [[ -f "$UMBRALES" ]]; then
    correr "$PY" "$M/huecos.py" "$E8/y26/exports/"*.sequence.npz --umbrales "$UMBRALES" \
      --out "$MEJORAS/e10/huecos-y26cal.json"
  fi
}

s8() {
  build base "$MEJORAS/e10/v8r" --relleno
  e4_datos base "$MEJORAS/e10/v8r" BOXTWIN_RELLENO=1
  entrenar "$MEJORAS/e10/v8r" "$RAIZ_V8"
  # S8b: Y26-cal+R solo si E8b ya corrio
  if ensayo || [[ -f "$MEJORAS/etapas/S5.hecho" ]]; then
    build y26 "$MEJORAS/e10/y26calr" --umbrales "$UMBRALES" --relleno
    e4_datos y26 "$MEJORAS/e10/y26calr" BOXTWIN_UMBRALES="$UMBRALES" BOXTWIN_RELLENO=1
    entrenar "$MEJORAS/e10/y26calr" "$RAIZ_Y26"
  fi
}

s9() { correr "$PY" "$M/lectura.py" e10; }
s10() { correr "$PY" "$M/medir_e11.py" desarrollo; }

s11() {
  local videos=${MEJORAS_E11_VIDEOS:-}
  if [[ -z "$videos" ]]; then
    ensayo || { echo "S11 necesita MEJORAS_E11_VIDEOS (L3)" >&2; return 2; }
    videos="<videos-de-L3>"
  fi
  local v
  for v in $videos; do
    correr "$PY" "$REPO/boxtwin-annotator/scripts/medir_c1.py" medir --video "$v" --modelo "$GUANTES" \
      --device 0 -o "$MEJORAS/e11/evidencia"
  done
  # shellcheck disable=SC2086
  correr "$PY" "$M/medir_e11.py" confirmacion --video $videos
}

s12() { correr "$PY" "$M/lectura.py" e11; }
s13() { correr "$PY" "$M/guantes_por_cuadro.py" --device 0 --modelo "$GUANTES"; }

s14() {
  build base "$MEJORAS/e12/v8g" --guantes "$MEJORAS/e12/guantes"
  local det=("$MEJORAS/e12/v8g/det/"*.det.npz)
  correr "$BIN/boxtwin-detector" folds "${det[@]}" --semillas 42 1 2 3 4 --out "$MEJORAS/e12/v8g/folds.json"
}

s15() { correr "$PY" "$M/lectura.py" e12; }

declare -A QUE=([S3]="k-reuso" [S4]="e8b calibracion y build" [S5]="e8b folds y E4" [S6]="e8b lectura"
                [S7]="e10 huecos" [S8]="e10 relleno" [S9]="e10 lectura" [S10]="e11 desarrollo"
                [S11]="e11 confirmacion" [S12]="e11 lectura" [S13]="e12 guantes por cuadro"
                [S14]="e12 build y folds" [S15]="e12 lectura")
pedidas=("$@")
if [[ ${#pedidas[@]} -eq 0 ]]; then
  echo "uso: mejoras.sh S3 [S4 ...]   etapas: S3 a S15 (10-pasos.md)" >&2
  exit 2
fi
for id in "${pedidas[@]}"; do
  [[ -n "${QUE[$id]:-}" ]] || { echo "etapa desconocida: $id" >&2; exit 2; }
  etapa "$id" "${QUE[$id]}" "${id,,}"
done
