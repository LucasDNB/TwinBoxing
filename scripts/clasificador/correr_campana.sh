#!/usr/bin/env bash
# BoxTwin - Corre la campana de entrenamiento y prueba del clasificador (14-10-2026).
#
# POR QUE EXISTE
#   Son 63 corridas de entrenamiento mas su prueba: horas de GPU que tienen que poder cortarse
#   y retomarse sin repetir nada ni dejar resultados a medias. Tambien comparten la placa con
#   el worker del producto y con las mediciones de tiempo (benchmark de perfiles, C5): una
#   corrida de entrenamiento al mismo tiempo que una medicion de latencia la arruina sin que
#   el numero lo delate. Por eso espera a que la GPU este libre antes de cada corrida.
#
# QUE HACE
#   Recorre corridas.tsv en orden. Saltea la corrida si ya tiene pred.pkl. Antes de cada una
#   espera a que no haya otro proceso de computo en la GPU. Entrena, prueba el ultimo
#   checkpoint con --dump y escribe tiempo.json. Se detiene limpio despues de la corrida en
#   curso si existe el archivo PAUSA en el directorio de trabajo.
#
# USO (dentro de tmux, para que sobreviva a la sesion)
#   tmux new -d -s campana 'bash scripts/clasificador/correr_campana.sh'
#   bash scripts/clasificador/correr_campana.sh --solo 1     # una corrida, para cronometrar
#   touch ~/boxtwin/clasificador-2026-10/PAUSA               # parar despues de la actual

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TRABAJO="${BOXTWIN_CLASIFICADOR:-$HOME/boxtwin/clasificador-2026-10}"
PY="${BOXTWIN_PY_MMACTION:-$HOME/miniforge3/envs/boxtwin_mmaction/bin/python}"
MM="$REPO/test-poseConvBoxing/mmaction2"
LOG="$TRABAJO/campana.log"
SOLO=0
[[ "${1:-}" == "--solo" ]] && SOLO="${2:-1}"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

gpu_ocupada() {
  # Cualquier proceso de computo que no sea de esta campana cuenta como ocupada.
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

[[ -f "$TRABAJO/corridas.tsv" ]] || { echo "falta $TRABAJO/corridas.tsv (correr generar_configs.py)"; exit 1; }
[[ -x "$PY" ]] || { echo "no encuentro el python de boxtwin_mmaction en $PY"; exit 1; }

hechas=0
cd "$REPO"
while IFS=$'\t' read -r exp fuente semilla cfg wd; do
  [[ -z "$exp" ]] && continue
  if [[ -f "$wd/pred.pkl" ]]; then continue; fi
  if [[ -f "$TRABAJO/PAUSA" ]]; then log "PAUSA encontrada: me detengo"; exit 0; fi
  if [[ "$SOLO" != 0 && "$hechas" -ge "$SOLO" ]]; then log "--solo $SOLO cumplido"; exit 0; fi

  while gpu_ocupada; do
    log "GPU ocupada por otro proceso; espero 5 min antes de $exp/$fuente/s$semilla"
    sleep 300 < /dev/null
  done

  mkdir -p "$wd"
  log "arranca $exp/$fuente/s$semilla"
  t0=$(date +%s)
  if ! "$PY" "$MM/tools/train.py" "$cfg" --work-dir "$wd" < /dev/null > "$wd/train.log" 2>&1; then
    log "FALLO entrenamiento $exp/$fuente/s$semilla (ver $wd/train.log)"; exit 1
  fi
  t1=$(date +%s)
  ckpt="$(cat "$wd/last_checkpoint" 2>/dev/null)"
  if [[ ! -f "$ckpt" ]]; then log "FALLO: no hay ultimo checkpoint en $wd"; exit 1; fi
  if ! "$PY" "$MM/tools/test.py" "$cfg" "$ckpt" --dump "$wd/pred_parcial.pkl" --work-dir "$wd/prueba" < /dev/null > "$wd/test.log" 2>&1; then
    log "FALLO prueba $exp/$fuente/s$semilla (ver $wd/test.log)"; exit 1
  fi
  mv "$wd/pred_parcial.pkl" "$wd/pred.pkl"
  t2=$(date +%s)
  printf '{"entrenamiento_s": %d, "prueba_s": %d, "checkpoint": "%s"}\n' $((t1-t0)) $((t2-t1)) "$ckpt" > "$wd/tiempo.json"
  log "listo $exp/$fuente/s$semilla en $((t2-t0)) s"
  hechas=$((hechas+1))
done < "$TRABAJO/corridas.tsv"
log "campana completa"
