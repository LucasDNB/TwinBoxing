#!/usr/bin/env bash
# BoxTwin - Campana de medicion del Capitulo 5. Corre la cadena sobre los siete videos
# anotados, tres corridas cada uno. Se lanza desde la raiz del repositorio principal.
#
# Pacquiao va acotado a 11400 cuadros: su anotacion cubre del 938 al 11225 de 136364, asi
# que procesarlo entero gastaria 1,75 h por corrida para que las etapas 3 a 5 trabajen
# sobre el 8% de los cuadros. Decidido con Lucas el 15-09-2026.
set -u
W=boxtwin-annotator/.claude/worktrees/boxtwin-chapter5-measurements-4f4c5e
M=modelos
PY=~/miniforge3/envs/twinboxing_env/bin/python
OUT=$W/mediciones/corridas.csv

correr () {
  echo "=========== $1 ==========="
  $PY $W/scripts/medicion_cadena.py --video "$1" --corridas 3 --out $OUT \
      --modelo-pose yolov8l-pose.pt --ensamble $M/detector_7fuentes.ens.pt \
      --config $M/poseC3D_7fuentes.py --checkpoint $M/poseC3D_7fuentes.pth ${2:-} 2>&1 \
    | grep -viE "warn|futurewarning|userwarning"
}

correr anotacion-spar-01/videos/01-sparring.mp4
correr anotacion-spar-02/videos/02-sparring.mp4
correr anotacion-spar-03/videos/03-sparring.mp4
correr anotacion-spar-04/videos/04-sparring.mp4
correr anotacion/videos/Sparring.mp4
correr anotacion-sparring-3/videos/sparring-3-rounds.mp4
correr anotacion-pacquiao/videos/pacquiao_margarito.mp4 "--max-cuadros 11400"
echo "=========== CAMPANA TERMINADA ==========="
