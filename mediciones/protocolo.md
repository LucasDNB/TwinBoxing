# Protocolo de medición — cierre de supuestos del Capítulo 5

Especificación de referencia: «Especificación de mediciones locales — cierre de
supuestos del Capítulo 5», 14-09-2026.
Ejecución: 15-09-2026, estación `desarrollo-lucas`.

Este archivo es el registro de configuración exacta. Los resultados van en
[resultados.md](resultados.md), las corridas crudas en [corridas.csv](corridas.csv)
y las muestras de potencia en `potencia.csv`.

---

## 1. Qué se mide y qué no

La especificación pide la cadena completa, seis etapas (M-01). En el repositorio
existen cuatro de esas seis como código que corre solo. Las otras dos son
decisiones que hoy toma una persona en el anotador:

| # | Etapa de la especificación | Estado en el repositorio | Se mide |
|---|---|---|---|
| 1 | Estimación de pose | `boxtwin/preprocess/runner.py`, YOLOv8l-pose | Sí |
| 2 | Seguimiento | BoT-SORT, dentro del mismo pase | Sí |
| 3 | Resolución de identidad | `core/identity.py` **aplica** intervalos que asignó una persona; no hay nada que los infiera | Sí, pero sólo el costo de aplicarlos |
| 4 | Detección de golpes | ensamble de 5 semillas, `boxtwin_detector` | Sí |
| 5 | Clasificación de familia | PoseConv3D, entorno `boxtwin_mmaction` | Sí |
| 6 | Estimación de resultado | **No existe.** `landed` es un campo que se completa a mano en el diálogo de clasificación; se lee, se exporta y se cuenta, nunca se infiere | No |

Esto está anotado como desvío D-1 y D-2 en el apartado 6.

## 2. Equipo

| | |
|---|---|
| CPU | Intel Core i7-7700K @ 4.20 GHz |
| RAM | 15 GiB |
| GPU | NVIDIA GeForce RTX 2080 SUPER, 8 GB |
| Sistema | Ubuntu 26.04 LTS, kernel 7.0.0-30-generic |
| Módulo NVIDIA | 595.84 (open kernel module) |
| Biblioteca NVML | 595.91 |

El desacople entre el módulo del kernel y NVML es el motivo del desvío D-3.

## 3. Software

| Componente | Versión |
|---|---|
| Repositorio | commit `b0f4f4d`, rama `claude/boxtwin-chapter5-measurements-4f4c5e` |
| Entorno de pose, identidad y detección | `twinboxing_env` |
| · Python | 3.12 |
| · torch | 2.6.0+cu124 |
| · ultralytics | 8.4.41 |
| · OpenCV | 4.13.0 |
| · NumPy | 2.4.3 |
| Entorno de clasificación | `boxtwin_mmaction` |
| · Python | 3.10 |
| · torch | 2.1.0+cu118 |
| · mmaction2 | 1.2.0 |
| · mmcv | 2.1.0 |

## 4. Modelos

| Etapa | Archivo | sha256 |
|---|---|---|
| Pose | `yolov8l-pose.pt` | `4be6ab6c6eca601c…` |
| Seguimiento | `boxtwin/configs/botsort.yaml` | en el `meta.json` de cada corrida |
| Detección | `modelos/detector_7fuentes.ens.pt`, 5 semillas | `1e7720b822033673…` |
| Clasificación | `modelos/poseC3D_7fuentes.pth` | `86e9a7ed409aed64…` |

Parámetros de inferencia de pose: `imgsz=640`, `conf=0.25`, `iou=0.7`, `half=False`,
`device=0`, detección de cortes de plano activada con umbral 0,3. Lote de 1 cuadro:
el pase es cuadro a cuadro porque el seguimiento tiene estado y no se puede
paralelizar sobre el eje temporal.

Umbral de decodificación del detector: 0,80, el que fijó el barrido de umbral.

## 5. Instrumentación

- **Tiempo por etapa.** `scripts/medicion_cadena.py` envuelve en tiempo de ejecución
  las funciones que ya existen (`_inferir`, el callback `on_predict_postprocess_end`
  donde BoT-SORT hace la asociación, `detectar_cortes`, `sha256_file`) y acumula por
  etapa. **No se editó una sola línea de código de producción**, así que la lógica que
  corre bajo medición es idéntica bit a bit a la que corre en serio.
- **Verificación de que la instrumentación no cambia el resultado.** Cada corrida
  compara el cache de pose recién calculado contra el que ya vive en el proyecto
  (`track_id` y `keypoints`). La columna `cache_identico` de `corridas.csv` guarda el
  veredicto. Una corrida con `no` no se usa para derivar nada.
- **Reloj.** Las duraciones se miden con `time.perf_counter()`; el inicio de cada
  etapa se guarda además en epoch UTC (`time.time()`) en `marcas.json` dentro del
  directorio de trabajo de la corrida, para poder cruzarlas con las muestras de
  potencia.
- **Potencia.** `scripts/medicion_potencia.py` muestrea
  `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used` a **0,2 s** en un
  proceso aparte, con marca de tiempo en el mismo reloj. Ver desvío D-3.

## 6. Registro de desvíos

Anotados antes de ejecutar, como pide el apartado 8 de la especificación.

| ID | Fecha | Desvío | Motivo |
|---|---|---|---|
| D-1 | 15-09 | No se mide la etapa de estimación de resultado (conectado / bloqueado / esquivado) | No existe como código. Es un campo de anotación manual. Medir su tiempo sería inventarlo |
| D-2 | 15-09 | La etapa de identidad se mide sólo como el costo de **aplicar** asignaciones ya hechas por una persona | Tampoco hay inferencia automática de identidad. El número que sale es un piso, no el costo de la etapa en un producto que la resuelva solo |
| D-3 | 15-09 | S-17 (potencia) **no se mide en esta campaña** | `nvidia-smi` no arranca: módulo del kernel 595.84 contra NVML 595.91, tras una actualización de driver sin reiniciar. CUDA computa igual, pero NVML es la única vía de telemetría de potencia y la máquina no expone ninguna alternativa por `hwmon` ni por sysfs. Se resuelve reiniciando; el muestreador queda escrito y verificado |
| D-4 | 15-09 | El proxy de reproducción queda apagado (`make_proxy=False`) | Es el video liviano que usa la interfaz del anotador para reproducir, no una etapa del análisis. Dejarlo sumaría un ffmpeg en paralelo que no aporta al informe táctico |
| D-5 | 15-09 | El ensamble del detector se **carga** de disco, no se entrena en cada corrida | Entrenar es costo de desarrollo y ocurre una vez. Meterlo en el costo por hora de video inflaría S-19 con trabajo que en producción no se repite. `tools/pipeline.py` sí entrena, y por eso no se lo usó como arnés |
| D-6 | 15-09 | El conjunto de videos son los **siete que tienen anotación**, no una muestra elegida por duración | Las etapas 3 a 5 necesitan identidad resuelta, que sólo existe donde alguien anotó. `amateur_estatico` (120 s) y el video de BoxingVI (803 s) tienen cache de pose pero no anotación, así que no pueden recorrer la cadena entera |
| D-7 | 15-09 | El ensamble usado (`detector_7fuentes`) vio las siete fuentes en entrenamiento | Irrelevante para el tiempo, que es lo que se mide acá. **Invalida cualquier número de precisión** que salga de estas corridas: para eso están los folds que dejan una fuente afuera |

## 7. Conjunto de videos

Duración tomada del contenedor con `ffprobe`, no del nombre del archivo.

| Video | Material | Duración (s) | Resolución | fps | Golpes anotados |
|---|---|---|---|---|---|
| `01-sparring.mp4` | gimnasio | 168,43 | 1920×1080 | 30,00 | 99 |
| `02-sparring.mp4` | gimnasio | 170,63 | 1920×1080 | 30,00 | 67 |
| `03-sparring.mp4` | gimnasio | 174,93 | 1920×1080 | 30,00 | 131 |
| `04-sparring.mp4` | gimnasio | 182,63 | 1920×1080 | 30,00 | 124 |
| `Sparring.mp4` | gimnasio | 186,60 | 640×360 | 29,64 | 125 |
| `sparring-3-rounds.mp4` | gimnasio | 579,97 | 1920×1080 | 30,00 | 400 |
| `pacquiao_margarito.mp4` | competencia | 2275,01 | 1920×1080 | 59,94 | 151 |

Siete videos (M-05 pide seis), cuatro resoluciones/velocidades distintas y tres
órdenes de duración: ~170 s, ~580 s y ~2275 s. Material de gimnasio y de
competencia.

**Salvedad que va al capítulo:** no es una muestra aleatoria del material que
procesaría el producto en operación. Son los videos que este proyecto anotó, y están
sesgados hacia sparring de gimnasio con cámara fija.

## 8. Ejecución

Tres corridas por video (M-06). La primera se marca `frio` e incluye la carga de
modelos a memoria; se informa aparte y no entra en la mediana (M-07). El proceso no
se reinicia entre videos.

Antes de cada campaña se verifica que no haya otra carga en la GPU (M-09). Con NVML
caído eso se verifica con `fuser -v /dev/nvidia*` en vez de `nvidia-smi`.

### Comandos

```bash
# desde /home/lucasb/Proyectos/TwinBoxing
W=boxtwin-annotator/.claude/worktrees/boxtwin-chapter5-measurements-4f4c5e
M=modelos

# muestreo de potencia, en otra terminal (requiere nvidia-smi funcionando)
~/miniforge3/envs/twinboxing_env/bin/python $W/scripts/medicion_potencia.py \
    --out $W/mediciones/potencia.csv --intervalo 0.2 --etiqueta campana-15-09

# una campaña por video
for v in anotacion-spar-01/videos/01-sparring.mp4 \
         anotacion-spar-02/videos/02-sparring.mp4 \
         anotacion-spar-03/videos/03-sparring.mp4 \
         anotacion-spar-04/videos/04-sparring.mp4 \
         anotacion/videos/Sparring.mp4 \
         anotacion-sparring-3/videos/sparring-3-rounds.mp4 \
         anotacion-pacquiao/videos/pacquiao_margarito.mp4; do
  ~/miniforge3/envs/twinboxing_env/bin/python $W/scripts/medicion_cadena.py \
      --video "$v" --corridas 3 --out $W/mediciones/corridas.csv \
      --modelo-pose yolov8l-pose.pt \
      --ensamble $M/detector_7fuentes.ens.pt \
      --config $M/poseC3D_7fuentes.py --checkpoint $M/poseC3D_7fuentes.pth
done
```
