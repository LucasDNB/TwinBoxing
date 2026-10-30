# E9 · RTMO-l en lugar de YOLOv8l-pose (pre-registro, fase 2)

30-10-2026. Fase 2: no corre antes del 17-11. El commit de este archivo es el pre-registro; se
puede corregir hasta antes de su humo (H9-1), con la corrección anotada como decisión. Si E8b
cambia algo de la regla de calibración que este experimento hereda, la corrección se anota
antes del humo.

## 1. Por qué este estimador

Dos razones, y la segunda es la que lo distingue de E8:

1. **Precisión de referencia.** RTMO es un estimador de una etapa que codifica los keypoints
   con clasificación de coordenadas sobre una arquitectura tipo YOLO; RTMO-l da 74,8 de AP en
   COCO val2017 y 141 fps en una V100 según su paper (arXiv 2312.07526), contra 67,6 de
   YOLOv8l-pose y 70,4 de YOLO26l-pose. **E8 ya mostró que el AP de COCO no predice el
   resultado sobre este material**, así que esto es motivo para medir, no evidencia.
2. **Licencia.** `rtmlib` (github.com/Tau-J/rtmlib) corre RTMPose, RTMO, RTMW y DWPose sin
   mmcv, mmpose ni mmdet, solo con numpy, opencv y onnxruntime, y su repositorio declara
   Apache-2.0. El Capítulo 5 carga 5.000 USD por año de licencia empresarial de Ultralytics y
   nombra "reemplazar el estimador por uno de licencia permisiva" como alternativa no adoptada.
   E9 es el primer paso medido de esa alternativa. Ver §8 para lo que **no** resuelve.

## 2. Qué cambia y qué queda fijo

Cambia el estimador: `rtmo-l_16xb16-600e_body7-640x640` en ONNX, vía `rtmlib`, a 640×640.

Queda fijo: el tracker BoT-SORT sin ReID (`botsort-sin-reid.yaml`, el de la base de E8),
alimentado con las cajas y puntajes de RTMO en lugar de los de YOLO; el puente de identidad de
E8; el detector, E4, semillas, folds y el umbral 0,80; y **la calibración de umbrales de E8b
§2**, aplicada desde el inicio (fracción por articulación igualada a la de v8 con 0,3).

## 3. Entorno

`rtmlib` necesita `onnxruntime-gpu` compatible con CUDA 12.4. Para no mezclar entornos (regla del
proyecto), va en uno nuevo, `boxtwin_rtm` (Python 3.12, numpy, opencv 4.13.0 como la estación,
`onnxruntime-gpu`, `rtmlib` fijado por versión y sha256 de la rueda). El re-caché corre ahí y
escribe el mismo formato de caché (`pose.npz` y `meta.json`) que lee `PoseCache`; el tracker
corre en `twinboxing_env` sobre las detecciones guardadas, en un segundo paso. Los dos pasos se
comunican por archivo, como `twinboxing_env` y `boxtwin_mmaction`.

## 4. Humo (antes de cachear)

| Id | Qué | Pasa si |
|---|---|---|
| H9-1 | Orden de keypoints: 17 en orden COCO | en 240 pares de cuadros de sparring-3-rounds, el error medio contra v8 con el índice propio es menor que con izquierda y derecha cambiadas (como H1 de E8) |
| H9-2 | Tracker con cajas externas | 600 cuadros de pacquiao_margarito: tracks estables, sin errores; reinicios en cortes de plano como en H2 |
| H9-3 | Tiempo y memoria en la 2080 | se informa: mediana y p95 de ms por cuadro de pose más tracking, pico de VRAM; sin umbral |
| H9-4 | Licencias | archivo `licencias.md` en `~/boxtwin-mejoras/e9/` con la licencia de `rtmlib`, la de los pesos ONNX descargados y los datasets de entrenamiento de body7 con sus términos, cada uno con la fuente consultada y la fecha |

H9-4 no frena el experimento (es investigación), pero frena cualquier cambio al Capítulo 5.

## 5. Preguntas y reglas

| Id | Pregunta | Regla |
|---|---|---|
| P17a | F1 del detector, RTMO-cal contra base v8 de E8 | la de P12a |
| P17b | Familia de E4, RTMO-cal contra base v8 | la de P12b |
| P17c | Trackeo sobre el denominador común | la de P12c, sin umbral |
| P17d | C1 sobre los cachés de RTMO, seis fuentes | sin umbral, como P12d |
| P17e | Tiempo y memoria | sin umbral, dirección declarada: más rápido que v8 |
| K | K1, K3 y K4 de E8 aplicados a RTMO; K-cal de E8b | como en E8 y E8b |

La base es la de E8 reusada, con K-reuso (sha256 de tensores) como en E8b.

## 6. Qué decide cada resultado

- **Iguala o mejora en P17a y P17b, H9-4 limpio**: el Capítulo 8 recomienda migrar el
  estimador, con el número; el Capítulo 5 puede sumar un escenario sin la licencia del
  estimador, **pero no sin la del resto de la cadena** (§8). Para A, PR propio con C1 y C5
  medidos otra vez.
- **Peor**: se informa; la licencia sigue como costo.
- Nada cambia A antes de un PR propio.

## 7. Lo que E9 no separa

- RTMO de RTMPose top-down (más preciso y más lento, necesita un detector de personas aparte):
  si RTMO queda cerca, RTMPose-l es el siguiente candidato con este mismo pre-registro.
- ViTPose (Apache-2.0, top-down, pesado) y Sapiens: Sapiens tiene licencia CC BY-NC 4.0 y queda
  descartado para el producto.

## 8. Lo que E9 no resuelve de la licencia

Sacar a Ultralytics del producto necesita tres piezas, y E9 mide una:

| Pieza | Hoy | Alternativa |
|---|---|---|
| Pose | YOLOv8l-pose (AGPL-3.0 o Enterprise) | RTMO vía rtmlib (E9) |
| Tracker | BoT-SORT de Ultralytics | OC-SORT o ByteTrack standalone (E13) |
| Detector de guantes | `guantes-v2.pt`, entrenado con Ultralytics | reentrenar con otro framework de licencia permisiva; fuera de esta campaña |

Además, `boxtwin-guantes` y el preproceso importan `ultralytics`. Hasta que las tres piezas
estén reemplazadas y medidas, el Capítulo 5 no puede sacar la licencia.
