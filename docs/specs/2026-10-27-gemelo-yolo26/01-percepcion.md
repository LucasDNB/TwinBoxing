# Percepción: YOLO26l-pose, re-caché y puente de identidad

## 1. Qué cambia y qué queda fijo

Cambia una sola cosa: el checkpoint de pose, de `yolov8l-pose.pt` a `yolo26l-pose.pt`.

Queda fijo, byte a byte donde se puede comprobar:

| Pieza | Valor | Cómo se comprueba |
|---|---|---|
| Parámetros del preproceso | `imgsz` 640, `conf` 0,25, `iou` 0,7, `half` false, cortes de plano con 0,3 | `inference` del `meta.json` de cada caché |
| Tracker | `boxtwin-annotator/src/boxtwin/configs/botsort.yaml` | `tracker_config_sha256` igual en los catorce cachés (siete por condición) |
| Entorno | `twinboxing_env`: python 3.12.13, torch 2.6.0+cu124, ultralytics 8.4.41, opencv 4.13.0 | `ultralytics_version` y `torch_version` del `meta.json`; `pip freeze` en la huella |
| Identidad | `guantes-v2.pt`, perfil de color, piso 0,55, coexistencia | el `config.json` de cada medición |
| Detector | 20 features, puntaje mínimo de keypoint 0,3 (`features.MIN_SCORE`), TCN, hiperparámetros, semillas 42 1 2 3 4, siete folds, umbral 0,80 | `config` de cada corrida de `folds` |
| Clasificador | E4: 26 features, `ClasificadorGolpe`, semillas 0 a 4, los 1047 golpes, siete folds | `config.json` de cada corrida, igual salvo `exp`, `que` y `commit` |

Con YOLO26 la inferencia es de punta a punta y no hace NMS, así que `iou` no actúa. Se deja en
0,7 igual: está en el hash de configuración y sacarlo sería un segundo cambio. Se declara.

## 2. El checkpoint

- Origen: los assets de Ultralytics, nombre `yolo26l-pose.pt`. Se baja una vez a
  `~/boxtwin-gemelo/modelos/yolo26l-pose.pt` (S4) y su sha256 se anota en `05-pasos.md`. El
  runner ya lo hashea y lo escribe en el `meta.json` de cada caché.
- Entorno: la rueda de ultralytics 8.4.41, la de `twinboxing_env`, trae
  `ultralytics/cfg/models/26/yolo26-pose.yaml` (verificado el 27-10). No se instala nada.
- Licencia: AGPL-3.0 o Enterprise, la misma que YOLOv8. El costo de licencia del Capítulo 5 no
  cambia.
- Referencia en COCO, que no es la medida: mAP de pose 70,4 contra 67,6 de YOLOv8l, con 25,9 M de
  parámetros contra 44,4 M y 91,7 GFLOPs contra 168,6. COCO no mide clinch, desenfoque de golpe
  ni cuerpos chicos. Por eso E8.

## 3. Humo (fase 1, antes de cachear nada)

Sobre copias, en `~/boxtwin-gemelo/humo/`, con el código del worktree de C.

| Id | Qué | Pasa si |
|---|---|---|
| H1 | Carga y forma, en GPU y en CPU | `kpt_shape` es `[17, 3]` y una imagen de prueba da 17 keypoints por persona. El orden se comprueba sobre 50 cuadros de `01-sparring`: para cada persona detectada por las dos poses con IoU ≥ 0,5, la distancia mediana keypoint a keypoint es menor con el índice propio que con cualquier permutación izquierda-derecha |
| H2 | Tracking | `preprocess` sobre los primeros 600 cuadros de `01-sparring` con `botsort.yaml` (ReID `model: auto`) termina sin error, y sobre el tramo de `pacquiao_margarito` con los dos primeros cortes de plano el tracker se reinicia sin error (`reset()` existe) |
| H3 | Tiempo y memoria | 1.000 cuadros de `sparring-3-rounds` a 1080p, después de 100 de calentamiento, nada más en la GPU: ms por cuadro de pose y tracking, mediana y p95, y pico de VRAM, para las dos poses. Es P12e |

Si H1 falla en el orden, E8 se frena: cambiar `COCO17_NAMES` sería un segundo cambio y necesita
su propio spec.

Si H2 falla por el ReID con `model: auto`, entra la rama D-H2 del pre-registro: las dos
condiciones corren con `with_reid: false` en un yaml aparte (`botsort-sin-reid.yaml`, el mismo
archivo con esa línea cambiada) y la línea de base deja de usar los cachés originales: se
re-cachea v8 sin ReID y pasa por el puente igual que YOLO26. El ReID ya se midió sin efecto el
18-08 (31 cambios de identidad con ReID, 29 sin ReID), así que el costo para la comparación es
bajo, pero se declara. El `botsort.yaml` de A no cambia.

## 4. Re-caché de las siete fuentes

Las siete fuentes de `scripts/clasificador/comun.py`, con 180.193 cuadros en total;
`pacquiao_margarito` tiene 136.364 (el 76%). A los 29 fps medidos con YOLOv8l son unas 1 h 45;
con YOLO26 se mide y se anota.

Layout, que copia el del repo para que los scripts existentes corran con solo cambiar la raíz:

```
~/boxtwin-gemelo/raiz-y26/
├── anotacion/                Sparring
│   ├── videos/Sparring.mp4   enlace simbolico al video original (solo lectura)
│   ├── cache/                pose.npz y meta.json de YOLO26
│   ├── annotations/          el annot.json que escribe el puente (§5)
│   └── exports/
├── anotacion-spar-01/ ... anotacion-pacquiao/
```

Reglas duras, porque el preproceso ya pisó una anotación dos veces el 19-08 armando pruebas
aisladas:

1. El preproceso se llama siempre con `--project` explícito, nunca infiriendo del video.
2. `scripts/gemelo/recache.py` (T8) se niega a correr si el proyecto resuelto no está bajo
   `~/boxtwin-gemelo/`, o si el destino ya tiene un `annot.json` que no escribió el puente.
3. Antes y después de la fase 1: sha256 de las siete anotaciones, `meta.json` y `pose.npz`
   originales de la copia principal (CG4). Se listan los archivos directamente y no con
   `huellas.py`, que toma como raíz el checkout desde donde corre y desde el worktree de C miraría
   sus propias copias.
4. Reanudable por shards, como el preproceso de siempre. Cada fuente deja su tiempo y sus
   cuadros por segundo en `~/boxtwin-gemelo/e8/recache.json`.

## 5. Puente de identidad

### El problema

Las anotaciones guardan la identidad como tramos de `track_id` del caché de v8
(`identity.assignments`), más las interpolaciones que aceptó Lucas (`identity.interpolations`).
Un caché nuevo renumera los tracks y cambia sus cortes, así que esas asignaciones apuntarían a
otra persona. Los eventos no tienen el problema: van por peleador, brazo y cuadro, no por track.

### El algoritmo (`scripts/gemelo/puente_identidad.py`, T6)

Por fuente, leyendo la anotación y el caché originales de la copia principal, sin escribirlos:

1. **Verdad por cuadro.** Para cada cuadro y cada peleador, la caja observada que resuelve
   `IdentityResolver.by_fighter` sobre el caché original, sin las interpoladas. Es la misma
   verdad que usa `tools/comparar_tracker.py`.
2. **Emparejamiento.** En el mismo cuadro del caché nuevo, la detección con mayor IoU contra esa
   caja, aceptada si el IoU es ≥ 0,5 (el umbral MOT que ya usa `comparar_tracker`). Si la misma
   detección es la mejor para A y para B, se la queda el de mayor IoU y el otro queda sin
   emparejar en ese cuadro; se cuenta como conflicto.
3. **Asignaciones nuevas.** Las corridas de cuadros consecutivos con el mismo `track_id` nuevo y
   el mismo rol se escriben como `assignments`, con `origin.op` `puente` y anotador `puente-e8`.
   Un track nuevo que cubre a A en un tramo y a B en otro queda con dos asignaciones, igual que
   ya pasa en las anotaciones originales.
4. **Interpolaciones transportadas.** Una interpolación original se transporta si en el caché
   nuevo ese rol queda sin emparejar en todo el hueco y emparejado en el cuadro anterior y en el
   siguiente: se escribe con los tracks nuevos, el mismo método y `accepted_by`
   `puente-e8 (de in_XXXX)`. Las demás no se transportan. El puente no inventa interpolaciones
   que Lucas no aceptó.
5. **Todo lo demás del documento se copia igual**: `events`, `fighters`,
   `unreliable_segments`, `video`, `combo_overrides`, `process` y `counters`. `pose` apunta al
   caché nuevo, con su `meta_sha256` y `keypoint_sources` del modelo (T1). Ninguna de las siete
   fuentes tiene `manual_tracks` (relevado el 27-10); si apareciera uno, el puente se frena en
   vez de adivinar.
6. **Escritura** en `raiz-y26/<dir>/annotations/<fuente>.annot.json`, validada con
   `boxtwin.core.annotations.load`. El script se niega a escribir dentro del repo.

Deja `raiz-y26/<dir>/puente.json` con: cuadros con verdad por peleador, emparejados, conflictos,
mediana y p10 del IoU de los emparejados, interpolaciones transportadas y no transportadas,
tracks nuevos usados, y cuadros con detección de YOLO26 sin verdad.

### Lo que el puente no puede hacer

La verdad está armada sobre detecciones de YOLOv8. Eso sesga a favor de v8 en dos sentidos: el
puente castiga a YOLO26 donde dibuja la caja distinto (IoU < 0,5 con la misma persona) y no
puede premiarlo donde ve a un peleador que v8 no veía, porque ahí no hay verdad. Las métricas de
trackeo de E8 (P12c) se calculan por eso sobre los cuadros que las dos condiciones tienen
emparejados, y los cuadros sin verdad se informan aparte.

### Controles del puente

- **K1, identidad.** El puente del caché original contra sí mismo reproduce, cuadro a cuadro, el
  `track_id` de cada peleador en todos los cuadros observados, sin conflictos. Si no da 100%, el
  puente tiene un error y E8 no se lee.
- **K4, calidad.** Mediana del IoU de los emparejados ≥ 0,70 en cada fuente. Si una fuente queda
  debajo, sus números de E8 se informan con la marca "puente débil".

## 6. Cambios de código (fase 0)

Pruebas escritas antes del código, en rojo y después en verde (S1). Diff mínimo: ningún script
existente se reescribe entero. Ninguno cambia A, que corre un commit desacoplado.

| Id | Dónde | Qué cambia | Pruebas, antes del código |
|---|---|---|---|
| T1 | `boxtwin-annotator/src/boxtwin/core/project.py` | `keypoint_sources` sale del modelo del `meta.json` del caché (el nombre de `inference.model_path` sin extensión), no fijo en `yolov8l-pose` | meta con `yolo26l-pose.pt` da `yolo26l-pose`; meta sin `model_path` da `yolov8l-pose`, lo de hoy |
| T2 | `boxtwin-api/src/boxtwin_api/app.py` | `/salud` suma `modelo_pose`: `archivo` y los 12 primeros del sha256 de `cfg.modelo_pose`, calculado una vez al arrancar | archivo temporal da nombre y hash; archivo inexistente da `null` sin excepción |
| T3 | `boxtwin-annotator/src/boxtwin/mvp/orquesta.py` | `info_detector` suma las métricas medidas del archivo `<checkpoint>.metricas.json` si existe (`precision_medida`, `recall_medido`, `medido_sobre`) y un bloque `pose` con modelo y sha256 del `meta.json` de la sesión | con el archivo, la Fight-Card muestra esos números; sin él, 0,885 y 0,484 como hoy; en los dos casos valida contra `prototipo/contrato/fightcard-0.3.schema.json` |
| T4 | `scripts/clasificador_geo/entrenar_final.py` | variantes `E4-y26` y `E4a-y26`: `golpes.npz` del trabajo de E8, métricas del JSON de E8, salida en `~/boxtwin-gemelo/modelos/` | las rutas de cada variante; `E4` y `E4a` con las mismas rutas y métricas que hoy |
| T5 | `scripts/clasificador/comun.py` | `BOXTWIN_RAIZ_FUENTES`: raíz de los proyectos de anotación, por defecto la del repo | sin la variable, `ruta_annot` igual que hoy; con ella, apunta a la raíz dada |
| T6 | `scripts/gemelo/puente_identidad.py` | el puente de §5 | sobre cachés sintéticos: K1 (contra sí mismo da 100%); caja corrida con IoU < 0,5 queda sin emparejar; conflicto A-B resuelto por IoU; interpolación transportada y no transportada; un track mixto da dos asignaciones; se niega a escribir dentro del repo; la salida carga con `load` |
| T7 | `scripts/gemelo/medir_trackeo.py` | `tools/comparar_tracker.py` generalizado a las siete fuentes, con salida JSON y denominador común (§5) | cambios de id, ids por peleador y tracks mixtos sobre secuencias sintéticas; v8 contra sí mismo da cero diferencias; el denominador es la intersección |
| T8 | `scripts/gemelo/guardas.py`, `recache.py`, `e8.sh` | guardas de ruta, candado de GPU (`04-convivencia.md`, regla 3) y las etapas de la fase 1, reanudables | proyecto fuera de `~/boxtwin-gemelo` se niega; export sin `--out` se niega; base sqlite falsa con un trabajo `en_cola` o `tomado` hace esperar; salida simulada de `nvidia-smi` con otro proceso hace esperar; cada espera queda en el log |
| T9 | `despliegue/produccion.sh` | `worktree-gemelo`, `env-gemelo`, `previo c`, `verificar c` y `huella b` | ensayo fuera de la estación con rutas por entorno, como el del 26-10 (A en 8100, B en 8101, C en 8102); `verificar a` y `verificar b` dan lo mismo que antes |
| T10 | `despliegue/gemelo/` | `boxtwin-gemelo.env.ejemplo`, las dos unidades y `huella-B.sh` | `diff` de las unidades contra las de B: solo nombre, descripción, directorio, entorno y puerto |
