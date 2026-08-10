# CLAUDE.md — BoxTwin / TwinBoxing

Contexto persistente del repositorio. Leelo entero antes de tocar nada.

---

## 1. Qué es esto

BoxTwin es el Proyecto Final Integrador de Ingeniería en Informática (USAL) de
Lucas Benítez: análisis táctico automatizado de boxeadores mediante visión por
computadora. El sistema detecta peleadores, estima pose cuadro a cuadro,
mantiene identidad, clasifica acciones y arma un perfil táctico (Fight-Card) que
se muestra en un tablero web.

Lucas es practicante de boxeo. El criterio de dominio lo pone él, no vos.

**Director académico:** Ing. Esteban Tissera MBA. Sus clases rigen la estructura
de cada capítulo de la tesis.
**Tutores:** Ing. Norman Funes, Lic. Sebastián Sarasate.
**Experto de dominio consultado:** Miguel Suarez, técnico de la FAB.

---

## 2. Arquitectura del producto (decidida, no reabrir sin motivo)

Arquitectura híbrida, tres vías de despliegue:

1. Navegador móvil responsive. El usuario graba con la cámara nativa, sube el
   video, el procesamiento corre 100% en servidor con fidelidad completa.
   **Este es el caso base del producto.**
2. Navegador de escritorio con subida de archivo. Idéntico al anterior.
3. Cliente local con GPU dedicada para modo LIVE. Pose y tracking corren
   localmente con modelo reducido; la clasificación de acciones y la Fight-Card
   se delegan al servidor enviando solo la serie de keypoints (~15 MB/h contra
   3,6 GB/h de video, ratio 240:1).

El principio que sostiene la separación es la latencia: el overlay tiene un
presupuesto duro de ~40 ms y exige local, mientras que el conteo de acciones
tolera varios cientos de ms y admite nube.

El modo LIVE es la vía de alta exigencia técnica, no la puerta de entrada.

### Posición epistemológica (esto condiciona todo)

Un sistema monocular **no establece contacto físico**. De ahí se derivan tres
consecuencias que no se negocian:

- La Fight-Card distingue golpes lanzados (vía clasificador, alta precisión) de
  conectados / bloqueados / esquivados / recibidos (vía heurísticos, precisión
  esperada 60–80%, siempre declarada como estimación).
- El sistema **no emite puntuación de rounds ni veredictos de combate**.
- Toda estimación de conexión se reporta con margen de error explícito.

Si escribís prosa o código que contradiga esto, está mal.

---

## 3. Stack y entorno

- **Pose:** YOLOv8l-pose (`yolov8l-pose.pt`). BlazePose/MediaPipe es candidato
  bibliográfico, **no** el estimador elegido.
- **Tracking:** BoT-SORT.
- **Reconocimiento de acciones:** PoseConv3D (baseline). Fine-tuning previo sobre
  dataset Bhargav: 84,51% top-1 en epoch 8, 270 train / 71 val, 6 clases.
- **Backend:** Python + FastAPI. **Frontend:** React + Three.js.
  **Persistencia:** PostgreSQL. **Optimización:** TensorRT.
- **Hardware:** desktop `desarrollo-lucas` con NVIDIA RTX 2080 Super (8 GB VRAM),
  Ubuntu, más una notebook. El límite de 8 GB de VRAM es real: verificalo antes
  de proponer batch sizes o modelos grandes.

### Entornos conda (`~/miniforge3/`)

| Entorno | Para qué |
|---|---|
| `twinboxing_env` | Trabajo general, dataset, scripts de pipeline |
| `boxtwin_mmaction` | Pipeline PoseConv3D. Python 3.10, torch 2.1.0+cu118, mmcv 2.1.0, mmdet 3.2.0, mmpose 1.3.2 |

Las versiones de `boxtwin_mmaction` están pinneadas y el stack de OpenMMLab es
frágil. **No actualices nada ahí sin pedir confirmación explícita.**

### Rutas

```
~/Proyectos/TwinBoxing/          repo, branch testeoDS
├── scripts/                     scripts versionados
├── test-BoxingVI/               directorio de trabajo del dataset
│   ├── clips/<Clase>/<Video>_<inicio>_<fin>.mp4
│   └── ...
└── docs/experiments/            changelog de experimentos
```

Convención de invocación: se corre **desde `test-BoxingVI/`** llamando
`python ../scripts/nombre.py`.

---

## 4. Estado del dataset BoxingVI

Dataset público de Kumar et al. (NCVPRIPG 2025). Lo que encontramos ejecutando:

- El repo de GitHub está casi vacío, los datos viven en Google Drive.
- Se publican 10 de los 20 videos nominales. Se descargaron 9 (V6 tiene el link
  muerto: 685 anotaciones sin video, 12,6% del total nominal).
- Las anotaciones vienen en 10 formatos Excel incompatibles entre sí. Resuelto
  con un loader multi-estrategia (detección por nombre con fallback posicional).
- **El paper declara anotaciones a 30 fps y es empíricamente falso.** Los fps
  nativos son cuatro valores distintos: 24 (V1), 25 (V10), 24000/1001 = 23,976
  (V2, V3, V4, V9) y 30000/1001 = 29,97 (V5, V7, V8). Ninguno es VFR.
  Verificado con un test discriminante
  usando los videos a 29,97 como control. Los índices de frame son nativos.
  `boxingvi_clip.py` v0.2 corta sin reencodear.
- Corte completo: 4.757 de 4.757 clips, cero fallos. Distribución sobre
  `manifest.csv`: Cross 1371, Jab 1282, Lead Hook 908, Rear Uppercut 472,
  Lead Uppercut 428, Rear Hook 296.
- **El pipeline no corre sobre ese total sino sobre `manifest_filtrado.csv`:
  4.751 clips, seis menos.** Los seis descartados son degenerados: tres de 2
  frames y tres demasiado largos para contener un solo golpe (63, 37 y 111
  frames; 111 frames a 25 fps son 4,4 s). Los dos manifests conviven, así que
  todo número que vaya a la tesis tiene que declarar sobre cuál se calculó.

### Calidad de anotación (verificación manual)

Columna Clips contada sobre `manifest_filtrado.csv`.

| Video | Clips | Verificados | Correctos | Estado |
|---|---|---|---|---|
| V1 | 1863 | 6 | 1 | Roto: clases **y** segmentación temporal |
| V2 | 232 | 232 | — | Reanotado completo. Irrecuperable: 134 usables, sesgo fuerte a Jab/Cross. Ver abajo |
| V3 | 810 | 4 | 1 | Clases mal, ventanas temporales bien. Recuperable reanotando |
| V4 | 559 | 5 | 5 | Confiable |
| V5 | 594 | 4 | 3 | Confiable |
| V7 | 195 | 3 | 3 | Confiable |
| V8, V9, V10 | 498 | 0 | — | **Sin verificar.** V9 y V10 están en el split de validación |

Hallazgo importante: **la calidad no correlaciona con el tamaño del video.** V2
tiene 232 clips y está tan roto como V1 con 1863. No hay forma de inferir la
calidad de V8, V9 y V10 sin mirarlos.

Lo que dejó la reanotación completa de V2, medido sobre `reanotado.csv`:

- **98 de 232 clips (42%) no contienen ningún golpe.** La segmentación temporal
  también está rota, no solo las clases. Esto desmiente la hipótesis con la que
  se armó `boxingvi_annot.py`, que daba las ventanas de V2 y V3 por correctas.
  Para V3 la hipótesis sigue sin verificar: salió de mirar 4 clips.
- Entre los 134 usables, la etiqueta original acierta **17 veces, 12,7%**. Con 6
  clases el azar es 16,7%, así que V2 no está cerca del azar: está por debajo.
  Etiquetar al voleo habría dado mejor resultado.

### Placas de título en V1

**V1 trae 209 clips que no contienen una persona: 181 son placas de título
enteras ("ROUND 2 / 2-2 / 1 minute") y 28 cruzan el corte entre la placa y el
metraje. Es el 11,2% del video, todos con etiqueta de golpe.** Que el anotador
original le pusiera Cross a una pantalla de texto dice bastante sobre cómo se
generaron esas etiquetas.

V1 es el único video afectado. Los otros ocho no tienen un solo clip con
fracción de negro por encima del corte, así que acá el cero es una afirmación
fuerte y no un umbral que no aplica.

Las 209 caen **todas en train, ninguna en validación**, así que hoy no ensucian
ninguna métrica reportada, solo el entrenamiento. Sobre los pesos de clase el
efecto es chico: recalcularlos sin las placas los mueve como mucho 1,9%, porque
están repartidas proporcionalmente entre clases. Es un problema de calidad de
muestra, no de balance.

Advertencia metodológica, porque costó: un primer detector por brillo medio con
umbral global marcó 528 de 810 clips de V3 como placas, todos falsos positivos.
V3 es metraje real de estudio con fondo oscuro. El brillo medio confunde video
oscuro con pantalla negra, y un umbral calibrado en un video no transfiere a
otro con otra exposición. Si volvés a tocar esto, `boxingvi_placas.py` ya tiene
la prueba de dos poblaciones que evita repetir el error.

Esto no es un contratiempo, es un hallazgo metodológico sobre un dataset
publicado y va documentado como tal en el Capítulo 4.

---

## 5. Tarea en curso

Verificación ciega de V8, V9 y V10 con muestreo estratificado, 18 clips por
video, mínimo 2 por clase presente.

Criterio de decisión **fijado antes de mirar los resultados**:

| Aciertos sobre 18 | Decisión |
|---|---|
| 16 o más | El video se usa tal cual |
| 11 a 15 | Reanotación completa del video |
| 10 o menos | Se descarta |

Con 18 de 18 el límite inferior del intervalo de confianza queda cerca del 82%.
Alcanza para declarar el video usable, no para afirmar calidad del 95%. Eso se
escribe así en el capítulo, sin inflarlo.

### Después de la verificación

1. Rehacer el split. Si V9 o V10 caen, la validación se rehace entera sobre V4,
   V5 y V7, y probablemente haya que pasar a validación cruzada por video en vez
   de split fijo.
2. Recalcular pesos de clase sobre el dataset consolidado.
3. Reanotar V3 completo (~810 clips). Medición real de productividad: 2,2 s por
   clip mediana, o sea alrededor de una hora.
4. Decidir V1 con el dato de tasa de descarte que deje V3.
5. Recién ahí, Fase D: extracción de pose sobre el dataset final.

### Herramientas relevantes

- `boxingvi_clip.py` v0.2 — corte de clips sin reencoding.
- `boxingvi_split.py` — split por video y pesos de clase.
- `boxingvi_pose.py` — extracción de pose. Usa heurística de desplazamiento de
  muñeca normalizado por longitud de torso para elegir al boxeador atacante.
  Ojo: YOLOv8-pose detecta como personas a los boxeadores pintados en las
  paredes del gimnasio, y V1 trae 209 clips que son placas de título (ver §4).
- `boxingvi_placas.py` — detecta placas de título. Mide fracción de píxeles
  negros y movimiento entre primer y último frame, dos cosas que no dependen de
  la exposición del video, y antes de contar verifica que la distribución tenga
  dos poblaciones separadas. Reclasifica desde las métricas guardadas sin releer
  los videos, así que mover umbrales es instantáneo. Salida: `clips/placas.csv`.
- `boxingvi_annot.py` — reanotación web. Servidor HTTP local, playback 0,25x,
  anotación ciega con reveal opcional (tecla E), reanudable, registra tiempo de
  decisión por clip.
- `boxingvi_muestra.py` — muestreo estratificado para verificación. Determinista
  por semilla, mezcla las filas para no filtrar la etiqueta por el orden, y se
  niega a pisar una muestra ya escrita salvo `--force`: resortear después de ver
  resultados parciales anula el criterio pre-registrado. Sale en el esquema de
  `manifest_filtrado.csv`, así que `boxingvi_annot.py` la consume sin cambios.
  Muestra vigente: `clips/muestra_v8v9v10.csv`, 54 clips, seed 42, commit
  `7f6dcd5` (anterior a toda anotación).

---

## 6. Experimentos pendientes que bloquean la escritura

Ninguno de estos depende del dataset. Se pueden correr en paralelo a la
anotación. **El Capítulo 4 no se escribe sin el primero.**

1. **Benchmark de perfiles de despliegue.** YOLOv8-pose n/s/m/l a 416/480/640 px,
   exportados a TensorRT y ONNX, en desktop y notebook. Medir FPS sostenidos con
   2 peleadores y mAP de keypoints. Los perfiles A/B/C **no se declaran, se
   miden**. Cuidado con el warm-up: los benchmarks de inferencia mienten si se
   promedia desde el primer frame.
2. **Degradación del clasificador** con muestreo de pose cada dos cuadros. Un jab
   dura 200–300 ms: son ~8–9 muestras a 30 fps y ~4 a 15 fps. Decide si el
   perfil B es viable.
3. **PoC de despliegue en nube.** Containerizar el worker, procesar ~20 videos en
   instancia GPU alquilada (presupuesto < 20 USD), medir throughput y costo real
   por hora de video.
4. **Pérdida de precisión de pose entre 1080p y 720p.** Decide la estrategia de
   transcodificado en dispositivo antes de subir.

---

## 7. Cómo trabajar en este repo

### Reglas duras

- **Validar empíricamente antes de declarar.** Este proyecto ya se comió dos
  errores por creerle a una fuente: los 30 fps del paper de BoxingVI y la
  calidad de sus anotaciones. Si un número va a terminar en la tesis, se mide.
- **No corras jobs largos sin avisar.** Entrenamientos, extracción de pose sobre
  miles de clips, exports de TensorRT: proponelo, esperá el ok.
- **No toques `clips/` ni las anotaciones originales.** Todo output nuevo va a
  archivo aparte.
- **No reescribas un script existente entero.** Diff mínimo, y explicá qué
  cambia y por qué. Si el cambio es grande, mostrá el plan primero.
- **No actualices dependencias** de `boxtwin_mmaction` sin confirmación.
- Los scripts van a `scripts/`, versionados. Nada de carpetas scratch sin
  versionar: ese patrón ya causó correr versiones viejas repetidas veces.

### Estilo de código

Los scripts del proyecto llevan un docstring de cabecera con tres bloques:
**POR QUÉ EXISTE** (el problema empírico que lo motivó), **QUÉ HACE** y **USO**.
Sin tildes en los docstrings, por compatibilidad. Comentarios en español.
Argparse con defaults sensatos. Todo lo que produzca datos, reanudable.

### Git

Branch `testeoDS`. Commits con mensaje fechado.

### Documentación de experimentos

`docs/experiments/`, y el README raíz lleva entradas en formato terso:
fecha `DD-MM`, título, lista numerada de líneas cortas. Sin tablas, sin prosa
larga. Ejemplo:

```
30-07 Pipeline de dataset BoxingVI: descarga, corte y split

1. Fix de fps en boxingvi_clip.py: la v0.1 reencodeaba a 30 fps segun declara
   el paper, verificacion empirica muestra fps mixtos
2. Corte completo: 4757 de 4757 clips, cero fallos
```

---

## 8. Si escribís prosa para la tesis

Español rioplatense, primera persona, tono directo. Sin negrita en el cuerpo del
texto, sin guiones em, sin construcciones simétricas tripartitas. Párrafos
cortos, vocabulario llano. El Capítulo 2 es la referencia de registro.

Para editar texto existente: formato **ANTES / DESPUÉS / QUÉ CAMBIÓ** a nivel de
oración. Verificá si el cambio ya está aplicado antes de proponerlo. Prohibido
reescribir lo que no hace falta.

Citas bibliográficas: solo con identificador arXiv o venue real verificado.
YOLOv8 se cita como software (repositorio Ultralytics), no como paper
peer-reviewed.

Los errores tipográficos que quedaron en las entregas anteriores son
deliberados, señal de autoría humana. No los marques.

---

## 9. Cómo interactúa Lucas

Execution-first. Espera recomendaciones concretas con el razonamiento detrás, no
preguntas abiertas. Cuando haya un punto de decisión real, surfacealo con
opciones concretas y una recomendación tuya, y decí de qué depende.

Si un dato nuevo invalida una recomendación anterior, decilo explícito. Pasó con
la estimación de reanotación de V3: la medición real de 2,2 s por clip tiró abajo
la estimación previa de 4 a 8 horas.

Declará las limitaciones abiertamente. Si algo no se puede saber sin medirlo,
esa es la respuesta correcta.