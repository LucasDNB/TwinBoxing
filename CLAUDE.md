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
  nativos van de 24 a 29,97 según el video. Verificado con un test discriminante
  usando los videos a 29,97 como control. Los índices de frame son nativos.
  `boxingvi_clip.py` v0.2 corta sin reencodear.
- Corte completo: 4.757 de 4.757 clips, cero fallos.
- Distribución: Cross 1371, Jab 1282, Lead Hook 907, Rear Uppercut 473,
  Lead Uppercut 428, Rear Hook 296.

### Calidad de anotación (verificación manual)

| Video | Clips | Verificados | Correctos | Estado |
|---|---|---|---|---|
| V1 | 1863 | 6 | 1 | Roto: clases **y** segmentación temporal |
| V2 | 232 | 232 | — | Reanotado completo. Irrecuperable: 134 clips usables, sesgo fuerte a Jab/Cross |
| V3 | 811 | 4 | 1 | Clases mal, ventanas temporales bien. Recuperable reanotando |
| V4 | 559 | 5 | 5 | Confiable |
| V5 | 595 | 4 | 3 | Confiable |
| V7 | 195 | 3 | 3 | Confiable |
| V8, V9, V10 | 549 | 0 | — | **Sin verificar.** V9 y V10 están en el split de validación |

Hallazgo importante: **la calidad no correlaciona con el tamaño del video.** V2
tiene 232 clips y está tan roto como V1 con 1863. No hay forma de inferir la
calidad de V8, V9 y V10 sin mirarlos.

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
3. Reanotar V3 completo (~811 clips). Medición real de productividad: 2,2 s por
   clip mediana, o sea alrededor de una hora.
4. Decidir V1 con el dato de tasa de descarte que deje V3.
5. Recién ahí, Fase D: extracción de pose sobre el dataset final.

### Herramientas relevantes

- `boxingvi_clip.py` v0.2 — corte de clips sin reencoding.
- `boxingvi_split.py` — split por video y pesos de clase.
- `boxingvi_pose.py` — extracción de pose. Usa heurística de desplazamiento de
  muñeca normalizado por longitud de torso para elegir al boxeador atacante.
  Ojo: YOLOv8-pose detecta como personas a los boxeadores pintados en las
  paredes del gimnasio, y cuatro clips de V1 son placas de título.
- `boxingvi_annot.py` — reanotación web. Servidor HTTP local, playback 0,25x,
  anotación ciega con reveal opcional (tecla E), reanudable, registra tiempo de
  decisión por clip.
- `boxingvi_muestra.py` — muestreo estratificado para verificación.

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