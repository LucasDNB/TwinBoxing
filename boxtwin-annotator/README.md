# boxtwin-annotator

Anotador de eventos de golpe sobre video continuo, con overlay de pose y gestion de
identidad de peleadores. Herramienta interna de investigacion de BoxTwin, no un producto.

Existe porque los datasets publicos de boxeo clasifican por clip completo y no por evento
dentro de un video, y porque su calidad de anotacion no resistio la verificacion: en
BoxingVI, la reanotacion completa de V2 encontro 42% de clips sin ningun golpe y 7,3% de
acierto de etiqueta, y V1 trae 209 clips que son placas de titulo con etiqueta de golpe.

## Estado

Bloques 1 a 3 de 7 terminados: modelo de datos, preproceso y reproductor con overlay.
Todavia no se pueden anotar eventos.

| Bloque | Que | Estado |
|---|---|---|
| 1 | Modelo de datos en `core/` con validaciones y tests | hecho |
| 2 | CLI de preproceso, reanudable, con proxy | hecho |
| 3 | Reproductor de escritorio con overlay y navegacion frame-exacta | hecho |
| 4 | Anotacion de eventos con atajos de teclado y persistencia | pendiente |
| 5 | Gestion de identidad | pendiente |
| 6 | Exports (clips, mmaction, sequence, stats) | pendiente |
| 7 | Reanotacion ciega y reporte de acuerdo | pendiente |

## Instalacion

En la maquina de desarrollo corre sobre `twinboxing_env`, que ya trae PySide6,
ultralytics, opencv y numpy:

```bash
conda run -n twinboxing_env pip install -e ".[dev]"
```

Desde cero en otra maquina, ver `environment.yml`.

`boxtwin_mmaction` no se toca: el anotador nunca lo importa y ese entorno esta pinneado.

## Tests

```bash
conda run -n twinboxing_env python -m pytest
```

## Preproceso

```bash
boxtwin-annotator preprocess proyecto/videos/spar.mp4
```

Escribe `cache/spar.pose.npz`, `cache/spar.meta.json` y `cache/spar.proxy.mp4`. Es
reanudable: si el proceso se corta, el mismo comando retoma desde el ultimo shard.

Para verificar metadatos sin procesar nada, que en este proyecto ya hizo falta dos veces:

```bash
boxtwin-annotator probe proyecto/videos/spar.mp4 --count
```

Numeros medidos en esta maquina (RTX 2080 Super, yolov8l-pose, imgsz 640, BoT-SORT con
ReID), no estimados:

| Que | Valor |
|---|---|
| Inferencia con tracking | 29 fps sostenidos, 34,5 ms por cuadro |
| VRAM | 0,26 GB de pico |
| Proxy 960 px de 60 s de 1080p | 5,5 s |
| Costo del ReID | cero, `model: auto` reusa las features del detector |

O sea, un video de 24.000 cuadros son unos 14 minutos, y el proxy se genera en paralelo
sin costo de reloj porque la inferencia es GPU-bound y el encoding es CPU-bound.

### Reanudacion y costuras de identidad

BoT-SORT tiene estado. Si el proceso muere en el cuadro 10.000 y arranca de nuevo, el
tracker vuelve a numerar desde 1 y esos ids colisionarian con los del tramo anterior: una
asignacion de identidad hecha sobre el track 1 del primer tramo se aplicaria en silencio
al track 1 del segundo, que es otra persona.

Por eso los ids que emite el tracker se desplazan por un offset persistido, y cada punto de
reanudacion queda registrado en `meta.json` como una costura:

```json
"resume_seams": [{"frame": 122, "id_offset": 2}]
```

En una costura ningun track cruza de un lado al otro, asi que hay un corte de identidad
garantizado que se resuelve con un re-seed. Verificado sobre un clip real: interrumpir y
reanudar produce exactamente el mismo conteo de detecciones que la corrida entera y una
sola division de track, sin ninguna colision de id.

### Que no se persiste

Solo las detecciones a las que el tracker les puso id. Una deteccion sin track no se puede
asignar a un peleador ni encadenar con el cuadro siguiente, asi que en el cache seria
ruido. Se cuentan aparte en `counts.untracked_dropped` para que el descarte sea medible.

### Limitacion conocida

`cv2.VideoCapture.read()` devuelve False tanto al terminar el archivo como ante un error de
decodificacion a mitad, y opencv no distingue los dos casos. Lo unico observable es que
salgan bastantes menos cuadros de los estimados; cuando pasa, el comando lo reporta como
`short_decode` y hay que verificar con `probe --count`. No se puede afirmar cual de las dos
cosas ocurrio.

## Como se usa

```bash
boxtwin-annotator annotate proyecto/videos/spar.mp4
```

Es una aplicacion de escritorio en PySide6. Necesita sesion grafica; si no la hay, falla
con una explicacion en vez de abortar.

### Sin pantalla en la maquina

El equipo con GPU puede ser headless. Hay dos caminos y ninguno necesita instalar nada del
lado del servidor.

**Servir la ventana por VNC.** El plugin viene con Qt:

```bash
boxtwin-annotator annotate proyecto/videos/spar.mp4 --platform vnc
```

Escucha en el 5900 y sirve una pantalla virtual de 1600x1000, porque el default del plugin
es 1024x768 y ahi la ventana queda recortada. Desde la otra maquina, tunel y cliente VNC:

```bash
ssh -L 15900:localhost:5900 usuario@maquina
```

El puerto local se sugiere distinto del remoto a proposito: en Windows, Hyper-V y WSL
reservan rangos que suelen incluir el 5900, y ahi ssh falla con `bind: Permission denied`,
que parece un problema del servidor y no lo es.

**Anotar en otra maquina.** Anotar no necesita GPU, asi que se puede mover el cache en vez
de servir la ventana:

```bash
boxtwin-annotator bundle proyecto/videos/spar.mp4 --out /tmp/spar_bundle
pip install -e ".[gui]"     # del otro lado: PySide6, opencv y pyyaml, sin torch
```

Con `--with-video` incluye el original, que habilita el zoom en resolucion completa. Por
cada 10 minutos de video: el npz son 5 MB, el proxy 97 MB, el original 54 MB en 1080p y
490 MB en 4K. En 1080p el proxy pesa mas que el original porque usa GOP 12; ahi conviene
mover el original y generar el proxy del otro lado con `boxtwin-annotator proxy`, que solo
necesita ffmpeg.

## Reproductor

Navegacion frame-exacta: todo se indexa por numero de cuadro, nunca por timestamp. Con un
fps de 30000/1001, indexar por tiempo acumula deriva y el cuadro que devuelve un salto deja
de ser el que se pidio.

Latencia medida por cuadro, incluyendo decodificacion, resolucion de identidad y repintado:

| | mediana | p95 | peor |
|---|---|---|---|
| Adelante | 4,97 ms | 5,48 ms | 7,62 ms |
| Atras | 3,63 ms | 4,10 ms | 60,40 ms |
| Atras dentro del buffer | 3,62 ms | 3,84 ms | 4,18 ms |
| Saltos al azar (scrub) | 3,70 ms | 36,83 ms | 46,03 ms |

El presupuesto es 33,4 ms por cuadro a 1,0x y 133,5 ms a 0,25x. El peor caso hacia atras
son 60 ms y es el rellenado del buffer, que ocurre cada ~48 cuadros.

Velocidades 0,10x / 0,25x / 0,50x / 1,0x, hacia adelante y hacia atras, con 0,25x por
defecto, que es donde se distingue el inicio de la extension del codo.

**El reloj avanza exactamente un cuadro por tic y nunca descarta.** Si la maquina no llega,
la reproduccion se pone lenta, que es visible; descartar cuadros seria invisible y haria
que el anotador etiquete sobre una version del video que no es la que va a exportar.

### Fuente de imagen

Se reproduce del proxy. En pausa y con zoom sobre 1,6x se decodifica el cuadro del video
original, que cuesta unos 6 ms sobre 1080p. Ampliar el proxy 4x deja la imagen tan borrosa
que juzgar si un guante llego a la cara se vuelve adivinanza, y ese juicio es justamente lo
que se esta anotando.

Cambiar de fuente no mueve ni un keypoint porque hay un solo sistema de coordenadas:
pixeles del video original. La imagen se estira a ese tamano al dibujarla y todo lo demas
se pinta ahi. Si cada capa aplicara su propia conversion, el esqueleto se correria del
cuerpo lo suficiente para no notarlo y anotar mal.

### Overlay

Color por ROL y nunca por track_id. Un track_id cambia en cada oclusion y en cada
reanudacion del preproceso; si el color lo siguiera, el mismo peleador cambiaria de color
solo y se perderia la senal que sirve para detectar un intercambio de identidad.

Los keypoints por debajo del umbral de confianza se dibujan atenuados y no se ocultan: una
pose mala y una pose incompleta son cosas distintas y se corrigen distinto.

## Arquitectura

Dos fases separadas. **No se infiere pose durante la reproduccion**: la inferencia en vivo
hace que retroceder sea inusable y no alcanza framerate con la UI encima.

1. **Preproceso** (CLI, una vez por video). Extrae metadatos reales con ffprobe, corre
   YOLOv8l-pose con BoT-SORT sobre todos los frames y escribe `<video>.pose.npz` mas
   `<video>.meta.json`. El archivo de pose es inmutable: la anotacion nunca lo modifica.
2. **Anotador** (GUI). Lee el video y su cache de pose, dibuja el overlay y permite
   anotar. La identidad y los eventos viven en `<video>.annot.json`.

Separacion estricta de dependencias: `core/` no importa PySide6, torch, ultralytics ni
opencv. `gui/` y `preprocess/` van encima. Todo el pipeline de export tiene que poder
correr sin GUI. Lo verifica `tests/test_core_no_qt.py`.

### Layout del proyecto de anotacion

```
project/
  videos/
  cache/
    <video>.pose.npz        # inmutable
    <video>.meta.json
    <video>.proxy.mp4       # 960 px, GOP 12, para el reproductor
  annotations/
    <video>.annot.json      # fuente de verdad
    <video>.reanno.json
  exports/
  config.yaml
```

## Convenciones del esquema

- **Los eventos son inclusivos** (`start_frame`, `end_frame`). Los rangos de identidad y
  los tramos no confiables son **semiabiertos** y por eso su campo se llama
  `end_frame_excl`. La diferencia de nombre es la que evita confundirlos leyendo el JSON.
- Se anota por lado (`left`/`right`), no por rol (`lead`/`rear`), porque el lado es lo que
  el modelo observa. El rol se deriva combinando lado y guardia. La relacion no se invierte.
- Tipo, altura y lado son campos independientes. La agregacion a clases se decide en el
  export, asi un error de etiqueta se localiza en la dimension que lo produjo.
- El orden de las claves es el de declaracion de los modelos, no alfabetico: alfabetico
  separaria `start_frame` de `end_frame` y el archivo dejaria de leerse solo.
- Los `null` se escriben explicitos. Omitirlos ahorraba 7% de lineas en un archivo real y
  costaba que cada registro dejara de mostrar su forma completa.

### Fronteras temporales, definicion operacional

La consistencia entre sesiones depende de estas definiciones. La GUI las muestra en un
panel fijo y `settings_snapshot.boundary_definitions_version` registra bajo cual se anoto
cada evento.

- `start_frame` (onset): primer cuadro en que el puno inicia el desplazamiento hacia el
  objetivo, con el codo empezando a extenderse o el hombro rotando. No el cuadro en que se
  carga el peso.
- `peak_frame`: cuadro de maxima extension del brazo o de contacto, lo que ocurra primero.
- `end_frame` (offset): cuadro en que el puno retrocedio aproximadamente la mitad del
  recorrido de vuelta hacia la guardia.
- `feint`: el movimiento inicia pero se aborta antes de alcanzar el 60% de la extension
  esperada y no hay retraccion de recuperacion completa.

### Combinaciones

Un golpe puede empezar antes de que termine el anterior. El solapamiento entre golpes de
**lados distintos** del mismo peleador es una combinacion y no genera ninguna advertencia:
avisarlo haria saltar la advertencia en casi todos los intercambios y entrenaria al
anotador a ignorarlas. Solo se avisa el solapamiento del **mismo brazo**, que salvo doble
jab es fisicamente sospechoso.

Por eso el export `sequence` usa un carril BIO por brazo y no por peleador: con un solo
carril, un 1-2 obliga a descartar uno de los dos golpes.

### Guantes

Los indices 17 y 18 estan reservados para el guante izquierdo y derecho. No viven en el
npz: los derivados se extrapolan al leer sobre el vector codo-muneca y los detectados
saldrian de un detector propio.

El guante derivado **no aporta informacion a un clasificador** que ya recibe codo y
muneca, porque es una funcion determinista de los dos. Sirve para el overlay, donde el
anotador ve la posicion estimada mientras juzga `landed`, y para los heuristicos de
conexion. El detectado si seria observacion nueva y resolveria el caso en que la muneca se
pierde por desenfoque de movimiento, que es justo el instante del golpe.

## Codigos de validacion

`core/validation.py` devuelve Issues que no impiden abrir ni guardar el archivo. La lista
completa esta en `ISSUE_CODES`.

| Nivel | Codigos |
|---|---|
| error | `EV_OUT_OF_BOUNDS`, `ID_ASSIGNMENT_OVERLAP`, `ID_ROLE_COLLISION`, `ID_ORPHAN_INTERP`, `CB_UNKNOWN_EVENT` |
| warning | `EV_TOO_LONG`, `EV_OVERLAP_SAME_SIDE`, `EV_OVERLAP_SAME_SIDE_LONG`, `EV_REFIRE_TOO_FAST`, `EV_UNRELIABLE_BUT_CLEAN`, `EV_NO_IDENTITY`, `FG_GUARD_OVERRIDE_OVERLAP`, `ID_INTERP_TOO_LONG` |
| info | `EV_GUARD_MISMATCH`, `PM_MISSING_METRICS` |

## Esquema de teclas

Se define en `gui/keymap.py` y es remapeable desde `config.yaml`. Las acciones de
anotacion e identidad ya estan declaradas aunque sus manejadores lleguen en los bloques 4
y 5, para que el archivo del usuario no cambie de forma despues.
