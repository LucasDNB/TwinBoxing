# boxtwin-annotator

Anotador de eventos de golpe sobre video continuo, con overlay de pose y gestion de
identidad de peleadores. Herramienta interna de investigacion de BoxTwin, no un producto.

Existe porque los datasets publicos de boxeo clasifican por clip completo y no por evento
dentro de un video, y porque su calidad de anotacion no resistio la verificacion: en
BoxingVI, la reanotacion completa de V2 encontro 42% de clips sin ningun golpe y 7,3% de
acierto de etiqueta, y V1 trae 209 clips que son placas de titulo con etiqueta de golpe.

## Estado

Los siete bloques terminados y sin pendientes abiertos. El pipeline va del video crudo al
dataset exportado, con su medida de acuerdo intra-anotador.

| Bloque | Que | Estado |
|---|---|---|
| 1 | Modelo de datos en `core/` con validaciones y tests | hecho |
| 2 | CLI de preproceso, reanudable, con proxy | hecho |
| 3 | Reproductor de escritorio con overlay y navegacion frame-exacta | hecho |
| 4 | Anotacion de eventos con atajos de teclado y persistencia | hecho |
| 5 | Gestion de identidad | hecho |
| 6 | Exports (clips, mmaction, sequence, stats) | hecho |
| 7 | Reanotacion ciega y reporte de acuerdo | hecho |

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

## Anotacion

Se marca sobre el reproductor y se clasifica en un dialogo que se completa entero con el
teclado. La anotacion con mouse no escala: la medicion previa del proyecto dio 2,2 s por
decision, y ese numero solo se sostiene si la mano no se mueve del teclado.

| Tecla | Que hace |
|---|---|
| `[` | marca el inicio y abre un evento |
| `]` | marca el final y abre el dialogo de clasificacion |
| `Z` | descarta el evento en curso |
| `1` `2` | elige fighter_A / fighter_B |
| `Supr` | borra el evento seleccionado |
| `Ctrl+Z` / `Ctrl+Shift+Z` | deshacer / rehacer |
| `Ctrl+S` | guardar |

Dentro del dialogo:

| Tecla | Que hace |
|---|---|
| `Q` `W` | lado izquierdo / derecho |
| `A` `S` `D` | recto / hook / uppercut |
| `H` `B` | cabeza / cuerpo |
| `F` | amague (se aprieta de nuevo y vuelve a completo) |
| `Shift+F` | abortado |
| `4`–`8` | conecta, bloqueado, esquivado, falla, sin datos |
| `C` `V` `N` | calidad limpia, con oclusion, ambigua |
| `P` | marca el pico en el cuadro del preview |
| `[` `]` | remarca las fronteras sobre el preview |
| `Espacio` | pausa el preview |
| `Enter` / `Esc` | confirma / cancela |

Las teclas se validan **por contexto** y no globalmente. `Espacio` es reproducir en el
reproductor y pausar el preview en el dialogo, y son la misma tecla a proposito: el dialogo
esta modal encima, asi que nunca compiten.

### Decisiones que no son de comodidad

**El clip se reproduce en loop a 0,25x mientras se clasifica.** Decidir si un golpe fue
hook o uppercut mirando un cuadro congelado es adivinar: lo que distingue los dos es la
trayectoria. El loop ademas cuenta las vueltas, que es el proxy directo de dificultad de la
decision y queda en las metricas.

**Ningun campo que define la clase arranca elegido.** Preseleccionar el tipo mas frecuente
ahorraria teclas y sesgaria el dataset hacia esa clase cada vez que se confirme sin mirar.
Los que tienen default en el esquema (`landed` sin datos, `quality` limpia) si vienen
puestos, porque su default es una afirmacion honesta y no una suposicion.

**La guardia se hereda de la vigente para ese peleador en el inicio del golpe.** El rol
lead/rear se deriva de ahi en el export, nunca al reves.

**Los ids no se reusan aunque se cancele el dialogo.** El contador no retrocede: una
referencia externa, por ejemplo la de un `reanno.json`, no puede pasar a apuntar a otro
evento.

### Persistencia y deshacer

Se guarda en cada evento confirmado y ademas cada 30 segundos. Con guardado solo por
temporizador, un corte costaria hasta media docena de golpes.

Todo cambio pasa por el historial, incluidas las correcciones en la tabla: una edicion
tambien se puede errar, y es el momento en que menos se esta prestando atencion. El
historial guarda 100 pasos y los comandos son reversibles, no copias del documento: mil
eventos con sus metricas pesan varios megas y cincuenta copias serian cientos.

### Metricas de proceso

Por evento se registra quien anoto, en que sesion, cuando se creo y se confirmo, los
milisegundos de trabajo activo, cuantas vueltas del preview hubo y cuantas ediciones
posteriores.

El tiempo que se cuenta es **activo**, no de reloj de pared: el cronometro se detiene
cuando la ventana pierde el foco y cuando pasan 15 segundos sin actividad. Medido de reloj
de pared, una sesion de tres horas con dos de almuerzo reporta tres horas y el numero deja
de servir para estimar, para comparar la dificultad de dos videos o para escribirlo en el
capitulo.

### Balance de clases

La pestana Balance cuenta los eventos por tipo y lado, y ademas por mano adelantada y
atrasada, que es el espacio de clases del export. Resalta la clase mas escasa.

Se ve mientras se anota y no al final a proposito: en BoxingVI la clase mas frecuente tiene
1371 ejemplos y la menos frecuente 296, casi cinco a uno, y eso se supo al terminar de
cortar los clips.

## Gestion de identidad

Es la parte del sistema donde un error no se ve. Si un rol queda mal, el esqueleto sigue
dibujandose sobre un cuerpo y el overlay se ve perfecto; lo unico que cambia es a que
peleador se le atribuyen los keypoints en el export.

Un `track_id` no significa nada estable: cambia en cada oclusion, en cada clinch y en cada
reanudacion del preproceso. Por eso el rol no vive en el cache de pose sino como intervalos
en el archivo de anotacion, y el color del overlay va por ROL y nunca por id.

La pestana Identidad ofrece cinco operaciones, todas reversibles con `Ctrl+Z`:

**Asignar un rol a un track**, desde el cuadro actual hasta la proxima decision manual. No
desde el principio del video: el track pudo haber sido otra persona antes, y pisar todo el
rango borraria correcciones ya hechas.

**Corregir un intercambio** desde el cuadro actual. Los dos assignments nuevos comparten
`op_id`, asi que en un diff se lee como un solo gesto y no como dos cambios sueltos que hay
que correlacionar a ojo. El alcance corta en la proxima decision manual posterior: sin ese
limite, corregir en el minuto dos pisaria lo que ya se habia corregido en el minuto cinco.

**Re-sembrar** dibujando una caja sobre el peleador. Dos modos y la diferencia importa: si
la caja se superpone con un track existente se le asigna el rol a ese track, que es el caso
comun porque el tracker no perdio al peleador sino que le cambio el id; si no se superpone
con nada se crea un track manual sin keypoints, y esos cuadros se marcan solos como no
confiables porque no hay pose que exportar.

**Marcar tramos no confiables** como `occluded` o `pose_unreliable`. No borra la pose: la
deteccion se sigue viendo y se sigue pudiendo juzgar. Que entre o no al dataset es politica
del export, y esa politica puede cambiar sin volver a mirar el video.

**Unir tracks separados por un hueco corto**, propuestos por IoU y ordenados por confianza.
Se confirman de a uno. Aplicarlos solos seria comodo y peligroso: en un clinch las cajas de
los dos peleadores se superponen casi por completo, y ahi es donde la heuristica se
equivoca. El costo de errarle es un tramo entero atribuido a la persona equivocada.

### Interpolacion

Los cuadros de un hueco aceptado se sintetizan **al leer** y nunca se escriben en el npz. El
cache guarda lo que el modelo observo; una pose inventada no es observacion, y mezclarlas en
el mismo archivo haria imposible saber despues cual era cual. Salen marcadas con
`interpolated` y con `det_conf` en cero.

La interpolacion es lineal y no pretende ser trayectoria: en tres cuadros de un golpe rapido
la muneca recorre bastante y una recta no describe eso. Sirve para que el tramo no tenga
agujeros. El score de un punto sintetizado es el minimo de sus dos extremos, porque un punto
inventado no puede tener mas confianza que los datos con que se invento.

Una deteccion real siempre gana sobre una sintetizada en el mismo cuadro.

### Por que estas operaciones deshacen distinto

Los comandos de evento guardan lo minimo para revertirse. Los de identidad guardan una
instantanea del bloque entero, que es lo contrario, y la razon es el tamano: mil eventos con
sus metricas son varios megas y copiarlos cincuenta veces costaria cientos, mientras que las
asignaciones son unas decenas de registros y una copia no se nota.

Las correcciones de identidad son reescrituras estructurales: partir un intervalo, truncar
otro, insertar dos nuevos. Calcular la inversa exacta de cada una a mano es donde se cuelan
los errores. Donde la copia es barata, conviene la opcion que no se puede equivocar.

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

## Exports

```bash
boxtwin-annotator export proyecto/videos/spar.mp4 --format mmaction --classes 12
```

Corre sin GUI: el pipeline de export tiene que poder ejecutarse en el entorno de
entrenamiento, que no tiene Qt ni pantalla.

Los cuatro formatos llevan en su metadata el **sha256 de la anotacion** que los genero. Es
lo unico que permite saber, meses despues y con el annot.json ya modificado, si un modelo se
entreno sobre los datos que uno cree.

| Formato | Que produce |
|---|---|
| `stats` | distribucion, duraciones, cobertura y tiempo de anotacion, en json y en texto |
| `mmaction` | pickle para PoseConv3D, `keypoint` (M,T,V,C) y `keypoint_score` (M,T,V) |
| `sequence` | npz con etiquetas BIO por cuadro y brazo, mas mascara de validez |
| `clips` | un mp4 por evento en carpetas por clase, mas `manifest.csv` |

### Espacios de clases

El default es **`--label-space side`**: el lado observado, sin ningun paso de derivacion.
El conjunto de 6 queda

```
straight-left   straight-right
hook-left       hook-right
uppercut-left   uppercut-right
```

`--label-space lead-rear` sigue disponible y es el espacio tactico que usan los datasets del
area, pero depende de `guard`, que la pone el anotador y puede estar mal o cambiar a mitad
del combate; un error de guardia intercambia sistematicamente dos clases sin que nada lo
delate. En ese espacio el conjunto de 6 coincide con las de BoxingVI (Jab, Cross, Lead/Rear
Hook, Lead/Rear Uppercut), asi que es el que hay que usar para concatenar con lo que ya esta
cortado de ese dataset.

`--classes 6` es tipo por eje, `12` agrega la altura, `14` agrega amague y fondo.

Los amagues quedan fuera de 6 y 12. Un amague no es un recto mal hecho: meterlo en la clase
recto ensucia justamente los ejemplos que definen la frontera. En el espacio de 14 tiene
clase propia. Los abortados quedan fuera de los tres: no describen ninguna categoria.

El orden de la lista de clases es **contrato**, porque el indice es la etiqueta numerica. El
espacio de 14 extiende al de 12 sin renumerarlo.

### mmaction

Por defecto una sola persona, el que pega. `--persons both` agrega al rival, que da contexto
sobre si el golpe llega, pero cambia el problema: se pasa de clasificar un movimiento a
clasificar un intercambio, y deja de ser comparable con lo ya entrenado.

Por defecto 17 keypoints. `--keypoints coco17+gloves` sube V a 19 y con eso los pesos
preentrenados en NTU dejan de cargar directo, porque cambia la primera convolucion. Es una
rama comparativa, no un cambio de base.

`--background N` muestrea ejemplos de fondo, solo en el espacio de 14. Se exige margen a los
bordes de todo evento del peleador, se excluyen los tramos no confiables y se pide identidad
resuelta en toda la ventana: un fondo que contiene la cola de un golpe le ensena al modelo
que ese movimiento es fondo.

**Partir train/val por evento filtra datos.** En una combinacion las ventanas de dos golpes
comparten cuadros. Hay que partir por video o por round, y la advertencia va escrita en el
`meta.json` porque el que arme el split seis meses despues no va a leer la consola.

### sequence

Un carril BIO por **brazo**, no por peleador. En una combinacion el jab izquierdo todavia
esta retrayendo cuando arranca el cross derecho, y BIO no puede representar dos segmentos
solapados en un mismo carril: habria que descartar uno. Un 1-2 es el golpe mas frecuente del
boxeo, asi que el formato perderia datos exactamente donde mas hay.

```
labels        (n_peleadores, 2, T)   int16, esquema O / B-clase / I-clase
valid         (n_peleadores, T)      bool
interpolated  (n_peleadores, T)      bool
keypoints     (n_peleadores, T, V, 2)
kp_score      (n_peleadores, T, V)
```

La mascara de validez distingue "no hay golpe" de "no sabemos": un cuadro sin identidad
resuelta o dentro de un tramo no confiable no es fondo, es ausencia de dato. Sin esa
distincion el modelo aprende que las oclusiones son fondo.

El unico solapamiento posible dentro de un carril es el doble jab. Se trunca en el export,
se reporta, y el `annot.json` no se toca.

### Determinismo

`mmaction`, `sequence` y `stats` son deterministas byte a byte, verificado corriendo el
mismo comando dos veces. El muestreo de fondo va por semilla y la semilla queda en la
metadata.

`clips` **no** puede serlo. El corte es frame-exacto, lo que obliga a reencodear (los cortes
sin reencodear caen en keyframes), y la salida de libx264 depende de su version. La garantia
esta sobre el `manifest.csv` y los rangos de cuadros, no sobre los bytes del mp4. Verificado
sobre material real: los seis clips salieron con exactamente la cantidad de cuadros esperada.

## Reanotacion ciega y acuerdo

Un dataset propio no vale mas que uno publico solo por ser propio. Lo que lo hace defendible
es poder decir cuanto se contradice a si mismo.

```bash
boxtwin-annotator reanno proyecto/videos/spar.mp4 --fraction 0.10 --seed 42
```

Sortea la muestra y la **congela**. Volver a correr el comando muestra el avance; resortear
exige `--force` explicito, porque resortear despues de ver resultados parciales convierte el
numero en lo que uno quiera que sea. Es el mismo criterio que ya se aplico en la
verificacion de BoxingVI, donde la muestra vivia en un commit anterior a toda anotacion.

Si la anotacion cambio desde el sorteo, el comando se niega a seguir: comparar contra un
estado distinto del que se sorteo invalida el pre-registro.

Se reanota desde la pestana **Reanotacion** del anotador. Mientras el modo esta activo:

- **se ocultan las marcas del timeline y la lista de eventos.** La marca dice exactamente
  donde empieza y termina el golpe, y con eso a la vista el error de fronteras mide cero por
  construccion;
- **la ventana de cada intento lleva relleno aleatorio** a los dos lados, entre 8 y 24
  cuadros, distinto en cada intento. Con relleno fijo, restarlo recuperaria las fronteras
  originales;
- **se puede revelar la etiqueta previa**, y el intento queda marcado. Prohibirlo no
  serviria, porque el archivo esta ahi para abrirlo; registrarlo hace que no cuente como
  ciego y que el reporte lo diga.

### El reporte

```bash
boxtwin-annotator reanno proyecto/videos/spar.mp4 --report
```

```
dimension         n  acuerdo   azar   kappa
side              7    1.000  0.510   1.000
punch_type        7    0.857  0.429   0.750
target            7    1.000  0.755   1.000
completeness      7    1.000  0.755   1.000

fronteras, en cuadros
  start     mae   1.57  sesgo  -0.71  mediana  2.0  max   2  exactos 0/7
```

Se reporta **kappa de Cohen** y no porcentaje de acuerdo. Con clases desbalanceadas, un
anotador que responda siempre la clase mas frecuente saca un porcentaje alto sin saber nada;
kappa descuenta el acuerdo esperado por azar y ese caso da cero. El porcentaje se reporta al
lado porque es lo que se entiende de un vistazo, pero no es la medida.

Se reporta **por dimension** y no sobre la clase colapsada. Si el acuerdo baja hace falta
saber si baja en el tipo, en el brazo o en la altura, porque cada uno se corrige distinto:
el brazo con mejor overlay, el tipo con mejor definicion operacional.

De las fronteras se reporta el error absoluto medio y tambien el **sesgo con signo**. Son
dos problemas distintos: ruido simetrico significa que la definicion es dificil de aplicar
con precision; un sesgo consistente significa que se esta interpretando distinto, y eso se
arregla cambiando la definicion, no esforzandose mas.

`landed` y `quality` quedan fuera del reporte principal: son estimaciones sobre algo que un
sistema monocular no observa, y mezclarlas bajaria el kappa por una razon que no es calidad
de anotacion.

Kappa queda en `n/d` cuando los dos anotadores usaron una sola categoria: ahi la formula
divide por cero y el valor no existe. No es acuerdo perfecto, y decir 1.0 seria mentir.

Con menos de 20 intentos el reporte avisa: kappa es inestable con n bajo y no conviene
citarlo sin su n.

## Herramientas

### Renderizar el overlay a video

```bash
python tools/render_overlay.py proyecto/videos/spar.mp4 --from 100 --to 400 --fps 8
```

Quema el overlay usando el mismo camino de dibujo que la aplicacion y estampa, por cuadro,
de que cuadro es la imagen y de que cuadro son los keypoints. Sirve para revisar anotaciones
de corrido sin abrir la interfaz, para producir figuras, y para zanjar cualquier duda sobre
sincronia sin depender de una sesion remota.

Nacio como diagnostico. En agosto de 2026 se reporto un desfasaje aparente entre el
esqueleto y la persona; con esta herramienta se verifico que **los dos numeros van siempre
iguales, o sea que el camino de dibujo de escritorio esta sincronizado cuadro a cuadro**. Lo
que se veia era el cliente de la variante web pidiendo cuadros de mas y dejando la imagen
atrasada respecto del esqueleto, arreglado en `feat/boxtwin-annotator-web`.

Antes ya se habia verificado, por separado, que el proxy y el original estan alineados
(mismo conteo de cuadros y `proxy[N]` corresponde a `original[N]`), y que el render directo
sobre el `.npz` cae sobre el cuerpo. Las tres verificaciones juntas descartan el desfasaje
como problema de datos, de proxy y de dibujo.

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

Se define en `gui/keymap.py` y es remapeable desde `config.yaml`, que ademas lleva el
identificador del anotador:

```yaml
annotator: lucas
keymap:
  event.mark_start: "F1"
  event.mark_end: "F2"
```

Lo del archivo se superpone al default en vez de reemplazarlo, asi que agregar acciones en
una version posterior no deja sin teclas a quien ya tenia su config. Un nombre de accion
mal escrito falla al abrir en vez de ignorarse.
