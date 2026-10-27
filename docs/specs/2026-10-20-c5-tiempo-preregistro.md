# C5 · Pre-registro de la medición del tiempo de procesamiento

20-10-2026 · criterio original en `docs/specs/MVP.md` §7 · decisiones de Lucas del 20-10 ·
script `boxtwin-api/scripts/medir_c5.py` · se commitea antes de tocar el código que se mide y
antes de las sesiones de prueba

## 1. Qué se mide

C5, tal como está en la spec: mediana sobre las sesiones de prueba en la estación del tiempo
de procesamiento dividido por la duración del video, umbral **≤ 2,0**. Si no se cumple, se
baja a YOLOv8m-pose y se mide la pérdida.

## 2. Lo que la spec no fijaba

**Reloj (decisión de Lucas): hasta la Fight-Card.** Por sesión, la suma de lo que tardó el
worker en tres etapas, leído de la tabla `trabajos` como `terminado_en − tomado_en`:

1. el primer `procesar` terminado en `listo`;
2. el primer `completar` terminado en `listo` que se tomó después de ese `procesar`;
3. el primer `clasificar` terminado en `listo` que se tomó después de ese `completar`, si la
   estación tiene clasificador.

Incluye el arranque de cada subproceso y de su entorno conda, porque el usuario lo espera
igual. No incluye la espera en la cola, el tiempo de la persona en la siembra ni el `render`
del video procesado, que no hace falta para leer la Fight-Card. Los `completar` posteriores
(re-siembra, corrección por track) no cuentan: son otra espera, que el usuario eligió.

**Duración del video:** `video.duracion_s` de `sesion.json` de la sesión. La tabla `sesiones`
tiene la columna pero hoy queda vacía, medido el 20-10 sobre las diez sesiones de la base.

**Sesiones:** las sesiones de prueba de C4 y C6 (semana del 27-10) que llegan a Fight-Card. Las
que no llegan cuentan en C4 y acá se informan aparte, sin entrar a la mediana. Una sola
estación (`desarrollo-lucas`, RTX 2080 Super), sin otro trabajo de GPU corriendo; si lo hubo,
la sesión se marca y se informa con y sin ella.

**Al lado, sin umbral:** el factor de cada sesión, el máximo, el reparto por etapa (de
`sesion.json`), el `render` y la espera en cola.

## 3. Lo que no cuenta

Las diez sesiones que ya están en la base de la estación (14-10 al 20-10) son el ensayo que
motivó esto y no entran en C5. Dieron entre 2,3x y 3,0x en seis de ocho, con la mitad del
tiempo en la evidencia de guantes. La causa medida: `analizar` salta de posición en el video
cada 5 cuadros, y en H.264 cada salto decodifica desde el keyframe anterior (61,3 s contra
3,2 s leyendo en orden, sobre 50 s de `02-sparring`).

Ese arreglo se hace **antes** de las sesiones de prueba y no cambia este criterio: lo que se
mide es el sistema que se va a presentar. Se declara que el arreglo se decidió después de ver
el ensayo.
