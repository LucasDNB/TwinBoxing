# C5 · Ensayo del tiempo de procesamiento y la lectura de video de la evidencia de guante

20-10-2026 · pre-registro de C5: `docs/specs/2026-10-20-c5-tiempo-preregistro.md` · scripts
`boxtwin-api/scripts/medir_c5.py` y `boxtwin-annotator/scripts/comparar_evidencia.py` ·
arreglo en `boxtwin-annotator/src/boxtwin/core/identidad_auto.py` (`analizar`)

**Esto no es C5.** C5 se mide sobre las sesiones de prueba de la semana del 27-10. Esto es el
ensayo sobre las diez sesiones que ya estaban en la base de la estación, que mostró que con el
sistema como estaba C5 no pasaba, encontró la causa y la arregló antes de medir.

## 1. El ensayo: el sistema como estaba no entraba en 2x

Reloj del pre-registro (hasta la Fight-Card, desde la tabla `trabajos`), leído con
`medir_c5.py` en modo solo lectura sobre `~/boxtwin/datos/boxtwin.db`:

| Sesión | Video | Pose | Guantes | Hasta la Fight-Card | Factor |
|---|---|---|---|---|---|
| 4ad5553a | 174,9 s | 255 s | 223 s | 484 s | 2,77 |
| c297b71a | 182,6 s | 262 s | 251 s | 518 s | 2,84 |
| 198c6000 | 168,4 s | 231 s | 158 s | 393 s | 2,33 |
| 4dad5694 | 170,6 s | 247 s | 260 s | 512 s | 3,00 |
| da40de0b | 174,9 s | 255 s | 223 s | 514 s | 2,94 |
| 044e133e | 580,0 s | 815 s | 744 s | 1575 s | 2,71 |
| b249a1bb | 186,6 s | 165 s | 33 s | 202 s | 1,08 |
| 117e6a1f | 186,6 s | 166 s | 33 s | 202 s | 1,08 |

Pose y guantes salen de `sesion.json`. Dos sesiones más no llegaron a la Fight-Card (esperan
siembra). Mediana 2,741 sobre ocho; b249a1bb y 117e6a1f son el mismo video subido dos veces.
Solo da40de0b tiene `clasificar` (1,7 s, clasificador geométrico).

El 1,098x del 22-09 era sobre un video de 2 minutos con 20 tracks y no representaba el
sparring de gimnasio. De las tres sesiones re-corridas en §3, las dos lentas tienen 351 y 401
tracks, y la rápida 40: el costo de la evidencia crece con los tracks.

## 2. La causa: un salto por cuadro en la lectura de video

La mitad del tiempo era la evidencia de guante. `analizar` leía un cuadro cada 5 saltando con
`CAP_PROP_POS_FRAMES`, y en H.264 cada salto decodifica desde el keyframe anterior (cada 5 s
en `02-sparring`). Medido sobre los primeros 50 s de `02-sparring`: **61,3 s saltando contra
3,2 s leyendo en orden**, 19 veces.

El plan B pre-registrado para C5 (bajar a YOLOv8m-pose) ataca solo la pose. No habría tocado
esto.

## 3. El arreglo y su verificación

Lectura en orden con `grab()` para los cuadros intermedios. Test con un video sintético
numerado: se leen exactamente los cuadros del paso, para paso 1, 5 y 7.

Como esa evidencia alimenta la identidad, que C1 ya midió con la versión vieja, se rehízo con
el código nuevo sobre tres sesiones y se comparó campo por campo con la guardada
(`comparar_evidencia.py`, tolerancia 1e-3):

| Sesión | Guantes antes | Guantes ahora | Tracks idénticos |
|---|---|---|---|
| 4dad5694 | 260,5 s | 66,1 s | 401 de 401 |
| c297b71a | 251,5 s | 50,9 s | 351 de 351 |
| b249a1bb | 33,5 s | 12,8 s | 40 de 40 |

Idéntica en las tres: el salto caía en los cuadros correctos y solo era lento. C1 no cambia.

## 4. Proyección, no medida

Reemplazando el tiempo de guantes por el nuevo en las dos sesiones re-corridas, el factor
pasaría de 3,00 a unos 1,86 (4dad5694) y de 2,84 a unos 1,74 (c297b71a). Es una cuenta sobre
etapas medidas por separado, no una corrida por el worker. Debajo de 2,0, con poco margen: la
pose sola ya ocupa unas 1,45 veces la duración del video.

## 5. Lo que hay que declarar

- El arreglo se decidió después de ver el ensayo. No cambia el criterio de C5, que se
  pre-registró antes del arreglo y se mide sobre las sesiones de prueba.
- `medir_c5.py` marca una sesión como solapada solo si otro trabajo del worker corrió a la
  vez. Uso de GPU por fuera del worker no lo ve.
- La tabla `sesiones` deja `duracion_s` vacío; la duración sale de `sesion.json`.
- La app en la estación sigue con el código viejo hasta mergear y actualizar la copia
  principal. Tiene que estar antes del 27-10.
