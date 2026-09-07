# El detector generaliza a una fuente nueva; el clasificador no lo hacía

**01-09-2026 · 639 golpes sobre tres fuentes · TCN por cuadro, cuatro particiones, cinco
semillas cada una.**

## Qué se preguntaba

El clasificador de ventanas aprendió en distribución y quedó **en o por debajo de su línea de
base** en los tres folds que dejan una fuente afuera. Esa brecha es el problema abierto del
proyecto, y la hipótesis del detector era que se debía a la representación: PoseConv3D trabaja
sobre volúmenes de heatmaps en píxeles, y por ahí se cuela la cámara.

El detector usa features geométricas centradas en los hombros, escaladas por
`max(ancho de hombros, largo del torso)` y espejadas por brazo. Todas esas vías están cerradas
por construcción. La pregunta era si eso alcanza.

## Montaje

TCN de convoluciones dilatadas, campo receptivo 63 cuadros, salida por cuadro sobre O/B/I.
Decodificación a segmentos con largo mínimo 5 y hueco máximo 2. Evaluación por evento con
`boxtwin.core.agreement.emparejar` a IoU 0,3, **el mismo emparejador con que se midió el
acuerdo intra-anotador**, emparejando carril por carril.

La heurística de extensión de muñeca pasa por el **mismo decodificador, el mismo emparejador y
la misma región**. Sin eso, la comparación mediría la diferencia de decodificador y no la de
detector.

Cinco semillas por partición, y se reporta la media. Es imprescindible: ver más abajo.

## Resultado

| Partición | Golpes | F1 medio | Desvío | recall | precisión | F1 heurística |
|---|---|---|---|---|---|---|
| **En distribución** (`sparring-3`) | 81 | **0,445** | 0,090 | 0,674 | 0,359 | 0,248 |
| sin `Sparring` | 114 | **0,348** | 0,083 | 0,582 | 0,280 | 0,243 |
| sin `Pacquiao` | 145 | **0,506** | 0,101 | 0,583 | 0,493 | 0,303 |
| sin `sparring-3` | 380 | **0,416** | 0,093 | 0,589 | 0,349 | 0,274 |
| *Techo humano* | | *0,907* | | *0,911* | *0,903* | |

Dos cosas, y la segunda es la importante.

**Le gana a la heurística en las cuatro particiones**, por 0,10 a 0,20 de F1. Con desvío 0,09
sobre cinco semillas, el error estándar de la media es ~0,04: son márgenes de 2,5 a 5 errores
estándar, consistentes en dirección en las cuatro.

**No hay brecha entre distribución y fuente nueva.** En distribución da 0,445; los tres folds
cruzados dan 0,348, 0,506 y 0,416, con media 0,423. Uno de los cruzados queda *por encima* del
de distribución. Cualquier diferencia es más chica que el ruido entre semillas.

## Por qué eso importa

El clasificador, sobre los mismos datos y las mismas tres fuentes:

| | Línea de base | Clasificador |
|---|---|---|
| En distribución | 37,5% | **62,5%** |
| sin sparring | 41,2% | 33,3% |
| sin pacquiao | 60,7% | 58,6% |
| sin sparring3 | 37,8% | 34,1% |

Los tres folds cruzados **en o por debajo de predecir la clase mayoritaria**, contra 25 puntos
de ventaja en distribución.

No son números comparables entre sí —aquello es top-1 de seis clases, esto es F1 por evento de
un detector— y no hay que ponerlos en la misma tabla como si lo fueran. Lo comparable es la
**forma**: el clasificador se derrumba al cambiar de fuente y el detector no.

La lectura más probable es que la representación era el problema. Detectar *cuándo* hay golpe,
sobre geometría normalizada, es una tarea que transfiere entre gimnasios, cámaras y peleadores.
Decir *qué* golpe es, sobre heatmaps en píxeles, no transfirió.

Lo que **no** prueba: que la familia del golpe sea aprendible. Son dos tareas distintas y el
detector no intenta la segunda.

## El ruido entre semillas es más grande que casi todo

| Partición | F1 medio | mín | máx |
|---|---|---|---|
| En distribución | 0,445 | 0,383 | 0,623 |
| sin `Sparring` | 0,348 | 0,299 | 0,512 |
| sin `Pacquiao` | 0,506 | 0,405 | 0,683 |
| sin `sparring-3` | 0,416 | 0,348 | 0,601 |

**Entre la mejor y la peor semilla hay 0,24 de F1 en una misma configuración.** Con 81 a 380
golpes en validación, una corrida sola no dice nada, y la tentación de reportar la buena es
exactamente del tamaño del efecto que uno quiere mostrar.

Pasó en este mismo experimento. El primer barrido de `alpha`, con una semilla, dio esto:

| alpha | F1, una semilla | F1 medio de cinco | desvío |
|---|---|---|---|
| 0,00 | 0,389 | 0,421 | 0,097 |
| **0,25** | **0,634** | 0,445 | 0,135 |
| 0,50 | 0,623 | 0,456 | 0,116 |
| 0,75 | 0,595 | 0,451 | 0,076 |
| 1,00 | 0,556 | 0,527 | 0,095 |

Se había elegido `alpha` 0,25 por ese 0,634, que es el máximo de su distribución y no su
centro. **El barrido no tiene ganador**: los cinco valores caen dentro de un desvío. Se dejó el
default 0,5 y se reporta inconcluso, que es lo que dicen los datos.

## Dos defectos propios, encontrados midiendo

**La verdad fundamental no se recortaba a la región evaluada.** En la partición en distribución
el tensor de etiquetas es el de la fuente entera y sólo la máscara separa train de val, así que
se contaban como no encontrados los ~300 golpes de la mitad que ni siquiera se evaluó. El
recall salía 0,145 en lugar de 0,679.

**Dos corridas con la misma semilla daban distinto.** `entrenar` fijaba la semilla, pero el
modelo se construye antes, así que la inicialización de los pesos quedaba con el estado global
que hubiera. Sumado a que cuDNN elige kernels no deterministas, la diferencia llegaba a 0,1 de
F1: más que cualquier efecto reportable. Se agregó `sembrar()`, que hay que llamar **antes** de
construir el modelo, y `cudnn.deterministic`.

## Dónde está el error hoy

El recall es estable entre 0,58 y 0,67 en las cuatro particiones. **La precisión es lo que
falla**: entre 0,28 y 0,49 contra 0,903 del humano. El detector encuentra la mayoría de los
golpes y marca de más.

Y el umbral de decodificación no lo va a arreglar: medido sobre la validación, el 63% de los
cuadros tiene P(golpe) < 0,05 y el 14,7% > 0,95, así que sólo el 9% cae en la zona indecisa.
Mover el umbral de 0,1 a 0,9 reclasifica menos de un cuadro de cada diez. El modelo está
saturado.

## Qué sigue

**Sumar fuentes.** Sigue siendo lo mismo que dijo el experimento del clasificador, y ahora con
un argumento más fuerte: si el detector ya transfiere con tres fuentes, es la pieza que más
tiene para ganar con seis. Más cámaras, más gimnasios, más pares de peleadores; no más rounds
del mismo video.

**Calibrar antes que tunear.** La precisión no se arregla con el umbral porque la salida está
saturada. Calibración por temperatura, o pérdida focal, atacan eso directamente.

**Y no confundir esto con un sistema que cuenta golpes.** Con precisión 0,35 hay dos marcas
falsas por cada tres golpes reales. Para el bucle de preanotación hace falta 0,5 o 0,6, que es
la próxima frontera concreta.
