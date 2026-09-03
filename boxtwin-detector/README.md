# boxtwin-detector

Detector temporal de golpes sobre video continuo. Decide, por cuadro y por brazo, si ahí
empieza un golpe, si hay uno en curso, o si no pasa nada.

Existe porque el clasificador de ventanas no tiene clase "no hay golpe": fue entrenado sobre
ventanas que siempre contienen uno, así que a cualquier ventana le devuelve una de las seis
familias. Medido sobre 30 segundos de `sparring-3`, el pipeline completo encuentra 14 de 21
golpes reales pero dispara 66 veces. Cuatro de cada cinco marcas no son un golpe.

Y no hay atajo heurístico: el disparador por extensión de muñeca que tapaba el agujero
[no supera al azar](../boxtwin-annotator/docs/experiments/2026-09-01-disparador-vs-azar.md).
La señal hay que aprenderla en el tiempo.

## Estado

Se entrega en bloques. **Bloques 1 a 5 terminados**, 128 tests.

| Bloque | Qué | Estado |
|---|---|---|
| 1 | Dataset: remuestreo, features, máscaras, particiones | hecho |
| 2 | Modelo temporal y entrenamiento | hecho |
| 3 | Decodificación a segmentos y evaluación | hecho |
| 4 | Los tres folds y el reporte | hecho |
| 5 | Ensamble de semillas | hecho |

## Instalación

```bash
conda run -n twinboxing_env pip install -e ".[dev]"
```

Corre sobre `twinboxing_env`, que ya trae torch 2.6 con CUDA. **`boxtwin_mmaction` no se
toca**: está pinneado y es donde vive el clasificador viejo.

`torch` es un extra (`.[train]`) y no una dependencia base, porque armar y auditar el dataset
no necesita GPU y se hace mucho más seguido que entrenar.

## Cómo se arma el dataset

```bash
# 1. exportar desde el anotador, con 14 clases para que los amagues sean visibles
cd ../boxtwin-annotator
python -m boxtwin.cli export --format sequence --classes 14 --label-space side <video>

# 2. armar los tensores del detector
boxtwin-detector build <proyecto>/exports/*.sequence.npz --out data/
```

Cuatro carriles por fuente: dos peleadores por dos brazos. Cada carril es una secuencia
independiente.

## Las cuatro decisiones del armado

Entre lo que el anotador exporta y lo que un modelo por cuadro puede consumir hay cuatro
cosas que no son de forma sino de contenido. Cada una arruina el resultado en silencio.

### 1. Remuestreo a un fps común

Pacquiao corre a 59,94 fps y las otras dos fuentes a 30. En milisegundos los golpes duran
parecido, pero en cuadros no:

| Fuente | fps | Duración mediana |
|---|---|---|
| Sparring | 29,6 | 10 cuadros = 337 ms |
| Pacquiao | **59,9** | 12 cuadros = 200 ms |
| sparring-3 | 30,0 | 7 cuadros = 233 ms |

Un modelo con receptive field fijo en cuadros tendría que aprender dos escalas temporales
para la misma acción física. Y el fold que deja Pacquiao afuera es justo el que cambia el
fps: se confundiría un problema de unidades con uno de generalización.

El remuestreo se hace sobre los **keypoints**, y las features se calculan después. Al revés
las derivadas quedarían en cuadros de 60 fps y no habría arreglo posterior.

Los segmentos se remuestrean como intervalos **semiabiertos**. Mapeando los dos extremos por
separado, un golpe de 12 cuadros a 60 fps da 7 a 30 en lugar de 6: engorda un 17% al cambiar
de fps. Con el mapeo semiabierto, además, dos golpes que no se solapaban tampoco se solapan
después.

### 2. Las features cierran la vía de aprender la cámara

El clasificador no generalizó a una fuente nueva, y la sospecha razonable es que aprendió la
cámara y la escala, porque trabaja sobre heatmaps en píxeles. Acá eso está cerrado por
diseño: **centradas** en el punto medio de los hombros, **escaladas** por el ancho de
hombros, y **espejadas** por brazo, así un jab y un cross se ven iguales.

El espejado es un espejo real —niega la x **y** permuta los pares izquierda/derecha—, y el
brazo derecho se resuelve espejando el esqueleto entero y leyendo después el izquierdo. La
invariancia queda por construcción y no por dos ramas que hay que mantener de acuerdo.

Los ángulos que dan la vuelta van como (coseno, seno). A −π y a +π el cuerpo está en la misma
pose, y un escalar que salta de uno a otro le enseña al modelo un borde que no existe.

Son 20 features por cuadro. Lo que esto **no** arregla es la familia del golpe: está medido
tres veces que no es separable en pose monocular, y el detector no lo intenta.

### 3. Los amagues no son fondo

Un amague es movimiento de brazo diseñado para parecer un golpe: es el negativo más difícil
que existe en este dominio. Etiquetarlo `O` le enseña al modelo que el gesto de golpe es
fondo. Se enmascara, que es lo mismo que hace el anotador con los cuadros sin identidad
resuelta: no es fondo, es "no sabemos".

Por eso el export tiene que ser `--classes 14`: es el único espacio donde el amague tiene
clase propia y por lo tanto se lo puede ver.

### 4. Fuera del tramo anotado no hay fondo, hay ignorancia

El export cubre el video entero, pero de Pacquiao sólo está anotado el round 1: 2,9 minutos
de 87. Los cuadros del round 7 con identidad resuelta entrarían como `O`, o sea que el modelo
aprendería "acá no hay golpe" sobre metraje donde nadie miró. Medido antes de acotarlo: 1228
cuadros usables de Pacquiao, 971 de Sparring y 512 de sparring-3.

El tramo se deduce del primer y el último cuadro etiquetado, que es lo único que el export
permite saber. Cuesta el fondo legítimo de antes del primer golpe y después del último, y ese
error va en la dirección segura: se pierden negativos verdaderos en vez de inventarlos.

## Lo que hay, medido

639 golpes sobre tres fuentes, 84.177 cuadros-carril usables:

| | cuadros | % de los usables |
|---|---|---|
| `O` | 80.256 | **95,34%** |
| `I` | 3.384 | 4,02% |
| `B` | 537 | **0,64%** |

**Decir siempre `O` acierta 95,34% por cuadro.** La exactitud por cuadro no es una métrica
acá y hay que sacarla de la vista desde el principio: es el número que hace sentir que algo
funciona cuando no aprendió nada. Todo se mide por evento.

`B` se conserva aunque cueste un cuadro por golpe, porque sin ella dos golpes pegados del
mismo brazo se leen como uno solo largo. Medido: pasa 1 vez en 400 eventos.

## El modelo

Una TCN de convoluciones dilatadas (1, 2, 4, 8, 16), campo receptivo **63 cuadros = 2,1 s**,
que es ~9 veces la duración mediana de un golpe. Entrada `(4, T, 20)`, salida `(4, T, 3)`.

**No causal**, a propósito. Se ve en los datos por qué hace falta: en el golpe del cuadro 161
de `sparring-3`, los primeros cuatro cuadros ya etiquetados son indistinguibles de la guardia
de los tres anteriores. La `B` está donde el anotador dijo que arranca el movimiento, no donde
el movimiento se hace evidente, y eso recién pasa en el cuadro 166. Ningún clasificador cuadro
a cuadro puede acertar ahí; hay que mirar hacia adelante. El detector corre sobre video
grabado, así que puede.

Una sola etapa, también a propósito: el refinamiento multi-etapa es lo que se prueba **después**
de saber que una etapa no alcanza. Empezar por el grande deja sin saber cuál de las dos cosas
aportó.

Nada de heatmaps ni PoseConv3D. Ese modelo clasifica un clip recortado y trabaja en píxeles,
que es por donde se cuela la cámara.

## Entrenamiento

```bash
boxtwin-detector train data/*.det.npz --fold en-distribucion --fuente sparring-3-rounds
boxtwin-detector train data/*.det.npz --fold sin-Sparring
```

Cuatro cosas que un loop genérico no hace:

- **La pérdida se enmascara.** Los cuadros sin pose confiable, los amagues y todo lo que cae
  fuera del tramo anotado no entran.
- **Pesos por clase** `(1/frecuencia)^0,5`. El inverso puro le da a `B` un peso de 149 contra
  1 de `O` y el gradiente queda dominado por un puñado de cuadros.
- **Las ventanas se sortean dentro del tramo anotado.** El tensor de Pacquiao tiene 68.250
  cuadros y sólo 5,8% usables; sorteando sobre el largo total, 19 de cada 20 ventanas caerían
  donde no hay nada que aprender.
- **La inferencia va por trozos con solape**, y el borde se descarta. Sin eso aparece un
  artefacto cada N cuadros que ninguna métrica agregada muestra.

La estandarización se ajusta **sólo sobre el entrenamiento**. Ajustarla sobre todo el corpus
filtra la validación a través de la escala: sutil, y da una mejora chica y falsa.

### Lo que da hasta ahora

| | F1 macro por cuadro | recall O / B / I |
|---|---|---|
| Decir siempre `O` | ~0,33 | 1,00 / 0,00 / 0,00 |
| En distribución (`sparring-3`) | **0,635** | 0,98 / 0,39 / 0,67 |
| Dejando `Sparring` afuera | **0,512** | 0,96 / 0,15 / 0,45 |

**Estos números son una escalera, no el piso.** F1 macro por cuadro sirve para elegir un
checkpoint y para saber que el modelo aprende algo; no dice cuántos golpes encuentra. La
medida real —precisión y recall **por evento**— necesita la decodificación a segmentos, que es
el bloque 3.

El sobreajuste llega temprano: en distribución, la pérdida sigue bajando de 0,124 a 0,012
mientras el recall de `B` cae de 0,57 a 0,16. Por eso hay parada temprana y se guarda el mejor
checkpoint, no el último.

## Decodificación y evaluación

```bash
boxtwin-detector eval modelos/detector-en-distribucion-sparring-3-rounds.pt data/*.det.npz
```

Tres números por cuadro todavía no son una lista de golpes. La decodificación es donde se
**elige el punto de operación**, y por eso `eval` devuelve la curva entera y no un punto: el
disparador heurístico quedó clavado en 21% de precisión justamente porque no tenía esta
perilla.

Los tres parámetros salen de lo medido: largo mínimo 5 cuadros (los golpes duran 7 de mediana,
p10 en 5), hueco máximo 2 (un bajón de un cuadro es un error de pose, no el final del golpe) y
corte por `B` apagado por defecto (los carriles ya son por brazo, así que dos golpes seguidos
del mismo brazo pasan 1 vez en 400).

Se evalúa con `boxtwin.core.agreement.emparejar` a IoU 0,3 — **el mismo emparejador con que se
midió el acuerdo intra-anotador**. Con otro, los números no se podrían poner al lado del techo
humano, que es el punto. El emparejamiento es **por carril**: dos golpes simultáneos de
peleadores distintos no son el mismo golpe.

### Cómo localiza cuando acierta

Sobre la partición en distribución de `sparring-3`, el error de fronteras del modelo es de
**2,0 cuadros al inicio y 1,9 al final**, con IoU medio 0,665. El humano, medido por
reanotación ciega, da 1,12 y 1,55. Cuando el detector encuentra un golpe, lo ubica a menos de
un cuadro de donde lo ubicaría una persona.

Los resultados por evento están abajo, en **Los cuatro folds**. Un solo entrenamiento no
alcanza para reportarlos: el desvío entre semillas es de ~0,09 de F1.

### El umbral tiene poco recorrido

Medido sobre la validación: el 63% de los cuadros tiene P(golpe) < 0,05 y el 14,7% > 0,95.
**Sólo el 9% cae en la zona indecisa.** El modelo está saturado, así que mover el umbral de 0,1
a 0,9 reclasifica menos de un cuadro de cada diez.

La consecuencia práctica: el punto de operación lo fija más el `alpha` de los pesos por clase,
en el entrenamiento, que el umbral de decodificación. Barrer `alpha` —o calibrar la salida— es
trabajo del bloque 4.

## Los cuatro folds

```bash
boxtwin-detector folds data/*.det.npz --en-distribucion sparring-3-rounds --out reporte.json
```

Cinco semillas por partición, y se reporta la media. Ver
[`docs/experiments/2026-09-01-detector-folds.md`](docs/experiments/2026-09-01-detector-folds.md).

| Partición | Golpes | F1 medio | Desvío | recall | precisión | F1 heurística |
|---|---|---|---|---|---|---|
| **En distribución** (`sparring-3`) | 81 | **0,445** | 0,090 | 0,674 | 0,359 | 0,248 |
| sin `Sparring` | 114 | **0,348** | 0,083 | 0,582 | 0,280 | 0,243 |
| sin `Pacquiao` | 145 | **0,506** | 0,101 | 0,583 | 0,493 | 0,303 |
| sin `sparring-3` | 380 | **0,416** | 0,093 | 0,589 | 0,349 | 0,274 |
| *Techo humano* | | *0,907* | | *0,911* | *0,903* | |

**Le gana a la heurística en las cuatro**, por 0,10 a 0,20 de F1, que son 2,5 a 5 errores
estándar de la media.

**Y no hay brecha entre distribución y fuente nueva**: 0,445 contra 0,348 / 0,506 / 0,416. Uno
de los cruzados queda por encima del de distribución. Eso es lo contrario de lo que le pasó al
clasificador, que cayó a o por debajo de su línea de base en los tres folds cruzados teniendo
25 puntos de ventaja en distribución. No son números comparables entre sí —aquello es top-1 de
seis clases— pero la **forma** sí lo es: uno se derrumba al cambiar de fuente y el otro no.

La lectura más probable es que la representación era el problema: geometría normalizada
transfiere, heatmaps en píxeles no.

### El ruido entre semillas es más grande que casi todo

Entre la mejor y la peor semilla de una misma configuración hay **0,24 de F1**. Con 81 a 380
golpes en validación, una corrida sola no dice nada.

Pasó en este mismo trabajo: el primer barrido de `alpha`, con una semilla, eligió 0,25 por un
F1 de 0,634 cuya media real sobre cinco semillas es 0,445. El barrido **no tiene ganador** —los
cinco valores caen dentro de un desvío— así que quedó el default 0,5 y se reporta inconcluso.

Por eso `sembrar()` hay que llamarla **antes** de construir el modelo, y por eso está
`cudnn.deterministic`. Sin las dos cosas, dos corridas con la misma semilla difieren hasta 0,1
de F1: más que cualquier efecto reportable.

## Ensamble de semillas

```bash
boxtwin-detector train data/*.det.npz --fold sin-Sparring --semillas 42 1 2 3 4
```

El desvío entre semillas de una misma configuración es ~0,10 de F1, y es **igual con 62 golpes
de validación que con 380**. Si viniera del muestreo de la evaluación tendría que caer como la
raíz del tamaño. No cae: la varianza está en el entrenamiento.

Promediar las probabilidades de cinco semillas gana en los cuatro folds cruzados:

| Partición | Golpes | Una corrida | Desvío | **Ensamble** | recall | precisión |
|---|---|---|---|---|---|---|
| En distribución (`sparring-3`) | 81 | **0,445** | 0,090 | 0,409 | 0,691 | 0,290 |
| sin `02-sparring` | 62 | 0,520 | 0,144 | **0,602** | 0,500 | 0,756 |
| sin `03-sparring` | 129 | 0,496 | 0,074 | **0,515** | 0,395 | 0,739 |
| sin `Sparring` | 114 | 0,397 | 0,101 | **0,470** | 0,482 | 0,458 |
| sin `Pacquiao` | 145 | 0,494 | 0,097 | **0,548** | 0,393 | **0,905** |
| sin `sparring-3` | 380 | 0,496 | 0,111 | **0,615** | 0,518 | 0,755 |
| *Techo humano* | | | | *0,907* | *0,911* | *0,903* |

Cinco fuentes, 830 golpes. El fold más confiable —`sin sparring-3`, con 380 golpes en
validación— da **F1 0,615**. Y la precisión de `sin Pacquiao` es **0,905**, contra 0,903 del
humano: sobre transmisión profesional que el modelo nunca vio, nueve de cada diez marcas son
un golpe real.

Pierde en distribución, y es coherente: la ganancia viene de suprimir detecciones espurias, que
son idiosincrasia de cada corrida. Sobre la misma fuente en que se entrenó, las manías de un
modelo encajan; cruzando de fuente, no.

**La precisión pasa de 0,28–0,43 a 0,60–0,85**, a un paso del 0,903 humano. El costo es recall,
que baja de ~0,59 a 0,31–0,49.

No se compara contra la mejor de las cinco semillas (0,638 en el fold grande): elegirla mirando
la validación es seleccionar sobre el test.

Ver [`docs/experiments/2026-09-03-ensamble.md`](docs/experiments/2026-09-03-ensamble.md), que
documenta además una teoría equivocada sobre la cuantización del voto que escondió el mejor
punto de operación durante una medición entera.

## ¿Sirve sumar fuentes?

La cuarta no; la quinta sí. Pasando de tres a cuatro fuentes de entrenamiento, el cambio medio
sobre los folds cruzados fue **+0,046** en una corrida y **+0,048** en ensamble — contra
**−0,002** cuando se sumó la anterior.

El control lo respalda: la partición en distribución no toca las fuentes nuevas y da números
**idénticos bit a bit** entre las dos rondas, así que la diferencia viene sólo de los datos
agregados.

`03-sparring` es el doble de grande que `02` (131 golpes contra 67) y bastante más diverso
(28% hooks contra 13%), así que no se puede separar *más fuentes* de *más datos* con dos
incrementos. Lo que queda establecido es que el techo no estaba donde parecía.

Ver [`docs/experiments/2026-09-03-quinta-fuente.md`](docs/experiments/2026-09-03-quinta-fuente.md).

## Dónde está el error hoy

Con el ensamble, **la precisión dejó de ser el problema**: 0,60 a 0,85 según el fold, contra
0,903 del humano. Cuatro de cada cinco marcas que hace son un golpe real.

**Ahora manda el recall**, entre 0,39 y 0,52 contra 0,911 del humano. Ahí queda todo el error.

## Particiones

**Dejando una fuente afuera** es la que importa. El clasificador aprendió en distribución y
quedó en o por debajo de su línea de base en los tres folds cruzados; si el detector no se
mide así, no se entera.

**En distribución** no dice nada sobre generalizar, pero dice si el modelo aprende algo, que
es otra pregunta y hay que poder responderla por separado cuando la primera da mal. Es por
tramo continuo y con una banda muerta en el medio: sin la banda, una ventana centrada cerca
del corte ve cuadros de los dos lados y la validación filtra.

Una partición al azar por cuadro no mide nada sobre video continuo, y el proyecto ya pagó una
versión de ese error: el baseline público de 84,51% comparte el 96% de sus sujetos entre
train y val.

## Contra qué se mide

Cuando el detector exista, se evalúa con `boxtwin.core.agreement.emparejar` a IoU 0,3, que es
el mismo criterio con el que se midió el acuerdo intra-anotador. Evaluar con otro emparejador
haría que los números no se puedan comparar contra el techo humano, que es el punto.

| | recall | precisión |
|---|---|---|
| **Techo humano** (reanotación ciega) | 0,911 | 0,903 |
| Dardos al azar, a ±0,5 s | — | 0,46 |
| Disparador de muñeca | 0,67 | 0,21 |

## Limitación conocida

Los eventos `aborted` no llegan al export en ningún espacio de clases, así que sus cuadros
quedan como `O`. Son 9 sobre 676 (1,3%). Se declara y no se corrige: corregirlo pedía que el
detector leyera el `annot.json` y dejara de consumir la interfaz.

## Tests

```bash
conda run -n twinboxing_env python -m pytest
```
