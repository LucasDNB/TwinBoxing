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

Se entrega en cuatro bloques. **Bloque 1 terminado**, 51 tests.

| Bloque | Qué | Estado |
|---|---|---|
| 1 | Dataset: remuestreo, features, máscaras, particiones | hecho |
| 2 | Modelo temporal y entrenamiento | pendiente |
| 3 | Decodificación a segmentos y evaluación | pendiente |
| 4 | Los tres folds y el reporte | pendiente |

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
