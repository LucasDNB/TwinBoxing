# La cámara en mano no era el problema, y la curva se aplanó

**04-09-2026 · siete fuentes, 1046 golpes · cuarto incremento con el mismo protocolo.**

## Tres preguntas

`01-sparring` es la única fuente filmada con **cámara en mano desde adentro del ring**: las
otras seis van desde afuera, con cámara fija o casi. Era el régimen que faltaba, y se
esperaba que fuera el más difícil.

1. ¿Se sostiene la progresión de las tres rondas anteriores?
2. ¿Cuánto le cuesta al detector una cámara que nunca vio?
3. ¿Ayuda a las demás fuentes?

## 1. La progresión se aplanó

Los tres folds presentes en las cuatro rondas, entrenando sobre 3, 4, 5 y 6 fuentes:

| Fold | 3 | 4 | 5 | 6 | | Ens. 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|---|
| sin `Sparring` | 0,356 | 0,397 | 0,407 | 0,400 | | 0,429 | 0,470 | 0,535 | 0,519 |
| sin `Pacquiao` | 0,479 | 0,494 | 0,540 | 0,585 | | 0,580 | 0,548 | 0,618 | **0,651** |
| sin `sparring-3` | 0,430 | 0,496 | 0,530 | 0,530 | | 0,527 | 0,615 | **0,664** | 0,643 |
| **media** | 0,422 | 0,462 | 0,492 | 0,505 | | **0,512** | **0,544** | **0,605** | **0,604** |

El ensamble venía +0,032 y +0,061; este incremento da **+0,000**. Sobre todos los folds el
efecto medio de sumar `01` es **+0,013**, dentro del ruido.

**Control**: la partición en distribución da idéntico bit a bit por cuarta vez (0,4447 y
0,4088).

No alcanza para declarar saturación con un solo punto, pero sí para decir que **el cuarto
incremento no se pagó como los dos anteriores**, y que seguir sumando fuentes de este tipo ya
no es la palanca que era.

## 2. La cámara en mano no era el problema

| Fold | Golpes | Ensamble | recall | **precisión** |
|---|---|---|---|---|
| **sin `01-sparring`** | 96 | 0,658 | 0,510 | **0,924** |
| sin `03-sparring` | 129 | **0,676** | 0,589 | 0,792 |
| sin `Pacquiao` | 145 | 0,651 | 0,572 | 0,754 |
| sin `sparring-3` | 380 | 0,643 | 0,545 | 0,784 |
| sin `04-sparring` | 120 | 0,596 | 0,467 | 0,824 |
| sin `02-sparring` | 62 | 0,585 | 0,500 | 0,705 |
| sin `Sparring` | 114 | 0,519 | 0,482 | 0,561 |
| *Techo humano* | | *0,907* | *0,911* | *0,903* |

Entrenado sobre las otras seis y evaluado sobre la cámara en mano que nunca vio, el detector
saca **precisión 0,924** — la más alta de todos los folds, y por encima del 0,903 humano. No
es el fold más difícil: es el segundo mejor.

La explicación más probable es la que motivó el diseño de las features: están centradas en el
punto medio de los hombros y escaladas por `max(ancho de hombros, largo del torso)`, así que
el movimiento de cámara se cancela. Un paneo mueve a los dos peleadores en la imagen y no
mueve nada en el espacio de features.

## 3. El fold difícil es otro

`sin Sparring` es el peor de los siete por margen amplio: **0,519**, con precisión 0,561
cuando el resto está entre 0,70 y 0,92. Y viene siendo el peor en las cuatro rondas.

`Sparring` es la única fuente a **640x360**; las otras seis son 1080p. La hipótesis es que la
pose se degrada lo suficiente como para que las features pierdan precisión, y es comprobable:
alcanza con reprocesar una de las fuentes de 1080p a 640x360 y ver si su fold cae al mismo
lugar.

## Qué sigue

**El recall.** La precisión está entre 0,56 y 0,92 contra 0,903 del humano; el recall entre
0,47 y 0,59 contra 0,911. Ahí queda casi todo el error, y el ensamble lo empeora: compra
precisión sacrificando recall.

**Y no más fuentes del mismo tipo**, al menos no como primera palanca. Cuatro incrementos
medidos dicen que los dos del medio se pagaron y el último no.
