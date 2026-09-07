# `Sparring` no es difícil por su resolución: un cuarto de sus golpes está sobre pose inventada

**04-09-2026 · siete fuentes · descarta dos hipótesis y encuentra el mecanismo.**

## Qué se preguntaba

`sin Sparring` es el peor de los siete folds y viene siendo el peor en las cuatro rondas:
ensamble 0,519 con precisión 0,561, cuando el resto está entre 0,70 y 0,92. La pregunta era
si convenía descartar esa fuente.

## Descartar del entrenamiento no cambia nada

| Fold | con `Sparring` | sin `Sparring` | Δ |
|---|---|---|---|
| sin `01-sparring` | 0,658 | 0,679 | +0,021 |
| sin `02-sparring` | 0,585 | 0,583 | −0,002 |
| sin `03-sparring` | 0,676 | 0,679 | +0,003 |
| sin `04-sparring` | 0,596 | 0,586 | −0,010 |
| sin `Pacquiao` | 0,651 | 0,672 | +0,021 |
| sin `sparring-3` | 0,643 | 0,645 | +0,002 |
| **media** | | | **+0,006** |

Dentro del ruido. `Sparring` es **difícil de predecir, no dañino para entrenar**. Y quitar sus
114 golpes —el 11% del dataset— no cuesta nada medible, lo que es coherente con que la curva
ya se haya aplanado.

Descartarlo **del reporte** sería otra cosa: quitar el caso difícil sube la media sin que el
sistema mejore, que es seleccionar sobre el test.

## Dos hipótesis caídas

**No es la resolución.** `Sparring` es nativamente 640x360 contra 1080p de las otras seis,
pero el preproceso corre con `imgsz=640` en las siete: un cuadro de 1920x1080 se reduce a
640x360 antes de la inferencia. **El modelo de pose ve la misma resolución en todas.**

**No es la calidad de la pose.** El `kp_score` mediano de `Sparring` es el **más alto** de las
siete: 0,945, contra 0,792 de `03-sparring`, que es de los mejores folds.

## El mecanismo

| Fuente | Golpes | Con pose interpolada adentro | % |
|---|---|---|---|
| **`Sparring`** | 123 | **29** | **23,6%** |
| `sparring-3` | 396 | 13 | 3,3% |
| `03-sparring` | 131 | 0 | 0,0% |
| `04-sparring` | 122 | 0 | 0,0% |

**Casi uno de cada cuatro golpes de `Sparring` contiene cuadros cuya pose no se midió**: se
rellenó por interpolación lineal para tapar huecos de identidad. En las otras fuentes eso es
3,3% o cero.

El detector no puede detectar un golpe en una pose que nadie observó: una interpolación es una
recta entre dos puntos y no tiene la firma temporal que el modelo busca. Así que el fold `sin
Sparring` mide, en buena parte, la interpolación de la anotación y no el detector.

Encaja con el resto de las señales. `Sparring` tiene 34 avisos de validación contra 1 a 18 del
resto, y **20 de ellos son `ID_INTERP_TOO_LONG`**, que ninguna otra fuente tiene. Fue el primer
video anotado, antes del click-para-asignar y antes de las mejoras de identidad de agosto.

## Consecuencia

Los cuadros interpolados entran hoy en la máscara `usable` del dataset. Fue una decisión
tomada al armarlo —son la mejor estimación del anotador y sacarlos quitaría cuadros positivos—
pero choca con el criterio que el propio dataset aplica en todos los demás casos: distinguir
*no hay golpe* de *no sabemos*. Una pose interpolada es lo segundo.

Queda como experimento a correr: sacarlos de la máscara y volver a medir las siete fuentes.
