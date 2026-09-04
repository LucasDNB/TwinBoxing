# El detector casi no encuentra golpes sobre pose interpolada — y aun así no es lo que hunde a `Sparring`

**04-09-2026 · siete fuentes · hipótesis confirmada, explicación insuficiente.**

## Qué se probó

Cuando la identidad tiene un hueco, el anotador lo rellena interpolando linealmente. La
hipótesis era que el detector no puede encontrar un golpe ahí —una recta entre dos puntos no
tiene la firma temporal que busca— y que eso explicaba por qué `sin Sparring` es el peor fold
en las cuatro rondas: el 23,6% de sus golpes contiene algún cuadro interpolado.

Se agregó al reporte el recall separado por si el golpe está sobre **pose medida** o sobre
**pose rellenada**. Es puramente aditivo: no toca el entrenamiento, ni la máscara, ni los
defaults. **Verificado: F1, recall global y precisión salen idénticos a los de la rama
anterior en las ocho particiones.**

## La hipótesis se confirma

| Fold | recall medidos | recall rellenados | Brecha |
|---|---|---|---|
| sin `Sparring` | 0,535 (86) | **0,321** (28) | **−0,214** |
| sin `Pacquiao` | 0,592 (130) | **0,400** (15) | **−0,192** |
| sin `sparring-3` | 0,561 (367) | **0,077** (13) | **−0,484** |

En las tres fuentes con interpolación, y en la misma dirección. En `sparring-3` es brutal:
**1 de 13** golpes con pose rellenada encontrado, contra 0,561 sobre los 367 medidos.

**Control**: las cuatro fuentes anotadas con el flujo nuevo tienen 0% de interpolación, y su
`recall_medidos` sale **exactamente igual** al recall global —0,510, 0,500, 0,589 y 0,467—
con `-` en la columna de rellenados. La máscara se propaga bien.

## Pero no explica lo que se quería explicar

| Fold | recall | precisión |
|---|---|---|
| sin `01-sparring` | 0,510 | 0,924 |
| sin `02-sparring` | 0,500 | 0,705 |
| sin `03-sparring` | 0,589 | 0,792 |
| sin `04-sparring` | 0,467 | 0,824 |
| **sin `Sparring`** | **0,482** | **0,561** |
| sin `Pacquiao` | 0,572 | 0,754 |
| sin `sparring-3` | 0,545 | 0,784 |

**El recall de `Sparring` (0,482) está dentro del rango de todos los demás** (0,467 a 0,589).
No es un valor atípico. Lo atípico es su **precisión: 0,561 contra 0,705–0,924 del resto.**

Y la interpolación afecta al recall, no a la precisión: es una propiedad de los golpes
anotados, no de las marcas que el modelo produce de más. El efecto confirmado corrige un
déficit que **no es el déficit de `Sparring`**.

El contrafáctico lo cuantifica. Si todos sus golpes tuvieran pose medida y las predicciones no
cambiaran:

| Fold | F1 real | F1 contrafáctico | Δ |
|---|---|---|---|
| sin `Sparring` | 0,519 | **0,575** | +0,056 |
| sin `Pacquiao` | 0,651 | 0,674 | +0,023 |
| sin `sparring-3` | 0,643 | 0,662 | +0,020 |

Aun corregido del todo, `Sparring` quedaría en 0,575 y **seguiría siendo el peor fold**: el
segundo peor es `sin 02-sparring` con 0,585.

## Qué queda

**Confirmado y útil**: el detector encuentra los golpes sobre pose rellenada a menos de la
mitad de la tasa. La medición separa el error del detector del error de la anotación, y eso
es información que antes se promediaba en un solo número.

**Sin explicar**: por qué la precisión de `Sparring` es 0,561 cuando el resto está entre 0,705
y 0,924. Ya se descartaron la resolución —el preproceso corre a `imgsz=640` en las siete, así
que la pose ve lo mismo— y la calidad de pose, cuyo `kp_score` mediano en `Sparring` es el más
alto de las siete. La interpolación era la tercera candidata y tampoco alcanza.

Que el modelo marque de más en esa fuente y no en las otras seis sigue siendo una pregunta
abierta, y conviene declararla como tal en vez de darla por resuelta.
