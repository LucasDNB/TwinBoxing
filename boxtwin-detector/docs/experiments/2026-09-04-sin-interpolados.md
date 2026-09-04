# Sacar la pose interpolada del entrenamiento no paga, y no probó lo que se quería probar

**04-09-2026 · siete fuentes · resultado negativo, y un experimento mal dirigido.**

## Qué se corrió

Los cuadros de pose interpolada —los que el anotador rellena linealmente para tapar un hueco
de identidad— entraban en la máscara `usable`. La propuesta fue sacarlos, con el argumento de
que una interpolación no es un dato observado y el propio dataset ya distingue *no hay golpe*
de *no sabemos* en todos los demás casos.

Cuesta poco: **1,02% de los cuadros usables**, concentrado donde el diagnóstico predecía.

| Fuente | Pérdida |
|---|---|
| `Sparring` | 4,37% |
| `Pacquiao` | 2,71% |
| `sparring-3` | 0,90% |
| `01` a `04` | **0,00%** |

Las cuatro fuentes anotadas con el flujo nuevo no pierden un solo cuadro.

## Resultado

| Fold | Ensamble antes | después | Δ |
|---|---|---|---|
| En distribución | 0,409 | 0,471 | +0,063 |
| sin `01-sparring` | 0,658 | 0,658 | +0,000 |
| sin `02-sparring` | 0,585 | 0,629 | +0,044 |
| sin `03-sparring` | 0,676 | 0,660 | −0,015 |
| sin `04-sparring` | 0,596 | 0,600 | +0,004 |
| sin `Sparring` | 0,519 | **0,490** | **−0,029** |
| sin `Pacquiao` | 0,651 | 0,641 | −0,010 |
| sin `sparring-3` | 0,643 | 0,665 | +0,022 |
| **media (cruzados)** | | | **+0,002** |

Media **+0,002** en ensamble y +0,011 en una corrida: dentro del ruido. La subida en
distribución es la más grande, pero su desvío entre semillas también subió de 0,090 a 0,129
sobre 81 golpes, así que no se sostiene.

## El experimento no probó la hipótesis

La hipótesis era que `sin Sparring` es el peor fold porque **el 23,6% de sus golpes contiene
pose interpolada**, y el detector no puede detectar un golpe en una pose que nadie midió. Eso
es una afirmación sobre la **evaluación**.

Pero la máscara `usable` no entra en la evaluación. En `evaluacion.py` sólo alimenta
`recall_alcanzables`; el recall y la precisión que se reportan salen de las etiquetas y de la
cobertura, y ninguna de las dos cambia. **Sacar los interpolados de la máscara afecta
únicamente al entrenamiento.**

Así que lo que se midió es otra pregunta, también válida: *¿la pose interpolada es señal
dañina para entrenar?* La respuesta es no, no de forma medible. Y `sin Sparring` incluso bajó
−0,029, que es lo contrario de lo que la hipótesis predecía si el mecanismo fuera ese.

La hipótesis original sigue **sin probar**. Para probarla hay que reportar el fold separando
los golpes que tienen pose medida de los que no, que es la propuesta 1 y usa la maquinaria de
`alcanzables` que ya existe.

## Decisión

**No conviene seguir por acá.** El cambio no paga, y el default se deja como estaba: los
cuadros interpolados siguen entrando en la máscara.

Se conserva la opción `--sin-interpolados`, apagada por defecto, porque es lo que hace
reproducible esta medición y cuesta cuatro tests. Es el mismo criterio con que quedó
`tools/barrido_disparador.py` después de su resultado negativo.
