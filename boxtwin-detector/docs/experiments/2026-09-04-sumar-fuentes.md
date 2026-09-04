# Sumar fuentes sí sirve, y la primera medición no alcanzaba para verlo

**04-09-2026 · seis fuentes, 950 golpes · tres incrementos con el mismo protocolo.**

## Qué se preguntaba

El experimento del clasificador cerró con *"sigue sumar fuentes"*, y la primera prueba
directa —[sumar `02-sparring`](2026-09-03-ensamble.md)— dio un cambio medio de **−0,002**
sobre los folds cruzados. La conclusión entonces fue, con cuidado, *no se detectó mejora*.

Con `03-sparring` y `04-sparring` anotados hay tres incrementos medidos con el mismo
protocolo, y la respuesta cambia.

## La progresión

Los tres folds presentes en las tres rondas, entrenando sobre 3, 4 y 5 fuentes:

| Fold | Golpes | 3 fuentes | 4 | 5 | | Ensamble 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| sin `Sparring` | 114 | 0,356 | 0,397 | 0,407 | | 0,429 | 0,470 | **0,535** |
| sin `Pacquiao` | 145 | 0,479 | 0,494 | 0,540 | | 0,580 | 0,548 | **0,618** |
| sin `sparring-3` | 380 | 0,430 | 0,496 | 0,530 | | 0,527 | 0,615 | **0,664** |
| **media** | | **0,422** | **0,462** | **0,492** | | **0,512** | **0,544** | **0,605** |

**Monótona en las dos columnas.** El ensamble sube 0,09 de F1 en dos incrementos, y el fold
más confiable —`sin sparring-3`, con 380 golpes en validación, tres veces más que cualquier
otro— va de 0,527 a 0,664.

**Control.** La partición en distribución entrena sólo sobre la mitad de `sparring-3` y no
toca las fuentes nuevas. Da **idéntico bit a bit** en las tres rondas: 0,4447 en una corrida
y 0,4088 en ensamble. El pipeline es determinista y la diferencia viene sólo de los datos.

## Por qué la primera medición no lo vio

No fue un error de medición sino de potencia. Con desvío ±0,10 entre semillas y cinco
semillas, el error estándar de la media es ~0,045: el piso de detección ronda 0,06 de F1, y
el primer incremento valía menos que eso.

Lo que lo hizo visible fueron dos cosas a la vez:

- **El ensamble**, que es determinista y no tiene varianza de semilla, así que sus deltas se
  leen sin ruido encima.
- **Incrementos más grandes.** `02` sumó 67 golpes; `03` sumó 131 y `04` sumó 124.

| | `02-sparring` | `03-sparring` | `04-sparring` |
|---|---|---|---|
| Golpes | 67 | 131 | 124 |
| Golpes por minuto | 23,9 | 45,9 | 41,6 |
| Hooks | 13% | 28% | 21% |
| Uppercuts | 1% | 5% | **19%** |

`04` es el que trajo los uppercuts: 24 en un solo video, cuando en las cinco fuentes
anteriores eran la clase marginal con 2, 5 y 6 ejemplos.

Sigue sin poder separarse *más fuentes* de *más datos* de *más diversidad*: los tres
cambiaron juntos en cada incremento. Lo honesto es decir que el conjunto se paga, no cuál de
los tres factores lo hace.

## La tabla completa, seis fuentes

| Partición | Golpes | Una corrida | Desvío | **Ensamble** | recall | precisión | Heurística |
|---|---|---|---|---|---|---|---|
| En distribución (`sparring-3`) | 81 | 0,445 | 0,090 | 0,409 | 0,691 | 0,290 | 0,248 |
| sin `02-sparring` | 62 | 0,451 | 0,134 | 0,542 | 0,419 | 0,765 | 0,309 |
| sin `03-sparring` | 129 | 0,577 | 0,104 | **0,651** | 0,550 | 0,798 | 0,388 |
| sin `04-sparring` | 120 | 0,508 | 0,112 | 0,581 | 0,450 | 0,818 | 0,286 |
| sin `Sparring` | 114 | 0,407 | 0,113 | 0,535 | 0,509 | 0,563 | 0,243 |
| sin `Pacquiao` | 145 | 0,540 | 0,115 | 0,618 | 0,469 | **0,907** | 0,303 |
| sin `sparring-3` | 380 | 0,530 | 0,114 | **0,664** | 0,566 | 0,802 | 0,274 |
| *Techo humano* | | | | *0,907* | *0,911* | *0,903* | |

**La precisión de `sin Pacquiao` es 0,907 contra 0,903 del humano.** Sobre metraje de
transmisión profesional que el modelo nunca vio, empata el techo. El recall ahí es 0,469.

## Lo que no encaja

`sin 02-sparring` **bajó** al sumar `04`: de 0,602 a 0,542 en ensamble. Es el fold más chico
—62 golpes— y el de la fuente más atípica, con la mitad de golpes por minuto que el resto.
Con ese tamaño no se puede distinguir ruido de fold chico de un efecto real. Se deja
anotado sin explicación en vez de inventarle una.

## Nota sobre las guardias

Estas mediciones se corrieron con `fighter_A` de `02` y `04` anotados como diestros, siendo
zurdos. **No las afecta**: el detector exporta con `--label-space side`, que es el lado
observado y no pasa por la guardia, y colapsa las clases a O/B/I. Verificado después de
corregir: los `.det.npz` salen **idénticos byte a byte**.

Lo que sí cambia es el espacio `lead-rear`, o sea el clasificador, donde 175 eventos de tres
peleadores tenían jab y cross intercambiados.

## Qué sigue

**Anotar `01-sparring`**, la séptima y la única con cámara en mano dentro del ring, un
régimen que no existe en el dataset. Si la tendencia se sostiene, es el incremento que más
tiene para decir sobre invariancia a la cámara.

**El recall.** La precisión ya llegó al techo humano en un fold; el recall se mueve entre
0,42 y 0,57 contra 0,911. Ahí queda todo el error, y el ensamble lo compra caro: mejora
precisión sacrificando recall.
