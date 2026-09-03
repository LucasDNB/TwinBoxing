# La quinta fuente sí movió la aguja, y la cuarta no

**03-09-2026 · cinco fuentes, 830 golpes · revisa la primera sección de
[`2026-09-03-ensamble.md`](2026-09-03-ensamble.md).**

## Qué revisa

Ese doc midió que sumar `02-sparring` —pasar de dos a tres fuentes de entrenamiento— dio un
cambio medio de **−0,002**, y concluyó, con cuidado, que *no se detectó mejora* y no que
sumar fuentes no sirva. Con `03-sparring` anotado hay un segundo incremento, y el resultado
es distinto.

## Resultado

Los folds que existen en las dos rondas, pasando de tres a cuatro fuentes de entrenamiento:

| Fold | Golpes | Una corrida: 3 → 4 fuentes | Δ | Ensamble: 3 → 4 | Δ |
|---|---|---|---|---|---|
| sin `02-sparring` | 62 | 0,458 → 0,520 | **+0,062** | 0,505 → 0,602 | **+0,097** |
| sin `Sparring` | 114 | 0,356 → 0,397 | **+0,041** | 0,429 → 0,470 | **+0,041** |
| sin `Pacquiao` | 145 | 0,479 → 0,494 | +0,014 | 0,580 → 0,548 | **−0,031** |
| sin `sparring-3` | 380 | 0,430 → 0,496 | **+0,066** | 0,527 → 0,615 | **+0,087** |
| **media** | | | **+0,046** | | **+0,048** |

Contra el **−0,002** de la ronda anterior.

**Control.** La partición en distribución entrena sólo sobre la mitad de `sparring-3` y no
toca las fuentes nuevas. Sus números salen **idénticos bit a bit** entre las dos rondas —
0,4447 en una corrida, 0,4088 en ensamble. El pipeline es determinista y la diferencia viene
sólo de los datos agregados, no de otra cosa que se haya movido.

## La tabla completa, cinco fuentes

| Partición | Golpes | Una corrida | Desvío | **Ensamble** | recall | precisión | Heurística |
|---|---|---|---|---|---|---|---|
| En distribución (`sparring-3`) | 81 | **0,445** | 0,090 | 0,409 | 0,691 | 0,290 | 0,248 |
| sin `02-sparring` | 62 | 0,520 | 0,144 | **0,602** | 0,500 | 0,756 | 0,309 |
| sin `03-sparring` | 129 | 0,496 | 0,074 | **0,515** | 0,395 | 0,739 | 0,388 |
| sin `Sparring` | 114 | 0,397 | 0,101 | **0,470** | 0,482 | 0,458 | 0,243 |
| sin `Pacquiao` | 145 | 0,494 | 0,097 | **0,548** | 0,393 | **0,905** | 0,303 |
| sin `sparring-3` | 380 | 0,496 | 0,111 | **0,615** | 0,518 | 0,755 | 0,274 |
| *Techo humano* | | | | *0,907* | *0,911* | *0,903* | |

Dos cosas para mirar:

**El fold más confiable es el mejor.** `sin sparring-3` tiene 380 golpes en validación, tres
veces más que cualquier otro, y da F1 **0,615** con recall 0,518 y precisión 0,755.

**La precisión de `sin Pacquiao` es 0,905**, contra 0,903 del humano. Sobre metraje de
transmisión profesional que el modelo nunca vio, nueve de cada diez marcas que hace son un
golpe real. El recall ahí es 0,393.

## Por qué una fuente sí y la otra no

No se puede separar limpiamente con dos incrementos, y conviene decirlo antes que inventar
una explicación:

| | `02-sparring` | `03-sparring` |
|---|---|---|
| Golpes | 67 | **131** |
| Golpes por minuto | 23,9 | **45,9** |
| Hooks | 13% | **28%** |
| Aporte al dataset de entonces | +10% | +18% |

`03` es el doble de grande y bastante más diverso en composición. El incremento cambió en
volumen, en densidad y en reparto de clases a la vez, así que la pregunta *"¿fue por sumar
una fuente o por sumar 129 golpes?"* queda abierta. Lo que sí queda establecido es que el
techo no estaba donde parecía después de la primera medición.

## Qué sigue

**Anotar `04` y `01`.** Con siete fuentes el leave-one-source-out tiene margen de verdad, y
ahora hay evidencia de que el incremento se paga.

**El recall sigue siendo la frontera.** La precisión ya llegó al nivel humano en uno de los
folds; el recall se mueve entre 0,39 y 0,52 contra 0,911. Ahí es donde queda todo el error.
