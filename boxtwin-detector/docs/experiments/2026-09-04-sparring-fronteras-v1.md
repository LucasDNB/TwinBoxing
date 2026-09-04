# `Sparring` marca de más porque su anotación usa otra convención de fronteras

**04-09-2026 · siete fuentes · causa encontrada, y no es del detector.**

## El síntoma

`sin Sparring` es el peor fold en las cuatro rondas. Su recall (0,482) está dentro del rango
de las otras seis (0,467–0,589); lo atípico es su **precisión: 0,561 contra 0,705–0,924**.

## Cuatro hipótesis descartadas

| Hipótesis | Por qué cae |
|---|---|
| Resolución 640x360 | El preproceso corre con `imgsz=640` en las siete: la pose ve lo mismo en todas |
| Calidad de pose | Su `kp_score` mediano es el **más alto** de las siete (0,945) |
| Pose interpolada | Afecta al recall, no a la precisión, y su recall está en rango |
| Densidad de golpes | 42,5/min, justo en el medio del rango (23,9 a 52,8) |

## Dónde caen los falsos positivos

| Fuente | FP/min | **FP pegados a un golpe** | FP lejos |
|---|---|---|---|
| **`Sparring`** | 8,2 | **50%** | 50% |
| `03-sparring` | 3,9 | 9% | 91% |
| `pacquiao` | 1,4 | 25% | 75% |
| `01-sparring` | 6,6 | 22% | 78% |

La mitad de lo que `Sparring` marca de más cae **a menos de medio segundo de un golpe
anotado**. Y no es que marque mucho: predice 23,3 marcas por minuto, la **más baja** de las
cuatro. `03-sparring` predice 27,0 y tiene precisión 0,792.

Descartada también la fragmentación: subir `hueco_maximo` de 2 a 12 no absorbe esas marcas y
empeora las cuatro fuentes de forma monótona.

Mirando los casos uno por uno aparece el patrón. `B-left` predice `[714,720]` contra un golpe
anotado `[717,727]`: se solapan 4 cuadros, IoU 0,29, **falla el umbral de 0,3 por un pelo**.
No son alucinaciones, son detecciones **mal localizadas** de golpes reales.

## La causa

| Fuente | `boundary_definitions_version` | Duración mediana |
|---|---|---|
| **`Sparring`** | **1** | **337 ms** |
| `Pacquiao` | 2 | 200 ms |
| `sparring-3` | 2 | 233 ms |
| `01` a `04` | 2 | 200–233 ms |

`Sparring` es **la única fuente anotada bajo la versión 1** de las definiciones de frontera, y
sus golpes son un **45% más largos** que los de todas las demás. Está documentado: el 19-08 se
escribió la definición operacional del tipo de golpe y `boundary_definitions_version` pasó de
1 a 2, con la nota de que *"los 125 de Sparring.mp4 son versión 1"*. Nunca se reanotaron.

El número que lo cierra es la razón entre lo que el modelo marca y lo que la anotación dice:

| Fuente | Marca del modelo | Anotación | Razón |
|---|---|---|---|
| **`Sparring`** | 6,0 | **10,0** | **0,60** |
| `03-sparring` | 7,0 | 7,0 | 1,00 |
| `Pacquiao` | 7,0 | 6,0 | 1,17 |
| `01-sparring` | 7,0 | 7,0 | 1,00 |

**El modelo predice 7 cuadros en todas las fuentes** —aprendió la duración de la convención
v2, que es la de seis de los siete— y en `Sparring` se le compara contra golpes de 10. Una
marca de 6 cuadros dentro de un golpe de 10 da IoU 0,60 en el mejor caso, y con tres o cuatro
cuadros de desplazamiento cae por debajo de 0,3.

Bajando el umbral de IoU, la brecha se cierra en parte y **sólo en `Sparring`**:

| IoU mínimo | `Sparring` | Las otras tres | Brecha |
|---|---|---|---|
| 0,30 | 0,647 | 0,849 | −0,202 |
| 0,20 | 0,706 | 0,849 | −0,143 |
| 0,10 | 0,721 | 0,849 | −0,128 |

Las otras tres no se mueven **un solo punto** en todo el barrido: el efecto es específico de
`Sparring` y es del emparejador. Se cierra el 37% de la brecha.

## Qué queda sin explicar

El 63% restante. La convención de fronteras es la causa principal y está medida, pero no es la
única: con IoU 0,10 `Sparring` sigue 0,128 por debajo. Conviene declararlo en vez de dar el
caso por cerrado.

## Consecuencia

**El fold `sin Sparring` mide, en buena parte, un cambio de convención de anotación y no la
calidad del detector.** Es el mismo tipo de error que el proyecto ya documentó dos veces —
comparar contra un número producido bajo otro protocolo— y acá se coló por dentro del propio
dataset.

Tres caminos, en orden de costo:

1. **Que el reporte muestre `boundary_definitions_version` por fuente**, para que un fold con
   otra convención se vea como no comparable en lugar de promediarse con los demás.
2. **Reanotar las fronteras de `Sparring` bajo la v2.** Son 125 eventos y no hace falta volver
   a clasificar nada: sólo mover inicio y fin.
3. **Dejarlo como está y reportarlo aparte**, declarando la versión.
