# El producto contra la anotación manual: recall 0,432, y el 18% de los golpes ni siquiera se ve

**22-09-2026 · cinco sesiones subidas por la web, 546 golpes anotados a mano · detector
`detector_7fuentes.ens.pt` · emparejador a IoU 0,3, el mismo del resto del proyecto.**

## Por qué

El detector está medido y la identidad está medida. El producto no. Entre las piezas y lo
que el usuario ve hay una etapa que ninguna de las dos mediciones incluye: **la identidad
automática**. En los folds del detector la identidad venía resuelta a mano, porque las
fuentes estaban anotadas; en el producto la resuelve el sistema con una sola respuesta
humana.

El disparador fue concreto: sobre `Sparring` la Fight-Card mostró 27 golpes y la anotación
manual de ese mismo video tiene 125.

## Método

Las cinco sesiones se subieron por la web y se procesaron enteras por el circuito del
producto, con la siembra elegida mirando los recortes. Después se emparejó cada Fight-Card
contra el `annot.json` del mismo video —verificado por sha256 que es el mismo archivo— con
`agreement.emparejar` a IoU 0,3.

Se prueban las dos orientaciones de A y B y se toma la mejor: los roles de la anotación los
puso una persona y los de la sesión salen de la semilla que eligió el usuario, así que no
tienen por qué coincidir.

Lo que falta se parte en dos, y esa partición es el punto del experimento:

- **Sin identidad**: el peleador no estaba resuelto en al menos la mitad de los cuadros del
  golpe. Ahí no hay detector que valga; el carril está enmascarado.
- **Visible y no encontrado**: estaba resuelto y el detector no lo marcó.

```bash
boxtwin-annotator medir ~/boxtwin/datos/sesiones/*/ \
    --anotaciones anotacion anotacion-spar-0{1,2,3,4}
```

## Resultado

| fuente | anotados | marcas | empar. | recall | precisión | sin identidad | recall visible | cobertura |
|---|---|---|---|---|---|---|---|---|
| 01-sparring | 99 | 63 | 59 | 0,596 | 0,937 | 11% | 0,670 | A 85% / B 82% |
| 02-sparring | 67 | 45 | 40 | 0,597 | 0,889 | 1% | 0,606 | A 97% / B 89% |
| 03-sparring | 131 | 83 | 61 | 0,466 | 0,735 | 10% | 0,517 | A 79% / B 93% |
| 04-sparring | 124 | 75 | 56 | 0,452 | 0,747 | 12% | 0,514 | A 81% / B 91% |
| **Sparring** | **125** | **27** | **20** | **0,160** | 0,741 | **47%** | 0,303 | **A 27%** / B 73% |
| **total** | **546** | **293** | **236** | **0,432** | **0,805** | **18%** | **0,528** |

**Advertencia que va primero y no al final: las cinco fuentes están adentro del
entrenamiento de `detector_7fuentes.ens.pt`.** Esto mide el comportamiento del producto de
punta a punta sobre material que el detector ya vio. No mide generalización, y ningún
número de acá se puede comparar con los 0,885 / 0,484 de la spec, que se midieron sobre
fuente no vista.

## Lo que dice

**El 18% de los golpes anotados cae en cuadros sin identidad resuelta.** Son 98 golpes de
546 que ningún detector podría encontrar, porque el carril de ese peleador está en cero.
Esa pérdida no aparece en ninguna medición anterior del detector y es enteramente de la
identidad automática.

Descontándola, el detector encuentra 0,528 de lo que sí pudo ver. Ese número sí es
comparable en forma con el 0,484 medido sobre fuente no vista, y está apenas arriba, que
es lo que corresponde para material en distribución.

**La precisión se sostiene: 0,805.** El sistema no inventa golpes. Sobre `Sparring`, las 27
marcas caen **todas** a menos de medio segundo de un golpe anotado.

## Sparring es el caso que hay que mirar

Con recall 0,160 está tres veces abajo del resto, y la causa no es el detector:

- **El peleador A está identificado el 26,8% del video.** Tres de cada cuatro cuadros de A
  son un carril vacío.
- 47% de los golpes anotados caen en cuadros sin identidad.
- De 453 tracks, el núcleo son 30 y la coexistencia resolvió 22. Quedaron 327 sin asignar.
- La separación de color entre los dos perfiles dio **0,354 contra un piso de 0,55**, así
  que el color se abstuvo, como corresponde. Sin color, lo único que reparte es la
  coexistencia, y no alcanzó a los fragmentos.

Se descartó la explicación que el propio proyecto tenía a mano. `Sparring` es la única
fuente con `boundary_definitions_version: 1` y sus golpes duran 337 ms de mediana contra
270 ms de las marcas del sistema, así que el IoU 0,3 podría estar castigando marcas
buenas. **No es eso**: relajando el IoU de 0,30 a 0,05 el recall se mueve de 0,160 a 0,184.
El problema no es que las marcas fallen el solapamiento, es que hay 27 marcas para 125
golpes.

## Consecuencia

El cuello de botella del producto hoy es **la cobertura de identidad**, no el detector.
Donde la identidad cubre bien —02-sparring, con 97% y 89%— el recall llega a 0,597 con
apenas 1% de golpes invisibles. Donde se cae, arrastra todo.

Eso ordena el trabajo:

1. La siembra fija el ancla y decide quién es A, y eso ya está medido y funciona. Lo que no
   resuelve es **el alcance**: cuántos de los 453 tracks terminan con rol. Ahí hoy no hay
   más que la coexistencia, y cuando el color se abstiene no queda nada.
2. El plan B de C1 declarado en la spec —corrección manual por track en la web— deja de ser
   un plan B y pasa a ser lo que decide si el producto sirve sobre material como `Sparring`.
3. C1 sigue sin medirse como tal. Esto no la reemplaza: C1 pregunta cuántos tracks quedan
   **bien** asignados, y acá se midió cuántos golpes se encuentran, que es otra cosa.

## Lo que este experimento no dice

- Nada sobre generalización: todo es en distribución.
- Nada sobre si los tracks asignados son los correctos. Un track con rol equivocado cuenta
  como "identidad resuelta" y sus golpes como visibles, así que el 0,528 de recall sobre lo
  visible es, si acaso, optimista.
- Nada sobre el tipo de golpe: el clasificador no corrió en ninguna de las cinco.
