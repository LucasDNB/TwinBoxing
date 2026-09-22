# El MVP sobre video real: 1,09x tiempo real, y la siembra saca la identidad de 3 tracks a 11

**22-09-2026 · `amateur_estatico`, 120,1 s, 636x360, 30 fps · detector `detector_7fuentes.ens.pt`
(ensamble de 5 semillas sobre las siete fuentes) · guantes `guantes-v2.pt` · RTX 2080 Super.**

## Por qué

El comando de inferencia, la API y el frontend estaban construidos y testeados, pero todos
los tests usan pose sintética y una TCN sin entrenar: prueban que los datos lleguen enteros
de una punta a la otra, no que el sistema funcione. Esto es la primera corrida del circuito
completo sobre un video, y sobre el más difícil que hay disponible.

`amateur_estatico` es boxeo amateur de competencia: cámara lejana, árbitro adentro del ring,
público, y un ancho de hombros mediano de **18,4 px** contra 132 en `sparring-3`. El sistema
nunca lo vio: no está entre las siete fuentes de entrenamiento.

## Método

La sesión se armó en un directorio aparte, con el video enlazado. El preproceso se rehizo
entero (el cache viejo tenía otra configuración), así que los `track_id` son nuevos y no
comparables con los de `anotacion-amateur/`.

```bash
boxtwin-annotator procesar amateur_estatico.mp4 --out sesion/ \
    --modelo-guantes guantes-v2.pt --modelo-pose yolov8l-pose.pt --round 120
boxtwin-annotator completar sesion/ --semilla-a 11 --semilla-b 16 \
    --detector detector_7fuentes.ens.pt
```

Las dos semillas salieron de mirar los dos recortes que propuso el sistema, que es lo que
haría el usuario. Los dos son del cuadro 2315, aislados: uno de casco azul y camiseta roja,
otro de casco rojo y camiseta celeste. La pregunta se contesta en dos segundos.

## Resultado 1: el tiempo entra en el criterio, con margen

| Etapa | Segundos |
|---|---|
| Preproceso (pose + tracking, 3601 cuadros) | 106,9 |
| Evidencia de guante (20 tracks) | 24,5 |
| Identidad con la siembra | 0,01 |
| Series por peleador | 0,12 |
| Detector (4 carriles, 5 modelos) | 0,41 |
| Guardia | 0,00 |
| **Total** | **131,9 s sobre 120,1 s de video = 1,098x** |

El criterio C5 pide 2x o menos. Queda a menos de la mitad del presupuesto, **sin la etapa de
clasificación**, que corre en el otro entorno y no está incluida acá.

Lo que llama la atención es el reparto: el 99,6% del tiempo es pose y guantes. Todo lo que el
MVP agregó —identidad, features, detector, indicadores— suma **0,54 segundos**. Si hay que
bajar el tiempo, el lugar es el preproceso y no el resto, y eso ya tiene una palanca conocida
que es el benchmark de perfiles (§6 de CLAUDE.md).

## Resultado 2: la siembra hace lo que se esperaba, y el motivo no era el que se creía

Misma evidencia, las dos rutas:

| | automático | con siembra |
|---|---|---|
| Tracks resueltos por coexistencia | 3 | **6** |
| Componentes sin orientar | 2 | 1 |
| **Separación de perfiles de color** | **0,097** | **1,375** |
| Tracks asignados a A | 1 | 4 |
| Tracks asignados a B | 2 | 7 |
| Tracks sin asignar | **15** de 20 | **7** de 20 |

El camino automático se abstiene, y hace bien: con separación 0,097 contra un piso de 0,55,
los dos perfiles son el mismo color y repartir por color sería tirar una moneda. Deja 3
tracks asignados por geometría y 15 sin decidir.

Con las dos semillas, la separación pasa a **1,375**. Ese número no es un ajuste: la
diferencia es cuál componente de coexistencia se usa como referencia. El automático elige el
más grande que aparezca temprano, y en este video ese componente tenía los dos lados
contaminados —de ahí el 0,097—. La semilla fija el componente que contiene al track que una
persona señaló, y ahí los dos lados sí son los dos peleadores.

**Esto corrige la hipótesis anterior.** El riesgo 1 de la spec decía que en el amateur los
guantes son casi del mismo color (separación 0,133 medida antes). Lo que esta corrida muestra
es que el problema no era el color de los guantes sino **el ancla**: con el componente
correcto, los mismos guantes separan 1,375, catorce veces más que el piso. No alcanza para
declararlo general con un video, pero sí para decir que la explicación que estaba escrita no
es la única y probablemente no es la que manda.

Queda sin medir si los 11 tracks asignados son los correctos. `anotacion-amateur/` no tiene
identidad anotada, así que C1 no se puede puntuar acá: hay que correrlo sobre las seis
fuentes de gimnasio, que sí la tienen.

## Resultado 3: 46 golpes detectados, y el conteo se lee comparando

| | total | izq | der |
|---|---|---|---|
| Peleador A | 22 | 15 | 7 |
| Peleador B | 24 | 16 | 8 |

Cobertura de identidad: A el 67,7% del video, B el 84,7%, y un 10,5% del tiempo con alguien
en pantalla sin identificar. Son 23 golpes por minuto entre los dos, contra los 45,9 por
minuto de `03-sparring`, que es la fuente más intensa del dataset. Con recall medido 0,484 el
conteo está por debajo del real y no hay anotación de este video contra la cual medir, así
que **el número no se reporta como cuántos golpes hubo**. Lo que sí queda es lo que el
producto propone leer: los dos peleadores trabajaron parejo (cociente 0,92) y los dos tiraron
el doble de izquierda que de derecha.

## Resultado 4: el indicador de guardia no mide lo que dice medir

Acá está el hallazgo caro, y aparece antes de C3.

Sobre los 41 golpes con mano opuesta medible, el indicador marca **40 con la mano caída**. Y
de los 46 golpes, en 35 la mano que pegó **no vuelve** a la zona de guardia en un segundo y
medio. Leído como táctica, eso diría que dos amateurs de competencia pelean dos minutos con
la guardia abajo, lo cual no pasa.

La causa está en la definición operativa, no en los peleadores. Midiendo la distancia de la
muñeca a la nariz en anchos de hombro sobre los 3127 cuadros con pose confiable:

| | p10 | mediana | p90 | bajo 0,6 |
|---|---|---|---|---|
| A, muñeca izquierda | 1,09 | 1,75 | 4,25 | 1,9% |
| A, muñeca derecha | 0,93 | 1,49 | 3,42 | 1,9% |
| B, muñeca izquierda | 0,75 | 1,53 | 4,20 | 5,5% |
| B, muñeca derecha | 0,77 | 1,39 | 4,21 | 3,7% |

**La muñeca casi nunca está a menos de 0,6 anchos de hombro de la nariz**, ni siquiera en
guardia. Tiene sentido geométrico: en guardia el puño está al lado del mentón, y del mentón a
la nariz ya hay una distancia vertical que en esta proyección se suma. El umbral de 0,6 no
describe "mano arriba", describe "mano tocándose la cara".

El umbral es un parámetro y **no se cambió**: la definición está pre-registrada en la spec
como criterio de dominio a confirmar por Lucas, y moverla mirando estos datos sería elegir el
criterio después de ver el resultado, que es el error que este proyecto ya documentó dos
veces. Lo que esta corrida aporta es el rango sobre el que calibrarla: el p10 anda entre 0,75
y 1,09, así que la zona de guardia probablemente esté cerca de 1,0 y no de 0,6.

C3 sigue siendo la medición que decide, y ahora se sabe que va a fallar con el umbral actual.

## Lo que esta corrida no prueba

- No hay anotación de este video, así que **no se midió ni precisión ni recall** sobre él. Los
  46 golpes no se compararon contra nada.
- El tipo de golpe no se estimó: la etapa de clasificación no se corrió.
- Un video de 2 minutos no dice nada sobre uno de 30, que es el límite declarado en RF1.
- La imagen del worker sigue sin construirse.
