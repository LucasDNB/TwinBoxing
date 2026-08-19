# El protocolo de reanotación ciega medía su propia ambigüedad

**19-08-2026 · Sparring.mp4 · hallazgo metodológico, no un bug puntual.**

## Qué pasó

Primera corrida real de reanotación ciega sobre 35 eventos de los 118 anotados. Resultado
según el protocolo v1:

| Dimensión | Kappa |
|---|---|
| side | 0,349 |
| punch_type | 0,432 |
| target | 0,433 |
| completeness | −0,030 |

Error de fronteras: MAE de 8,1 cuadros al inicio y 8,4 al final, sobre golpes que duran 9,9
cuadros de media. O sea, un error casi tan grande como el objeto medido.

## La señal que no cerraba

El error de **duración** era de 2,2 cuadros, muy por debajo del error de posición. Eso no es
compatible con un anotador que no sabe dónde empiezan los golpes: significa que marcaba
ventanas del largo correcto, pero corridas.

## La causa

El protocolo mostraba una ventana de video y pedía reanotar "el evento X" **sin decir cuál
de los golpes visibles era**. En boxeo los golpes se encadenan, así que la ventana contiene
seguido dos o tres.

Medido sobre la muestra: **26 de las 35 ventanas contienen más de un golpe del mismo
peleador.** En la corrida, **15 de 34 intentos ciegos reanotaron un golpe distinto del
objetivo**, y en 12 de esos 15 la etiqueta puesta coincidía con la del golpe que realmente
se marcó.

El caso más claro es el primero: objetivo `ev_0004` (right hook, 237–248), se marcó 229–236
con etiqueta left-straight. Eso es `ev_0003`, el jab que viene justo antes.

Había además un defecto de interfaz que contribuía: el panel nunca decía de qué peleador era
el intento, y el golpe se registraba bajo el que estuviera seleccionado en la UI.

## Reanálisis

Reemparejando cada marca con el evento de máximo solapamiento temporal del mismo peleador,
una regla que no mira las etiquetas:

| Dimensión | v1 | Reemparejado |
|---|---|---|
| side | 0,349 | 0,606 |
| punch_type | 0,432 | 0,803 |
| target | 0,433 | 0,713 |
| completeness | −0,030 | 0,473 |
| MAE de frontera | 8,1 / 8,4 | **1,4 / 2,0** |

**Este número no se reporta.** Es un reanálisis hecho después de ver que el resultado era
malo, y aunque la regla de reemparejamiento sea objetiva, elegirla a posteriori es una
decisión tomada con los datos a la vista. Lo que sí establece con solidez es el diagnóstico:
el problema estaba en el protocolo, no en el criterio de anotación.

Nota sobre `completeness`: su kappa de −0,03 con 94% de acuerdo es la paradoja de kappa con
marginales extremos, 32 de 34 casos en una sola categoría. No es informativo en ninguna de
las dos versiones.

## El protocolo v2

La tarea pasa a ser **"marcá todos los golpes de este peleador en esta ventana"**, y el
emparejamiento se vuelve un paso explícito de la evaluación con su umbral declarado, que es
como se evalúa detección temporal de acciones.

Umbral de solapamiento 0,3 y no 0,5: los golpes duran unos 10 cuadros, así que exigir 0,5
descartaría las parejas que difieren en 3 o 4 cuadros de frontera, y ese error es justo lo
que se quiere medir. Un umbral que descarta los casos difíciles reporta el error de los
fáciles.

Beneficio extra: mide **detección** además de clasificación. Un golpe de la anotación sin
pareja es una omisión; uno de la reanotación sin pareja puede ser un golpe que la primera
pasada se perdió, y en ese caso el dataset está incompleto. La v1 no podía dar ese número
porque cada intento admitía un solo golpe por construcción.

## Alcance

Esto no es específico de esta herramienta. **Cualquier protocolo de anotación por evento
sobre acciones que se solapan tiene el mismo problema**, y afecta también a los datasets
cortados en clips: un clip recortado alrededor de un golpe contiene los golpes vecinos y no
dice cuál es el etiquetado. Es plausible que sea parte de por qué la verificación manual de
BoxingVI resultó tan difícil, porque ahí el revisor tampoco sabía a quién ni a cuál mirar.

## Estado

Los 35 intentos de la v1 se conservan como evidencia y se migran a v2 marcados con
`protocolo_v1`. El reporte declara que sus números de detección no significan nada. La
muestra hay que rehacerla con semilla nueva: los resultados de la primera corrida ya se
vieron, incluida la lista de los 15 casos mal emparejados.

---

## Resultado con el protocolo v2

Muestra nueva, semilla 7, 35 ventanas, 35 intentos ciegos. Medido contra la anotación de
**118 eventos**, es decir el estado previo a las correcciones que la propia reanotación
motivó.

### Detección

| | |
|---|---|
| Golpes exigibles (anotados y enteros en alguna ventana) | 56 |
| Encontrados | 51 |
| Omitidos | 5 |
| Marcas del reanotador | 72 |
| Sin correspondencia | 7 |
| **Recall** | **0,911** |
| **Precisión** | **0,903** |

### Clasificación, sobre 51 golpes emparejados

| Dimensión | Acuerdo | Kappa |
|---|---|---|
| side | 1,000 | **1,000** |
| punch_type | 0,882 | 0,755 |
| completeness | 0,961 | 0,651 |
| target | 0,941 | 0,635 |

`target` y `completeness` tienen kappa bajo con acuerdo altísimo: es la paradoja de kappa
con marginales extremos, casi todo es `head` y `full`. Se reportan con el acuerdo y el n al
lado, no con el kappa solo.

### Fronteras

| | MAE | Sesgo | Mediana | Máx | Exactos |
|---|---|---|---|---|---|
| start | 1,12 | −0,92 | 1,0 | 6 | 16/51 |
| end | 1,55 | −0,80 | 1,0 | 8 | 13/51 |
| duración | 1,65 | +0,12 | 2,0 | 8 | 12/51 |

Sobre golpes de 9,9 cuadros de media, un error de 1,1 cuadros es el 11% de la duración.
**Es el techo contra el que hay que reportar el error del modelo**, no cero.

## Dos correcciones al análisis, encontradas leyendo este resultado

Ninguna estaba en la anotación.

**Golpes en el borde.** Un evento que asomaba por el borde de la ventana quedaba fuera de
los exigibles, pero si el reanotador lo marcaba, la marca se contaba como inventada. 8 de
las 12 marcas "sin correspondencia" eran eso. Ahora se emparejan sin entrar al acuerdo.

**Observaciones repetidas.** Las ventanas se solapan, así que un mismo evento caía en dos y
aportaba dos observaciones al kappa: 60 parejas eran 52 eventos distintos. Ahora entra una
vez, con su mejor pareja.

Con las dos corregidas, recall pasó de 0,857 a 0,911 y precisión de 0,833 a 0,903.

## Los 7 golpes que faltaban

Las 7 marcas sin correspondencia se revisaron a ojo y **las 7 son golpes reales**. Se
agregaron como `ev_0121` a `ev_0127`, con dos correcciones de tipo respecto de lo marcado en
la reanotación. La anotación pasó de 118 a 125 eventos: un subconteo del 5,6%.

**El número de acuerdo no se recalcula sobre la anotación corregida.** Sería circular: esos
7 golpes salieron de la reanotación, así que coincidirían por construcción. Corriendo el
reporte contra la versión corregida da recall 0,900 y precisión 0,958, y ese 0,958 no
significa nada. La medición válida queda congelada en `exports/Sparring.agreement.previo.*`
y es la que va al capítulo, declarando que se midió sobre 118 eventos.

## Hook contra straight: no hay descriptor geométrico

De los 8 desacuerdos de `punch_type`, 6 son `hook` anotado y `straight` reanotado. La
confusión es direccional, no simétrica.

Se probaron tres descriptores sobre los 103 golpes completos, buscando una definición
operacional medible:

| Descriptor | AUC |
|---|---|
| Ángulo del codo en máxima extensión | 0,608 |
| Rectitud de la trayectoria de ida | 0,638 |
| Barrido angular alrededor del hombro | 0,510 |

El mejor umbral sobre el ángulo del codo acierta 62% contra 59% de predecir siempre la clase
mayoritaria. **No hay separación.** La proyección monocular es la explicación más probable:
un hook tirado hacia la cámara se proyecta como un straight.

Consecuencia para la anotación: la definición operacional no puede ser un umbral, tiene que
ser fenomenológica y la pone el anotador.

Consecuencia para el modelo, que es la que importa: **el eje difícil de este problema es la
familia del golpe, no el brazo.** Coincide con `side` en kappa 1,000 contra `punch_type` en
0,755, y con lo que ya se había medido en BoxingVI, donde V4 fallaba en familia conservando
lateralidad. No prueba que PoseConv3D no pueda —un modelo espaciotemporal ve mucho más que
tres descriptores a mano— pero es una señal que conviene declarar antes de entrenar y no
después.
