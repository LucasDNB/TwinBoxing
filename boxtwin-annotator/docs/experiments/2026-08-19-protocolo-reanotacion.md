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
