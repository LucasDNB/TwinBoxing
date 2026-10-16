# Cambios a la tesis por E4 a E6 (16-10)

Fuentes: `docs/experiments/2026-10-16-clasificador-geometrico.md` (E4, E4a, E5, F) y
`docs/experiments/2026-10-16-rol-por-brazo.md` (E6). Pre-registro en `703fd11`. Las fuentes LaTeX
no están en el repositorio: estos textos se pegan a mano en el `.tex`. Igual que el 15-10, R-01
va como DESPUÉS con la indicación de qué reemplaza; el ANTES lo pone Lucas.

## 1. Cap 6, R-01 (generalización del clasificador)

DESPUÉS (reemplaza el párrafo del 15-10): "El clasificador de familia se midió dejando afuera
cada una de las siete fuentes anotadas. El de producción, PoseConv3D sobre mapas de calor,
acierta la familia en el 65,6% de los golpes de una fuente que no vio, contra el 74,6% en
distribución y el 63,2% de decir siempre recto. Un modelo chico sobre las mismas features
geométricas que usa el detector de golpes, más seis del rival, llega al 72,7% (Wilson 69,9% a
75,3%) y mejora al de producción en cinco de las siete fuentes, que era el criterio fijado antes
de medir. El riesgo baja, pero no se cierra: el modelo nuevo no está en producción y su número
sale de un ensamble de cinco semillas, contra una sola del de producción."

QUÉ CAMBIÓ: R-01 tenía el número fuera de distribución pero ninguna alternativa medida. Ahora
tiene una que lo supera con una regla escrita de antemano, y la salvedad de las semillas.

## 2. Cap 7, apartado nuevo: clasificador geométrico (E4) y 3D (E5)

Párrafos propuestos:

"El detector de golpes generaliza a fuentes nuevas y el clasificador no. La diferencia más
visible está en la entrada: el detector ve veinte medidas del brazo centradas en los hombros,
escaladas por el torso y espejadas para leer siempre el mismo brazo; PoseConv3D ve mapas de
calor en píxeles. Probé el mismo tipo de modelo chico que usa el detector sobre esas veinte
medidas, más seis que ubican la muñeca respecto del torso del rival, con las mismas siete
fuentes y los mismos golpes que la campaña anterior.

La familia pasó de 65,6% a 72,7%, con cinco de siete fuentes mejor, justo el mínimo que había
fijado. La altura no: sobre las cuatro fuentes de C8, la exactitud balanceada fue 0,719 contra
0,729 de la regla geométrica recalculada sobre los mismos golpes. Las seis medidas del rival
son las que llevan la altura; sin ellas, la familia casi no cambia y la altura cae a 0,613.

Después elevé la pose a 3D con MotionBERT y sumé dieciséis medidas del brazo en un marco pegado
al cuerpo, para que el arco de un hook no dependa de dónde está la cámara. La familia quedó en
73,8%, una diferencia de 1,1 puntos que no alcanza el mínimo de 3. El hook, que era la razón de
probarlo, no mejoró: 59,0% de recall contra 62,0% sin 3D.

Con ocho clases, y con jab y cross separados por la anotación, la exactitud queda en 62,3% sin
3D y 63,2% con 3D, contra 52,2% de la campaña anterior con el rival. Sigue lejos del 80% de la
sub-hipótesis 1."

## 3. Cap 7, apartado nuevo: jab y cross por el brazo (E6)

Párrafo propuesto:

"En producción jab y cross salen del clasificador. Pero el detector ya dice con qué brazo se
tiró el golpe, y con la guardia del peleador eso alcanza: un ortodoxo que pega con la izquierda
tira un jab. Sobre los 259 rectos que el detector encontró en las cuatro fuentes de C8, esa
regla acertó la mano en el 98,5% de los casos, contra 72,6% del clasificador medido sobre una
fuente que no vio. Ese 98,5% es un techo: como la guardia la tomé de la anotación, la regla
acierta justo cuando el detector acierta el brazo, y el detector se entrenó con esas fuentes.
Suponiendo a todos ortodoxos, la regla baja a 81,1%, que sigue arriba del clasificador."

## 4. Cap 8, comparación con el estado del arte monocular

Párrafo propuesto (verificar el identificador arXiv 2601.11492 de BoxMind antes de citarlo; lo
trae el spec, no lo verifiqué en esta sesión):

"BoxMind clasifica técnica y altura en boxeo de élite con pose 3D estimada desde una cámara,
sobre unos 10.900 eventos anotados por entrenadores y con partición por round. Con pose sola
reporta un F1 de 0,692 en técnica y 0,747 en altura. Mis números no son comparables uno a uno:
mido exactitud y exactitud balanceada, no F1, con 1.047 golpes y una partición más dura, por
fuente entera. Sirven como orden de magnitud: en altura, mi modelo y la regla geométrica quedan
en 0,719 y 0,729 de balanceada; en técnica, el 3D no me dio la ventaja que ellos atribuyen a la
pose 3D."

## 5. Declaración de uso de IA, una línea

"Claude Code implementó y corrió E4 a E6 por especificación (datos, features, modelo,
entrenamiento, elevación 3D, evaluación, auditoría y estos borradores) el 16-10; los criterios,
los umbrales y las configuraciones estaban fijados en el pre-registro antes de la primera
corrida."
