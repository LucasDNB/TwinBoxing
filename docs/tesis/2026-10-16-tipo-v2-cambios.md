# Cambios a la tesis por el tipo de golpe v2 (16-10)

Fuente: `docs/experiments/2026-10-16-tipo-v2.md` (CA1, CA2, CA7 y M1). M1 pre-registrada en
`4ff0712`. Como las notas anteriores, los textos se pegan a mano en el `.tex`; el ANTES lo pone
Lucas. v2 está construida y medida pero no encendida: hasta que Lucas encienda la bandera, el
producto sigue con PoseConv3D y estos párrafos se escriben en condicional donde corresponde.

## 1. Cap 4 §3.3 (clasificación del tipo de golpe) y descripción del prototipo

DESPUÉS (párrafo nuevo, después de la descripción de PoseConv3D): "El prototipo tiene una
segunda forma de clasificar el tipo, detrás de una bandera de configuración. La familia del
golpe (recto, hook o uppercut) la decide un modelo chico sobre las mismas medidas del brazo
que usa el detector, más seis que ubican la muñeca respecto del torso del rival. Si un recto
es jab o cross no lo decide ningún modelo: sale del brazo que marcó el detector y de la
guardia del boxeador, que el entrenador carga en su perfil. Si no la cargó, el sistema supone
guardia ortodoxa y lo dice al lado del tipo. Esta forma corre en el mismo entorno que el
detector, sin el stack de OpenMMLab."

QUÉ CAMBIÓ: el capítulo describía un solo camino de clasificación. Ahora describe el segundo,
con lo que lo diferencia: familia por modelo, mano por regla.

## 2. Cap 6, R-01 (generalización del clasificador)

DESPUÉS (agrega al párrafo del 16-10, al final): "Sobre los golpes que encuentra el detector en
cuatro sesiones completas, con el modelo entrenado sin la fuente de cada una, la familia sale
bien en el 78,0% (Wilson 73,6% a 81,8%, 390 golpes) y el tipo de cuatro clases en el 77,2%
con la guardia correcta. Estos números no se comparan con el 72,7%: el detector encuentra más
fácil los golpes más claros. Lo que sí se compara es la ventana: con la ventana anotada, el
mismo modelo da 79,0% sobre los mismos golpes, así que las fronteras del detector casi no
cuestan."

QUÉ CAMBIÓ: R-01 tenía la alternativa medida con la ventana anotada. Ahora tiene también lo que
pasaría en el producto, con el aviso de por qué ese número sale más alto.

## 3. Cap 6, R-10 (imagen del worker)

DESPUÉS (agrega): "La clasificación v2 no usa el entorno de OpenMMLab. Con v2 encendida, el
worker puede correr entero en un solo entorno, y la parte de la imagen que más riesgo tenía,
mmcv compilado contra CUDA 11.8, deja de ser necesaria para el tipo de golpe. La imagen sigue
sin construirse."

QUÉ CAMBIÓ: R-10 dependía de que la imagen llevara los dos entornos. Con v2 deja de depender
para el tipo, y se dice que la imagen todavía no existe.

## 4. Cap 7, apartado nuevo: el tipo v2 en el producto

Párrafos propuestos:

"Llevé al producto el modelo geométrico detrás de una bandera apagada y medí, con una medición
escrita antes de correrla, cuánto acierta sobre los golpes que encuentra el detector. Sobre
cuatro sesiones completas, cada una clasificada con el modelo entrenado sin su fuente, la
familia sale bien en el 78,0% de 390 golpes emparejados con los anotados. Con la ventana
anotada en vez de la del detector, el mismo modelo da 79,0%."

"Jab o cross sale del brazo y de la guardia. Con la guardia de cada boxeador, el tipo de
cuatro clases da 77,2%, casi lo mismo que la familia: la mano casi no agrega error. Sin
guardia cargada, el sistema supone ortodoxa, y en las dos sesiones con un zurdo el tipo de
cuatro clases cae a 43,1% y 50,7%. Por eso la Fight-Card avisa cuando supuso la guardia."

"Para verificar que el producto calcula lo mismo que se midió, comparé las medidas de entrada
de los 1.047 golpes anotados: coinciden exactamente. Al reproducir el modelo apareció otra
cosa: el entrenamiento en GPU no es determinista, y con la misma semilla dos corridas difieren
en sus probabilidades. El modelo del producto se entrenó en CPU, donde sí lo es, con la misma
receta. El 72,7% de la medición anterior es un número de cinco semillas entrenadas en GPU, y
una reejecución daría otro cercano."

QUÉ CAMBIÓ: el Cap 7 tenía E4 como experimento. Ahora tiene su paso al producto, el número de
punta a punta y el hallazgo sobre reproducibilidad, que vale también para los demás modelos
entrenados en GPU del proyecto.

## 5. Cap 7, bolsa y sombra con v2 (agregado 16-10 a la tarde)

Párrafo propuesto, a continuación del apartado de v2:

"En bolsa y sombra hay una sola persona en cuadro. El modelo de familia usa seis medidas que
ubican la muñeca respecto del torso del rival, y sin rival entrarían en cero durante todo el
golpe, un caso que casi no vio al entrenar. Para esas sesiones uso la misma receta sin esas
seis medidas, que ya había medido dejando una fuente afuera: acierta la familia en el 72,4%,
contra 72,7% con el rival. Ninguno de los dos se midió sobre golpes a la bolsa o al aire, así
que la Fight-Card de una sesión individual muestra el número como medido en sparring. La
altura no se estima: la regla geométrica necesita al rival."

QUÉ CAMBIÓ: el Cap 7 no decía con qué modelo se clasifican las sesiones de una sola persona.

Si la medición de B8 se hace, su resultado va en el mismo apartado, pase o no pase.

