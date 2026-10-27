# Cambios a la tesis por la campaña del clasificador (14 y 15-10)

Fuentes: `docs/experiments/2026-10-14-piloto-etiquetador.md` (E0) y
`docs/experiments/2026-10-15-clasificador-tipo-altura.md` (E1 a E3). Spec pre-registrado en
`d359247`. Las fuentes LaTeX no están en el repositorio: estos textos se pegan a mano en el
`.tex`. No tengo el texto actual de R-01, así que va como DESPUÉS con la indicación de qué
reemplaza; el ANTES lo pone Lucas al aplicarlo.

## 1. Cap 6, R-01 (generalización del clasificador)

DESPUÉS: "El clasificador de familia se midió por primera vez dejando afuera cada una de las
siete fuentes anotadas, con el mismo config de producción. Sobre una fuente que no vio acierta
la familia en el 65,6% de los golpes (intervalo de Wilson al 95%: 62,7% a 68,4%), contra el
74,6% que daba en distribución. Decir siempre recto acertaría el 63,2%, así que el modelo casi
no le agrega nada a esa regla. El número sale de una sola semilla de entrenamiento. El riesgo
sigue abierto, ahora medido."

QUÉ CAMBIÓ: R-01 describía el riesgo sin un número fuera de distribución. Ahora lo tiene, y
muestra que el 74,6% estaba inflado.

## 2. Cap 7, apartado nuevo: etiquetado asistido (E0)

Párrafo propuesto:

"Antes de usar a Claude para preetiquetar material nuevo, medí cuánto le acierta a mi
anotación mirando tiras de cuadros a ciegas. Sobre 150 golpes de las siete fuentes, acertó la
familia en el 40% (Wilson 32,5% a 48%) y la altura con una exactitud balanceada de 0,644. Los
umbrales, fijados antes de correr, eran 0,85 y 0,80. En familia no se separa del azar: su
exactitud balanceada (0,315) queda por debajo de un tercio. Confunde el hook con el recto y el
uppercut con el hook, la misma trampa monocular que le cuesta al modelo. No hay
preetiquetado."

## 3. Cap 7, apartado nuevo: espacio de clases y rival (E1 a E3)

Párrafos propuestos:

"Probé tres configuraciones del clasificador, dejando afuera una fuente por vez. La primera
replica la de producción, con jab, cross, hook y uppercut por mano adelantada o atrasada. La
segunda cambia el espacio de clases a tipo por altura, sin lado. La tercera le suma al rival
en la entrada. Corrí una sola semilla por configuración: las 63 corridas previstas llevaban
unas 75 horas de GPU.

Cambiar el espacio de clases no mejoró la familia: 64,1% contra 65,6%. La altura tampoco salió:
sobre las cuatro fuentes donde se había medido la regla geométrica, la exactitud balanceada
fue 0,541 y el recall de cuerpo 0,133, contra 0,75 y 0,60 del criterio. Sumar al rival mejoró
la altura según la regla fijada antes (balanceada de 0,557 a 0,605, mejor en cinco de siete
fuentes), pero tampoco llega al criterio.

Con ocho clases (jab, cross, hook y uppercut, a la cabeza o al cuerpo), y con jab y cross
separados por la anotación, la exactitud queda en 52,6% y 52,2%. La sub-hipótesis 1 pedía más
de 80%, así que la parte de golpes no alcanza el umbral. El número es además optimista, porque
supone que el lado del golpe llega siempre bien."

## 4. Cap 7, hallazgo sobre el código (H-F)

Párrafo propuesto:

"Revisé si el espejado de datos en el entrenamiento podía estar enseñándole al modelo
etiquetas contradictorias. No: el espejo invierte la imagen y además intercambia izquierda y
derecha, así que un jab espejado sigue siendo un jab. Lo que sí encontré es que en producción
jab y cross salen solo del clasificador, sin cruzarse con el brazo que da el detector ni con
la guardia del peleador. Sobre fuentes nuevas el clasificador acierta mano adelantada o
atrasada en el 65,6% de los golpes, menos que decir siempre adelantada."

## 5. Cap 6, riesgo nuevo

No aplica. El sesgo de anclaje del preetiquetado solo entraba si se usaba E5, y E0 no lo
habilitó.

## 6. Declaración de uso de IA, una línea

"Claude Code corrió en modo agente la campaña del clasificador (exports, particiones,
entrenamiento, evaluación y auditoría) y el piloto de etiquetado a ciegas; los criterios, los
umbrales y las decisiones de alcance los fijé yo antes de medir."
