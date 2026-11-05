# Todos los cambios a realizar en el `.tex`

5-11-2026. Reúne en un solo lugar los ocho archivos `*-cambios.md` de esta carpeta, en orden de capítulo.
Donde un cambio reemplazó a otro, queda solo el vigente. Si difiere de algún archivo viejo, manda este.

El `.tex` no está en el repositorio ni en esta máquina, así que no pude ver qué ya está aplicado.
Los ANTES que aparecen se verificaron el 13-10 contra la Entrega 6. Donde no hay ANTES, lo ponés vos.

## Antes de empezar

**Una decisión tuya afecta a todo el archivo.** Dijiste que BoxTwin quedó fuera del alcance. Estos textos
hablan de BoxTwin como el producto, porque es lo medido. Si el capítulo no explica ese cambio de alcance,
hace falta una frase que lo diga. No la escribí: no tengo una definición de ese cambio ni de BoxTwinZ.

**Textos viejos que corregí al consolidar**, marcados con `CORREGIDO`, porque ya no eran ciertos:

1. El clasificador geométrico figuraba apagado: en A corre encendido (`/salud` dice `geometrico`).
2. La imagen del worker figuraba sin construir: se construyó el 20-10.
3. El mapa de exposición y el avatar figuraban sin construir: los dos existen.
4. E8 usaba 0,623 y 0,433 de F1: van 0,600 y 0,479, los de R1, que es lo que R1 pide para la tesis.
5. La comparación con BoxMind decía que el 3D no ayudó: E7 mostró que con otro estimador sí.

**Textos nuevos que no estaban en ningún archivo**, marcados con `NUEVO`: C1, C3, C2 y C7, BoxMind y E7.

---

## Capítulo 3

### 3.1 Glosario, "Mapa de calor de exposición"

ANTES: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre las zonas
corporales de un peleador, la frecuencia con la que cada zona resultó alcanzada en las sesiones
registradas. Es descriptivo de lo observado y no predictivo."

DESPUÉS: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre las zonas
corporales de un peleador, la frecuencia con la que sus rivales dirigieron golpes a cada zona en las
sesiones registradas. Cuando exista la heurística de conexión, suma la frecuencia estimada con la que cada
zona resultó alcanzada, con su margen. Es descriptivo de lo observado y no predictivo."

Por qué: "resultó alcanzada" afirmaba contacto, y un sistema monocular no lo establece.

---

## Capítulo 4

### 4.1 Tabla 4.1, fila "Clasificador de acciones"

ANTES: "Detección de eventos de golpe dentro de la secuencia y clasificación por tipo, lado y altura"

DESPUÉS: "Detección de eventos de golpe dentro de la secuencia, con el brazo que lanza, y clasificación por
tipo"

### 4.2 Tabla 4.1, fila "Fight-Card y perfil"

ANTES: "Conteos por asalto, heurísticas de conexión con margen declarado y agregación en el perfil táctico"

DESPUÉS: "Conteos por asalto, heurísticas de conexión con margen declarado y agregación en el perfil
táctico. La altura del golpe por posición de la mano respecto del rival se midió y no alcanzó su criterio
(apartado del Cap 7)"

### 4.3 §3.3, clasificación del tipo de golpe (`CORREGIDO`)

Va después de la descripción de PoseConv3D.

DESPUÉS: "El prototipo tiene una segunda forma de clasificar el tipo, que es la que corre hoy en la
estación. La familia del golpe (recto, hook o uppercut) la decide un modelo chico sobre las mismas medidas
del brazo que usa el detector, más seis que ubican la muñeca respecto del torso del rival. Si un recto es
jab o cross no lo decide ningún modelo: sale del brazo que marcó el detector y de la guardia del boxeador,
que el entrenador carga en su perfil. Si no la cargó, el sistema supone guardia ortodoxa y lo dice al lado
del tipo. Esta forma corre en el mismo entorno que el detector, sin el stack de OpenMMLab."

### 4.4 Tabla 4.5, estimadores de pose, fila nueva debajo de YOLOv8l-pose

DESPUÉS: "YOLO26l-pose (Ultralytics, pesos de COCO). Evaluado el 29 y el 31-10 con el resto del sistema
fijo: con el umbral de 0,3 pensado para YOLOv8 empeora al detector y al clasificador; con un umbral
calibrado a su propia distribución de puntajes, los iguala sobre las siete fuentes anotadas (Cap. 7)."

### 4.5 Perfiles de despliegue

DESPUÉS: "Los perfiles no los declaré, los medí. En la estación, con una RTX 2080 SUPER y dos peleadores en
cuadro, YOLOv8l-pose a 640 exportado a TensorRT FP16 tarda 10,3 ms de mediana entre decodificar y estimar
pose, con p95 de 11,3 ms y 93 cuadros por segundo sostenidos. Ese es el perfil A. El perfil B colapsa en
A: en esta máquina bajar a 15 cuadros por segundo no compra calidad, porque lo mejor que hay ya entra en
40 ms. El perfil C, solo ONNX, es el mismo modelo a 26,0 ms y 38 cuadros por segundo. El mAP de pose sobre
las personas de COCO val2017 da 0,672 en los dos."

"Esto vale para la estación. La notebook quedó sin medir, así que no extrapolo estos perfiles a otra
máquina. Tampoco dice que el modo en vivo entre en 40 ms: mide la pose, sin tracker, identidad ni dibujo."

### 4.6 Transcodificado en el dispositivo (Cap 4 o Cap 5)

DESPUÉS: "Comparé la pose sobre video transcodificado a 720p contra la de 1080p, en los mismos cuadros. Se
empareja más del 99% de las cajas y el error mediano de muñeca es de 0,8% y 1,1% del alto de la caja. Con
la regla que fijé antes de medir, 720p es apto. Es acuerdo con 1080p y no precisión contra una verdad
anotada, y vale para un codificador y una calidad."

### 4.7 §3.1, licencia

Hay una edición pendiente de licencia registrada en `docs/specs/2026-10-30-mejoras-post-e8/09-integracion-y-tesis.md`
como "memoria del 29-10". No tengo ese texto. E9 no corrió, así que no se suma nada. Revisala vos.

---

## Capítulo 6

### 6.1 R-01, generalización del clasificador (`CORREGIDO`)

Reemplaza el párrafo actual de R-01.

DESPUÉS: "El clasificador de familia se midió dejando afuera cada una de las siete fuentes anotadas.
PoseConv3D sobre mapas de calor acierta la familia en el 65,6% de los golpes de una fuente que no vio,
contra el 74,6% en distribución y el 63,2% de decir siempre recto. Un modelo chico sobre las mismas
features geométricas que usa el detector de golpes, más seis del rival, llega al 72,7% (Wilson 69,9% a
75,3%) y mejora a PoseConv3D en cinco de las siete fuentes, que era el criterio fijado antes de medir. Hoy
el producto usa este modelo. El riesgo baja, pero no se cierra: su número sale de un ensamble de cinco
semillas, contra una sola de PoseConv3D."

"Sobre los golpes que encuentra el detector en cuatro sesiones completas, con el modelo entrenado sin la
fuente de cada una, la familia sale bien en el 78,0% (Wilson 73,6% a 81,8%, 390 golpes) y el tipo de cuatro
clases en el 77,2% con la guardia correcta. Estos números no se comparan con el 72,7%: el detector
encuentra más fácil los golpes más claros. Lo que sí se compara es la ventana: con la ventana anotada, el
mismo modelo da 79,0% sobre los mismos golpes, así que las fronteras del detector casi no cuestan."

### 6.2 R-02, identidad

Sin cambios. Bajaba de probabilidad solo si E11 mejoraba y se integraba, y E11 quedó sin decidir.

### 6.3 R-10, imagen del worker (`CORREGIDO`)

Se agrega al final.

DESPUÉS: "La clasificación geométrica no usa el entorno de OpenMMLab, así que la parte de la imagen que más
riesgo tenía, mmcv compilado contra CUDA 11.8, dejó de ser necesaria para el tipo de golpe. La imagen se
construyó el 20-10 sin ese entorno. Falta probarla con GPU, y el despliegue en nube quedó sin medir."

### 6.4 Apartado 8 (y Cap 4, apartado 2.7): nota fechada (`CORREGIDO`)

No se reescriben esos apartados. Se agrega la nota.

DESPUÉS: "El 13 de octubre especifiqué una ampliación de alcance con siete pedidos, ordenada en tres tandas
con una regla de corte. Entraron las combinaciones izquierda, derecha, izquierda. La identificación de una
sola persona en bolsa y sombra entró el 21 de octubre por decisión de producto, sin la medición de
cobertura, y es función sin validar. La altura del golpe se midió y no pasó su criterio, así que se
muestra rotulada como estimación. El mapa de exposición cuenta golpes dirigidos, no recibidos. El avatar 3D
es ilustrativo y no mide profundidad."

---

## Capítulo 7

### 7.1 Criterios pre-registrados nuevos

DESPUÉS: "Con la ampliación del 13 de octubre sumé cuatro criterios, escritos antes de medir como los siete
anteriores. C8 pide que la altura del golpe, calculada con la posición de la mano respecto de los hombros y
el torso del rival, llegue a una exactitud balanceada de 0,75 y a un recall de golpes al cuerpo de 0,60. C9
compara el recall del detector sobre golpes a la cabeza y al cuerpo. C10 pide coherencia y plausibilidad al
avatar. C11 registra, sin umbral, cuántas combinaciones izquierda, derecha, izquierda encuentra el sistema
de punta a punta."

### 7.2 Dato descriptivo del conjunto propio

DESPUÉS: "En las seis fuentes con identidad anotada hay 972 golpes con altura marcada, 162 al cuerpo, un
16,7%. Con esa proporción la exactitud simple no sirve: decir que todo va a la cabeza da 0,83 sin saber
nada, y su exactitud balanceada es 0,50. En todo el material hay 28 combinaciones izquierda, derecha,
izquierda con menos de medio segundo entre golpes, una cada 30 golpes, así que ese bloque va a mostrar
números chicos."

### 7.3 Resultado de C8 (`CORREGIDO`)

DESPUÉS: "Elegí el umbral sobre dos fuentes y medí una sola vez sobre las otras cuatro, 806 golpes. La regla
llegó a 0,726 de exactitud balanceada y a 0,818 de recall al cuerpo. No pasa: le gana con claridad a la
línea de base, pero llama cuerpo a 204 de 674 golpes a la cabeza. Antes de medir fijé que un golpe sin rival
en cuadro cuenta como error; sin esos 53 golpes la cifra sería 0,781, y no la uso porque la miré después.
Por eso la altura se muestra rotulada como estimación, y el mapa de exposición cuenta golpes dirigidos."

### 7.4 C8, nivel 2, punta a punta

DESPUÉS: "Medí también lo que ve el entrenador: las cuatro fuentes procesadas por el producto, con la altura
calculada sobre los golpes que el sistema encontró y emparejó con uno anotado. Sobre 391 golpes la exactitud
balanceada baja a 0,608 y el recall al cuerpo a 0,587, aun con un detector que vio estas fuentes al entrenar.
Hay un segundo dato que pesa más que la regla: el detector encuentra la mitad de los golpes a la cabeza y un
tercio de los golpes al cuerpo. Un mapa de exposición hecho con este detector le restaría peso al cuerpo si
no se corrige ese sesgo."

### 7.5 C1, identidad con siembra humana (`NUEVO`)

DESPUÉS: "C1 pedía un 95% de tracks bien asignados con siembra humana. Sobre las seis fuentes de gimnasio dio
85 de 105, un 0,810 (Wilson 0,724 a 0,873). No pasa, y como el techo del intervalo queda debajo del umbral no
es un problema de muestra chica. De los 20 errores, 18 vienen de la geometría: tracks que mezclan a los dos
peleadores y generan una coexistencia que no existe. Por eso la identidad entró con el plan B, la corrección
manual por track en la web."

### 7.6 C2, C3 y C7 (`NUEVO`)

DESPUÉS: "C3 tampoco pasó. Contra 100 golpes marcados a mano, el indicador de vuelta a la guardia acordó en 51
de 88 (0,580, kappa −0,01) y el de mano opuesta caída en 31 de 86 (0,360, kappa 0,09), contra 0,80 pedido.
Ninguno se separa del azar, así que los indicadores de guardia salieron del MVP y quedan como línea futura."

"C2, la estabilidad del recall entre rounds, y C7, la línea de base del clasificador sobre las sesiones,
quedaron sin medir."

### 7.7 C4, C5 y C6

DESPUÉS: "Medí el tiempo hasta la Fight-Card sobre nueve sesiones de la estación. La mediana dio 1,949 veces
la duración del video, debajo del máximo de 2,0, pero ese número esconde lo importante. La muestra tiene
ensayos, bolsa y material de transmisión, y solo dos sesiones nuevas de gimnasio. Esas dos, filmadas en 4K,
tardaron 8,88 y 4,24 veces la duración del video. El criterio se cumple por la letra y no se cumple para
video 4K de celular."

"A 4K cada cuadro cuesta unas 2,5 veces más que a 1080p, y a 60 cuadros por segundo hay el doble de cuadros.
Las nueve sesiones llegaron a la Fight-Card sin errores de procesamiento."

"Probé bajar la resolución antes de procesar. El costo por cuadro vuelve al de 1080p, lo que muestra que no
estaba en el modelo. Aun así, ninguna condición cumplió la regla: con el transcodificado en el servidor,
sparr01 queda en 2,21 a 1080p y sparr en 3,02 a 1080p y 30 cuadros por segundo. Si el video llegara ya a
1080p, el de 30 cuadros por segundo daría 1,75. El de 60 no entra si se queda a 60."

"La prueba con usuarios (C6) no se realizó. BoxTwin quedó fuera del alcance y para BoxTwinZ no hace falta.
Ningún entrenador revisó la Fight-Card, así que no tengo evidencia de que le sirva a uno."

### 7.8 Etiquetado asistido (E0)

DESPUÉS: "Antes de usar a Claude para preetiquetar material nuevo, medí cuánto le acierta a mi anotación
mirando tiras de cuadros a ciegas. Sobre 150 golpes de las siete fuentes, acertó la familia en el 40% (Wilson
32,5% a 48%) y la altura con una exactitud balanceada de 0,644. Los umbrales, fijados antes de correr, eran
0,85 y 0,80. En familia no se separa del azar: su exactitud balanceada (0,315) queda por debajo de un tercio.
Confunde el hook con el recto y el uppercut con el hook, la misma trampa monocular que le cuesta al modelo.
No hay preetiquetado."

### 7.9 Espacio de clases y rival (E1 a E3)

DESPUÉS: "Probé tres configuraciones del clasificador, dejando afuera una fuente por vez. La primera replica
la de producción, con jab, cross, hook y uppercut por mano adelantada o atrasada. La segunda cambia el espacio
de clases a tipo por altura, sin lado. La tercera le suma al rival en la entrada. Corrí una sola semilla por
configuración: las 63 corridas previstas llevaban unas 75 horas de GPU."

"Cambiar el espacio de clases no mejoró la familia: 64,1% contra 65,6%. La altura tampoco salió: sobre las
cuatro fuentes donde se había medido la regla geométrica, la exactitud balanceada fue 0,541 y el recall de
cuerpo 0,133, contra 0,75 y 0,60 del criterio. Sumar al rival mejoró la altura según la regla fijada antes
(balanceada de 0,557 a 0,605, mejor en cinco de siete fuentes), pero tampoco llega al criterio."

"Con ocho clases (jab, cross, hook y uppercut, a la cabeza o al cuerpo), y con jab y cross separados por la
anotación, la exactitud queda en 52,6% y 52,2%. La sub-hipótesis 1 pedía más de 80%, así que la parte de
golpes no alcanza el umbral. El número es además optimista, porque supone que el lado del golpe llega siempre
bien."

### 7.10 Hallazgo sobre el código (H-F) (`CORREGIDO`)

DESPUÉS: "Revisé si el espejado de datos en el entrenamiento podía estar enseñándole al modelo etiquetas
contradictorias. No: el espejo invierte la imagen y además intercambia izquierda y derecha, así que un jab
espejado sigue siendo un jab. Lo que sí encontré es que en ese momento jab y cross salían solo del
clasificador, sin cruzarse con el brazo que da el detector ni con la guardia del peleador. Sobre fuentes
nuevas el clasificador acertaba mano adelantada o atrasada en el 65,6% de los golpes, menos que decir siempre
adelantada."

### 7.11 Clasificador geométrico (E4) y 3D (E5)

DESPUÉS: "El detector de golpes generaliza a fuentes nuevas y el clasificador no. La diferencia más visible
está en la entrada: el detector ve veinte medidas del brazo centradas en los hombros, escaladas por el torso y
espejadas para leer siempre el mismo brazo; PoseConv3D ve mapas de calor en píxeles. Probé el mismo tipo de
modelo chico que usa el detector sobre esas veinte medidas, más seis que ubican la muñeca respecto del torso
del rival, con las mismas siete fuentes y los mismos golpes que la campaña anterior."

"La familia pasó de 65,6% a 72,7%, con cinco de siete fuentes mejor, justo el mínimo que había fijado. La
altura no: sobre las cuatro fuentes de C8, la exactitud balanceada fue 0,719 contra 0,729 de la regla
geométrica recalculada sobre los mismos golpes. Las seis medidas del rival son las que llevan la altura; sin
ellas, la familia casi no cambia y la altura cae a 0,613."

"Después elevé la pose a 3D con MotionBERT y sumé dieciséis medidas del brazo en un marco pegado al cuerpo,
para que el arco de un hook no dependa de dónde está la cámara. La familia quedó en 73,8%, una diferencia de
1,1 puntos que no alcanza el mínimo de 3. El hook, que era la razón de probarlo, no mejoró: 59,0% de recall
contra 62,0% sin 3D."

"Con ocho clases, y con jab y cross separados por la anotación, la exactitud queda en 62,3% sin 3D y 63,2% con
3D, contra 52,2% de la campaña anterior con el rival. Sigue lejos del 80% de la sub-hipótesis 1."

### 7.12 3D con 4D-Humans (E7) (`NUEVO`)

Va a continuación de 7.11.

DESPUÉS: "Cambié MotionBERT por 4D-Humans, con las mismas dieciséis medidas. La familia subió a 0,777, 0,039
sobre MotionBERT y 0,051 sobre el modelo sin 3D, y el recall de hook pasó a 0,683. Las dos mejoras pasan con lo
justo: ganan cinco de siete fuentes, el mínimo que había fijado, y un golpe en contra en una sola fuente las
dejaría en no demostradas. Con ocho clases llega a 0,673, todavía debajo del 80%. Este modelo quedó en la
instancia académica; el producto sigue con el modelo sin 3D."

### 7.13 Modelo de BoxMind con mis poses (`NUEVO`)

DESPUÉS: "Probé el modelo de atributos de BoxMind con mis poses. Antes de comparar, intenté acercarme a lo que
reporta el paper sobre BoxingWeb y no llegué: 0,478 de F1 de técnica con mis poses y 0,617 con su código sin tocar,
contra un piso de 0,642. Por eso lo que sigue es exploratorio y no dice cuánto vale BoxMind. Entrenado con mis
fuentes, quedó 0,054 debajo de mi modelo geométrico en familia. Sumar 5668 golpes de BoxingWeb no movió la familia
sobre fuentes no vistas (+0,006) y el recall de hook bajó de 0,417 a 0,263."

### 7.14 Jab y cross por el brazo (E6) (`CORREGIDO`)

DESPUÉS: "Hasta ese momento jab y cross salían del clasificador. Pero el detector ya dice con qué brazo se tiró
el golpe, y con la guardia del peleador eso alcanza: un ortodoxo que pega con la izquierda tira un jab. Sobre
los 259 rectos que el detector encontró en las cuatro fuentes de C8, esa regla acertó la mano en el 98,5% de
los casos, contra 72,6% del clasificador medido sobre una fuente que no vio. Ese 98,5% es un techo: como la
guardia la tomé de la anotación, la regla acierta justo cuando el detector acierta el brazo, y el detector se
entrenó con esas fuentes. Suponiendo a todos ortodoxos, la regla baja a 81,1%, que sigue arriba del
clasificador."

### 7.15 El tipo v2 en el producto (`CORREGIDO`)

DESPUÉS: "Llevé al producto el modelo geométrico detrás de una bandera, que después encendí, y medí, con una
medición escrita antes de correrla, cuánto acierta sobre los golpes que encuentra el detector. Sobre cuatro
sesiones completas, cada una clasificada con el modelo entrenado sin su fuente, la familia sale bien en el
78,0% de 390 golpes emparejados con los anotados. Con la ventana anotada en vez de la del detector, el mismo
modelo da 79,0%."

"Jab o cross sale del brazo y de la guardia. Con la guardia de cada boxeador, el tipo de cuatro clases da 77,2%,
casi lo mismo que la familia: la mano casi no agrega error. Sin guardia cargada, el sistema supone ortodoxa, y
en las dos sesiones con un zurdo el tipo de cuatro clases cae a 43,1% y 50,7%. Por eso la Fight-Card avisa
cuando supuso la guardia."

"Para verificar que el producto calcula lo mismo que se midió, comparé las medidas de entrada de los 1.047
golpes anotados: coinciden exactamente. Al reproducir el modelo apareció otra cosa: el entrenamiento en GPU no
es determinista, y con la misma semilla dos corridas difieren en sus probabilidades. El modelo del producto se
entrenó en CPU, donde sí lo es, con la misma receta. El 72,7% de la medición anterior es un número de cinco
semillas entrenadas en GPU, y una reejecución daría otro cercano."

### 7.16 Bolsa y sombra con v2

DESPUÉS: "En bolsa y sombra hay una sola persona en cuadro. El modelo de familia usa seis medidas que ubican la
muñeca respecto del torso del rival, y sin rival entrarían en cero durante todo el golpe, un caso que casi no
vio al entrenar. Para esas sesiones uso la misma receta sin esas seis medidas, que ya había medido dejando una
fuente afuera: acierta la familia en el 72,4%, contra 72,7% con el rival. Ninguno de los dos se midió sobre
golpes a la bolsa o al aire, así que la Fight-Card de una sesión individual muestra el número como medido en
sparring. La altura no se estima: la regla geométrica necesita al rival."

### 7.17 Clasificador a 15 fps

DESPUÉS: "Simulé pose a 15 cuadros por segundo y reentrené el clasificador. La exactitud de familia baja de
0,732 a 0,718, dentro de la caída de 0,03 que había fijado. El uppercut es la excepción: su recall baja de
0,541 a 0,412, con 85 golpes. No medí el detector en esas condiciones, así que esto no alcanza para decir que
el perfil B funciona de punta a punta."

### 7.18 Actualizar el estimador de pose (E8 y E8b) (`CORREGIDO`)

Reglas de redacción: no escribir "YOLO26 estima peor la pose". Los números del detector son los de R1.

DESPUÉS: "Probé si cambiar el estimador de pose por uno más nuevo mejora el sistema. Cambié una sola cosa:
YOLOv8l-pose por YOLO26l-pose, el mismo tamaño y los mismos pesos de COCO. El preproceso, el tracker, la
identidad, el detector y el clasificador quedaron iguales, y las reglas de lectura las escribí antes de medir.
En COCO el modelo nuevo es mejor (70,4 contra 67,6 de mAP de pose) y más chico, pero COCO no tiene clinch,
desenfoque de golpe ni cuerpos chicos."

"Sobre las siete fuentes anotadas, dejando una afuera cada vez, el detector de golpes, medido como corre el
producto, bajó de 0,600 a 0,479 de F1 por evento en el umbral de producción, y el clasificador de familia de
0,732 a 0,675. El nuevo no ganó en ninguno de los seis folds comparables del detector y ganó en uno de los siete
del clasificador. Siguiendo a cada peleador, cambió de track un poco menos (306 contra 326 veces) pero mezcló
más a los dos en un mismo track (29 contra 23)."

"La explicación más probable no está en dónde ubica las articulaciones sino en cuánto confía en ellas. El
detector descarta una muñeca con puntaje menor que 0,3, un umbral pensado para YOLOv8. Con YOLO26 queda debajo
una de cada cuatro muñecas, contra una de cada diez, y el detector se queda con un 25% menos de pose usable."

"Lo que sí mejora es el costo: 37,1 contra 40,6 ms por cuadro en la RTX 2080 Super y 350 contra 544 MiB de
memoria."

"El umbral de 0,3 lo elegí para YOLOv8, y YOLO26 puntúa más bajo en todas las articulaciones. Lo recalibré por
articulación, sin mirar ningún golpe: para cada una, el umbral de YOLO26 es el que acepta la misma fracción de
keypoints que YOLOv8 acepta con 0,3. Salieron entre 0,10 y 0,18. Con esos umbrales, y nada más cambiado, el
detector quedó en 0,575 de F1 contra 0,600 de YOLOv8, medido como corre el producto, y el clasificador de familia
en 0,713 contra 0,732. Por la regla que escribí antes de medir, los dos igualan, pero los dos quedan del lado
bajo y el detector no gana en ningún fold. Puedo decir que YOLO26 calibrado no es peor por más de lo que fijé; no
que sea igual de bueno."

"La lectura cambia respecto de la primera comparación. No es que YOLO26 estime peor la pose: el sistema estaba
calibrado a cómo puntúa YOLOv8. Recalibrar recupera 0,096 de F1 y 0,037 de familia, y el 78% de los cuadros que se
perdían. Lo que no recupera está concentrado en una fuente, Sparring."

### 7.19 Mejoras posteriores a E8: relleno (E10)

DESPUÉS: "Probé también rellenar por interpolación los huecos cortos de keypoints, de hasta 0,10 s, en vez de
descartar esos cuadros. Recupera un 7,6% de cuadros usables. El detector, medido como corre el producto, queda en
0,619 de F1 contra 0,600 y gana en cinco de seis fuentes; el clasificador de familia, en 0,743 contra 0,732.
Ninguna de las dos diferencias llega a la mejora que fijé antes de medir, aunque la del detector queda cerca. La
razón está en los huecos: solo el 27% de los cuadros de muñeca que se pierden cae en huecos tan cortos. La pose
que se pierde en el golpe no se recupera interpolando."

### 7.20 Mejoras posteriores a E8: guante (E12)

DESPUÉS: "Sumé el guante detectado como entrada del detector de golpes. Con la evaluación corregida para que use
la máscara de pose del producto, la diferencia es de +0,001 de F1: no demostrada. El primer número, 0,429, era un
defecto de la evaluación y no del guante. Además cuesta 12,6 ms por cuadro, y con eso C5 se iría de 2,0."

### 7.21 Mejoras posteriores a E8: corte de tracks mixtos (E11)

Regla de redacción: separar desarrollo de confirmación.

DESPUÉS: "El corte de tracks mixtos se diseñó sobre las seis fuentes del diagnóstico de C1. Sobre ese material
sube la precisión por cuadro de 0,850 a 0,923 y la cobertura de 0,788 a 0,872, sin empeorar ninguna fuente. El
número es exploratorio porque diseñé y medí sobre lo mismo. La confirmación pedía dos sparrings nuevos anotados a
mano, que no llegué a tener, así que la hipótesis queda sin decidir y el corte sigue apagado."

### 7.22 Mejoras posteriores a E8: tracker (E13)

DESPUÉS: "Probé si otro tracker produce menos tracks mixtos, re-trackeando las mismas detecciones. Antes verifiqué
que el camino offline reproduce el tracker original en los 180193 cuadros. Ninguno cumplió la regla, que pedía
bajar los mixtos un 30% sin subir los cambios de track por minuto más de un 20%. OC-SORT y ByteTrack empeoraron
las dos cosas. BoostTrack con reidentificación bajó los mixtos de 23 a 18 y subió los cambios un 49,8%."

"BoostTrack igual deja un dato que la regla no mira: corta a la mitad los cuadros dentro de tracks mixtos y la
precisión de identidad por cuadro pasa de 0,882 a 0,977, sobre las fuentes de desarrollo. Cambia mezcla por
fragmentación. Lo dejo como línea futura, sin número de aceptación."

### 7.23 Identidad estable

DESPUÉS: "Con la revisión por tramos y un usuario simulado, la identidad queda estable en desarrollo con una
precisión de 0,985, a un costo de unas 23 decisiones por minuto de video. Es material de desarrollo, y el usuario
simulado contesta la verdad, cosa que una persona no siempre hace. Falta medirlo con personas."

---

## Capítulo 8

### 8.1 Comparación con el estado del arte monocular (`CORREGIDO`)

Cita: arXiv 2601.11492, "BoxMind: Closed-loop AI strategy optimization for elite boxing validated in the 2024
Olympics". Verifiqué identificador, título y autores el 5-11. Las cifras de abajo vienen del spec: cotejalas con el
paper antes de citarlas.

DESPUÉS: "BoxMind clasifica técnica y altura en boxeo de élite con pose 3D estimada desde una cámara, sobre unos
10.900 eventos anotados por entrenadores y con partición por round. Con pose sola reporta un F1 de 0,692 en técnica
y 0,747 en altura. Mis números no son comparables uno a uno: mido exactitud y exactitud balanceada, no F1, con 1.047
golpes y una partición más dura, por fuente entera. Sirven como orden de magnitud: en altura, mi modelo y la regla
geométrica quedan en 0,719 y 0,729 de balanceada; en técnica, el 3D de MotionBERT no me dio ventaja y el de
4D-Humans sí."

### 8.2 Líneas futuras

DESPUÉS: "Cualquier estimador de pose nuevo tiene que entrar con su umbral calibrado por igualación de fracciones,
como hice con YOLO26; si no, la comparación mide el umbral y no el estimador. YOLO26 iguala a YOLOv8 con un 8% menos
de tiempo por cuadro y un 36% menos de memoria, lo que importa para el modo LIVE. Evaluarlo para producción, con C5
medido otra vez, es el paso siguiente."

DESPUÉS, en el párrafo de la pérdida de pose en el golpe: "Interpolar huecos más largos tapa el golpe entero, así que
la vía que queda es otra fuente de la posición de la mano, como el guante detectado, o más cuadros por segundo y otra
cámara."

DESPUÉS: "Quedan con pre-registro escrito y sin número: la confirmación de E11, E9, E14 y E15. Sin medir quedan la
notebook, el despliegue en nube y el detector a 15 cuadros por segundo. Con E13, el pre-registro decía que si ningún
tracker mejoraba, el problema era la oclusión del clinch y no la asociación. Los datos de BoostTrack no me dejan
afirmarlo así: una asociación con apariencia reduce la mezcla, a costa de fragmentar. Medirlo bien pide material nuevo
y resolver la licencia."

Si querés sumar la línea del 4K: "Para video 4K, lo que queda por probar es pedir 1080p a 30 cuadros por segundo en la
subida, midiendo identidad y golpes sobre ese video."

---

## Declaración de uso de IA

Reemplaza las líneas anteriores, si todavía no están.

DESPUÉS: "Claude Code corrió en modo agente la campaña del clasificador (exports, particiones, entrenamiento,
evaluación y auditoría), el piloto de etiquetado a ciegas, E4 a E6 por especificación, y la campaña posterior a E8 con
las mediciones del cierre. Los criterios, los umbrales y las decisiones de alcance los fijé yo antes de medir, salvo
donde el capítulo declara lo contrario."

---

## No va al `.tex`

- La línea de E8 que proponía recalibrar el umbral: ya se hizo, y la reemplaza 8.2.
- El riesgo nuevo sobre el sesgo de anclaje: no aplica, E0 no habilitó el preetiquetado.
- Cap 5: sin cambios. E9 no corrió y E13 no dio positivo.
