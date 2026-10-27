# E-05 — Bloques para pegar en el Capítulo 5

Registro del Capítulo 2. Sin negrita en el cuerpo, párrafos cortos, primera persona.

Aviso sobre el destino de cada bloque: el capítulo no está en este repositorio, así que
no pude verificar contra la redacción actual de 5.5 y 5.14 ni armar el ANTES / DESPUÉS
que pide la forma de trabajo. Los bloques están escritos para reemplazar la redacción de
supuesto por la de medición, y cada uno dice qué supuesto cubre. Hay que confirmar que
caigan en el apartado correcto antes de pegarlos.

---

## Bloque A — para 5.5, reemplaza el supuesto S-19

El costo de procesamiento dejó de ser una estimación. Lo medí sobre la estación de
desarrollo, con la cadena completa que hoy existe y sobre siete videos anotados, tres
corridas por video. El protocolo, las versiones y el registro de desvíos están en
mediciones/protocolo.md, y las corridas crudas en mediciones/corridas.csv.

Procesar una hora de video cuesta 1,447 horas de cómputo acelerado. Esa es la mediana
sobre catorce corridas en caliente. El rango va de 1,134 a 2,624, y la dispersión entre
videos, tomando la mediana de cada uno, va de 1,136 a 2,620. El valor que este capítulo
declaraba antes, 1,8 horas, sobreestimaba el cómputo un 24 por ciento.

El rango parece ancho y conviene explicar de dónde sale, porque no es ruido. Normalizado
por cuadro el costo es casi constante: entre 43,7 y 51,6 milisegundos por cuadro sobre
1080p. Lo que cambia entre videos no es el costo del cuadro sino cuántos cuadros trae
cada segundo. El video de competencia de la muestra corre a 59,94 cuadros por segundo y
por eso cuesta 2,62 veces el tiempo real, aunque su costo por cuadro sea el más bajo de
los siete.

De ahí sale algo que afecta al flujo de fondos más que la relación misma. El cómputo
escala con la cantidad de cuadros y no con la duración. Un usuario que suba video a 60
cuadros por segundo cuesta el doble que uno que lo suba a 30, y eso se decide en el
transcodificado antes de la subida, no en el servidor. La duración, en cambio, no mueve
la relación: el video de diez minutos de la muestra da 1,447, que cae justo entre los de
tres minutos.

También quedó resuelta una tensión que arrastraba este capítulo. La única medición
propia anterior daba 2,3 veces el tiempo real y cubría sólo pose y seguimiento, contra
el 1,8 declarado para la cadena entera. Los dos números medían cosas distintas. Aquella
corrida no incluía la clasificación, que se lleva el 21,7 por ciento del tiempo, pero
tampoco usaba precisión reducida. Los caches de pose de este proyecto se construyeron
con FP16 y el valor por defecto del código es FP32, y esa diferencia sola explica 33
segundos sobre los 60 que tarda la pose en un video de tres minutos. Todo lo que informo
acá está medido con la configuración que corre en serio.

## Bloque B — para 5.5, dónde se va el cómputo

Medí además el tiempo por etapa, y las etapas suman el cien por ciento del tiempo de
pared en las catorce corridas, así que no hay cómputo sin atribuir.

El seguimiento se lleva el 47,7 por ciento, la estimación de pose el 25,4 y la
clasificación de familia el 21,7. Lo que queda se reparte entre la carga de modelos, la
decodificación del video, la resolución de identidad y la detección de golpes.

El resultado que no esperaba es que la etapa más cara no sea la red de pose sobre la
GPU sino el seguimiento, que corre en el procesador. Lo verifiqué por separado sobre
cuatrocientos cuadros, cambiando sólo el método de compensación de movimiento global: con
el que usa el proyecto el seguimiento cuesta 21,7 milisegundos por cuadro, y sin él 1,5.
La compensación explica el 93 por ciento del costo de la etapa, y apagarla multiplica por
2,8 el rendimiento de toda la cadena.

No propongo apagarla todavía. La compensación está para cámara en mano y el único
material de transmisión de la muestra es uno. Lo que puedo afirmar es que sobre cámara
fija de gimnasio se pagan veinte milisegundos por cuadro para compensar un movimiento que
no existe. Falta medir si apagarla degrada la identidad, y hasta que eso esté medido la
optimización queda enunciada y no aplicada. Vuelvo sobre esto en el Capítulo 8.

Conviene decir también lo contrario, porque va contra la intuición. Las dos etapas que
construí para este trabajo, la resolución de identidad y la detección de golpes, juntas
no llegan al medio por ciento del tiempo. El costo está en los dos modelos de terceros y
en el seguidor.

## Bloque C — para 5.5, el supuesto que sigue abierto

El consumo eléctrico sigue siendo un supuesto. No lo pude medir.

La telemetría de la placa no responde en la estación: el módulo del kernel quedó en una
versión y la biblioteca de gestión en otra, después de una actualización de driver sin
reiniciar. El cómputo funciona igual y toda esta campaña corrió sobre esa GPU, pero la
potencia sólo se lee por esa vía y verifiqué que la máquina no expone ninguna
alternativa. Se resuelve reiniciando.

Dejo escrito y verificado el muestreador que la va a medir, y dejo dicho que los 250 W
que este capítulo declara no tienen respaldo. Cuando se mida va a ser el consumo de la
placa, que no es el del equipo ni el de una instancia de nube con su plataforma
asociada.

## Bloque D — para 5.14, reemplaza el supuesto S-04

El costo variable por hora de video queda derivado de la medición y no estimado.

Con 1,447 horas de cómputo por hora de video, y tomando el precio de lista de una
instancia g4dn.xlarge en us-east-1 al 15 de septiembre de 2026, que es de 0,5260 dólares
por hora, el cómputo sale 0,7613 dólares. Sumando un quince por ciento por almacenamiento
y transferencia, el costo variable queda en 0,88 dólares por hora de video. El capítulo
declaraba 1,25.

Ese quince por ciento es un supuesto que no medí y lo declaro como tal.

Elegí esa instancia porque su GPU comparte arquitectura y generación con la placa sobre
la que medí. Es lo más cercano que hay en catálogo, y cercano no es equivalente. La placa
de la instancia es de setenta vatios y la de la estación de doscientos cincuenta, así que
lo más probable es que el trabajo tarde más ahí y el costo real sea mayor. Esa
equivalencia de rendimiento es el único supuesto que la campaña no toca, y se cierra con
la prueba de concepto en nube.

Para acotar por arriba tomé una instancia claramente más rápida que la estación, la
g5.xlarge, a 1,006 dólares por hora de lista a la misma fecha. Con la misma relación da
1,68 dólares por hora de video, y es un techo holgado porque esa placa haría el trabajo
en menos horas que las medidas.

Lo que corresponde escribir, entonces, es que el costo variable está entre 0,88 y 1,68
dólares por hora de video según qué instancia se alquile. Los precios son de lista, con
fecha, y cambian.

## Bloque E — salvedad de alcance, va donde se presenten los tres números

Corresponde decir qué parte del sistema está medida.

La cadena que cronometré tiene cuatro de las seis etapas que el producto necesita:
estimación de pose, seguimiento, detección de golpes y clasificación de familia. La
resolución de identidad está medida sólo como el costo de aplicar asignaciones que hizo
una persona en el anotador, porque hoy no hay nada que las infiera. La estimación de
resultado no está medida porque no existe como código: es un campo que se completa a
mano.

Un producto que resuelva la identidad por su cuenta y estime el resultado del golpe va a
costar más que estas 1,447 horas por hora de video. Cuánto más, no lo sé, y no lo voy a
saber hasta que esas dos etapas existan.

El conjunto de siete videos tampoco es una muestra aleatoria del material que procesaría
el producto en operación. Son los videos que este trabajo anotó, seis de los siete son
sparring de gimnasio con cámara fija y el único de competencia entra recortado a sus
primeros tres minutos.
