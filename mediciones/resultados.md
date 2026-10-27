# Resultados — cierre de supuestos del Capítulo 5

Campaña del 15-09-2026 sobre `desarrollo-lucas` (RTX 2080 Super, 8 GB).
Configuración, versiones y desvíos en [protocolo.md](protocolo.md).
Datos crudos en [corridas.csv](corridas.csv).

24 corridas registradas, 21 válidas: 7 en frío y 14 en caliente, sobre 7 videos.
Las 3 restantes son un intento fallido que quedó anotado en el CSV (desvío D-8).

---

## 1. Resumen

| ID | Supuesto | Valor en el capítulo | Medido | Estado |
|---|---|---|---|---|
| S-19 | Cómputo por hora de video | 1,8 h/h | **1,447 h/h** (rango 1,134 a 2,624) | Medido |
| S-17 | Consumo por instancia | ~250 W | — | **Sigue siendo supuesto** (desvío D-3) |
| S-04 | Costo variable por hora de video | 1,25 USD | **0,876 USD** (rango 0,686 a 1,588) | Derivado de S-19 medido y un precio de lista con fecha |

---

## 2. S-19, relación de cómputo por hora de video

Mediana sobre las 14 corridas en caliente: **1,447 veces el tiempo real**.
Rango 1,134 a 2,624. Dispersión entre videos, tomando la mediana de cada uno:
1,136 a 2,620.

Corridas en frío, informadas aparte como pide M-07: mediana 1,445, rango 1,142 a
2,609. La diferencia contra las corridas en caliente es despreciable, y tiene una
explicación: cargar los modelos cuesta unos 2 segundos sobre corridas de entre 200
y 840 segundos. En esta cadena el arranque en frío no se paga.

### Por video

| Video | Duración (s) | fps | Cuadros | Ratio (mediana) | ms por cuadro | Segmentos |
|---|---|---|---|---|---|---|
| `Sparring.mp4` (640×360) | 186,6 | 29,64 | 5531 | 1,136 | 38,3 | 79 |
| `01-sparring.mp4` | 168,4 | 30,00 | 5053 | 1,389 | 46,3 | 80 |
| `02-sparring.mp4` | 170,6 | 30,00 | 5119 | 1,370 | 45,7 | 48 |
| `03-sparring.mp4` | 174,9 | 30,00 | 5248 | 1,548 | 51,6 | 101 |
| `04-sparring.mp4` | 182,6 | 30,00 | 5479 | 1,490 | 49,7 | 93 |
| `sparring-3-rounds.mp4` | 580,0 | 30,00 | 17399 | 1,447 | 48,2 | 277 |
| `pacquiao_margarito.mp4` (recorte) | 191,0 | 59,94 | 11451 | 2,620 | 43,7 | 99 |

### Lo que el ratio esconde, y que importa para el costo

El ratio va de 1,14 a 2,62, casi un factor 2,3. Parece dispersión y no lo es.
Normalizado por cuadro, el costo es casi constante: entre 43,7 y 51,6 ms por cuadro
sobre 1080p, con mediana 47,3.

El video de Pacquiao no es más caro por ser de competencia ni por tener 181 cortes
de plano. Es más caro por segundo de video porque corre a 59,94 fps, o sea el doble
de cuadros por segundo. Su costo por cuadro, 43,7 ms, es de hecho el más bajo de los
siete a 1080p.

De ahí sale la conclusión que sirve para el flujo de fondos: **el cómputo escala con
la cantidad de cuadros, no con la duración**. Un usuario que suba video a 60 fps
cuesta el doble que uno que lo suba a 30. Eso es una decisión de producto, no una
constante de la naturaleza, y está del lado del transcodificado en dispositivo que el
Capítulo 4 ya tiene como experimento pendiente.

La duración, en cambio, no mueve el ratio. `sparring-3-rounds` dura 3,4 veces más que
los sparrings cortos y su ratio, 1,447, cae justo en el medio de los de ellos.

---

## 3. Reparto por etapa

Mediana del porcentaje del tiempo total, corridas en caliente. Las etapas suman el
100,0% del tiempo de pared en las 14 corridas, así que no hay tiempo sin atribuir.

| Etapa | % del total | Rango |
|---|---|---|
| Seguimiento (BoT-SORT) | **47,7%** | 33,9% a 52,7% |
| Estimación de pose (YOLOv8l-pose) | 25,4% | 22,9% a 41,3% |
| Clasificación de familia (PoseConv3D) | 21,7% | 12,5% a 23,8% |
| Preparación (carga de modelo, hash, cortes) | 4,1% | 0,1% a 5,1% |
| Decodificación | 3,0% | 0,7% a 3,4% |
| Resolución de identidad | 0,2% | 0,2% a 0,4% |
| Detección de golpes | 0,0% | 0,0% a 0,0% |
| Estimación de resultado | **no medida** | no existe como código |

### El seguimiento cuesta más que la pose

Es el resultado que no esperaba. La etapa más cara de la cadena no es la red de pose
sobre la GPU sino BoT-SORT, que corre en CPU y se lleva casi la mitad del tiempo.

Lo medí por separado para no quedarme con la sospecha. Sobre 400 cuadros de
`01-sparring`, con el mismo modelo y la misma configuración, cambiando sólo el método
de compensación de movimiento global:

| `gmc_method` | Total 400 cuadros | Seguimiento | ms por cuadro de seguimiento |
|---|---|---|---|
| `sparseOptFlow` (el que usa el proyecto) | 13,22 s | 8,70 s (65,8%) | 21,7 |
| `none` | 4,74 s | 0,60 s (12,6%) | 1,5 |

La compensación de movimiento global explica 20,2 de los 21,7 ms, o sea el 93% del
costo del seguimiento. Apagarla multiplica por 2,8 el rendimiento de toda la cadena.

Esto no es una recomendación de apagarla. La compensación existe para cámara en mano,
y el único material de transmisión de la muestra es el de Pacquiao. Lo que sí puedo
afirmar es que sobre cámara fija de gimnasio se está pagando 20 ms por cuadro para
compensar un movimiento que no hay. Lo que falta medir antes de tocarla es si
apagarla degrada la identidad, y eso no lo medí: `tools/comparar_tracker.py` ya tiene
el circuito armado para hacerlo contra las asignaciones manuales.

Va al Capítulo 8 como la primera optimización a evaluar, con esa salvedad.

### Las dos etapas propias son gratis

La resolución de identidad y la detección de golpes juntas no llegan al 0,5% del
tiempo. El ensamble de cinco semillas corre sobre una serie de features ya calculada
y tarda centésimas de segundo sobre un video entero.

Conviene decirlo en el capítulo porque va contra la intuición: lo que este proyecto
construyó no es lo que cuesta. El costo está en los dos modelos de terceros y en el
tracker.

---

## 4. S-17, consumo eléctrico

**No medido.** El supuesto de 250 W sigue sin respaldo.

`nvidia-smi` no arranca en esta máquina: el módulo del kernel quedó en 595.84 y la
biblioteca NVML en 595.91, después de una actualización de driver sin reiniciar. CUDA
computa igual, y de hecho toda esta campaña corrió sobre esa GPU sin problemas, pero
NVML es la única vía de telemetría de potencia y está caída. Verifiqué que la máquina
no expone ninguna alternativa: no hay sensor de la placa por `hwmon` ni por sysfs, sólo
`nvme` y `coretemp`.

Se resuelve reiniciando. El muestreador está escrito y verificado en
`scripts/medicion_potencia.py`, se niega a escribir un CSV vacío si NVML no responde, y
queda listo para correr:

```bash
python scripts/medicion_potencia.py --verificar
```

Cuando dé `OK`, correr la campaña de nuevo con el muestreador en paralelo cierra S-17.

Cuando se mida, va con esta salvedad: es el consumo de la placa, no el del equipo, y
mucho menos el de una instancia de nube con su plataforma asociada.

---

## 5. S-04, costo variable por hora de video

Derivación, con S-19 medido:

```
1,447 h de cómputo por hora de video
  × 0,5260 USD/h   (AWS g4dn.xlarge, us-east-1, precio de lista al 15-09-2026)
  = 0,7613 USD
  + 15% de almacenamiento y transferencia
  = 0,8755 USD por hora de video
```

Rango, arrastrando el rango de S-19: 0,686 a 1,588 USD por hora de video. El extremo
alto es el video a 60 fps.

El 15% de almacenamiento y transferencia es un supuesto que no medí y que declaro como
tal. Sale de la relación entre cómputo y egreso en cargas parecidas, no de una factura
de este proyecto.

### Por qué esa instancia, y qué queda sin cerrar

Elegí `g4dn.xlarge` porque su GPU es una NVIDIA T4, que comparte arquitectura Turing y
generación con la RTX 2080 Super. Es lo más cercano que hay en catálogo.

Cercano no es equivalente, y en la dirección que no conviene. La T4 es una placa de
70 W con menos núcleos y menos reloj que una 2080 Super de 250 W. Si el trabajo corre
más lento ahí, y lo más probable es que así sea, el costo real es mayor que estos
0,88 USD. **La equivalencia de rendimiento entre las dos placas no está medida, y es
el único supuesto que esta campaña no toca.**

Para acotar por arriba, una instancia claramente más rápida que la 2080 Super:
`g5.xlarge`, con A10G, a 1,006 USD/h de lista a la misma fecha. Con el mismo 1,447 y
el mismo 15% da 1,675 USD por hora de video, y ese número es un techo holgado porque
la A10G haría el trabajo en menos horas de las medidas acá.

Lo honesto para el capítulo es decir que el costo variable está entre 0,88 y 1,68 USD
por hora de video según qué GPU se alquile, y que cerrarlo pide la prueba de concepto
en nube que el Capítulo 4 ya tiene en la lista.

---

## 6. Qué tan lejos estaba el capítulo

S-19 pasa de 1,8 a 1,447. El capítulo sobreestimaba el cómputo un 24%.

También queda resuelta la tensión que la especificación marcaba entre el 1,8 declarado
y el 2,3 medido antes sobre pose y seguimiento solos. Los dos números eran de cosas
distintas: aquella medición no incluía la clasificación, que se lleva el 21,7%, pero
tampoco corría con FP16. Los caches reales de este proyecto se construyeron con
`half=True` y el default del código es `half=False`; medir con el default cuesta 93,5 s
de pose contra 60,5 s, un 55% más. La configuración que corre en serio es la primera, y
es sobre esa que está medido todo lo de acá.

S-04 pasa de 1,25 a 0,88 USD, pero con el rango de arriba ese movimiento no es una
mejora que convenga festejar: el número viejo cae dentro del rango nuevo.

---

## 7. Salvedades que van sí o sí junto a los números

1. Todo esto está medido sobre una RTX 2080 Super de 8 GB. Trasladarlo a una instancia
   de nube supone equivalencia de rendimiento, y esa equivalencia no está medida.
2. S-17 no se midió. Los 250 W siguen siendo un supuesto sin respaldo.
3. S-04 sale de un precio de lista con fecha, y los precios de lista cambian.
4. El conjunto de siete videos no es una muestra aleatoria del material que procesaría
   el producto. Son los videos que este proyecto anotó, seis de siete son sparring de
   gimnasio con cámara fija, y el único de competencia entra recortado.
5. La cadena medida tiene cuatro de las seis etapas que el producto necesita. La
   resolución de identidad se mide sólo como el costo de aplicar asignaciones que hizo
   una persona, y la estimación de resultado no existe como código. Un producto que
   resuelva las dos solo va a costar más que estos 1,447.
