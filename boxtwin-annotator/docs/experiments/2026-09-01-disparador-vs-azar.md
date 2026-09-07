# El disparador por extensión de muñeca no es mejor que el azar

**01-09-2026 · las tres fuentes, 676 eventos · resultado negativo, y una herramienta que no
se construyó.**

## Por qué se buscó

El anotador gasta el 73-78% del tiempo en identidad y navegación, no en clasificar golpes.
De esos dos, la navegación —recorrer el video buscando dónde pasa algo— es la única parte
que una herramienta puede atacar sin tocar el problema de identidad.

La idea era un **salto a candidatos**: una lista de cuadros donde el disparador de
`tools/demo_vivo.py` detecta movimiento de brazo, y dos teclas para saltar de uno al
siguiente. No escribe etiquetas, así que no ancla al anotador ni contamina el dataset; solo
mueve el cursor. Y no necesita nada de lo que falta construir: corre sobre
`cache/<base>.pose.npz` con numpy, sin GPU ni modelo.

La pregunta previa a construirlo era el punto de operación. Para el demo el disparador se
calibró buscando precisión, porque cada falso positivo ensucia la pantalla. Para navegar
conviene lo contrario: un candidato de más cuesta un segundo de mirar, un golpe perdido
cuesta el golpe. Se esperaba que bajando el umbral el recall subiera a 90 y pico con un
número tolerable de paradas.

## Cómo se reproduce

```bash
python tools/barrido_disparador.py \
  ../anotacion-sparring-3/videos/sparring-3-rounds.mp4 \
  ../anotacion/videos/Sparring.mp4 \
  ../anotacion-pacquiao/videos/pacquiao_margarito.mp4
```

Corre sobre el cache de pose y la anotación, sin GPU y sin modelo. Las tres mediciones de
abajo salen de ahí.

## Lo que se midió

Barrido del umbral sobre las tres fuentes, con la ventana y el refractario expresados en
segundos y no en cuadros, porque Pacquiao corre a 59,94 fps y las otras dos a 30. Dos
variantes de disparo: por **pico** (máximo local, lo que hace el demo) y por **región**
(tramo continuo sobre el umbral). Una parada "encuentra" un golpe si cae a menos de 0,5 s
de él, que es la tolerancia con la que navegar sirve: al reproducir desde ahí el golpe se ve.

El recall llega a donde se esperaba. El costo, no:

**sparring-3, 400 eventos sobre 9,6 minutos, por región**

| Umbral | Paradas | Recall | Precisión | Paradas/min | s entre paradas |
|---|---|---|---|---|---|
| 0,4 | 569 | 0,858 | 0,540 | 59,3 | 1,0 |
| 1,0 | 934 | **0,988** | 0,524 | 97,3 | **0,6** |
| 1,6 | 738 | 0,988 | 0,625 | 76,9 | 0,8 |

Una parada cada 0,6 segundos: cubrir el video pide unas 930 pulsaciones. Eso no es navegar,
es avanzar cuadro a cuadro con otro nombre. Y ningún punto de operación baja de ~59 paradas
por minuto sin perder golpes.

## La línea de base que faltaba

Con 42 golpes por minuto, buena parte del video está cerca de un golpe **por construcción**.
Ese número es la precisión de poner paradas al azar, y sin él la tabla anterior no se puede
leer:

| Fuente | Golpes/min | Dentro de un golpe | ±0,5 s de un golpe | Tramos muertos >2 s |
|---|---|---|---|---|
| sparring-3 | 41,7 | 15% | **46%** | 40% |
| Sparring | 42,5 | 20% | **47%** | 44% |
| Pacquiao | 52,8 | 17% | **65%** | 18% |

Y entonces:

| Fuente | Precisión del disparador | Dardos al azar | Ventaja |
|---|---|---|---|
| sparring-3 | 0,52 | 0,46 | +6 pts |
| Sparring | 0,49 | 0,47 | **+2 pts** |
| Pacquiao | 0,70 | 0,65 | +5 pts |

**El disparador es, a efectos prácticos, poner paradas al azar.** En `Sparring` es
indistinguible del azar.

## Por qué, mecánicamente

La extensión de muñeca como discriminador de "hay golpe en este cuadro", medida por cuadro
y brazo sobre el brazo que efectivamente pega:

| Fuente | AUC extensión | AUC velocidad | Mediana con golpe | Mediana sin golpe |
|---|---|---|---|---|
| sparring-3 | 0,670 | 0,687 | 1,58 | 1,10 |
| Sparring | 0,608 | 0,655 | 1,92 | 1,55 |
| Pacquiao | 0,634 | 0,649 | 1,55 | 1,04 |

Las distribuciones se pisan casi por completo, y la razón es directa: **la guardia vive en
1,0-1,5 anchos de hombro**. El umbral que atrapa todos los golpes atrapa también al peleador
parado con las manos arriba. No hay umbral que separe porque no hay separación en ese número.

El paralelo con lo ya medido es exacto. Hook contra straight dio AUC 0,638 sobre el mejor de
cinco descriptores. Detectar *cuándo* hay golpe, con un descriptor 2D puntual, da 0,61-0,67.
Es el mismo orden de dificultad, y la misma forma de fracaso: un escalar hecho a mano sobre
un cuadro no lleva la información.

## Un defecto de medición, y por qué se cuenta

El primer barrido dio recall plano en 0,42 y **no subía al bajar el umbral**, que es lo
contrario de lo que tiene que pasar si el umbral es lo que ata. Eso delató dos defectos
propios, no del disparador:

1. El criterio de cobertura exigía que la parada cayera **dentro** de un golpe de 7 cuadros
   de mediana. Para navegar alcanza con caer cerca.
2. El juntado de paradas próximas conservaba la primera del grupo y descartaba las demás,
   que eran justamente las que caían sobre el golpe.

Corregidos los dos, el recall se comporta como debe y la conclusión cambia de signo: el
problema no es que el disparador no encuentre los golpes, es que los encuentra parando en
todas partes. Se deja anotado porque el número equivocado era más favorable a construir la
herramienta que el correcto.

## Lo que esto NO prueba

Que el detector sea imposible. Lo que se midió es el techo de **un número, en un cuadro,
elegido a mano**, que es precisamente la hipótesis que un modelo de secuencia no comparte:
ve la forma de la señal a lo largo de decenas de cuadros y combina varias features. El AUC
0,67 acota al descriptor puntual, no a la tarea.

Sí obliga a releer el demo. Los 66 disparos contra 21 golpes reales no son "un detector
rudimentario que se equivoca", son ruido con forma de propuesta. El 21% de precisión que se
reportó allí no es una línea de base débil de detección: es lo que produce un proponedor casi
aleatorio, y no dice nada sobre el clasificador.

## Qué se hizo en consecuencia

**No se construyó el salto a candidatos.** La medición costó una hora; la herramienta, una
semana de GUI, sidecar en `cache/`, marcas en el timeline y teclas nuevas en el keymap.

Queda además acotado el techo de cualquier versión futura de la idea, incluso perfecta: solo
el 40% de `sparring-3` y el 44% de `Sparring` está en tramos muertos de más de dos segundos,
y en Pacquiao apenas el 18%. Ese es todo el barrido que se puede llegar a evitar, y hay que
descontarlo del 22-27% del tiempo que no es identidad.

Y queda la línea de base correcta contra la que medir el detector cuando exista, que no es
el 21% del demo:

- Tarea de navegación —parar a menos de 0,5 s de un golpe—: **46% de precisión al azar**.
- Tarea real de segmentos: se mide con `emparejar()` a IoU 0,3, el mismo criterio del acuerdo
  intra-anotador, contra el techo humano de recall 0,911 y precisión 0,903.

## Qué queda abierto

Si el detector entrenado tampoco supera al azar por margen claro, hay dos lecturas que estos
datos no separan: que la pose monocular no contenga la señal de inicio de golpe, o que tres
fuentes no alcancen para aprenderla. La segunda es más probable —el mismo límite apareció en
el leave-one-source-out del clasificador— pero distinguirlas necesita material nuevo, no otro
modelo sobre el mismo material.
