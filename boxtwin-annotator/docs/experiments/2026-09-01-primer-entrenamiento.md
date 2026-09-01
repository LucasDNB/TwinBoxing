# Primer entrenamiento con dataset propio

**01-09-2026 · 640 muestras, 3 fuentes, PoseConv3D, 6 clases lead-rear.**

## Montaje

676 eventos anotados en tres fuentes; 640 caen en el espacio de 6 clases lead-rear. Se
descartó el espacio de 12 porque su clase más chica tiene 6 ejemplos contra 23, y con
validación por fuente eso deja dos casos en test.

Validación **dejando una fuente afuera**, tres folds, más un control **en distribución**
sobre `sparring-3` con partición estratificada 285/96. Inicializado desde el checkpoint de
Bhargav. Hiperparámetros sin tocar respecto de aquella corrida, a propósito: cambiar datos e
hiperparámetros a la vez deja sin saber cuál movió el número.

## Resultado

| Escenario | Línea de base | Modelo |
|---|---|---|
| **En distribución** (mismos peleadores y cámara) | 37,5% | **62,5%** |
| sin sparring | 41,2% | 33,3% |
| sin pacquiao | 60,7% | 58,6% |
| sin sparring3 | 37,8% | 34,1% |

**El modelo aprende en distribución y no generaliza a una fuente nueva.** Los tres folds
cruzados quedan en o por debajo de predecir siempre la clase mayoritaria, con exactitud
balanceada entre 0,15 y 0,22 contra 0,167 de azar.

## Descartado: la longitud de clip

Los clips tienen 7 cuadros de mediana y el pipeline los muestrea a 48, o sea que el modelo ve
cada cuadro real repetido casi siete veces. Parecía un candidato fuerte. No lo es:

| `clip_len` | Repetición | Mejor top-1 |
|---|---|---|
| 12 | 1,7x | 55,2% |
| 24 | 3,4x | 58,3% |
| 48 | 6,9x | 59,4% |

Los tres caen dentro del intervalo de confianza de ±10 puntos que corresponde a 96 muestras
de validación, y si algo hay, la tendencia favorece al clip largo. Hipótesis descartada.

## Hook contra straight: tres mediciones que coinciden

Matriz de confusión del control, colapsando lateralidad:

| Familia | Acierto |
|---|---|
| straight | 96% (49/51) |
| hook | 63% (22/35) |
| uppercut | 10% (1/10) |

**De los 13 hooks que falla, 11 los llama straight.** Es el mismo eje donde:

- El anotador saca kappa 0,881 distinguiéndolos.
- Ningún descriptor 2D pasó de AUC 0,64, sobre cinco probados, dos de los cuales codifican
  directamente la definición del anotador.

Las tres mediciones son independientes y dicen lo mismo: **la distinción existe y un humano la
hace con acuerdo alto, pero no está accesible en la pose monocular.** No es ambigüedad del
dominio ni criterio flojo del anotador.

## Comparación con el baseline público

Bhargav: 84,51% con 275 muestras de entrenamiento, en distribución con 96% de sujetos
compartidos. Este trabajo: 62,5% con 285 muestras, en distribución con los mismos dos
peleadores.

**Mismo volumen de datos, veintidós puntos menos.** La diferencia no es cantidad sino
dificultad: su material es una persona sola haciendo shadowboxing en plano fijo; el de acá son
dos boxeadores ocluyéndose en sparring real. El número publicado no es comparable con el de
este trabajo aunque compartan las seis clases.

## El agujero de arquitectura, ahora visible

`tools/demo_vivo.py` corre el pipeline completo sobre video. Medido contra la anotación sobre
30 segundos de `sparring-3`:

| | |
|---|---|
| Golpes reales encontrados | 14 de 21 (67%) |
| Disparos que caen sobre un golpe real | 14 de 66 (21%) |
| De los emparejados, familia correcta | 9 de 14 (64%) |

Cuatro de cada cinco marcas no son un golpe. **El clasificador no tiene clase "no hay golpe"**:
fue entrenado sobre ventanas que siempre contienen uno, así que a cualquier ventana le devuelve
uno de los seis. Hace falta un detector, y el modelo de secuencia con carriles BIO no está
construido. El disparador por extensión de muñeca que usa el demo es una heurística de
reemplazo, y los 52 falsos positivos son suyos, no del clasificador.

El 64% de familia sobre los emparejados es consistente con el 62,5% del clasificador.

## Qué sigue, y qué no

**No sigue tunear.** La brecha entre en distribución y cruzado es de más de 20 puntos y los
hiperparámetros mueven cinco. Bajar el learning rate sumaría unos puntos en distribución sin
cambiar que los folds cruzados estén en la línea de base.

**Sigue sumar fuentes.** Con tres no se aprenden features invariantes a la fuente, y eso es lo
que mide el leave-one-out. No más rounds del mismo video: más cámaras, más gimnasios, más pares
de peleadores.

**Y construir el detector**, que es lo que separa un clasificador de ventanas de un sistema que
cuenta golpes en video continuo. El 21% de precisión del disparador heurístico es la línea de
base contra la que se mide.
