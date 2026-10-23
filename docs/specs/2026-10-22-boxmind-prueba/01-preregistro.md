# BM · Pre-registro de la prueba académica con el modelo de BoxMind

22-10-2026 · scripts `scripts/boxmind_prueba/` (pruebas en `scripts/boxmind_prueba/tests/`) ·
referencia: Wang et al. (2026), *BoxMind*, arXiv:2601.11492v2, material suplementario B ·
código externo: `github.com/gouba2333/BoxingWeb`, commit `e06add2` · se commitea antes de
correr cualquier parte de la prueba

## 1. Qué se pregunta

BM es una prueba académica, no un cambio de producto. Responde cuatro preguntas sobre el
clasificador de tipo y altura, con los mismos 1.047 golpes y los mismos siete folds de E4:

1. **V0.** El modelo de atributos de BoxMind, alimentado con las poses de BoxTwin, ¿se acerca
   en su propio dominio a lo que reporta el paper? Si no, nada de lo que sigue se puede
   atribuir "al modelo de BoxMind".
2. **P7.** Con los mismos datos de BoxTwin, ¿la arquitectura y la representación de BoxMind le
   ganan a E4?
3. **P8.** Con la misma arquitectura, ¿sumar BoxingWeb (diez veces más golpes con el mismo
   esquema de etiquetas, otro dominio) mejora la familia sobre fuentes no vistas? Es la
   pregunta que decide si un dataset propio más grande es la palanca.
4. **Sub-hipótesis 1.** ¿Alguna configuración llega a 0,80 en ocho clases?

## 2. Condiciones de uso del material de terceros

El repositorio BoxingWeb no tiene archivo de licencia (verificado el 22-10 en `e06add2`: el
README remite a un LICENSE que no existe). El conjunto se publicó junto con el paper para
investigación; el video es transmisión olímpica de terceros. Sin licencia explícita, se usa
con estas condiciones, que valen para todo lo de BM:

1. Solo investigación. Ningún modelo entrenado con BoxingWeb entra al producto; el tipo v2 en
   producción sigue siendo E4 entrenado con fuentes propias.
2. Nada entra a git: ni video, ni poses, ni conjuntos armados, ni pesos, ni predicciones. Todo
   vive en `~/boxtwin/boxmind-prueba-2026-10/` (`bm.fuera_del_repo` frena cualquier salida que
   caiga dentro del repositorio).
3. El código de BoxMind se usa desde un clon externo y no se copia. Lo que hubo que
   reimplementar (representación, pérdidas, bucle) está escrito acá y tiene prueba de
   equivalencia contra el original (`test_equivalencia_boxmind.py`).
4. Se cita el paper como fuente del conjunto y del modelo, y la tesis declara en una línea
   que se usó sin licencia explícita, para investigación y sin redistribución.
5. La evaluación es siempre sobre fuentes propias. Si hubiera que retirar BoxingWeb, E4 y sus
   números no dependen de nada de esto.

## 3. Qué es "el modelo de BoxMind" en esta prueba

El repositorio publica la arquitectura del clasificador de atributos (`model/mfusion.py`:
I3D, codificadores de pose TCN, ST-GCN y PDETR, cabeza MoE), un `train.py` atado a sus datos y
el cargador de BoxingWeb (50 rondas de transmisión, partidas en `data_train` y `data_test`).
No publica pesos entrenados, ni el detector de golpes, ni el seguidor UVE, ni BoxingStudio.

Por eso BM cubre lo mismo que E4: clasificar tipo y altura de un golpe ya delimitado por la
anotación. No mide detección ni identidad. Se usa la variante **PoseOnly+TCN+MoE**, la de
pose sola del paper (Tabla S3: técnica 0,692, objetivo 0,747 de F1). La variante con video
(I3D) queda afuera de esta prueba por costo en la 2080 y porque el paper la gana por 0,033 en
técnica; se puede pre-registrar aparte.

## 4. Diseño

### 4.1 Representación (`representacion.py`)

Por cuadro, 28 articulaciones × 5 canales: las 14 de BoxMind del atacante y las 14 del rival,
con 2D normalizado al recorte del atacante (`get_region` y `prepare_video_data`) y 3D de cada
uno en su propio marco del cuerpo (`convert_pose_3d`). Desvíos respecto del original, todos
iguales para BoxTwin y para BoxingWeb:

- **B1. Fuente de las articulaciones.** 2D de YOLOv8l-pose (COCO-17) y 3D de MotionBERT
  (H36M-17), en vez de SMPL de 4D-Humans. Cabeza: promedio de nariz, ojos y orejas con score
  ≥ 0,3; cuello: medio de los hombros (COCO) o base del cuello (H36M).
- **B2. Escala del 3D.** Se divide por la mediana del largo del torso en la ventana, porque
  MotionBERT no da metros y la escala dependería de la resolución.
- **B3. BoxingWeb re-estimado.** BoxingWeb pasa por el mismo preproceso de BoxTwin
  (`boxtwin.preprocess`, configuración del producto) y la misma elevación de E5. La pose de
  BoxingWeb (`pose_gt.pkl`, la que usa su `train.py`; `uve.pkl` si no está) sirve solo para
  saber quién es rojo y quién azul en cada cuadro: emparejado por hombros, codos y caderas,
  aceptado debajo de 0,5 anchos de hombro. Un golpe de BoxingWeb se usa si el atacante quedó
  emparejado en al menos el 50% de sus cuadros; los descartados se cuentan.
- **B4. Ventana.** La anotada, de inicio a fin, sin el margen de 8 cuadros de E4 y sin
  remuestrear (como BoxMind). Por encima de 45 fps, un cuadro de cada dos (la regla de E5).
  En BoxingWeb, el filtro de su cargador: golpes de 4 a 30 cuadros. En BoxTwin entran los
  1.047, cualquiera sea su largo.
- **B5. Huecos.** 2D y 3D con huecos de hasta 10 cuadros interpolados; un golpe sin recorte
  válido o sin 3D entra con esos canales en cero y se evalúa igual (como en E4).

### 4.2 Modelo y entrenamiento (`entrenar.py`)

- Arquitectura importada del clon sin modificar: `MFusion` con `PoseOnly+TCN+MoE`, 4
  expertos, dropout 0,5.
- Hiperparámetros del paper (B.4.1): SGD, lr 0,05, momentum 0,9, weight decay 1e-4, batch 64,
  12 épocas, lr dividido por 10 en las épocas 5 y 10. Focal loss con gamma 2 y los alphas de
  `train.py`; pérdida total técnica + objetivo + distancia + 2 × efecto. Espejo con
  probabilidad 0,5.
- **B6.** Bucle propio: el de `train.py` evalúa contra el test en cada época y solo lee
  BoxingWeb. Se toma la **última época**, sin validación ni selección de época, igual que E4.
- **B7.** BoxTwin no tiene distancia ni efecto anotados: esas dos pérdidas cuentan solo en
  los golpes de BoxingWeb.
- **B8.** La cabeza de objetivo tiene 3 salidas (cabeza, pecho, abdomen) entrenadas con 2
  etiquetas, como en `train.py` con `tgt_cls=2`; la probabilidad de cuerpo es pecho más
  abdomen.
- **B9.** El TCN no usa la mano: se le pasa un tensor en cero, como pasa en el original.
- Semillas 0 a 4. Número primario: ensamble de las cinco, folds concatenados (como E4).

### 4.3 Experimentos

| | Entrena con | Prueba sobre | Pregunta |
|---|---|---|---|
| BM-0 | `train.py` original, sus poses, `data_train` | `data_test` | reproducción, una corrida (semilla 42 fija en su código) |
| BM-0b | BoxingWeb re-estimado, `data_train` | `data_test` | V0 |
| BM-1 | BoxingWeb completo | los 1.047 golpes de BoxTwin | transferencia sin ver BoxTwin, sin umbral |
| BM-1b | BoxTwin, seis fuentes | la séptima, siete folds | P7 |
| BM-2 | BoxTwin seis fuentes + BoxingWeb completo | la séptima, siete folds | P8, P9, P10 |
| BM-2-25, BM-2-50 | como BM-2 con el 25% y el 50% de las rondas de BoxingWeb | ídem | curva, sin umbral |

Las rondas de la curva se sortean una sola vez, con semilla 0, por ronda completa.

## 5. Lectura

Todas las comparaciones de familia usan `veredicto_mejora` de `clasificador_geo/evaluar.py`,
la misma regla de E4: mejora si el total supera a la base por 0,03 o más y gana en al menos 5
de los 7 folds; si el desvío entre semillas de la familia de cualquiera de los dos pasa de
0,03, una mejora se informa como no concluyente.

1. **V0.** F1 macro de técnica de BM-0b, mediana de las cinco semillas, sobre `data_test`:
   válida si es ≥ 0,642 (0,692 del paper menos 0,05). Si no es válida, BM-1 a BM-2 se
   calculan igual y se informan como exploratorios, sin veredicto. BM-0 se informa al lado:
   si BM-0 tampoco llega, el problema es del código publicado y no de nuestras poses.
2. **P7.** BM-1b contra E4 (0,727).
3. **P8.** BM-2 contra BM-1b. Es la que decide la sección 6.
4. **P9.** BM-2 contra E4.
5. **P10.** Altura de BM-2 sobre los 775 golpes de las cuatro fuentes de C8 contra la regla
   de C8 recalculada (0,729): mejora con 0,03 o más y 3 de las 4 fuentes; C8b al lado
   (balanceada ≥ 0,75 y recall de cuerpo ≥ 0,60).
6. **Sub-hipótesis 1.** Ocho clases con la lectura a/b/c del 14-10 (`d359247`): jab y cross
   salen del rol anotado.
7. **D9 de E4.** Familia ≥ 0,90 u ocho clases ≥ 0,80 obliga a buscar fuga antes de escribir.
8. **Curva.** Familia y ocho clases contra cantidad de golpes de BoxingWeb (0, 25%, 50%,
   100%), sin umbral. Sirve para dimensionar la sección 6 si corresponde.

## 6. Gatillo de la propuesta de dataset propio

La propuesta de `02-dataset-propio.md` se presenta como **respaldada por esta prueba** si V0
es válida y además P8 da mejora o BM-2 llega a 0,80 en ocho clases. Si P8 es no concluyente,
se presenta como no concluyente. En cualquier otro caso se presenta como **no respaldada**: la
propuesta queda escrita, pero la tesis dice que esta prueba no mostró que más datos del mismo
esquema muevan el clasificador, y la línea futura principal pasa a ser la segunda cámara. La
función es `evaluar_bm.gatillo`, con prueba.

## 7. Lo que hay que declarar

- **No es la comparación del paper.** El 0,725 de BoxMind es F1 macro, con la variante de
  video, sobre 10 rondas al azar del mismo corpus, y el `train.py` publicado entrena y evalúa
  con `pose_gt.pkl`. Por la historia del repositorio (el cargador de BoxingStudio, borrado en
  `eb71982`, distingue la captura multicámara de la salida de 4D-Humans, y en BoxingWeb el
  comentario de `pose_gt` dice "orden de 4D-Humans"), `pose_gt` sería 4D-Humans con la
  identidad verificada y `uve.pkl` la salida del seguidor automático: su número supone
  identidad resuelta, como la anotación de BoxTwin. Se confirma al descargar. BM mide exactitud de familia
  dejando una fuente afuera.
- **P7 no separa arquitectura de representación.** BM-1b cambia las dos cosas respecto de E4
  (articulaciones de los dos peleadores con 3D contra 26 features geométricas; ventana
  anotada contra ventana con margen remuestreada). Se lee como "el paquete de BoxMind contra
  E4". P8 sí aísla el efecto de los datos.
- **Dominios distintos.** BoxingWeb es boxeo olímpico de transmisión, con cambios de plano y
  etiquetas de entrenadores profesionales; BoxTwin es sparring de gimnasio con celular,
  etiquetado por el autor. "Recto" en BoxingWeb incluye jab y cross; "cuerpo" es pecho más
  abdomen; la mano es física. Si P8 no mejora, no se puede separar si faltan datos o si faltan
  datos *de este dominio*; eso es lo que respondería el dataset propio.
- **Identidad de BoxingWeb.** El emparejado contra su pose es una decisión de este
  pre-registro; su error, si lo hay, entra como ruido de etiquetas en el entrenamiento. Se
  informa la cobertura por ronda.
- **Determinismo.** En GPU el entrenamiento no es determinista bit a bit; las cinco semillas y
  la regla de dispersión cubren eso. No se re-entrena en CPU para producción porque nada de BM
  va a producción.

## 8. Orden y fechas

La GPU sigue la prioridad del 20-10: el benchmark R-11 primero; BM corre de noche después.
BoxingWeb lo descarga Lucas desde Tsinghua Cloud (el acceso automático está bloqueado). Antes
de correr, `python -m pytest scripts/boxmind_prueba/tests` con el clon en
`~/externos/BoxingWeb` (la prueba de equivalencia tiene que pasar, no saltearse).

| Paso | Dónde | Costo estimado |
|---|---|---|
| Descarga, `git clone`, `git checkout e06add2`, pruebas | estación | — |
| `bm0.py` | estación, GPU | menos de 1 h |
| `boxingweb.py` (preproceso de ~50 rondas y emparejado) | GPU, de noche | 2 a 4 h |
| `elevar.py --dominio boxingweb` y `--dominio boxtwin` | `boxtwin_mmaction`, GPU | 1 a 2 h |
| `armar.py` × 2, `entrenar.py` × 6 experimentos, `evaluar_bm.py` | GPU o CPU | 1 a 3 h |

Fecha de corte: **31-10-2026**, la misma de la sub-hipótesis 1. Si BM no termina, se informa
como no realizado; el corte no se corre. Nada de esto se cambia después de ver un número; si
algo del entorno obliga a un cambio, se anota en `TAREAS.md` con el motivo y antes de correr
la parte afectada.
