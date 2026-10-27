# Diseño E4 a E6

Todo el código nuevo va en `scripts/clasificador_geo/`, con pruebas en
`scripts/clasificador_geo/tests/` que corren sin GPU, sin videos y sin las anotaciones reales
(datos sintéticos chicos). Docstring de cabecera con POR QUÉ EXISTE, QUÉ HACE y USO, sin
tildes, como el resto de `scripts/`. Se reusa `scripts/clasificador/comun.py` (fuentes,
métricas, Wilson) importándolo; no se copia.

Directorio de trabajo: `~/boxtwin/clasificador-geo-2026-10/` (variable
`BOXTWIN_CLASIFICADOR_GEO`). Nada pesado entra a git.

## 1. Datos por golpe (`datos.py`)

Fuente de poses: `series_por_peleador(doc, cache)` de `boxtwin.mvp.orquesta`, la misma que usa
`boxtwin-annotator/scripts/medir_objetivo.py::series_de_fuente`. Devuelve, con la identidad
anotada, `kp (2, T, 17, 2)` en píxeles, `sc (2, T, 17)` y `valid (2, T)`. Copiar la
resolución de rutas de `series_de_fuente` (la caché puede estar en `<proyecto>/cache/`).

Por cada golpe con `completeness == "full"`:

- `p` = 0 si `fighter_A`, 1 si `fighter_B`; `r = 1 - p`.
- Ventana `[start - 8, end + 8]`, recortada a `[0, T-1]`.
- Atributos guardados: `frame_dir = f"{stem}_{event_id}"` (mismo nombre que el export de
  mmaction, para cruzar con E1 y E3), `fuente`, `event_id`, `fighter`, `side`, `tipo`
  (straight, hook, uppercut), `altura` (head, body), `rol` (lead o rear: `Event.arm_role` del
  documento cargado con `boxtwin.core.annotations.load`, que lo deriva de la guardia guardada
  en el evento y del lado; no reescribir la derivación), `fps`, `start_frame`, `end_frame`.

Salida: `<trabajo>/datos/golpes.npz` con `X (N, 32, F)`, `M (N, 32)` (máscara de pasos
usables), las etiquetas como enteros y una tabla `golpes.csv` con los atributos. `F` y los
nombres de features en `golpes.meta.json`, con el sha256 de cada anotación leída.

Chequeos que el script hace y que tienen prueba: 1.047 golpes, conteo por fuente igual a la
tabla del pre-registro, `frame_dir` únicos, mismo conjunto de `frame_dir` que los folds de E1
(`~/boxtwin/clasificador-2026-10/folds/E1/manifest.json` y sus pickles). Si no coincide, se
frena (decisiones delegadas D3).

## 2. Features 2D (`features2d.py`)

### 2.1 Brazo que golpea (20)

`boxtwin_detector.features.features_de(kp[p], sc[p], brazo=side)` sobre el video entero del
peleador (las derivadas necesitan vecinos), y después se corta la ventana. Importar del paquete
instalado en `twinboxing_env`; no copiar la función. Su máscara `usable` es la máscara por
cuadro.

### 2.2 Rival (6)

En cada cuadro, con `s_r` = punto medio de hombros del rival, `c_r` = punto medio de caderas del
rival, `torso_r = |s_r - c_r|`, `m` = muñeca del brazo que golpea del atacante, todo en píxeles
de imagen (sin espejar: la vertical no cambia con el espejo):

1. `h = (m_y - s_ry) / torso_r` (la altura relativa de la regla de C8, como serie).
2. `dx = |m_x - s_rx| / torso_r`.
3. `d = |m - s_r| / torso_r`.
4. y 5. inclinación del torso del rival: `(cos, sin)` del ángulo de `c_r → s_r` respecto de la
   vertical.
6. `rival_ok` en {0, 1}: hombros y caderas del rival con score ≥ 0,3 y `valid[r]`.

Si `rival_ok` es 0, las cinco primeras van en 0. No se rellena con vecinos.

### 2.3 Remuestreo a 32 pasos

Interpolación lineal por feature sobre el eje temporal de la ventana. La máscara se remuestrea
con vecino más cercano. Esto saca la diferencia de fps (Pacquiao es 59,94; el resto, cerca de
30) sin tratarla aparte.

## 3. Modelo y entrenamiento (`modelo.py`, `entrenar.py`)

`ClasificadorGolpe(n_features)`:

- `Conv1d(F, 64, 1)` → tres bloques residuales `Conv1d(64, 64, k=3, dilation=d, padding=d)`,
  `BatchNorm1d`, `ReLU`, `Dropout(0.2)`, con `d` en (1, 2, 4).
- Pooling enmascarado: promedio y máximo sobre los pasos con `M = 1`, concatenados (128).
- Dos cabezas lineales: tipo (3) y altura (2).

Pérdida: `CE_tipo + CE_altura`, cada una con pesos de clase = inversa de la frecuencia en el
fold de entrenamiento, normalizados a media 1.

Entrenamiento, fijo: Adam, lr 1e-3, weight decay 1e-4, batch 64, 60 épocas, sin scheduler, sin
early stopping. Aumentación temporal: en cada muestra de entrenamiento, recorte aleatorio de la
ventana con largo entre 80% y 100% y remuestreo a 32 (CropResize, el mejor de los simples en
arXiv 2410.23641). Estandarización (media y desvío por feature, sobre pasos con `M = 1`)
ajustada en el fold de entrenamiento.

Semillas: `torch`, `numpy` y `random` con la semilla; `torch.use_deterministic_algorithms`
no hace falta. Dispositivo: `cuda` si está libre (D5), si no `cpu`. Con 1.047 golpes de 32
pasos, una corrida en CPU tarda del orden de un minuto; las 7 × 5 de una configuración, menos
de una hora.

Salida por corrida: `<trabajo>/corridas/<exp>/<fuente>/s<semilla>/pred.npz` con `frame_dir`,
`p_tipo (n, 3)`, `p_altura (n, 2)`, más `config.json` (todos los hiperparámetros, la lista de
features, el commit) y `tiempo.json`. Reanudable: si existe `pred.npz`, se saltea.

## 4. Features 3D (`elevar3d.py`, `features3d.py`), solo E5

### 4.1 Elevación

Entorno `boxtwin_mmaction` (MMPose 1.3.2). Config
`configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py`
del paquete instalado (`mmpose/.mim/configs/...`), pesos
`https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth`,
descargados a `<trabajo>/pesos/` con su sha256 anotado. Uso: copiar el patrón de
`mmpose/.mim/demo/body3d_pose_lifter_demo.py` (`init_model`, `convert_keypoint_definition`
de `coco` a `h36m`, `extract_pose_sequence`, `inference_pose_lifter_model` con
`image_size=(ancho, alto)` del video).

Qué se eleva: solo el atacante, solo los cuadros de la ventana de cada golpe, cada uno con su
contexto de 243 cuadros alrededor tomado de `kp[p]` (relleno con el borde donde no hay pose).
Huecos de pose de hasta 10 cuadros se interpolan linealmente antes de elevar; los más largos
quedan con máscara 0. Para Pacquiao (59,94 fps) se toma un cuadro de cada dos para armar el
contexto, así los 243 cuadros cubren un tiempo parecido al de las otras fuentes.

Salida: `<trabajo>/datos/pose3d.npz` con `(N, 32, 17, 3)` ya remuestreado a la ventana del
golpe y máscara, alineado por `frame_dir` con `golpes.npz`.

Control de calidad que va al documento: coeficiente de variación del largo de antebrazo y de
brazo 3D a lo largo de cada ventana (mediana y p90 por fuente). No hay umbral: se informa.

### 4.2 Marco del cuerpo y 18 features

Por cuadro, con articulaciones H36M (0 pelvis, 1 cadera derecha, 4 cadera izquierda, 8 tórax,
11 hombro izquierdo, 12 codo izquierdo, 13 muñeca izquierda, 14 hombro derecho, 15 codo
derecho, 16 muñeca derecha):

1. Si el golpe es del brazo derecho, espejar: `x → -x` y permutar izquierda y derecha. Así,
   como en 2D, siempre se lee el brazo izquierdo.
2. Origen en el tórax. Escala: largo pelvis–tórax.
3. Rotación alrededor del eje vertical del cuerpo (pelvis → tórax) para que la línea de
   caderas quede sobre el eje x. Es lo que hace a las features independientes de dónde está la
   cámara, que es el problema del hook.

Features: muñeca xyz, codo xyz, hombro xyz (9); ángulo del codo; elevación del codo respecto del
tronco (abducción); ángulo de la muñeca en el plano horizontal alrededor del hombro (azimut,
como `cos` y `sin`); velocidad de la muñeca en el plano horizontal y en vertical; velocidad del
azimut (la curvatura que separa un arco de una recta) (18 en total). Nombres en el meta.

## 5. E6, rol por brazo (`rol_por_brazo.py`)

Insumos: las cuatro sesiones de C8 nivel 2 (`~/boxtwin/mediciones/c8-nivel2/`, una por fuente
de C8) con su `fightcard.json`, y las anotaciones.

1. Orientación A/B: `boxtwin.mvp.medicion.medir_sesion(fc, annot).invertido`, como
   `pares_nivel2`.
2. Emparejamiento por peleador, **sin separar por brazo**: todos los golpes del peleador de la
   Fight-Card contra todos los anotados completos del mismo peleador, con
   `boxtwin.core.agreement.emparejar` a IoU 0,3. (`pares_nivel2` empareja por carril, y así no
   puede ver un brazo cambiado.)
3. Acuerdo de brazo: `brazo` (izq, der) de la Fight-Card contra `side` anotado.
4. Sobre los emparejados cuyo `tipo` anotado es straight:
   - regla: rol = adelantado si (guardia ortodoxa y brazo izquierdo) o (zurda y brazo
     derecho); si no, atrasado. Guardia: la del evento anotado emparejado (`Event.guard`),
     con `boxtwin.core.types.arm_role(guardia, lado_del_detector)`.
   - variante: la misma regla con todos ortodoxos.
   - clasificador: `tipo` de la Fight-Card, jab = adelantado, cross = atrasado; un hook o
     uppercut del clasificador sobre un recto anotado cuenta como error de rol.
   - verdad: rol anotado.
5. Exactitud de cada una con Wilson, total y por fuente.

## 6. Evaluación (`evaluar.py`) y fusión

Junta los `pred.npz` por experimento. Para cada uno: ensamble de semillas por fold, folds
concatenados, y por semilla. Métricas con `comun.py`: familia (exactitud, balanceada, recall por
tipo con Wilson), altura (balanceada, recall de cuerpo con Wilson), ocho clases (tipo × altura,
con straight partido por el `rol` anotado), por fold y sobre las cuatro fuentes de C8.

`regla_c8.py` recalcula la regla de C8 sobre los mismos 775 golpes completos de las cuatro
fuentes con `boxtwin.mvp.objetivo.objetivo_de`, τ 0,20, sin dato = error.

Fusión F: para cada `frame_dir`, `p_tipo` y `p_altura` de E4 promediadas con las de E3,
reconstruidas desde `~/boxtwin/clasificador-2026-10/corridas/E3/<fuente>/s0/pred.pkl`
sumando las probabilidades de las clases de E3 por tipo y por altura (el orden del pickle es
el del fold, como verifica `scripts/clasificador/evaluar.py::esperado`). Si los pickles no
están, F queda como no realizado.

Salida: `<trabajo>/resultados/{E4,E4a,E5,F,E6,regla_c8}.json` y `resumen.md` con las
comparaciones del pre-registro §3, cada una con su veredicto (`mejora`, `no demostrada`,
`no concluyente`, `no realizado`).

## 7. Pruebas exigidas (sin GPU)

1. `datos`: con dos fuentes sintéticas, ventana y recorte en bordes, amagues afuera,
   `frame_dir` únicos, freno si el conjunto no coincide con el de los folds.
2. `features2d`: rival a la misma altura que los hombros da `h = 0`; rival sin pose da
   `rival_ok = 0` y ceros; remuestreo de una rampa lineal conserva la rampa.
3. `modelo`: forma de salida; la máscara importa (cambiar pasos con `M = 0` no cambia la
   salida).
4. `entrenar`: sobreajusta un lote sintético separable (exactitud 1 en entrenamiento); dos
   corridas con la misma semilla dan el mismo `pred.npz` en CPU; ninguna muestra de la fuente
   de prueba entra al entrenamiento ni a la estandarización.
5. `features3d`: un esqueleto rotado alrededor del eje vertical da las mismas features
   (invariancia); el espejo de un brazo derecho da las features del izquierdo.
6. `rol_por_brazo`: la regla en los cuatro casos de guardia y brazo; emparejamiento que
   detecta un brazo cambiado.
7. `evaluar`: predicciones perfectas dan 1; veredictos de la regla 0,03 y 5 de 7 en casos
   límite (0,029 no mejora; 0,031 con 4 de 7 no mejora; con 5 de 7 sí); desvío entre semillas
   mayor a 0,03 da `no concluyente`.

Comando: `~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/clasificador_geo/tests -q`.
Las pruebas de `elevar3d` que necesitan MMPose se marcan para saltearse si no se puede
importar.
