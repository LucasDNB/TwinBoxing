# El dataset público de guantes es casi todo fotografía de producto, y su parte útil son 13 segundos de una sola pelea

**18-09-2026 · `boxing-uuhxl/boxing-gloves-detection` v1 y v3 · caracterización con YOLOv8l-pose sobre muestras de 200-300.**

## Por qué

El detector de guantes tiene que decidir si un track es peleador o no, sobre video de ring:
cámara lejana, guante chico, movimiento. Antes de entrenar hacía falta saber si el dataset
público se parece a eso. El proyecto ya se comió dos veces el costo de creerle a un dataset
publicado sin medirlo, así que acá se midió primero.

## Qué hay adentro

La v1 son 2356 imágenes con 5694 cajas. La v3 son 3745 con 9465 cajas, y la diferencia no
es más de lo mismo: **853 de las nuevas son cuadros de video**, el resto más fotos web.

Caracterización sobre muestras de 200 de cada grupo de la v3:

| | Cuadros de video | Web / producto |
|---|---|---|
| Con persona detectada | **99,5%** | 57,0% |
| Cajas dentro de una persona | 70,8% | 52,7% |
| Guante / alto de imagen, mediana | 0,100 | 0,394 |
| Guantes de menos del 5% del alto | **10,3%** | 0,7% |
| **Guante / alto de persona, mediana** | **0,155** | 0,287 |

El último renglón es el que decide. Un guante de boxeo es cerca de **0,167 del alto de una
persona**, y eso no depende de la distancia a la cámara. Los cuadros de video dan 0,155, que
es la proporción real. Las fotos web dan 0,287, casi el doble: son primeros planos y
retratos donde el guante ocupa de más respecto del cuerpo que se ve. Entrenar recortes de
persona con esa proporción es entrenar para otra cosa.

En la v1, medido aparte, el 32,3% de las imágenes **no tiene ninguna persona** —guantes
Everlast sobre fondo blanco, guantes colgados de una pared de ladrillo— y solo el 12,3%
da un cuerpo entero con un guante adentro.

## Los 853 cuadros son un solo clip

Vienen todos de `trimed-box-match`, con números de cuadro entre 3 y 407 y **355 valores
distintos**: el resto son copias de la aumentación horneada. O sea que el material de
dominio del dataset son **355 cuadros únicos de unos 13 segundos de una sola pelea**.

Eso tiene una consecuencia que hay que tener presente al leer cualquier métrica de
validación. El reparto agrupa por clip de origen, porque cuadros vecinos de un video son
casi la misma imagen y repartirlos al azar infla la validación. Con un solo clip, el grupo
entero cae de un lado: **los 853 quedaron en train, y validación y test tienen cero
imágenes de dominio**. El mAP de validación de este dataset mide fotografía de producto y
no dice nada sobre rendimiento en ringside.

## Dos errores de lectura que costaron caro y no se veían

**Los polígonos.** La v3 mezcla formatos: los cuadros de video están anotados con polígonos
—`clase x1 y1 x2 y2 ...`— y el resto con cajas de cinco campos. Medido: 2404 líneas de
polígono contra **cero** de caja en los cuadros de video, y 5478 cajas más 1583 polígonos en
el resto. Un lector que exija cinco campos descarta el 100% del material de dominio sin
emitir un solo aviso. Reducir el polígono a su caja envolvente recuperó 3987 anotaciones,
de 5478 a 9465.

**El agrupado por clip.** El export renombra: `Trimed Box Match _mp4-0020.jpg` sale como
`Trimed-Box-Match-_mp4-0020_jpg.rf.<hash>.jpg`, con el número de cuadro en el medio y no
pegado a la extensión. La primera versión del agrupador buscaba dígitos al final, así que
dejaba cada cuadro en un grupo propio y repartía los 853 entre los tres splits: justo la
filtración que la función existe para evitar. Con el hash sacado antes, los 3745 grupos
bajaron a 1205 y el clip quedó entero de un lado.

Los dos fallan en silencio y ninguno se ve en una métrica. Es el modo de falla más caro que
tiene este tipo de pipeline.

## Una decisión que se revirtió

La primera versión de la bajada le devolvía la relación de aspecto 16:9 a todas las
imágenes, porque el estirado a 640×640 deforma el guante en elipse de 1,78 a 1 y todas las
muestreadas del catálogo eran 1920×1080. Está mal: el dataset mezcla cuadros de video 16:9
con fotos de producto cuadradas y 4:3, y un supuesto único deforma las que no lo cumplían.
Sin el tamaño original no se deshace nada y se deja como vino, declarándolo en el
manifiesto. El tamaño se puede recuperar cruzando con el catálogo, que necesita clave.

## Qué sigue

El dataset no alcanza como conjunto de entrenamiento, pero sirve de arranque. La vía es
entrenar un primer detector con la parte usable, correrlo sobre las siete fuentes anotadas
para que proponga cajas de guante, corregir unos cientos y reentrenar sobre material propio.
Las propuestas no arrancan de cero: hay 303 tracks de peleador con identidad confirmada y
`derive_gloves` ya estima la posición del guante sobre el vector codo-muñeca.

## Reproducir

```bash
conda activate twinboxing_env
boxtwin-guantes fetch --version 3 --zip dsglovesv3.zip --out data/roboflow-v3
```
