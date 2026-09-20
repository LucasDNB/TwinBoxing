# boxtwin-guantes

Detector de guantes de boxeo, y el filtro de identidad que se construye encima. Herramienta
interna de investigación de BoxTwin, no un producto.

## Por qué existe

La identidad de los peleadores es la última pieza enteramente manual del sistema. Se lleva
el 73% del tiempo de anotación medido, y es lo que hoy impide correr el pipeline sobre un
video nuevo sin que se siente una persona a resolverlo.

Medido sobre las siete fuentes anotadas, el problema tiene esta forma. Cada peleador está
fragmentado en muchos tracks y no en dos: en `sparring-3-rounds` son 38 ids para A y 36
para B. El trabajo manual son **390 relevos** de track, y la regla que hoy los propone
—IoU contra la última caja— acierta el **45,1%**. Agregarle predicción por velocidad la
empeora a 37,7%. Lo único que aportó fue la exclusión mutua, que la lleva a 62,1%: uno de
cada tres relevos sigue saliendo mal, y un relevo mal resuelto es un tramo entero de
keypoints atribuido al peleador equivocado, que en el overlay no se ve.

Y el filtro por altura que saca al público no alcanza. En 04-sparring lleva 375 tracks a
42, pero los que quedan no son público: son personas a distancia de ring. Entre 20 y 43
tracks por video tienen el tamaño de un peleador sin serlo —árbitro, entrenador,
cronometrista, la otra pareja del gimnasio— y uno de `sparring-3` tiene 1935 detecciones.

El guante es lo que separa esos dos grupos, porque el árbitro es el único adentro del ring
sin guantes de boxeo. Y se paga solo por dos razones que no dependen de la identidad:
calibra el `k` de `core/gloves.py`, que hoy sale de geometría de mano y no de una medición,
y resuelve el caso que ese módulo declara abierto, cuando la muñeca se pierde por desenfoque
justo en el instante del golpe.

## Experimentos

| Fecha | Pregunta | Resultado |
|---|---|---|
| [18-09](docs/experiments/2026-09-18-dataset-guantes.md) | ¿El dataset público se parece a video de ring? | Casi nada: su parte útil son 13 s de una pelea |
| [18-09](docs/experiments/2026-09-18-guante-separa-peleadores.md) | ¿El guante separa peleador de no peleador? | Sí, en las seis fuentes, y el árbitro de látex no lo engaña |
| [18-09](docs/experiments/2026-09-18-color-del-guante-separa-A-de-B.md) | ¿El color del guante distingue A de B? | 112 de 113 tracks; el único fallo es un empate |
| [20-09](docs/experiments/2026-09-20-negativos-del-propio-material.md) | ¿Sirven negativos duros del propio material? | Sí, pero solo con la unidad correcta: falsos de 17 a 8 |

## Estado

| Etapa | Qué | Estado |
|---|---|---|
| 1 | Bajada y caracterización del dataset | hecho |
| 2 | Recortes de persona | hecho |
| 3 | Entrenamiento | hecho |
| 4 | Medición contra las fuentes anotadas | hecho |
| 5 | Color del guante para A contra B | hecho |
| 6 | Partición global y siembra de perfiles | pendiente |
| 7 | Sidecar y consumo en el anotador | pendiente |

## Instalación

```bash
conda run -n twinboxing_env pip install -e ".[dev]"
```

El extra `train` trae ultralytics y opencv, y solo hace falta a partir de la etapa 2.

## El dataset

Es `boxing-uuhxl/boxing-gloves-detection` de Roboflow Universe: detección de objetos, clase
única `Boxing-Glove`, 4050 instancias en 1625 imágenes anotadas, licencia de dominio público.
Las imágenes son cuadros extraídos de videos de peleas, a 1920×1080.

Se baja el export de la **versión 1**, que es la única sin ecualización de contraste ni
escala de grises, y se le corrige el preprocesado que sí trae.

**El estirado se deshace acá.** Las cinco versiones publicadas reescalan con *Stretch to*
640×640 sobre imágenes 16:9, lo que comprime el eje horizontal a 0,5625 del vertical y deja
un guante redondo como elipse de 1,78 a 1. Como las etiquetas YOLO son normalizadas y el
estirado es un escalado lineal por eje, sus coordenadas son invariantes: alcanza con
devolverle a la imagen su relación de aspecto y las cajas quedan correctas sin tocar un solo
número. El tamaño original de cada imagen sale del catálogo, cruzando por nombre.

Lo que no vuelve es el detalle horizontal que el estirado ya descartó, y eso queda declarado
en el manifiesto. Para el enfoque de recortes de persona importa poco, porque el recorte se
reescala a 320×320 igual, y encima acerca la fuente al video amateur que es el caso duro.

**Lo que queda adentro y no nos gusta:** la aumentación horneada de la versión, que incluye
flip vertical. Un guante de boxeo dado vuelta no existe y eso le gasta capacidad al modelo.
No se saca porque quitarlo exige forkear el proyecto y generar una versión propia, y la
Etapa 4 va a decir si costó algo.

Lo que **no** se usa del export es su reparto en train/valid/test.

### El reparto es por clip, no al azar

Las imágenes son cuadros de video y se llaman `Trimed Box Match _mp4-2092.jpg`. Los cuadros
2092 y 2093 de la misma pelea son casi la misma imagen. Repartirlos al azar pone copias
cuasi-idénticas de los dos lados y la validación mide de más, que es exactamente el error
que el proyecto ya se comió una vez con BoxingVI.

Entonces el reparto agrupa por clip de origen y es determinista por sha1 del nombre del
clip: la misma máquina o cualquier otra dan el mismo reparto, sin semilla que recordar. El
reparto que trae Roboflow no se usa, porque no hay forma de saber si respeta la procedencia.

Lo primero que hay que mirar de la salida de `fetch` es `clips_en_mas_de_un_split`. Si no
está vacío, hay filtración y cualquier métrica posterior mide de más.

## Uso

```bash
export ROBOFLOW_API_KEY=...          # nunca se escribe en el repo
boxtwin-guantes fetch --version 1 --out data/roboflow
```

`data/` y `modelos/` están fuera de git: se regeneran.
