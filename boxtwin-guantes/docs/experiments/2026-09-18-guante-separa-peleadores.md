# El guante separa peleador de no peleador en las seis fuentes, y el árbitro de látex no lo engaña

**18-09-2026 · seis fuentes anotadas más el video amateur · YOLOv8n entrenado sobre recortes de persona del dataset público.**

## Por qué

La identidad es la última pieza manual del sistema. El filtro por altura saca al público
—en `04-sparring` lleva 375 tracks a 42— pero no saca a la gente a distancia de ring:
árbitro, entrenador, cronometrista, la otra pareja del gimnasio. Quedan entre 20 y 43 tracks
del tamaño de un peleador que no lo son.

La hipótesis a probar es que el guante los separa, porque el árbitro es el único adentro del
ring sin guantes de boxeo. El riesgo declarado de antemano era el árbitro con guantes de
látex, que en amateur es protocolo de sangre: si el detector los confunde, el discriminador
se cae.

## Método

Detector YOLOv8n, 3,0 M de parámetros, entrenado sobre 2139 recortes de persona con 4236
cajas derivados de la v3 del dataset público. La entrada no es el cuadro completo sino el
recorte de cada persona, porque el guante mide siempre cerca de 0,167 del alto de un cuerpo
y eso normaliza la escala.

Sobre cada fuente se recorta cada persona **del cache de pose** —no se vuelve a correr
pose— se le pasa el detector, y se mide la **fracción de recortes con al menos un guante
detectado, por track**. El rol de cada track sale de las asignaciones manuales.

Protocolo, y no fue uniforme: `01-sparring` se corrió sobre los primeros 900 cuadros con
paso 3; las otras cinco fuentes enteras con paso 5; el amateur entero con paso 3. Los
números son comparables entre las cinco que comparten protocolo, y `01-sparring` hay que
leerlo con esa salvedad.

## Resultado

| Fuente | Peleadores mediana / mín | Resto mediana / máx | Tracks que se solapan |
|---|---|---|---|
| Sparring | 0,608 / 0,364 | 0,327 / 0,340 | **0** |
| 01-sparring | 0,589 / 0,521 | 0,228 / 0,625 | 1 |
| 02-sparring | 0,960 / 0,796 | 0,328 / 0,950 | 5 |
| 03-sparring | 0,942 / 0,897 | 0,333 / 0,886 | **0** |
| 04-sparring | 0,834 / 0,753 | 0,218 / 0,716 | **0** |
| sparring-3 | 0,853 / 0,581 | 0,227 / 0,707 | 1 |

**Las medianas no se tocan en ninguna fuente**: peleadores entre 0,59 y 0,96, el resto entre
0,22 y 0,33. Tres de las seis separan sin un solo error.

## Los que se solapan no son errores del detector

Los siete tracks que superan al peleador más bajo tienen guantes de verdad. Son **otras
parejas del gimnasio**, más lejos de la cámara:

| Fuente | Track | Guante | Alto máx | ¿Lo saca el filtro de altura? |
|---|---|---|---|---|
| 01-sparring | 24 | 0,625 | 0,246 | **sí** |
| 02-sparring | 18 | 0,950 | 0,508 | **sí** |
| 02-sparring | 382 | 0,950 | 0,262 | **sí** |
| 02-sparring | 437 | 0,860 | 0,704 | **sí** |
| 02-sparring | 410 | 0,848 | 0,336 | **sí** |
| 02-sparring | 130 | 0,917 | 0,986 | no |
| sparring-3 | 140 | 0,707 | 0,710 | **sí** |

**Los dos filtros son complementarios.** La altura saca a los boxeadores lejanos, el guante
saca al árbitro y al entrenador, que están a distancia de ring. Combinándolos, sobre las seis
fuentes queda **un único track residual**: el 130 de `02-sparring`, con 290 recortes, alto
0,986 y guante 0,917. Con el tamaño de un peleador y guantes puestos, lo más probable es que
sea un fragmento de uno de los dos que quedó sin rol asignado, no un intruso. Sin verificar.

## El video amateur, que era la prueba dura

`anotacion-amateur` es el caso que nunca se había probado: **ancho de hombros mediano de
22 px contra 132 en `sparring-3`**, cámara lejana y estática, árbitro adentro del ring. El
guante ahí son 4 o 5 píxeles.

| Grupo | Tracks | Fracción |
|---|---|---|
| Con guante | 11 | 0,426 – 0,732 |
| **Árbitro (track 2)** | 1 | **0,197** |
| Track 18 | 1 | 0,205 |

**El riesgo declarado no se materializó.** El árbitro usa guantes de látex blancos
—verificado ampliando sus recortes— y el detector le dio **la fracción más baja del video**.
No confunde látex con guante de boxeo, y la explicación probable es la silueta: un guante de
boxeo es un bulto liso sin dedos, un guante de látex conserva la forma de una mano.

El corte entre 0,426 y 0,205 es un factor 2,1. Las fracciones son más bajas que en los
sparrings, 0,43–0,73 contra 0,75–0,96, que es lo esperable a esa escala, pero **el orden se
conserva**. Esa es la invariancia que nunca se había medido.

Limitación: este proyecto **no tiene anotación de identidad**, así que no hay tabla de
separación formal. La lectura del ranking y la identificación del árbitro salen de mirar el
video, no de ground truth.

## Lo que no hace

El guante separa **peleador de no peleador**. No separa A de B: en los 390 relevos de track
medidos el 65% tiene al otro peleador como candidato, y los dos tienen guantes. Eso sigue
necesitando la formulación global.

El detector además tiene falsos positivos sistemáticos sobre **zapatillas rojas, cabezales y
a veces el short**, visibles en los overlays. Era predecible: se entrenó con fotografía de
catálogo y ningún negativo, porque el dataset público tiene una sola clase y ningún ejemplo
de "esto no es un guante". Que la separación se sostenga igual dice que el margen es amplio.

## Lo que esto es

Un piso, no un techo. El detector se entrenó sobre un dataset que es casi todo fotografía de
producto, cuya parte de dominio son 355 cuadros únicos de una sola pelea
(ver [el análisis del dataset](2026-09-18-dataset-guantes.md)). Que con eso alcance para
separar en seis fuentes sugiere que con unos cientos de cajas corregidas sobre material
propio la cosa mejora bastante.

## Reproducir

```bash
conda activate twinboxing_env
cd ~/Proyectos/TwinBoxing/boxtwin-guantes

for P in anotacion anotacion-spar-02 anotacion-spar-03 anotacion-spar-04 anotacion-sparring-3; do
    python3 tools/probar_video.py "$HOME/Proyectos/TwinBoxing/$P" \
        --modelo modelos/guantes.pt --paso 5 --min-detecciones 40 \
        --out "salidas/prueba_${P#anotacion-}.mp4"
done

python3 tools/probar_video.py ~/Proyectos/TwinBoxing/anotacion-amateur \
    --modelo modelos/guantes.pt --paso 3 --min-detecciones 40 \
    --out salidas/prueba_amateur.mp4
```
