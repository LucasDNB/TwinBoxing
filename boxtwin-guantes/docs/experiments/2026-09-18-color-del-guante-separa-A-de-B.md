# El color del guante asigna A o B en 112 de 113 tracks, y el único fallo es un empate

**18-09-2026 · seis fuentes de sparring, 8150 guantes detectados · perfil por peleador, evaluado dejando cada track afuera.**

## Por qué

El detector de guantes ya separa peleador de no peleador, medido sobre las seis fuentes
([ver](2026-09-18-guante-separa-peleadores.md)). Lo que no resolvía es **A contra B**, que es
donde está el trabajo manual: de los 390 relevos de track medidos, el 65% tiene al otro
peleador como candidato y los dos tienen guantes.

El color era la única señal de apariencia disponible. El ReID genérico ya se había medido y
no aportaba a 640×360 (18-08-2026), pero ese encoder promedia el cuerpo entero, que a esa
resolución no deja señal; el guante es lo contrario, el objeto más saturado de la escena, y
el color es señal de baja frecuencia que sobrevive al downsampling mucho mejor que la forma.

El riesgo declarado de antemano era que los dos peleadores llevaran guantes del mismo color.
En sparring de gimnasio es perfectamente posible.

## Método

Sobre los cuadros donde la anotación dice quién es quién **y los dos están aislados** —IoU
entre sus cajas menor a 0,02— se recorta cada peleador, se le detecta el guante y se muestrea
el color del parche central. El aislamiento importa: en un clinch las cajas se superponen
casi por completo y el parche tendría cuerpo del rival adentro. Se descartó entre el 35% y el
52% de los cuadros por eso.

El tono se promedia **circularmente**. Es un ángulo y el rojo vive en los dos extremos de la
escala, así que una media aritmética de 5 y 175 daría verde.

La evaluación por track deja ese track **afuera** del perfil contra el que se lo compara. La
primera versión no lo hacía y el número salía inflado por construcción: el perfil incluía los
guantes que después clasificaba. Resultó que la diferencia era nula —112/113 en las dos
mediciones— pero eso se sabe después de corregirlo, no antes.

## Resultado

| Fuente | A | B | Por guante | **Por track** |
|---|---|---|---|---|
| Sparring | naranja | azul | 85,1% | **14/14** |
| 01-sparring | rojo (3,5) | naranja (9,1) | **70,9%** | 10/11 |
| 02-sparring | rojo | naranja | 94,3% | **11/11** |
| 03-sparring | rosa (164,4) | rojo (174,7) | 86,9% | **23/23** |
| 04-sparring | verde | violeta | 82,8% | **15/15** |
| sparring-3 | azul | rojo | 92,6% | **39/39** |

**87,7% por guante suelto sobre 8150 guantes, y 112 de 113 tracks: 99,1%.**

El único fallo es un **empate exacto**, 8 votos contra 8 sobre 16 guantes, en `01-sparring`,
que es justo la fuente donde los dos llevan el mismo naranja: tonos 3,5 y 9,1, a seis grados
uno del otro. El caso peor se dio y no colapsó — 70,9% por guante sigue 21 puntos arriba del
azar con guantes prácticamente iguales, y el voto por track lo recupera salvo en un fragmento
de 16 guantes.

## Una hipótesis que salió falsa

Parecía razonable que la distancia entre los dos perfiles sirviera como medida de confianza:
guantes parecidos, menos confianza. **No ordena.** La distancia más chica sí marca el peor
caso —0,538 da el 70,9%— pero de ahí en adelante no predice nada: 0,602 da 86,9% y 1,823 da
82,8%.

Sirve como alarma para el caso degenerado, no como score de confianza. Si el diseño va a
decidir cuándo confiar en el color, necesita otra señal, y no es esta.

## Lo que no cubre

**Los tracks cortos.** La evaluación exige al menos 10 guantes, y con eso entran 113 de 141
tracks: el 80,1%. Pero esos 113 cubren el **98,4% de los guantes detectados**, así que los 28
que quedan afuera son fragmentos breves que aportan el 1,6% de la evidencia. Es ahí donde el
color no alcanza y tiene que decidir la geometría.

**De dónde salen los perfiles.** Acá se arman con guantes ya etiquetados por la anotación
manual. En un video nuevo no hay tal cosa, y sembrar los dos perfiles sin intervención humana
es una pieza que no está resuelta ni medida.

**Las otras fuentes.** Solo sparring. `pacquiao_margarito` es metraje de transmisión con otra
iluminación y no se midió; `anotacion-amateur` no tiene anotación de identidad, así que ahí
no hay contra qué comparar.

## Qué habilita

Con esto el pipeline de identidad automática queda con tres piezas medidas y una abierta:

1. Filtro por altura: saca al público y a las parejas lejanas del gimnasio. Medido.
2. Filtro por guante: saca al árbitro y al entrenador, que están a distancia de ring. Medido,
   deja un residual en seis fuentes.
3. Color del guante por track: asigna A o B. Medido, 112 de 113.
4. Restricción global: dos cadenas con exclusión mutua, para los fragmentos demasiado cortos
   para votar y para sembrar los perfiles. **Abierto.**

El problema que le queda al punto 4 es mucho más chico que los 390 relevos originales.

## Reproducir

```bash
conda activate twinboxing_env
cd ~/Proyectos/TwinBoxing/boxtwin-guantes
for P in anotacion anotacion-spar-01 anotacion-spar-02 anotacion-spar-03 \
         anotacion-spar-04 anotacion-sparring-3; do
    python3 tools/color_guante.py "$HOME/Proyectos/TwinBoxing/$P" \
        --modelo modelos/guantes.pt --paso 8 --guardar "salidas/colores/${P#anotacion}.json"
done
```
