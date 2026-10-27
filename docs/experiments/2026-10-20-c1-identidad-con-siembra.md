# C1 · Identidad con siembra humana sobre las seis fuentes de gimnasio

20-10-2026 · pre-registro: `docs/specs/2026-10-20-c1-preregistro.md`, commiteado en `2e8fa17`
antes de correr · script `boxtwin-annotator/scripts/medir_c1.py` (código de `3a0261b`) ·
datos en `~/boxtwin/mediciones/c1/` (`resumen.json` y un `<fuente>.c1.json` por fuente)

## 1. Resultado

**C1 no pasa.** Primaria: 85 de 105 tracks evaluables, **0,810** (Wilson 95% 0,724 a 0,873),
contra un umbral de 0,95. El techo del intervalo queda 8 puntos debajo del umbral, así que no
es un caso de muestra chica. Las seis fuentes tuvieron siembra posible y en las seis la
persona simulada confirmó el par sugerido.

Al lado, sin umbral: cobertura de tracks 0,460 (85 de 185 peleadores de la anotación) y
cobertura pesada por cuadros 0,793 (65.628 de 82.809 cuadros de peleador).

Según el pre-registro, la identidad entra al MVP con el plan B: corrección manual por track en
la web (B3/W9), que ya está construida.

## 2. Por fuente

| Fuente | Sin siembra | Con siembra | Invertidos con siembra | Sin decidir | Mixtos |
|---|---|---|---|---|---|
| Sparring | 0/8 | 8/8 | 0 | 8 de 23 | 5 |
| 01-sparring | 9/9 | 9/9 | 0 | 4 de 14 | 0 |
| 02-sparring | 13/13 | 13/13 | 0 | 5 de 18 | 0 |
| 03-sparring | 4/20 | 16/20 | 4 | 19 de 39 | 4 |
| 04-sparring | 11/14 | 11/14 | 3 | 9 de 23 | 1 |
| sparring-3-rounds | 28/41 | 28/41 | 13 | 26 de 68 | 8 |
| **Total** | 65/105 | **85/105** | 20 | 71 de 185 | 18 |

Las dos columnas se puntúan igual, sin correspondencia cruzada: A tiene que ser A. Por eso
el 0,619 sin siembra de esta corrida no es el 82,4% del 18 al 22-09, que sí probaba el cruce.

## 3. Lectura

1. **La siembra arregla la polaridad, no las asignaciones.** En Sparring y 03-sparring la
   propuesta automática tenía A y B invertidos (0/8 y 4/20) y la siembra los dio vuelta. En
   04-sparring y sparring-3-rounds el puntaje con y sin siembra es idéntico: los 16 errores
   que quedan ahí son tracks puestos del lado equivocado, y sembrar no los toca.
2. **Los errores se concentran.** sparring-3-rounds aporta 13 de los 20; sin esa fuente el
   resto da 57/64. Tres fuentes salen perfectas. Con seis fuentes no da para decir si
   sparring-3-rounds es un caso raro o lo que pasa con videos largos (es la fuente más
   larga, 33.022 cuadros de peleador contra unos 10.000 de las demás).
3. **El color de guantes no separa en dos fuentes.** En 01-sparring y 03-sparring sale el
   aviso de guantes casi del mismo color (separación 0,384 y 0,348, piso 0,55). Es lo
   mismo que dejó el 22-09 sobre el amateur: la siembra sirve por el ancla y no por el color.
4. **La propuesta decide menos de la mitad de los tracks.** El 38% de los peleadores queda
   sin decidir (71 de 185) y 9 más descartados. Que la cobertura por cuadros (0,793) quede
   tan arriba de la de tracks (0,460) sugiere que lo no decidido son tracks cortos; no lo
   medí por track.

## 4. La caída desde el 99,1%, investigada

Diagnóstico con `boxtwin-annotator/scripts/diagnosticar_c1.py` sobre la evidencia ya
guardada, sin correr el detector de guantes otra vez.

**El 99,1% no es la misma medición, y la spec lo cita mal.** Sale de
`boxtwin-guantes/docs/experiments/2026-09-18-color-del-guante-separa-A-de-B.md`: perfiles de
color armados con todos los guantes de la anotación (dejando el track afuera), solo tracks con
10 guantes o más, y el color decidiendo cada track por separado. Ese mismo documento dice que
sembrar los perfiles en un video nuevo "es una pieza que no está resuelta ni medida". La spec
del MVP lo reescribió como "con perfiles sembrados desde tracks con rol conocido" y de ahí
sacó que "la diferencia entre 82,4% y 99,1% es toda de la siembra". Esa frase no tenía
medición atrás, y el pre-registro de C1 la heredó al decir que las dos cifras están sobre la
misma base.

**La señal de color sigue estando.** Con perfiles de oráculo como los del 18-09, sobre los 96
tracks evaluables de esta corrida que tienen 10 guantes o más, el color acierta 93 (0,969).
El producto, sobre esos mismos 96, acierta 79 (0,823). La caída no está en el color.

**Está en la geometría, y la causa son los tracks mixtos.** Por mecanismo:

| Mecanismo | Tracks | Errores |
|---|---|---|
| Componente de la semilla A (ancla) | 33 | 4 |
| Componente orientado por color contra el ancla | 63 | 14 |
| Suelto, voto por color | 6 | 1 |
| Rescate | 3 | 1 |

18 de los 20 errores caen en seis componentes de coexistencia (uno de 03-sparring, uno de
04-sparring, cuatro de sparring-3-rounds). Los seis tienen las dos cosas: un track mixto (el
tracker cambió de persona a mitad del track) y una arista de coexistencia entre dos tracks del
mismo rol según la anotación, y cada una de esas aristas toca a un track mixto. Ninguno de
los 13 componentes sin arista espuria tiene un solo error, y dos de ellos (en Sparring y
03-sparring) tienen un track mixto igual: el track mixto solo no alcanza, lo que rompe es que
genere una coexistencia con alguien de su mismo rol.

El mecanismo: `particionar` da por hecho que dos tracks en el mismo cuadro son dos personas
distintas ("la geometría no se equivoca"). Un track mixto rompe eso. Su tramo de B coexiste
con otro track de B, la bipartición los pone en lados opuestos y el reparto del componente
queda mal. Los errores salen de a pares A por B, como se espera de eso. En sparring-3-rounds los perfiles separan bien (1,694), así que el
problema no es el color. Lo que falla es el reparto adentro del componente, antes de que el
color intervenga.

Los dos errores restantes son fragmentos de 7 y 4 guantes (voto y rescate).

Esto contesta las tres hipótesis de la versión anterior de esta sección. Las semillas
explican la diferencia de base de comparación, no los errores. La definición de la verdad
toca justo a los tracks mixtos, pero el problema no es cómo se puntúan sino cómo los reparte
la propuesta. Y el detector de guantes no es la causa: el oráculo con `guantes-v2` da 0,969.

**Lo que no se puede concluir.** Este diagnóstico es exploratorio y se hizo después de ver el
resultado, sobre los mismos datos: describe por qué falló esta corrida, no predice. C1 sigue sin pasar: el pre-registro no se toca. Cualquier
arreglo (cortar tracks en el cambio de persona, exigir más coexistencia que
`min_coexistencia = 3`, orientar por track en vez de por componente) es un diseño nuevo, y se
mide con otro pre-registro sobre material que no sea este. Las aristas espurias tienen entre 3
y 69 cuadros compartidos: subir el umbral de coexistencia no las elimina todas sin medir
cuántas legítimas se pierden.

## 5. Desvíos y lo que hay que declarar

- **Ruta de Sparring.** La primera corrida cortó en la primera fuente: `FUENTES` apuntaba a
  `videos/Sparring.mp4` (carpeta vacía) y el proyecto vive en `anotacion/`. Es el caso que
  el pre-registro prevé (§3: se arregla el entorno y se corre). La corrida válida pasó las
  seis rutas con `--video` en el mismo orden; el arreglo del script va aparte en `6555213`.
  No se midió nada antes del arreglo.
- **Corrida en dos tramos.** La primera pasada válida se cortó con el cierre de la sesión
  durante `04-sparring`. Se relanzó igual; las cuatro fuentes ya medidas se saltearon por
  reanudación, sin recalcular. El número es de una sola regla y un solo código.
- Siembra simulada con la anotación: mide si la identidad alcanza dado que la persona
  contesta bien. Que la persona conteste bien es C6.
- En distribución para el detector de guantes, que vio negativos de este mismo material.
- Tiempo de la evidencia de guantes: 1679 s entre las seis fuentes, 745 s de ellos en
  sparring-3-rounds.
