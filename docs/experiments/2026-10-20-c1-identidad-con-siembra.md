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

## 4. Lo que no se explica todavía

La spec del MVP dice que con perfiles sembrados desde tracks de rol conocido el color asigna
112 de 113 tracks (99,1%), y el pre-registro los pone sobre la misma base que esta medición.
Acá da 0,810 sobre 105. No sé de dónde sale la diferencia sin medirla. Candidatos, sin
orden de probabilidad:

- Las semillas. Entonces eran tracks de rol conocido elegidos desde la anotación; acá es el
  par sugerido por el producto, que la regla del pre-registro confirma sin buscar otro.
- La verdad. El pre-registro usa el rol mayoritario por cuadros; la versión anterior
  (`puntuar_contra`) tomaba la última asignación de cada track. Solo afecta a los 18 tracks
  mixtos.
- El detector de guantes o la evidencia, si cambiaron entre el 22-09 y hoy.

No reportaría esto como regresión hasta reproducir el 99,1% con el código de hoy.

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
