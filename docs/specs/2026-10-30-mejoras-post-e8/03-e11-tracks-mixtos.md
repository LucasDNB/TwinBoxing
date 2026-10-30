# E11 · Corte de tracks mixtos antes de `particionar` (diseño y pre-registro)

30-10-2026. El commit de este archivo es el pre-registro. Correcciones solo antes de la primera
corrida de S10 (`10-pasos.md`), anotadas como decisión.

## 1. El problema, medido

C1 dio 0,810 contra 0,95 (`docs/experiments/2026-10-20-c1-identidad-con-siembra.md`). El
diagnóstico de §4 de ese documento ubicó 18 de los 20 errores en seis componentes de
coexistencia que tienen las dos cosas: un **track mixto** (el tracker pasó de una persona a la
otra a mitad del track) y una **arista espuria**: el tramo de B del track mixto coexiste con
otro track de B, `particionar` da por hecho que dos tracks en el mismo cuadro son dos personas,
y la bipartición reparte mal el componente entero. Con perfiles de oráculo el color acierta
0,969: la señal está, lo que falla es la unidad de decisión (el track entero).

E8 mostró que cambiar el estimador no arregla esto: con YOLO26 los tracks mixtos subieron de
23 a 29 (P12c).

## 2. El diseño: partir el track donde cambia el color

Antes de armar el grafo de coexistencia, cada track candidato se parte en **segmentos** donde
la evidencia de color dice que cambió de persona. Los segmentos reemplazan a los tracks como
nodos de `particionar`; todo lo demás (altura, guante, núcleo, rescate, orientación, votos,
semillas, piso de 0,55) queda igual.

### 2.1 La señal

Con siembra (el caso del producto, el de C1), los perfiles de referencia son los de las dos
semillas: `pA = perfil_de(colores de la semilla A)` y `pB` igual. Si la separación entre los dos
es menor que `separacion_minima` (0,55), **no se parte nada**: sin perfiles que separen, el
color no puede decir que alguien cambió.

Para cada muestra de color `(f, c)` de un track (las que ya guarda `EvidenciaTrack.colores`,
solo de cuadros aislados): `s(f) = distancia_color(c, pB) − distancia_color(c, pA)`. Positivo
es "más A", negativo es "más B".

### 2.2 La regla de corte (fija desde ahora)

Segmentación binaria, como mucho dos cortes por track:

1. Sobre la serie `s` ordenada por cuadro, con `n` muestras, se busca el índice `k`
   (`m ≤ k ≤ n − m`, `m = 5`, el `min_guantes_voto` actual) que maximiza
   `|mediana(s[:k]) − mediana(s[k:])|`.
2. Se corta si **las dos medianas tienen signo opuesto y las dos tienen valor absoluto ≥ 0,25**
   (`margen_orientacion` actual). Si no, el track queda entero.
3. Si se cortó, se repite una sola vez dentro de cada mitad con la misma regla.
4. El **cuadro de corte** es, dentro del intervalo entre la última muestra del primer tramo y
   la primera del segundo, el cuadro donde la caja del track tiene el IoU máximo con otra
   detección de altura comparable (`fraccion_alto_relevante`, 0,6), porque el tracker se
   confunde cuando los cuerpos se superponen. Si en ese intervalo ninguna detección se
   superpone, el punto medio.
5. Un segmento con menos de `m` muestras de color no puede salir de un corte (lo impide la
   regla 1).
6. Las semillas también se pueden partir. Los perfiles `pA` y `pB` de §2.1 salen de las
   semillas **antes** de partir; si una semilla se parte, la semilla pasa a ser el segmento que
   contiene el cuadro del recorte que vio la persona, y el resto entra como cualquier otro
   segmento. (Una semilla mixta contamina su propio perfil; se declara, no se corrige.)

Todos los parámetros son los que el módulo ya tiene: ninguno sale de mirar los errores de C1.
Eso no quita que **la idea** de cortar salió de mirarlos; por eso §4.

### 2.3 Qué cambia en el código

- `identidad_auto.py`: función nueva `segmentar(evidencia, cfg, semillas, cache)` que devuelve
  `dict[seg_id, Segmento]` con `track_id`, `[inicio, fin)` y su sub-`EvidenciaTrack`. `proponer`
  la llama antes del paso 3 si `cfg.cortar_mixtos` es verdadero (por defecto **falso**: A no
  cambia aunque corra este código).
- `particionar` no cambia: recibe segmentos con ids propios. `Propuesta.rangos` pasa a ser por
  segmento; `AsignarIdentidadAuto` escribe un `Assignment` por segmento con su
  `start_frame`/`end_frame_excl` (el esquema ya admite rangos).
- `diagnostico` suma `tracks_partidos`, `cortes` y, por corte, el cuadro y las dos medianas.
- La web de A no cambia. Una sesión procesada con cortes mostraría el mismo track con dos
  colores en la línea de tiempo, que es lo correcto; eso se revisa recién al integrar.

## 3. Preguntas

| Id | Pregunta | Tipo |
|---|---|---|
| P15a | Precisión por cuadro de la identidad (cuadros con rol correcto / cuadros con rol asignado), con y sin corte, en el material de confirmación | con regla |
| P15b | Cobertura por cuadro (cuadros con rol correcto / cuadros de peleador según la verdad) | con regla |
| P15c | C1 con su definición por track, con y sin corte, sobre las seis fuentes (desarrollo) | sin umbral; no es la medición de C1 |
| P15d | Cuántos tracks se parten, y cuántos de esos son mixtos según la verdad (precisión del corte) y cuántos mixtos se partieron (recall del corte) | sin umbral |

Por qué la medida primaria es por cuadro y no por track: con cortes, "un track" deja de ser la
unidad. C1 está definido por track y no se mide de nuevo: P15c se informa como referencia.

## 4. Material: desarrollo y confirmación

**Desarrollo** (lo que se miró para diseñar): las seis fuentes de gimnasio de C1, con la
evidencia ya guardada (`~/boxtwin/mediciones/c1/<fuente>.evidencia.json`) y las mismas semillas
simuladas. No necesita GPU: `analizar` no se vuelve a correr. Los números de desarrollo se
informan con esa etiqueta y **no** deciden P15a ni P15b.

**Confirmación** (lo que decide): al menos **dos sparrings de dos peleadores** que no estén
entre las siete fuentes, de 3 minutos o más, con guantes de distinto color (S-01 del Cap 6), con
la identidad anotada por Lucas en el anotador **antes** de correr E11 sobre ellos y sin ver la
propuesta automática (modo de asignación manual por track). Candidatos, en este orden: las
sesiones de prueba de A del 27-10 que tengan dos peleadores; el sparring de L4 del gemelo; dos
sparrings nuevos filmados con consentimiento. Lo elige Lucas (L3).

Si el 9-11 no hay material de confirmación anotado, E11 se informa **solo con desarrollo**,
rotulado como exploratorio, y la confirmación pasa al Cap 8 con este pre-registro.

## 5. Reglas de lectura (confirmación)

Sobre la suma de los videos de confirmación, cuadro por cuadro, con la siembra simulada con la
regla fija de C1 (`docs/specs/2026-10-20-c1-preregistro.md` §2.3: el par sugerido si son dos
peleadores distintos, si no el primer candidato de cada lado). Con cortes, la semilla es el
segmento que contiene el recorte que se le muestra a la persona:

- **Mejora** si la precisión por cuadro sube 0,03 o más **y** la cobertura no baja más de 0,02.
- **Empeora** si la precisión baja 0,03 o más, o la cobertura baja más de 0,05.
- En el medio: **sin diferencia demostrada**.
- Al lado: los dos números por video, intervalos de Wilson por cuadro (con la advertencia de que
  los cuadros no son independientes) y P15d.

La asimetría (precisión manda, cobertura tolera poco) es la del módulo: un rol equivocado se
dibuja perfecto, un sin asignar se ve.

## 6. Qué decide cada resultado

- **Mejora**: candidato para A después del 17-11 detrás de `BOXTWIN_CORTE_MIXTOS=1`, con C1 y
  C5 medidos otra vez sobre el material nuevo. Cap 7: el arreglo del mecanismo que el
  diagnóstico de C1 encontró, con su número. Cap 6: R-02 baja de probabilidad si se integra.
- **Sin diferencia o empeora**: Cap 7 lo informa; la corrección manual por track (B3/W9) sigue
  siendo la pieza decisiva, como dice hoy el Cap 6. Cap 8: E13 y E14 como las dos vías que
  quedan.

## 7. Lo que E11 no separa

- Cortes sin siembra: el producto siempre tiene siembra; sin ella no se corta.
- Tracks mixtos entre peleador y árbitro: el árbitro no tiene color de guante y no aparece en
  `colores`; el corte no los ve.
- Mixtos en tramos sin cuadros aislados (clinch largo): no hay muestras de color y no hay corte.

## 8. Auditoría

1. Commit anterior a la primera corrida de S10; la anotación de confirmación con fecha anterior
   a la corrida de confirmación (S11) y hecha con la propuesta automática apagada.
2. `cortar_mixtos=False` reproduce exactamente las propuestas de C1 sobre la evidencia guardada
   (las mismas 105 asignaciones y el mismo 85/105).
3. Recálculo de precisión y cobertura por cuadro con numpy desde los `annot.json`.
4. La regla de §5 aplicada como está escrita.

## 9. Salidas

`docs/experiments/<fecha>-e11-tracks-mixtos.md` (desarrollo y confirmación en secciones
separadas), JSON, entrada del README y `docs/tesis/<fecha>-e11-cambios.md`.
