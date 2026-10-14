# 01 · Combinaciones frecuentes: solo izq-der-izq

Pedido (13-10): en combinaciones frecuentes, marcar solo las secuencias izquierda-derecha-izquierda
cuyos golpes se ejecuten con menos de 0,5 s de diferencia entre cada uno.

Tareas: B10 (API y contrato), W13 (web). Esfuerzo: 1 día en total.

---

## 1. Qué cambia respecto de B7

Hoy `perfil.combinaciones()` (`boxtwin-api/src/boxtwin_api/perfil.py`) arma cadenas de golpes con
separación fin→inicio ≤ `pausa_combinacion_s`, las nombra por tipo de golpe y devuelve las cinco
más frecuentes de 2 o 3 golpes. Un golpe sin tipo corta la cadena.

Con el pedido nuevo cambian tres cosas:

1. Se cuenta un solo patrón: brazo izquierdo, derecho, izquierdo.
2. El patrón se define por el brazo, que sale del detector (medido), y no por el tipo, que sale
   del clasificador (estimación que no generaliza). Por eso el bloque pasa de "estimación" a
   "medido". El tipo de cada golpe se muestra al lado, como dato secundario y rotulado como
   estimación.
3. La regla de separación pasa de "≤" a "<", como dice el pedido.

Un golpe sin tipo ya no corta la combinación. Un golpe descartado por el entrenador (B4) sí la
corta, porque el entrenador dijo que ahí no hubo golpe.

## 2. Regla exacta, con las tres decisiones de dominio (tomadas el 13-10)

Golpes del mismo peleador en la misma sesión, ordenados por `t_inicio`, sin los descartados.
Una combinación izq-der-izq es una terna de golpes consecutivos `g1, g2, g3` con
`brazo = izq, der, izq` y separación < 0,5 s entre `g1` y `g2` y entre `g2` y `g3`.

DA-1 · Cómo se mide la separación. Tomada: inicio→inicio.

- Inicio→inicio: `g2.t_inicio − g1.t_inicio`. Es la lectura más literal de "se ejecuten con menos
  de 0,5 s de diferencia", y usa la frontera más precisa del detector: el error de anotación es
  1,12 cuadros al inicio contra 1,55 al final (metodo-y-escritura).
- Fin→inicio: `g2.t_inicio − g1.t_fin`, que es lo que hace B7 hoy. Mide la pausa entre golpes,
  no entre ejecuciones, y es más permisiva.
- Se toma inicio→inicio.

Lo que da cada combinación de reglas sobre lo anotado (seis `annot.json` con identidad, 962
golpes sin los abortados, fintas incluidas; medido el 13-10):

| Separación | Sin solapar | Solapadas |
|---|---|---|
| inicio→inicio | 28 | 29 |
| fin→inicio | 34 | 36 |

Sin las fintas (933 golpes) cada celda baja en uno o dos. Es poco: en todo el material anotado
hay del orden de 30 combinaciones izq-der-izq, una cada 30 golpes.

DA-2 · Solapamiento. Tomada: sin solapar. En izq-der-izq-der-izq hay dos ternas que comparten el
golpe del medio.

- Sin solapar (tomada): se recorre de izquierda a derecha y una terna contada consume sus
  tres golpes. Ahí cuenta una. Es lo que un entrenador cuenta mirando.
- Solapadas: cuenta dos. Infla la cifra en series largas.

DA-3 · Zurdos. Tomada: literal. El pedido dice izquierda-derecha-izquierda. Para un ortodoxo es adelante-atrás-
adelante (la base del 1-2-3); para un zurdo es atrás-adelante-atrás.

- Literal (tomada, porque es lo pedido): izq-der-izq anatómico para todos.
- Por guardia: adelante-atrás-adelante, que para un zurdo es der-izq-der. Requiere la guardia
  cargada; sin guardia no se puede calcular.
- Como se tomó literal, la pantalla dice "mano izquierda, derecha, izquierda" y no "1-2-3".

La regla vive solo en la API. La web no la copia (decisión ya tomada en W11).

## 3. B10 · API y contrato

### Fight-Card 0.3 (por sesión)

Se calcula al construir la Fight-Card y al aplicar correcciones (porque un "no fue un golpe"
puede romper una terna). Campo nuevo por peleador:

```json
"combinaciones": {
  "patron": "izq-der-izq",
  "separacion": "inicio_a_inicio",
  "max_s": 0.5,
  "lista": [
    {"golpes": ["g_0142", "g_0143", "g_0144"], "t_inicio": 347.2, "t_fin": 348.3,
     "tipos": ["jab", "cross", "hook"], "round": 2}
  ],
  "total": 7,
  "por_round": [{"round": 1, "total": 2}, {"round": 2, "total": 4}, {"round": 3, "total": 1}]
}
```

- `tipos` repite el tipo vigente de cada golpe (corregido si hay corrección). Es estimación.
- `patron`, `separacion` y `max_s` quedan escritos en el documento para que una Fight-Card vieja
  se pueda leer sin saber con qué regla se calculó.

### Perfil (B7 modificado)

`GET /boxeadores/{id}/perfil` cambia el campo `combinaciones`:

```json
"combinaciones": {
  "patron": "izq-der-izq", "total": 23, "por_minuto": 0.41,
  "por_contexto": {"sparring": 15, "bolsa": 8},
  "tipos_frecuentes": [{"tipos": ["jab", "cross", "hook"], "n": 9, "estimado": true}],
  "sesiones_con_combinaciones": 6
}
```

- `por_minuto` usa la duración con identidad resuelta del peleador (cobertura × duración), no la
  duración del video: si el sistema no lo vio, no puede decir que no combinó.
- La lista anterior (top 5 por tipo) se elimina. Esto rompe a un cliente que la lea; la única
  cliente es la web, que se cambia en el mismo PR. Se documenta en el CHANGELOG del contrato.

### `/producto`

`dominio.pausa_combinacion_s` se reemplaza por
`dominio.combinacion: {"patron": "izq-der-izq", "separacion": "inicio_a_inicio", "max_s": 0.5}`.

### Archivos

`boxtwin-annotator/src/boxtwin/mvp/combinaciones.py` (nuevo, función pura `ternas(golpes, regla)`),
`fightcard.py` (construir y `aplicar_correcciones` la llaman), `boxtwin-api/.../perfil.py`
(usa la misma función importada del anotador, no una copia), `config.py`, contrato 0.3.

### Pruebas (B10)

1. `izq(0,0) der(0,4) izq(0,8)` es una terna; `izq(0,0) der(0,5) izq(0,9)` no lo es (0,5 no es menor).
2. izq-der-izq-der-izq cuenta 1 (DA-2, sin solapar).
3. Un golpe sin tipo en el medio no corta; uno descartado sí.
4. Marcar `no_golpe` el golpe del medio y volver a leer la Fight-Card baja `total` en uno.
5. La suma de `por_round` es igual a `total`, y una terna que cruza el fin de un round cuenta en
   el round de su primer golpe.
6. Prueba de concordancia contra lo anotado: con los eventos de las seis `annot.json` como
   entrada, la función da el número de la tabla de §2 que corresponde a las decisiones tomadas
   (28 con las tomadas: inicio→inicio, sin solapar). Queda como prueba de
   regresión con un archivo fijo derivado de las anotaciones.

## 4. W13 · Web

- Análisis de sesión (04): en la línea de tiempo, cada terna se marca con un corchete que abarca
  sus tres ticks en el carril del peleador. En la lista de golpes, los tres golpes llevan la
  insignia "izq-der-izq". Tocar el corchete lleva el video a `t_inicio − 0,5 s`.
- Fight-Card (05 y 09): fila medida nueva "Combinaciones izq-der-izq", A contra B, por sesión y
  por round (por round solo con C2, como el resto de las comparaciones entre rounds).
- Perfil (06): el bloque "Combinaciones frecuentes" pasa a estado medido con el total, por minuto
  y por contexto; debajo, en estilo de estimación, los tres órdenes de tipos más frecuentes.
- Texto del margen, nuevo en `lib/textos.ts`:
  `margen.combinaciones`: "Para contar una combinación el sistema tiene que encontrar los tres
  golpes. Encuentra cerca de {r} de cada 10 golpes, así que se le escapan la mayoría de las
  combinaciones: el número sirve para comparar sesiones, no como total real."
  `{r}` sale de `detector.recall_medido`.

## 5. Lo que el número no dice, y cómo se mide (C11)

Si el detector encuentra cada golpe con probabilidad 0,484 y los fallos fueran independientes, el
sistema encontraría las tres de una terna en 0,484³ ≈ 11% de los casos. No hay por qué suponer
independencia: los golpes de una combinación salen rápido y seguidos, y el detector puede
fallarlos más o menos que al resto. Por eso no se declara el 11%: se mide.

Hay que decirlo antes de construir: con unas 30 combinaciones reales en todo el material anotado
y una fracción chica encontrada, la mayoría de las sesiones va a mostrar 0, 1 o 2. El bloque es
barato y correcto, pero su valor para el entrenador depende de que suba el recall del detector.
El estado vacío no puede leerse como "no combina": dice "el sistema no encontró combinaciones
completas en esta sesión".

C11 (sin umbral, como C7): sobre las fuentes anotadas, con la salida del sistema de punta a
punta, recall de ternas izq-der-izq (terna del sistema cuyos tres golpes emparejan con los tres
de una terna anotada a IoU 0,3) y precisión de ternas. Se registra por checkpoint del detector.
El texto del margen pasa a usar ese recall de ternas cuando exista (campo
`combinaciones.recall_medido` en `detector`).

## 6. Criterios de aceptación

- CA-13.1. Una sesión con ternas las marca en la línea de tiempo y suma igual en la Fight-Card,
  en sus rounds y en el perfil del peleador.
- CA-13.2. Ninguna secuencia distinta de izq-der-izq aparece como combinación.
- CA-13.3. El margen de combinaciones aparece siempre que aparece el número.
