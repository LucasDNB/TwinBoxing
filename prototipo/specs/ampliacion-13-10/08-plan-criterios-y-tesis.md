# 08 · Plan, criterios pre-registrados y lo que cambia en la tesis

---

## 1. Plan (DA-8, tomada el 13-10: tandas 1 y 2, avatar detrás de bandera y regla de corte)

Ventana real: del 14-10 al 03-11 son 15 días hábiles. Encima ya están comprometidos, y varios
vencidos según las fechas de control del Cap 6:

| Compromiso | Fecha de control (Cap 6) | Estado al 13-10 (CLAUDE.md) |
|---|---|---|
| Medir C1 (R-02) | 06-10 | pendiente, vencido |
| Radio de guardia antes de C3 (R-14) | 12-10 | pendiente, vencido |
| Imagen del worker (R-10) | 12-10 | escrita, nunca construida, vencido |
| Benchmark de perfiles (R-11) | 13-10 | pendiente |
| Cap 7 | 13-10 | hoy |
| Cap 8 y primer ensayo de demo (R-04) | 20-10 | |
| Prueba con usuarios (R-07) | 27-10 al 02-11 | |
| Medición de B8 (`02`) | antes de encender | pendiente |

Suma de los siete pedidos: unos 17 días de trabajo. No entran los siete antes del 03-11 sin
sacrificar alguna de las mediciones de arriba, y esas mediciones son las que el Cap 7 y el
tribunal van a mirar. Se construye en tres tandas, con una regla de corte:

### Tanda 1 · 14 al 20-10 (unos 5,5 días)

| Id | Tarea | Spec | Días | Hecho cuando |
|---|---|---|---|---|
| B10 | Combinaciones izq-der-izq: función, Fight-Card 0.3, perfil | `01` §3 | 0,5 | pruebas B10 en verde |
| W13 | Combinaciones en análisis, Fight-Card y perfil | `01` §4 | 0,5 | CA-13.1 a 13.3 |
| B8b | Casos de una persona, tramos con un rol, margen por contexto | `02` §2 | 1 | pruebas B8b |
| W19 | Identificación para una persona | `02` §3 | 0,5 | CA-19.1 a 19.4 |
| — | Medición de B8 sobre dos videos propios | `02` §2 | 0,5 + GPU | bandera encendida o declarado |
| B11a | Etapa `enriquecer`, regla de altura, contrato 0.3 | `03` §2, §3, §5 | 1,5 | CA-14.4 |
| B11b | Calibración en 01 y 02, medición C8 en las otras cuatro | `03` §4 | 1 | experimento documentado |

Es la tanda más rentable: las combinaciones y el boxeador único reutilizan casi todo, y C8 se
mide sobre anotaciones que ya existen, sin anotar nada.

### Tanda 2 · 21 al 27-10 (unos 4,5 días), solo si C8 pasa

| Id | Tarea | Spec | Días |
|---|---|---|---|
| W14 | Cabeza/cuerpo en análisis, Fight-Card y perfil | `03` §6 | 0,5 |
| — | Medición C9 (recall del detector por altura) | `04` §3 | 0,5 |
| B12 | Exposición en el perfil | `04` §3 | 1 |
| W15 | Mapa en el perfil | `04` §4 | 1 |
| B13 | Cruces y reglas en Python, mudanza de las reglas de W12 | `05` §2 | 0,5 |
| W16 | Cruce de zonas en el Rival Fantasma | `05` §3 | 1 |

Si C8 no pasa, la tanda 2 no se construye: no hay altura que mostrar, y el mapa y el cruce dependen
de ella. Ese resultado va al Cap 7 como cualquier otro.

### Tanda 3 · avatar (unos 7 días), detrás de la bandera `avatar_3d`

| Id | Tarea | Spec | Días |
|---|---|---|---|
| B14 | `elevar_3d`, postura promedio, perfil | `06` §3 | 3 |
| — | Medición C10 | `06` §4 | 0,5 |
| W17 | Visor del perfil | `06` §5 | 2 |
| W18 | Pareja en el Rival Fantasma | `07` | 1,5 |

Regla de corte: la tanda 3 arranca solo si las tandas 1 y 2 (o 1 sola, si C8 no pasó)
están cerradas y C1 y C5 están medidos. Si eso no pasa antes del 26-10, el avatar queda como lo
dejó el Cap 6: diseño rotulado, con este spec como su especificación. No se arranca a medias en la
semana de la prueba con usuarios.

## 2. Criterios pre-registrados nuevos

Se suman a C1 a C7 de `MVP.md`, con la misma regla: escritos antes de medir, no se tocan después.

| Id | Qué | Umbral | Si no pasa | Spec |
|---|---|---|---|---|
| C8 | Altura del golpe por regla geométrica, sobre 806 golpes de cuatro fuentes no usadas para calibrar | exactitud balanceada ≥ 0,75 y recall de cuerpo ≥ 0,60 | el bloque queda "No disponible" y no se construye la tanda 2 | `03` §4 |
| C9 | Recall del detector sobre golpes a la cabeza y al cuerpo | diferencia ≤ 0,10 | el mapa se muestra corregido por recall, declarado | `04` §3 |
| C10 | Coherencia y plausibilidad del avatar | CV de huesos ≤ 0,10, cambios de lado ≤ 2%, 16 de 20 posturas plausibles | postura 2D (B9) en lugar del avatar | `06` §4 |
| C11 | Recall y precisión de ternas izq-der-izq de punta a punta | sin umbral, registrado por checkpoint | — | `01` §5 |
| B8 | Persona asignada en bolsa y sombra (ya pre-registrado; se fija el procedimiento) | ≥ 95% de saltos de 2 s | bolsa y sombra quedan "En construcción" | `02` §2 |

Medición opcional, no comprometida: precisión y recall del detector en bolsa y sombra, anotando
los dos videos de B8. Hoy el margen que se muestra en esas sesiones es el de sparring, y `02` lo
rotula así.

## 3. Banderas nuevas en `/producto`

`banderas.objetivo` (C8), `banderas.exposicion` (C8, y C9 medido), `banderas.avatar_3d` (C10).
Por entorno, como C2 y C3: `BOXTWIN_C8`, `BOXTWIN_C9`, `BOXTWIN_C10`, con su `_FECHA`. La web no
decide ninguna.

## 4. Riesgos para el registro del Cap 6

Revisión de los martes, empezando hoy:

- R-03 (alcance no cierra) sube de probabilidad mientras la ampliación esté abierta. Respuesta: las
  tandas y la regla de corte de §1.
- R-21 nuevo · Licencia de los modelos de elevación 3D. Entrenados sobre Human3.6M, cuya licencia
  gratuita es solo académica. Afecta al producto, no al PFI. Respuestas: licenciar, reemplazar por
  un modelo de datos permisivos, o reconstrucción propia por largos de huesos para la postura.
  Disparador: antes de pasar al año cero del Cap 5.
- R-22 nuevo · El mapa de exposición describe al sparring partner más que al peleador, cuando casi
  todas sus sesiones son contra el mismo rival. Respuesta: `por_rival` y el aviso del 70% (`04`).
- R-23 nuevo · Una lectura del Rival Fantasma se lee como pronóstico. Respuesta: textos validados
  por Lucas antes de publicar, sin animación de pelea, frase fija de la pantalla.

## 5. Lo que cambia en el documento

Formato ANTES / DESPUÉS / QUÉ CAMBIÓ. Los textos de ANTES salen de la Entrega 6 en LaTeX; verificar
contra la versión vigente antes de aplicar.

### Glosario (Cap 3, marco conceptual)

ANTES: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre las
zonas corporales de un peleador, la frecuencia con la que cada zona resultó alcanzada en las
sesiones registradas. Es descriptivo de lo observado y no predictivo."

DESPUÉS: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre las
zonas corporales de un peleador, la frecuencia con la que sus rivales dirigieron golpes a cada zona
en las sesiones registradas. Cuando exista la heurística de conexión, suma la frecuencia estimada
con la que cada zona resultó alcanzada, con su margen. Es descriptivo de lo observado y no
predictivo."

QUÉ CAMBIÓ: "resultó alcanzada" afirmaba contacto, que el sistema no establece; ahora la definición
describe lo que el mapa de esta versión cuenta y deja el contacto estimado como extensión.

### Cap 4, Tabla 4.1, fila del clasificador de acciones

ANTES: "Detección de eventos de golpe dentro de la secuencia y clasificación por tipo, lado y altura"

DESPUÉS: "Detección de eventos de golpe dentro de la secuencia, con el brazo que lanza, y
clasificación por tipo"

QUÉ CAMBIÓ: el lado lo da el detector y la altura no la da ningún modelo de acciones; sale de una
regla geométrica de la Fight-Card (fila siguiente).

### Cap 4, Tabla 4.1, fila de Fight-Card y perfil

ANTES: "Conteos por asalto, heurísticas de conexión con margen declarado y agregación en el perfil
táctico"

DESPUÉS: "Conteos por asalto, altura del golpe por posición de la mano respecto del rival,
heurísticas de conexión con margen declarado y agregación en el perfil táctico"

QUÉ CAMBIÓ: se agrega dónde se calcula la altura del golpe.

### Cap 6, apartado 8 (alcance mínimo entregable) y Cap 4, 2.7

No se reescriben. En la entrega acumulativa se agrega una nota fechada: el 13-10 se especificó una
ampliación de alcance (los siete pedidos), con su plan en tandas y la regla de corte, y qué tandas
llegaron al 03-11. Si alguna no llega, queda como lo decía el Cap 6: diseño rotulado.

### Cap 7 (entrega de hoy)

Lo que puede entrar sin haber construido nada:

- C8 a C11 como criterios pre-registrados el 13-10, todavía sin medir. Pre-registrarlos el mismo
  día de la entrega y medirlos después es coherente con el resto del capítulo.
- Un dato descriptivo del conjunto propio que antes no estaba: 972 golpes con altura anotada, 16,7%
  al cuerpo, y unas 30 combinaciones izq-der-izq en todo el material. Explica por qué la exactitud
  simple no sirve para C8 y por qué el bloque de combinaciones va a mostrar números chicos.

### Cap 8 (20-10)

El experimento de `06` §6 (si la profundidad inferida separa hook de recto) como línea abierta,
con sus dos salidas posibles.
