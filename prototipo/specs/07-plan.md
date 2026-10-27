# 07 · Plan de construcción

Del 13-10 al 03-11, que es el congelamiento de funciones del Cap 6. Después del 03-11 solo se
corrige. Encima de esto están el Cap 8 (20-10), los ensayos de demo (20-10, 27-10 y 03-11) y la
prueba con usuarios (27-10 al 02-11). La regla del MVP sigue: si algo se corre, se corre el
desarrollo, no la entrega.

Esfuerzo en días de trabajo de una persona con Claude Code. Es una estimación, no una medición, y
se revisa cada martes junto con el registro de riesgos.

---

## 1. Tareas en orden

### Hito 0 · Base (13 al 16-10)

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| W0 | Crear `boxtwin-web-entrenador` con Next 16, TS estricto, Tailwind 4, `output: "export"` condicional, ESLint, Vitest, Playwright | `02` §2, §3, §9 | 0,5 | `npm run build` genera `out/` y `npm test` corre vacío |
| W1 | shadcn init, registro `@animate-ui`, instalar los componentes de `03` §4, fijar `motion@12.43.0`, tokens y fuentes, prueba de humo que monta cada componente | `03` §2, §4 | 1 | la prueba de humo pasa y el contraste de pares de tokens da ≥ 4,5:1 |
| W2 | Layout con barra inferior y lateral, guarda de sesión, ingreso, cliente de API con token y XHR, esquemas zod, portar `formato.js` | `02` §5 a §7, `04` §00 | 1,5 | CA-00.1 y CA-00.2 |
| B0 | Servir `out/` desde la API con cabeceras | `05` B0 | 0,5 | pruebas de B0 |
| B5a | `GET /producto` con criterios y banderas por entorno | `05` B5 | 0,5 | la web lee las banderas |

### Hito 1 · Cargar y procesar (16 al 21-10)

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| B1 | Contexto, rounds, consentimiento; migración SQL | `05` B1 | 1 | pruebas de B1, migración aplicada en la estación |
| W3 | Nueva sesión (02 y 08) con subida, cancelación y estimación | `04` §02 | 1,5 | CA-02.1 a CA-02.3 |
| B5b | `GET /panel`, `progreso` en `GET /jobs`; `sesiones` y `ultima_sesion` en `GET /boxeadores` (parte de B6 adelantada) | `05` B5, B6 | 1 | pruebas de B5 |
| W4 | Panel y lista de sesiones | `04` §01 | 1 | CA-01.1 a CA-01.3 |
| B2 | Candidatos con color, tramo y altura | `05` B2 | 1 | pruebas de B2 |
| W5 | Identificación, fase 1 | `04` §03 | 1 | CA-03.1, CA-03.3 |

### Hito 2 · Resultados (21 al 25-10)

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| W6 | Análisis de sesión: video, línea de tiempo, lista, estados de espera | `04` §04 | 2 | CA-04.1, CA-04.4 |
| B4 | "No fue un golpe"; JSON Schema de la Fight-Card 0.2 | `05` B4, `06` §3 | 0,5 | pruebas de B4 y de contrato |
| W7 | Corrección en hoja y popover | `04` §04 | 0,5 | CA-04.2 |
| W8 | Fight-Card 05 y 09, export, compartir | `04` §05 | 1,5 | CA-05.1 a CA-05.4 |
| B3 | Tramos sin asignar y asignación manual | `05` B3 | 2 | pruebas de B3 |
| W9 | Identificación, fase 2 | `04` §03 | 1 | CA-03.2 |

### Compuerta de reemplazo (lunes 26-10)

Ver §2. Si pasa, la prueba con usuarios del 27-10 se hace con la web nueva.

### Hito 3 · Perfiles, P1 (27-10 al 02-11, en paralelo con la prueba con usuarios)

| Id | Tarea | Lee | Días | Hecho cuando |
|---|---|---|---|---|
| B6 | Datos del peleador | `05` B6 | 0,5 | pruebas de B6 |
| W10 | Lista y alta de peleadores | `04` §peleadores | 0,5 | alta y edición desde el celular |
| B7 | Perfil extendido | `05` B7 | 1,5 | pruebas de B7 |
| W11 | Perfil táctico (06) | `04` §06 | 1,5 | CA-06.1, CA-06.2 |
| W12 | Rival Fantasma (07) | `04` §07 | 1 | CA-07.1, CA-07.2 |
| B8 | Sesiones de un solo peleador, con su medición | `05` B8 | 1,5 | medición hecha; bandera encendida o "próximamente" declarado |

### Congelamiento (03-11) y después

- 03-11: no entra ninguna función más. Lo que no llegó de P1 queda como diseño rotulado, como
  dice el Cap 6. P2 (W-F17, W-F18) no se empieza.
- 04 al 16-11: Lighthouse y accesibilidad, correcciones de la prueba con usuarios, borrar
  `boxtwin-web` si hubo reemplazo, actualizar `CLAUDE.md` (stack y §5), `boxtwin-api/README.md`,
  `despliegue/README.md`, y en la tesis el Cap 4 (frontend y apartado de diseño), el Anexo I y la
  sección del prototipo del Cap 7.

Total P0: unos 18 días de trabajo entre el 13-10 y el 25-10, que son 13 días corridos. No entra
sin solaparse con el Cap 8 y no entra si algún cambio de API se complica. Por eso existe la
compuerta: el MVP ya funciona con la web vieja y nada de esto pone en riesgo la entrega.

## 2. Compuerta de reemplazo de la web vieja

La web nueva reemplaza a `boxtwin-web` en la estación solo si, el 26-10, se cumplen todas:

1. Las cinco pruebas de integración de `06` §1 en verde.
2. El recorrido manual de `06` §4 completo en un Android y un iPhone reales, sobre el túnel,
   con una sesión nueva de gimnasio.
3. W0 a W8 hechos. W9 también, si C1 no llegó al 95% (si C1 pasó, W9 puede quedar para el
   hito 3).
4. Ninguna violación de accesibilidad seria o crítica.

Si no se cumple: la prueba con usuarios y los ensayos de demo siguen con `boxtwin-web`, la web
nueva sigue en desarrollo hasta el 03-11 y se vuelve a evaluar ese día. Si tampoco pasa el 03-11,
la entrega final usa la web vieja y la nueva se declara en el Cap 8 como trabajo en curso, con lo
que llegó a hacer.

Mientras no haya reemplazo, las dos webs conviven en el repo y la estación sirve la vieja.

## 3. Decisiones que necesitan a Lucas

Ninguna bloquea el hito 0. Cada una tiene fecha límite por la tarea que la usa:

| Decisión | Para | Antes del | Valor inicial propuesto |
|---|---|---|---|
| Nombres en pantalla de hook y uppercut | W6, W8 | 21-10 | **Decidido 13-10:** se quedan en inglés |
| Pausa máxima entre golpes de una combinación | B7 | 27-10 | **Decidido 13-10:** 0,5 s |
| Texto de las cinco reglas de oportunidades y amenazas | W12 | 30-10 | **Decidido 13-10:** el bloque se muestra "En construcción" hasta validar los textos |
| Sesiones de bolsa y sombra (B8) | W3 | — | **Decidido 13-10:** se muestran y al elegirlas avisan "En construcción" hasta medir el criterio de B8 |
| Estado de C2 y C3 a cargar en `/producto` | B5a | cuando se midan | `pendiente` |
| Si W-F14 y W-F15 (perfil y rival) entran o quedan como diseño | Hito 3 | 26-10 | entran si la compuerta pasa a tiempo |

## 4. Riesgos de este plan

Además de los tres de `01` §7, que son de ejecución:

- R-03 del Cap 6 (alcance no cierra) sube: este plan agrega trabajo a tres semanas del
  congelamiento. Se controla con la compuerta del 26-10, que deja la salida a la web vieja
  siempre abierta.
- R-04 (la demo falla ante el tribunal): una web nueva tiene menos horas de uso que la vieja. Los
  ensayos del 27-10 y del 03-11 se hacen con la web que esté en la estación ese día, y la demo
  local en la notebook (contingencia del Cap 6) se arma con la misma.
- R-07 (prueba con usuarios): no cambia de fecha por este plan. Si la web nueva no está, se hace
  con la vieja.
