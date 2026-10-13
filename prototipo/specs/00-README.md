# Specs de la web entrenador — BoxTwin

Estado: propuesta · 12-10-2026 · base: `main` en `0e7883e` (22-09)
Diseño de referencia: artifact "BoxTwin — Versión Entrenador", nueve pantallas (29-09-2026)
Stack pedido: Next.js + componentes de Animate UI, desarrollo mobile first
Fecha de congelamiento de funciones (Cap 6): 03-11-2026 · Entrega final: 17-11-2026

Estos specs reemplazan a `boxtwin-web` (React + Vite) por una web en Next.js que implementa
las nueve pantallas del diseño del prototipo final sobre la API que ya existe
(`boxtwin-api`), más los cambios de API que el diseño necesita y hoy no están.

## Qué significa "funcionando al 100%"

Que cada una de las nueve pantallas funciona con datos reales del sistema, sin datos de
demostración, en celular y en escritorio. No significa que se muestren números que el
sistema no mide. Donde el diseño muestra algo que todavía no está medido o no existe
(conectados, mapa de exposición, avatar 3D, guardia antes de C3), la pantalla funciona igual
y ese bloque dice qué falta para que exista. Es la misma línea del RF8 y del Cap 3: el
producto no muestra números que no puede sostener.

Por eso cada bloque de cada pantalla lleva uno de tres estados, definido en `01-producto.md`:

- **Medido**: sale de la API y tiene su margen a la vista.
- **Condicionado**: existe en el código pero depende de un criterio pre-registrado (C2, C3)
  o de una decisión de dominio. Se enciende con una bandera del servidor, no tocando la web.
- **No disponible**: requiere una pieza que no existe (sub-hipótesis 2, geometría 3D). El
  bloque se muestra con su explicación y sin números.

## Archivos

| Archivo | Contenido | Formato que sigue |
|---|---|---|
| `01-producto.md` | Alcance, funcionalidades ranqueadas, flujo, estados de cada bloque, desvíos del diseño | Prompt 9 de la Clase 5; prototipo vs MVP de la Clase 8 |
| `02-arquitectura.md` | Next.js con exportación estática servida por la API, estructura de carpetas, datos, autenticación, video | Clase 8, arquitectura en capas |
| `03-diseno-y-componentes.md` | Tokens del diseño, tipografía, grilla mobile first, mapa de componentes de Animate UI, movimiento, accesibilidad | Prompt 10 de la Clase 5; Clase 8, UX/UI |
| `04-pantallas.md` | Las nueve pantallas: ruta, datos, layout en celular y en escritorio, estados, interacción, criterios de aceptación | |
| `05-api-y-datos.md` | Endpoints que se usan tal cual y los cambios de API y del contrato de la Fight-Card | Prompt 11 de la Clase 5 (diseño antes del código) |
| `06-pruebas-y-aceptacion.md` | Cinco pruebas de integración críticas, e2e en celular, accesibilidad, rendimiento, CI | Prompt 11, paso 3 |
| `07-plan.md` | Orden de construcción con fechas, compuerta de reemplazo de la web vieja, riesgos | Plan de 8 semanas del MVP |

## Cómo usarlos con Claude Code

Cada tarea de `07-plan.md` tiene un identificador (W1, W2… para la web, B1, B2… para la API).
Una sesión de Claude Code toma una tarea, lee los apartados que esa tarea cita y nada más, y
cierra con sus pruebas en verde. El orden importa: las tareas B que una pantalla necesita van
antes que esa pantalla.

Regla del Prompt 11: para cada tarea B, primero el plan (endpoint, payload, tabla, prueba),
después el código.

## Lo que estos specs no cambian

- El pipeline de visión, los modelos y sus criterios de aceptación (C1 a C7 de `MVP.md`).
- El contrato `fightcard.json`: se extiende a 0.2 de forma compatible, no se rompe.
- La línea epistemológica: golpes detectados y no lanzados, sin conectados, sin puntuación,
  sin veredicto (RF6 y RF8).
- El despliegue de un nodo en la estación con túnel (RNF4).
