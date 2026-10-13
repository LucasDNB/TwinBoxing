# 06 · Pruebas y aceptación

Paso 3 del Prompt 11 (cinco pruebas de integración críticas y CI) y lo que la Clase 8 pide del
prototipo o MVP: cómo se probó y qué resultados dio. Lo que estas pruebas midan va al Cap 7 con
fecha y commit.

---

## 1. Cinco pruebas de integración críticas

Corren con Vitest, Testing Library y MSW, contra la API simulada con respuestas copiadas de una
sesión real (`tests/fixtures/sesion-real/`: un `GET /jobs/{id}` en cada estado, una Fight-Card
0.1 y una 0.2, un perfil). No dependen de la GPU ni del worker.

| # | Recorrido | Qué tiene que pasar | CA que cubre |
|---|---|---|---|
| I1 | Subir → espera → identificar → Fight-Card | Con la API simulada avanzando de estado, la web sube con progreso, sondea, muestra la identificación en `espera_siembra`, manda la siembra con los tracks elegidos y termina en la Fight-Card con los totales del documento | CA-02.1, CA-02.2, CA-03.1, CA-05.1 |
| I2 | Corrección de tipo y "no fue un golpe" | La lista cambia en el acto; si la API responde 422 vuelve atrás con aviso; un `no_golpe` no cambia "detectados" y suma "descartados" | CA-04.2 |
| I3 | Banderas | Con `comparacion_rounds` y `guardia` en falso no aparece en el DOM ningún texto ni número de guardia, de tendencia entre rounds ni de caída en R3; con las dos en verdadero aparecen. El texto "conectados" no aparece nunca | CA-01.3, CA-04.3, CA-05.4, CA-07.1 |
| I4 | Salto al instante | Tocar un golpe con `t_inicio = 347,0` deja `video.currentTime = 346,5`; la URL del video lleva `?t=<ticket>` y nunca el token de sesión; el ticket se renueva a los 25 min (tiempo simulado) | CA-04.1 |
| I5 | Fight-Card 0.1 y 0.2 | Las dos versiones se dibujan; una 0.1 muestra el aviso de tramos; un documento con `version: "0.3"` o un campo obligatorio faltante muestra el error de versión y no una tarjeta a medias | CA-05.1 |

## 2. Pruebas unitarias

- `lib/formato.ts`: las de `boxtwin-web/src/formato.test.js` portadas sin cambiar los casos.
- `lib/fightcard.ts`: conteo por tipo con correcciones, suma por round, combinaciones con pausa y
  numeración por guardia, texto del margen desde precisión y recall.
- `lib/banderas.ts`: `estadoDe(bloque)` para cada fila de `01` §4.
- Estimación de tiempo de la subida con el factor de `/producto`.

## 3. Pruebas de la API (pytest, en `boxtwin-api/tests/`)

Una por tarea B, como dice cada tarea en `05`. Además, una prueba de contrato: el esquema zod de
la web y el documento que produce `boxtwin.mvp.fightcard.construir` se validan contra el mismo
JSON Schema (`docs/specs/web-entrenador/fightcard-0.2.schema.json`, se escribe en B4), para que
la web y el worker no se separen en silencio.

## 4. De punta a punta (Playwright)

Contra el build exportado servido por la API real con SQLite y un worker falso que escribe los
archivos de una sesión real en cada etapa (sin GPU). Dos proyectos de Playwright:

- Celular: Pixel 7 (412 × 915) y iPhone 13 (390 × 844, WebKit).
- Escritorio: 1440 × 900, Chromium.

Recorridos: I1 completo; corrección; exportar CSV; perfil de un peleador con dos sesiones; rival
con dos perfiles. En cada página, `@axe-core/playwright` sin violaciones de impacto serio o
crítico.

Prueba manual obligatoria antes de la compuerta (`07` §2), en un celular Android y un iPhone
reales, sobre el túnel: filmar con "Filmar ahora", subir por datos móviles, identificar, corregir
un golpe, exportar. Se anota tiempo de subida, tiempo de procesamiento y cualquier error, y eso
alimenta C4 y C5.

## 5. Rendimiento y accesibilidad

Lighthouse CI en modo móvil sobre Panel, Sesión (Fight-Card) e Identificación, con los umbrales
de `03` §9. Un umbral roto falla el CI.

## 6. Integración continua

GitHub Actions, un flujo `web-entrenador.yml` que corre en cada push que toque
`boxtwin-web-entrenador/`, `boxtwin-api/` o `docs/specs/web-entrenador/`:

1. `npm ci`, `npm run lint`, `tsc --noEmit`.
2. `npm test` (unitarias e integración).
3. `npm run build` y verificación de que `out/` no contiene los nombres de los datos de
   demostración del diseño ("Tomás Ríos", "Gimnasio Demo", "M. Gómez").
4. `pytest` de `boxtwin-api` (sin GPU; las pruebas que necesitan el anotador ya se saltean con
   501).
5. Playwright y axe contra la API de prueba.
6. Lighthouse CI.

El despliegue no es parte del CI: sigue siendo manual en la estación (`despliegue/README.md`),
con `GET /salud` mostrando el commit para confirmar qué corre.

## 7. Qué va al Cap 7

El Cap 7 se entrega el 13-10 con lo que corre hoy. Esto entra en la versión del Cap 7 de la
entrega acumulativa del 17-11, con fecha y commit, en la sección del prototipo:

- Las cinco pruebas de integración y su resultado.
- El recorrido manual en dos celulares: tiempos de subida y procesamiento (insumo de C5), errores.
- Puntajes de Lighthouse por ruta.
- Cuántos tramos asignó a mano el entrenador en la prueba con usuarios (B3), como medida de uso
  real al lado de C1.
- Cuántas correcciones de tipo y de "no fue un golpe" hizo, como etiquetas nuevas (C7 y O-01 del
  Cap 6).
- La tabla de `01` §4 con el estado de cada bloque a esa fecha: qué está implementado, qué está
  condicionado y qué no está, que es lo que la Clase 8 pide declarar.
