# 01 · Producto: alcance de la web entrenador

Formato: Prompt 9 de la Clase 5 (funcionalidades ranqueadas por valor y esfuerzo, cuáles
entran, flujo de 5 a 7 pasos, stack, tres riesgos) y la distinción de la Clase 8: esto es un
MVP, no un prototipo. Ninguna pantalla simula funciones con datos inventados.

---

## 1. Usuario y propuesta

Usuario objetivo: el entrenador de un gimnasio de boxeo amateur que filma el sparring con el
celular, en el gimnasio, y lo revisa después, en el celular o en una computadora. Es el mismo
usuario de `MVP.md` y el de la prueba con usuarios del Cap 6 (R-07).

Lo que la web le da, en orden de valor:

1. Subir el video desde el celular sin pasar por una computadora.
2. Decir quién es quién con un toque por peleador.
3. Ver por peleador y por round cuántos golpes encontró el sistema, con el margen al lado.
4. Tocar cualquier golpe y verlo en el video, y corregir el tipo si está mal.
5. Juntar las sesiones de un mismo boxeador y ver cómo cambia.
6. Poner dos perfiles uno al lado del otro.

Los puntos 1 a 4 son el MVP aprobado el 21-09. Los puntos 5 y 6 son las pantallas 06 y 07
del diseño, rotuladas "alcance completo · post-MVP". Este spec las incluye porque el pedido es
dejar el prototipo final funcionando entero, pero solo con lo que se puede calcular de datos
reales (sección 4).

> Nota para la tesis: incluir las pantallas 06 y 07 cambia el alcance mínimo entregable del
> Cap 6 (sección 8, "lo post-MVP del diseño queda como diseño rotulado"). Si se construyen, se
> declara en la entrega acumulativa como ampliación de alcance con fecha y motivo, y R-03
> (alcance no cierra) se revisa con el plan de `07-plan.md`. Si no llegan al 03-11, quedan
> como estaban: diseño rotulado. Nada del MVP depende de ellas.

---

## 2. Funcionalidades ranqueadas

Prioridad: P0 entra al MVP y bloquea el reemplazo de la web vieja. P1 entra antes del
congelamiento del 03-11 si el plan se cumple. P2 queda especificada pero no se compromete.

| # | Funcionalidad | Pantalla | Valor | Esfuerzo | Prioridad | Depende de |
|---|---|---|---|---|---|---|
| W-F1 | Ingreso con email y clave, registro por invitación | Ingreso | Alto | Bajo | P0 | API existente |
| W-F2 | Subida desde el celular (cámara o galería) con progreso real y cancelación | 02, 08 | Alto | Medio | P0 | API existente |
| W-F3 | Contexto de la sesión (bolsa, sombra, sparring, combate), rounds y peleadores | 02, 08 | Alto | Bajo | P0 | B1 |
| W-F4 | Consentimiento de las personas filmadas y declaración de que no hay menores | 02, 08 | Alto (legal) | Bajo | P0 | B1 |
| W-F5 | Estado del procesamiento en vivo, con progreso por etapa | 01, 02 | Alto | Bajo | P0 | API existente |
| W-F6 | Siembra: elegir A y B entre las personas encontradas | 03 | Alto | Medio | P0 | B2 |
| W-F7 | Revisar tramos sin asignar y asignarlos a mano (plan B de C1) | 03 | Alto | Medio | P0 | B3 |
| W-F8 | Análisis de sesión: video, línea de tiempo por peleador, lista de golpes, salto al instante | 04 | Alto | Medio | P0 | API existente |
| W-F9 | Corrección del tipo de golpe, incluido "no fue un golpe" | 04 | Medio | Bajo | P0 | B4 |
| W-F10 | Fight-Card por sesión y por round, con el margen del detector a la vista | 05, 09 | Alto | Bajo | P0 | API existente |
| W-F11 | Exportar CSV, JSON y versión imprimible | 05 | Medio | Bajo | P0 | API existente |
| W-F12 | Panel del gimnasio: pendientes, sesiones recientes, peleadores | 01 | Alto | Medio | P0 | B5 |
| W-F13 | Peleadores: alta, edición, asignación a sesiones | 01, 06 | Medio | Bajo | P0 | B6 |
| W-F14 | Perfil del peleador con datos reales: mezcla de golpes, evolución, comparación por contexto, combinaciones | 06 | Medio | Medio | P1 | B7 |
| W-F15 | Rival Fantasma: frente a frente de dos perfiles y lecturas por reglas | 07 | Medio | Medio | P1 | B7 |
| W-F16 | Sesiones de un solo peleador (bolsa y sombra) | 02, 03 | Medio | Medio | P1 | B8 |
| W-F17 | Postura promedio en 2D y distancia de trabajo | 06, 07 | Medio | Alto | P2 | B9 y decisión de dominio |
| W-F18 | Instalable en la pantalla de inicio del celular (manifest, ícono) | todas | Bajo | Bajo | P2 | |
| — | Conectados, bloqueados, esquivados, recibidos | 05, 09 | Alto | Muy alto | Fuera | Sub-hipótesis 2 sin implementar |
| — | Mapa de exposición y cruces de zonas | 06, 07 | Alto | Muy alto | Fuera | Sub-hipótesis 2 |
| — | Golpes a la cabeza y al cuerpo | 05, 06 | Medio | Alto | Fuera | El clasificador del MVP da familia, no altura |
| — | Avatar 3D | 06, 07 | Bajo | Muy alto | Fuera | Geometría 3D (ver hallazgo del hook) |
| — | Planes, horas consumidas, pagos | 01, 02 | Bajo para el PFI | Medio | Fuera | F13 de `MVP.md` |

---

## 3. Flujo del usuario (6 pasos, celular)

1. Abre la web en el celular, entra con su cuenta. Ve el panel con lo pendiente arriba.
2. Toca "Subir sesión", filma o elige el video, marca qué se filmó, quiénes pelearon, los
   rounds, y confirma el consentimiento. Toca "Subir y procesar" y puede cerrar la pestaña.
3. Cuando el sistema termina la primera parte, el panel le muestra "Falta un paso". Entra y
   marca una vez a cada peleador.
4. Si quedan tramos que el sistema no pudo asignar, los revisa en la misma pantalla.
5. Recibe la Fight-Card. Cambia entre la sesión y cada round.
6. Toca un golpe, el video salta medio segundo antes. Si el tipo está mal, lo corrige ahí.
   Exporta o comparte.

---

## 4. Estado de cada bloque del diseño

Esta tabla es la que decide qué se dibuja. La web nunca decide sola si un bloque condicionado
se muestra: lo dice el servidor (`GET /producto`, B5).

| Pantalla | Bloque | Estado | Fuente o motivo |
|---|---|---|---|
| 01 | Aviso "falta un paso" | Medido | sesiones en `espera_siembra` |
| 01 | Cuatro indicadores | Medido | B5 |
| 01 | Sesiones recientes con estado y progreso | Medido | `GET /jobs`, progreso de `GET /jobs/{id}` |
| 01 | Tendencia por peleador ("el volumen cae en R3") | Condicionado a C2 | sin C2 se muestra la última sesión y la cantidad |
| 01 | Plan Entrenador y horas | Fuera | se quita del MVP, ver sección 5 |
| 02 | Video, contexto, peleadores, rounds | Medido | B1 |
| 02 | Checklist de filmación | Medido | texto fijo, viene de S-01 del Cap 6 |
| 02 | Estimación de tiempo | Medido | factor medido de la campaña del Cap 5, ver sección 5 |
| 02 | Consumo del plan | Fuera | |
| 03 | Personas encontradas con color de guante, tramo visible y altura | Medido | B2 |
| 03 | Resultado de la propagación (tramos asignados, separación de color) | Medido | `fightcard.identidad` |
| 03 | Revisar tramos dudosos | Medido | B3 |
| 04 | Video con esqueleto | Medido | `/videos/{id}/procesado` si existe, si no el original |
| 04 | Línea de tiempo y lista de golpes | Medido | `fightcard.peleadores.*.golpes` |
| 04 | Tipo de golpe con confianza | Medido como estimación | `tipo`, `confianza_tipo`, checkpoint |
| 04 | "guardia arriba / caída" por golpe | Condicionado a C3 | `guardia[]` |
| 05 | Golpes detectados por sesión y por round | Medido | `por_round`, `total` |
| 05 | Distribución por tipo | Medido como estimación | conteo de `tipo` con correcciones |
| 05 | Comparación entre rounds y gráfico por round | Condicionado a C2 | sin C2 se muestra solo la sesión |
| 05 | Guardia opuesta arriba | Condicionado a C3 | |
| 05 | A la cabeza / al cuerpo | No disponible | |
| 05 | Estimaciones heurísticas (conectados y demás) | No disponible | RF8; sub-hipótesis 2 |
| 05 | Exportar | Medido | `/fightcards/{id}/export` |
| 06 | Datos del peleador (categoría, altura, envergadura) | Medido | cargado a mano, B6 |
| 06 | Distribución de golpes por contexto | Medido como estimación | B7 |
| 06 | Combinaciones frecuentes | Medido como estimación | B7 |
| 06 | Volumen por round y caída en R3 | Condicionado a C2 | |
| 06 | Mismo peleador en distintos contextos | Medido, filas de guardia y R3 condicionadas | B7 |
| 06 | Evolución entre sesiones | Medido | `perfil.evolucion` |
| 06 | Avatar 3D | No disponible | se reemplaza por postura 2D en P2 (W-F17) |
| 06 | Mapa de exposición | No disponible | |
| 06 | Distancia de trabajo | No disponible hasta W-F17 | necesita definición de bandas por Lucas |
| 07 | Selección de perfiles A y B | Medido | `GET /boxeadores` |
| 07 | Dimensiones reales (siluetas a escala) | Medido | altura y envergadura cargadas |
| 07 | Frente a frente | Medido, filas condicionadas igual que en 06 | B7 |
| 07 | Cruces de zonas ataque/recepción | No disponible | |
| 07 | Oportunidades y amenazas | Medido con reglas | solo reglas que usan métricas medidas, `04-pantallas.md` §07 |
| 08 | Subida en el celular | Medido | igual que 02 |
| 09 | Fight-Card en el celular | Medido, igual que 05 | |

Cómo se ve cada estado está en `03-diseno-y-componentes.md` §6.

---

## 5. Desvíos deliberados respecto del diseño

El diseño del 29-09 quedó congelado con siete correcciones pendientes de la revisión de ese
día. Estos specs las resuelven así:

| # | En el diseño | En la web | Por qué |
|---|---|---|---|
| D1 | Guardia opuesta arriba con porcentajes | Bloque condicionado a C3 | El radio pre-registrado de 0,6 marca caída en 40 de 41 golpes; no mide lo que dice hasta corregir el radio y pasar C3 |
| D2 | "Tiempo de procesamiento ≤ 20 min" | "Entre X y Y min", calculado con el factor medido (1,13 a 2,62 veces la duración, mediana 1,45) | El ≤ 20 min no sale de ninguna medición |
| D3 | Comparaciones entre rounds siempre visibles | Condicionadas a C2 | Si el recall varía entre rounds, la caída de R2 a R3 puede ser del detector y no del boxeador |
| D4 | Barra del plan y horas consumidas en pantallas del MVP | Se quitan | Planes y pagos están fuera del MVP |
| D5 | Fight-Card del celular rotulada distinto que la de escritorio | Mismo rótulo y mismos bloques condicionados en las dos | |
| D6 | Texto de la siembra que promete cobertura completa | El texto dice cuántos tramos quedaron sin asignar y que esos golpes no se cuentan | La siembra deja tramos sin asignar; en `sparring` quedaron 327 de 453 |
| D7 | Hook y uppercut con nombre en inglés, menú "Era:" con opciones fijas | Los nombres salen de `fightcard.nomenclatura`; se suma "No fue un golpe" | La nomenclatura es decisión de dominio pendiente de Lucas, y la web no la fija |

Además, sin número de diseño:

- D8. El contexto "Sombra" pide qué se practicó (técnica libre, combinaciones, defensa). Se
  guarda como dato de la sesión y no cambia el análisis.
- D9. El consentimiento no estaba en el diseño. Es una contramedida comprometida en el Cap 6
  (R-09): sin la casilla marcada no se puede subir.
- D10. Las pantallas de escritorio del diseño miden 1440 px de ancho. La web se diseña desde
  360 px y crece; el escritorio es un layout más, no el original (ver `03` §3).

---

## 6. Stack, en una línea por capa

Detalle y justificación en `02-arquitectura.md`.

- Frontend: Next.js 16 (App Router, exportación estática), React 19, TypeScript, Tailwind 4,
  componentes de Animate UI instalados con el CLI de shadcn, Motion 12, TanStack Query 5.
- Backend: `boxtwin-api` (FastAPI) sin cambio de stack, con los cambios B1 a B9.
- Datos: PostgreSQL en la estación, SQLite en desarrollo, como hoy.
- Despliegue: la API sirve la web exportada desde el mismo origen, por el mismo túnel.

---

## 7. Tres riesgos de ejecución

1. Reescribir la web a tres semanas del congelamiento. La web en React + Vite ya hace los
   seis pasos y es la que se usa en el ensayo de demo del 20-10. Mitigación: la web nueva
   reemplaza a la vieja solo cuando pasa la compuerta de paridad de `07-plan.md` (las cinco
   pruebas de integración y el recorrido punta a punta en un celular real). Hasta entonces
   conviven y la demo usa la vieja.
2. Los cambios de API frenan las pantallas. B2 y B3 tocan el anotador (candidatos y
   reasignación de tramos) y son lo más caro. Mitigación: cada pantalla tiene un modo
   degradado con lo que la API ya da hoy (la siembra funciona con los recortes actuales aunque
   falten color y tramo visible), y B2 y B3 entran sin bloquear W-F6.
3. Animate UI y Motion cambian debajo. Animate UI copia el código al repositorio (no es una
   dependencia que se actualice sola), pero sus componentes piden `motion`, que publicó la
   versión 14 el 02-10. Mitigación: `motion` fijado en 12.43.0, que es la línea para la que
   está escrito el registro, y una prueba de humo que monta cada componente instalado.
