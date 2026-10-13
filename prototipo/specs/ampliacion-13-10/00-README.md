# Specs de la ampliación del 13-10 · BoxTwin web entrenador

Estado: propuesta · 13-10-2026 · base: `main` en `3a62a39` (web entrenador en producción desde el 13-10)
Sigue a: `prototipo/specs/00` a `07` (no los reemplaza)
Congelamiento de funciones (Cap 6): 03-11-2026 · Entrega final: 17-11-2026

Siete pedidos de Lucas del 13-10, especificados con el mismo formato que la web entrenador:
plan antes del código (Prompt 11 de la Clase 5), tareas B (API y worker) y W (web) con
identificador, criterio de aceptación por pantalla, y la regla de `01-producto.md` §4: ningún
bloque muestra un número que el sistema no pueda sostener.

## Los siete pedidos y dónde están

| # | Pedido | Spec | Tareas | Estado de partida |
|---|---|---|---|---|
| 1 | Separación entre cabeza y cuerpo en la Fight-Card | `03-cabeza-cuerpo.md` | B11, W14 | "No disponible" en 05 y 06 |
| 2 | Mapa de exposición | `04-mapa-de-exposicion.md` | B12, W15 | "No disponible" en 06 |
| 3 | Avatar 3D | `06-avatar-3d.md` | B14, W17 | Fuera (geometría 3D) |
| 4 | Avatar 3D en el Rival Fantasma | `07-avatar-3d-rival.md` | W18 | Fuera; hoy hay siluetas 2D a escala |
| 5 | Dónde ataca uno y dónde recibe el otro (Rival Fantasma) | `05-rival-ataque-recepcion.md` | B13, W16 | "No disponible" en 07 |
| 6 | Combinaciones frecuentes: solo izq-der-izq con menos de 0,5 s entre golpes | `01-combinaciones-izq-der-izq.md` | B10, W13 | B7 calcula las 5 secuencias más frecuentes de 2 o 3 golpes por tipo |
| 7 | Marcar un solo boxeador en bolsa y sombra | `02-un-solo-boxeador.md` | B8b, W19 | B8 con código hecho, bandera apagada, "En construcción" en la web |

El plan con fechas, los criterios pre-registrados nuevos (C8 a C11) y lo que esto le cambia a la
tesis están en `08-plan-criterios-y-tesis.md`.

## Dependencias

```
01 combinaciones ─────────────────────────────── (independiente)
02 un solo boxeador ──────────────────────────── (independiente; usa B8)
03 cabeza/cuerpo ──► 04 mapa de exposición ──► 05 ataque y recepción en el Rival
06 avatar 3D ──────► 07 avatar 3D en el Rival
```

Los tres primeros de la cadena 03 → 04 → 05 comparten una sola etapa nueva del worker
(`enriquecer`, en `03` §3), que lee `poses.npz` y la Fight-Card ya hechos. No corre pose ni
GPU para las sesiones existentes, salvo el avatar (B14), que sí usa GPU.

## Qué no cambia

- La posición epistemológica (CLAUDE.md §2): golpes detectados, sin conectados, sin puntuación,
  sin veredicto. El mapa de exposición de esta versión cuenta a dónde fue dirigido el golpe, no
  si tocó (`04` §1). Esto obliga a corregir una definición del glosario de la tesis (`08` §5).
- Los criterios C1 a C7 de `MVP.md` y sus umbrales.
- El contrato de la Fight-Card se extiende a 0.3 de forma compatible con 0.1 y 0.2.

## Decisiones tomadas (13-10)

Lucas delegó las ocho el 13-10 ("tomá las decisiones en base a tus recomendaciones"). Quedan
tomadas con la opción recomendada en cada spec, que conserva las alternativas descartadas y el
motivo. Si una medición muestra que alguna fue mala, se reabre declarándolo, no en silencio.
No cubre los textos de las reglas del Rival Fantasma (W12 y `05` §2): esos siguen esperando la
validación de Lucas como criterio de dominio y salen "En construcción" hasta entonces.

| Id | Decisión | Opciones | Tomada | Spec |
|---|---|---|---|---|
| DA-1 | Cómo se mide la separación de 0,5 s | inicio→inicio / fin→inicio | inicio→inicio | `01` §2 |
| DA-2 | izq-der-izq-der-izq, ¿una combinación o dos? | sin solapar / solapadas | sin solapar | `01` §2 |
| DA-3 | Zurdos | izq-der-izq literal / adelante-atrás-adelante | literal, como se pidió | `01` §2 |
| DA-4 | Lado de las zonas del mapa | por mano del rival / por lado del cuerpo que recibe | por mano del rival | `04` §2 |
| DA-5 | Cabeza/cuerpo en bolsa y sombra | no se calcula / respecto de los propios hombros | no se calcula | `03` §2 |
| DA-6 | Qué muestra el avatar | postura promedio / postura y repetición del golpe | postura promedio; golpe en P2 | `06` §1 |
| DA-7 | Modelo para elevar a 3D | MotionBERT vía mmpose 1.3.2 / reconstrucción propia por largos de huesos | MotionBERT | `06` §2 |
| DA-8 | Qué entra antes del 03-11 | todo / tandas 1 y 2 / solo tanda 1 | tandas 1 y 2; avatar detrás de bandera | `08` §1 |
