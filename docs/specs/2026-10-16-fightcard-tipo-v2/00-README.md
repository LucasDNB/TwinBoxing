# Fight-Card, tipo de golpe v2 · noche del 16-10-2026

Pedido de Lucas (16-10, 00:23): llevar al producto lo que E4 y E6 mostraron, por SDD y con
Claude Code trabajando de noche. El producto no cambia de comportamiento hasta que Lucas lo
apruebe: todo entra detrás de una bandera apagada.

## De dónde sale

`docs/experiments/2026-10-16-clasificador-geometrico.md` y `2026-10-16-rol-por-brazo.md`:

1. E4 (modelo chico sobre las 20 features del detector y 6 del rival) acierta la familia en
   0,727 dejando una fuente afuera, contra 0,656 de PoseConv3D (E1). Mejora según el
   pre-registro. Recall de hook 0,620 contra 0,337.
2. E6: el brazo del detector coincide con el anotado en 0,979 de los golpes emparejados. Con la
   guardia correcta, jab contra cross sale de ahí en vez del clasificador (0,726).
3. La altura de E4 no le gana a la regla de C8: la altura sigue con la regla.
4. E4 corre en `twinboxing_env` con las features que el detector ya calcula. Si reemplaza a
   PoseConv3D, el worker deja de necesitar `boxtwin_mmaction` para clasificar, que es la parte
   que hace difícil la imagen Docker (R-10 del Cap 6).

## Qué cambia en el producto (con la bandera encendida)

| Hoy (v1) | v2 |
|---|---|
| Tipo: argmax de PoseConv3D sobre 6 clases (jab, cross, hook y uppercut por mano), en `boxtwin_mmaction` | Familia (recto, hook, uppercut): ensamble de cinco semillas de E4 entrenado con las siete fuentes, en `twinboxing_env` |
| Jab o cross: lo decide el clasificador | Jab o cross: brazo del detector + guardia del boxeador asignado al lado A o B |
| Altura: regla de C8 (`enriquecer`) | Igual, sin cambios |
| Exactitud mostrada: 0,745 en distribución | 0,727 de familia con una fuente afuera (E4) |

Si el boxeador no tiene guardia cargada, se supone ortodoxa y la Fight-Card lo dice. Cuando el
entrenador carga la guardia o asigna el boxeador, la sesión se reclasifica sola.

## Archivos

| Archivo | Qué es |
|---|---|
| `01-requisitos.md` | Requisitos, criterios de aceptación y la medición de punta a punta pre-registrada. |
| `02-diseno.md` | Módulos, contratos (`tipos.json` 0.2, Fight-Card), worker, API, web, pruebas exigidas. |
| `03-decisiones-delegadas.md` | Qué hace Claude Code sin preguntar. |
| `docs/tareas/2026-10-16-fightcard-tipo-v2.md` | Registro SDD V0 a V12, con estado y evidencia. |
| `.claude/skills/noche-tipo-v2/SKILL.md` | Bucle desatendido (`/noche-tipo-v2`). |

## Cómo se lanza

1. Mergear antes el PR de `exp/clasificador-geometrico`: este trabajo usa sus scripts.
2. En la estación, en `main` actualizado, aplicar el parche, `claude` desde la raíz, modo
   automático, `/noche-tipo-v2`.
3. Leer antes `01-requisitos.md` §4 (la medición pre-registrada): al lanzar queda aprobada.

A la mañana: el registro de tareas, la bitácora `~/boxtwin/tipo-v2/NOCHE.md` y, si todo pasó,
la instrucción de una línea para encender la bandera en la estación.
