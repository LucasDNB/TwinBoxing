---
name: noche-tipo-v2
description: Trabajo desatendido para llevar al producto el tipo de golpe v2 (familia con el modelo geométrico E4, jab y cross por brazo del detector y guardia) detrás de una bandera apagada, por SDD, tarea por tarea. Solo la invoca Lucas.
disable-model-invocation: true
---

# Noche del tipo de golpe v2

Lucas se fue a dormir. Nadie contesta hasta la mañana. Tu trabajo es avanzar
`docs/tareas/2026-10-16-fightcard-tipo-v2.md` de V0 a V12 sin pedir nada y dejar un estado que
se entienda en dos minutos. **El producto no cambia de comportamiento esta noche**: todo queda
detrás de `BOXTWIN_CLASIFICADOR`, que por defecto sigue en `poseconv3d`.

## Antes de empezar, leé enteros

1. `CLAUDE.md` (las reglas duras valen; el aviso de jobs largos está dado al lanzar, y acá no
   hay ninguno largo).
2. `docs/specs/2026-10-16-fightcard-tipo-v2/` 00 a 03.
3. El registro de tareas.
4. Lo que vas a tocar, antes de tocarlo: `boxtwin-annotator/src/boxtwin/mvp/fightcard.py`
   (`aplicar_tipos`), `mvp/orquesta.py` (`aplicar_tipos_a_sesion`), `cli.py`,
   `boxtwin-detector/tools/clasificar_sesion.py` (el formato de hoy de `tipos.json` y el id del
   golpe), `boxtwin-api/src/boxtwin_api/worker.py`, `config.py`, `app.py`,
   `prototipo/contrato/fightcard-0.3.schema.json`, `prototipo/TAREAS.md` (cómo se registró la web)
   y `scripts/clasificador_geo/` (`entrenar.py`, `modelo.py`, `features2d.py`, `rol_por_brazo.py`).

## El bucle

Hasta que no quede ninguna tarea `pendiente` con dependencias `hechas`:

1. Elegí la primera disponible y marcala `en curso`.
2. Primero las pruebas que exige el diseño §7 para esa tarea, después el código. Diff mínimo en
   archivos existentes.
3. Corré la suite del paquete que tocaste y la de los que dependen de él. En rojo: arreglá.
   Tres intentos: T1. Una prueba vieja en rojo: T2.
4. `hecha` con la evidencia (comando y resultado). Commit `16-10 <Id> <qué>` con las líneas de
   coautoría que indique el sistema.
5. Una línea en `~/boxtwin/tipo-v2/NOCHE.md` (hora, tarea, resultado, decisión delegada).
6. Después de tareas que leen anotaciones: `huellas.py verificar`. Si falla: T4.

## Comandos

- `~/miniforge3/envs/twinboxing_env/bin/python` para todo. `boxtwin_mmaction` no se usa esta
  noche: el punto de v2 es no necesitarlo.
- Suites: las del diseño §7. Para la web, `npm --prefix prototipo/boxtwin-web-entrenador test`
  (y `run lint` y `run build` si la tarea toca componentes).
- Trabajo: `~/boxtwin/tipo-v2/` y `~/boxtwin/modelos/clasificador_geo_v1/`.
- Solo lectura: `~/boxtwin/clasificador-geo-2026-10/`, `~/boxtwin/clasificador-2026-10/`,
  `~/boxtwin/mediciones/c8-nivel2/` (M1 trabaja sobre copias).

## Reglas de la noche

- La bandera queda apagada por defecto. No se cambia ningún `.env`, servicio de systemd ni la
  base de la estación.
- La medición M1 está fijada en `01-requisitos.md` §4 y no se reinterpreta. Sin umbral: se
  describe.
- Ningún archivo de `scripts/clasificador_geo/` que ya produjo resultados se modifica. Se
  agregan archivos nuevos.
- No toques `README.md` raíz, `docs/specs/`, las anotaciones ni los checkpoints existentes.
- No hagas push.

## Al terminar

Al final de `NOCHE.md`, y como último mensaje, un resumen de no más de 25 líneas: tareas hechas,
bloqueadas o no realizadas; CA1 a CA8 con su resultado; los números de M1; decisiones delegadas
aplicadas; preguntas abiertas; y los pasos exactos para que Lucas encienda v2 en la estación
(variables de entorno, reinicio del worker, cómo volver atrás).
