---
name: noche-geometrico
description: Trabajo desatendido de la noche del 15 al 16-10-2026. Implementa y corre E4 a E6 (clasificador geométrico, 3D con MotionBERT y rol por brazo) por SDD, tarea por tarea, hasta terminar o quedar bloqueado. Solo la invoca Lucas.
disable-model-invocation: true
---

# Noche del clasificador geométrico

Lucas se fue a dormir y te dejó trabajando. Nadie va a contestar preguntas hasta la mañana.
Tu trabajo es avanzar el registro `scripts/clasificador_geo/TAREAS.md` de G0 a G12 sin
pedir nada, con criterio, y dejar a la mañana un estado que se entienda en dos minutos.

## Antes de empezar, leé enteros

1. `CLAUDE.md` (las reglas duras siguen valiendo, salvo "no corras jobs largos sin avisar":
   para esta noche el aviso es el pre-registro y Lucas ya lo dio al lanzar).
2. `docs/specs/2026-10-16-clasificador-geometrico/00-README.md`, `01-preregistro.md`,
   `02-diseno.md` y `03-decisiones-delegadas.md`.
3. `scripts/clasificador_geo/TAREAS.md`.
4. Como referencia de estilo y de lo ya medido: `scripts/clasificador/comun.py`,
   `scripts/clasificador/evaluar.py` y `docs/experiments/2026-10-15-clasificador-tipo-altura.md`.

## El bucle

Repetí hasta que no quede ninguna tarea `pendiente` cuyas dependencias estén `hechas`:

1. Elegí la primera tarea `pendiente` con sus dependencias `hechas`. Marcala `en curso` en
   `TAREAS.md`.
2. Implementá lo que dice el diseño. Primero las pruebas que el diseño §7 le exige a esa
   tarea, después el código. Diff mínimo en lo existente; código nuevo solo en
   `scripts/clasificador_geo/`.
3. Corré las pruebas:
   `~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/clasificador_geo/tests -q`.
   Si fallan, arreglá. Tres intentos y sigue fallando: D1.
4. Si la tarea corre algo largo (entrenamiento, elevación), lanzalo en segundo plano con
   `nohup ... > <trabajo>/logs/<tarea>.log 2>&1 &` y mientras tanto avanzá con otra tarea
   independiente. Para esperar, `sleep 300` entre chequeos del log; no te quedes sin hacer nada
   si hay otra tarea disponible.
5. Marcá la tarea `hecha` con la evidencia (comando y resultado, o archivo). Commit con mensaje
   fechado `16-10 <Id> <qué>` y las líneas de coautoría que te indique el sistema.
6. Una línea en `<trabajo>/NOCHE.md`: hora, tarea, resultado, decisión delegada si hubo.
7. Después de cada tarea que corre cosas sobre los datos: `huellas.py verificar`
   (`~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador/huellas.py verificar`).
   Si falla: D2.

## Comandos

- Intérpretes, escritos así para que coincidan con los permisos:
  `~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/<script>.py ...` y
  `~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador_geo/<script>.py ...`
  (este último solo para `elevar3d.py`).
- Directorio de trabajo: `~/boxtwin/clasificador-geo-2026-10/`. Creá `logs/`, `datos/`,
  `corridas/`, `resultados/` y `pesos/` adentro.
- Datos de la campaña anterior, solo lectura: `~/boxtwin/clasificador-2026-10/` (folds, `pred.pkl`
  de E3, huellas).

## Reglas de la noche

- **El pre-registro manda.** Ninguna lectura, umbral, hiperparámetro o configuración de
  `01-preregistro.md` se cambia. Si el diseño choca con el pre-registro, gana el pre-registro.
- **No mires resultados antes de tiempo para decidir nada.** Los números se miran cuando la
  tarea de evaluación los produce, y se leen contra la regla escrita.
- **Lo que no está en las decisiones delegadas y requiere decidir algo de fondo** va a
  "Preguntas abiertas" de `TAREAS.md`, y seguís con otra cosa.
- **No escribas prosa de tesis con números que no estén en un JSON de resultados.**
- **No toques** `README.md` raíz, `docs/specs/` (salvo el commit del pre-registro en G0), las
  anotaciones, el checkpoint de producción ni nada de `prototipo/`.
- **No hagas push.** Lucas abre el PR a la mañana.

## Al terminar (o al quedar todo bloqueado)

Escribí al final de `NOCHE.md` un resumen para Lucas, de no más de 25 líneas:

1. Qué tareas quedaron hechas, bloqueadas o no realizadas, y por qué.
2. Los números primarios de cada experimento con su veredicto contra el pre-registro
   (`mejora`, `no demostrada`, `no concluyente`, `no realizado`).
3. Las decisiones delegadas que se aplicaron.
4. Las preguntas abiertas.
5. Qué tiene que hacer Lucas: revisar, abrir el PR, aplicar notas de tesis.

Y dejá ese mismo resumen como tu último mensaje.
