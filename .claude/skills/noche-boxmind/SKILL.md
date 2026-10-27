---
name: noche-boxmind
description: Trabajo desatendido de la prueba académica con el modelo de BoxMind (BM). Corre por SDD las tareas BM0 a BM11 de scripts/boxmind_prueba/TAREAS.md hasta terminar o quedar bloqueado. Solo la invoca Lucas.
disable-model-invocation: true
---

# Noche de la prueba BM

Lucas te dejó trabajando en la estación y no va a contestar hasta la mañana. Tu trabajo es
avanzar `scripts/boxmind_prueba/TAREAS.md` de BM0 a BM11 sin pedir nada, con criterio, y
dejar a la mañana un estado que se entienda en dos minutos.

## Antes de empezar, leé enteros

1. `CLAUDE.md`. Las reglas duras siguen valiendo, salvo "no corras jobs largos sin avisar":
   para esta noche el aviso es el pre-registro y Lucas ya lo dio al lanzar.
2. `docs/specs/2026-10-22-boxmind-prueba/00-README.md`, `01-preregistro.md`,
   `02-dataset-propio.md` y `03-instructivo-noche.md`. El último es tu procedimiento:
   chequeo de entorno, bucle, comandos, decisiones delegadas, reglas y resumen final.
3. `scripts/boxmind_prueba/TAREAS.md` y los docstrings de `scripts/boxmind_prueba/*.py`.
4. Como referencia de formato y de lo ya medido: `scripts/clasificador_geo/evaluar.py` y
   `docs/experiments/2026-10-16-clasificador-geometrico.md`.

## Qué hacer

Seguí `03-instructivo-noche.md` al pie de la letra: §1 antes de tocar la GPU, §2 como bucle,
§3 para los comandos, §4 cuando algo no sale como se espera, §5 siempre, §6 al terminar.

Si `TAREAS.md` ya tiene tareas `hechas` de una noche anterior, no las rehagas: los scripts
son reanudables. Empezá por la primera `pendiente` con dependencias `hechas`.

## Recordatorio de lo que no se negocia

- El pre-registro no se cambia.
- Nada de BoxingWeb, ni nada entrenado con él, entra a git ni al producto.
- El código de BoxMind se usa desde el clon y no se modifica ni se copia.
- No hay push.
