# E7 modo académico · registro

25-10-2026 · enmienda y plan aprobados en `53012c9` · rama `e7-modo-academico`

## Estado

| Id | Tarea | Estado | Evidencia |
|---|---|---|---|
| Q | Enmienda y plan aprobados | hecha | `53012c9`, 25-10 |
| L | Texto de la licencia de SMPLify | **pendiente** | Lucas lo deja pendiente (25-10). Mientras siga así, la condición 3 de la enmienda impide encender el modo |
| M1 | Modelo final de E7 | hecha | `entrenar_final.py --exp E7 --dispositivo cuda` → `~/boxtwin/modelos/clasificador_e7_v1`: 5 semillas, 42 features, 1047 golpes, `metodo: geometrico-e7`, familia 0,7775, leyenda de licencia; `sin_Sparring/` para CA2-E7. Métricas de E7 copiadas a `docs/experiments/datos/2026-10-24-e7-4dhumans-*.json` (solo números) |
| M2 | Elevación por sesión | hecha | `boxtwin-annotator/tools/elevar_hmr2_sesion.py` (corre en `boxtwin_4dhumans`) → `pose3d_hmr2.npz` y su meta |
| M3 | Features 3D en el producto | hecha | `boxtwin.mvp.clasificador_geo`: `features_3d`, `features_3d_golpe`, `ClasificadorGeo.con_3d`, `clasificar_sesion(..., pose3d=)`; `boxtwin clasificar --pose3d`. Sin el archivo, error claro con un modelo E7 |
| M4 | Worker y configuración | hecha | `BOXTWIN_CLASIFICADOR=e7`, `BOXTWIN_CMD_HMR2`, `BOXTWIN_MODELO_CLASIFICADOR_E7`; `Config.tipo_v2`; la etapa `clasificar` corre elevar, clasificar con `--pose3d` y tipos; bolsa y sombra siguen con E4a; `/salud` dice `modelo_e7` |
| M5 | Contrato y Fight-Card | hecha | `geometrico-e7` y `licencia` en el schema 0.3 (`CHANGELOG.md`), en `fightcard.py` y en zod; la vista del tipo suma "Tipo en modo académico, no apto para uso comercial: …" |
| M6 | Tiempo contra C5 | hecha | `scripts/clasificador_geo/medir_tiempo_e7.py`, `~/boxtwin/e7-4dhumans/tiempo/tiempo_e7.json` (abajo) |

## Pruebas

Escritas antes del código y en rojo, después en verde. Corridas con `PYTHONPATH` al código de esta
rama, porque `twinboxing_env` tiene instalados el anotador y la API de la copia principal:

| Suite | Resultado |
|---|---|
| `boxtwin-annotator` | 764 pasadas (nuevas: `test_mvp_e7.py`, `test_tool_elevar_hmr2_sesion.py`) |
| `boxtwin-api` | 238 pasadas (nueva: `test_e7_modo_academico.py`) |
| `scripts/clasificador_geo/tests` | 79 pasadas, 1 salteada (mmpose, como antes) (nuevas: `test_equivalencia_e7.py`, `test_medir_tiempo_e7.py`, dos en `test_entrenar_final.py`) |
| `scripts/boxmind_prueba/tests` | 38 pasadas |
| web (`vitest`, `tsc`, `eslint`) | 209 pasadas, sin errores de tipos ni de lint |

CA-E7a sobre los 1047 golpes: las features 3D del producto contra `features3d.npz` de E7,
diferencia máxima 0. CA2-E7 en Sparring: producción contra `predecir` con el mismo modelo,
diferencia 0; contra E7 s0 (descriptivo), acuerdo de argmax 0,939.

## M6 · tiempo

Etapa `clasificar` con los comandos del worker, sobre copias de dos sesiones de sparring de la
estación, GPU sin otros procesos, 25-10:

| Sesión | Video | Segmentos | E4 | E7 | Agregado de E7 | Por golpe |
|---|---|---|---|---|---|---|
| `da40de0b` | 174,9 s | 88 | 1,8 s | 253,4 s | 251,6 s = 1,44x | 2,9 s |
| `044e133e` | 580,0 s | 244 | 1,9 s | 645,3 s | 643,4 s = 1,11x | 2,6 s |

Casi todo es la elevación con HMR2: clasificar y tipos tardan menos de 2 s. Con la proyección
de C5 del 20-10 (1,74x a 1,86x con E4), el modo E7 quedaría entre 2,9x y 3,3x: **no entra en
el umbral de 2x**. La enmienda (condición 4) lo permite igual para la demo, declarado, y C5 se
informa con el modo apagado.

En las dos sesiones, E7 y E4 coinciden en la familia en 84 de 88 y 232 de 244 golpes; todos los
segmentos salieron con 3D (pasos con 3D 0,995).

## Lo que falta para encender el modo (no es de este PR)

1. El texto de la licencia de SMPLify (L).
2. En la estación, solo para la demo y sin sesiones de prueba: `BOXTWIN_CLASIFICADOR=e7` y
   `BOXTWIN_CMD_HMR2=~/miniforge3/envs/boxtwin_4dhumans/bin/python <repo>/boxtwin-annotator/tools/elevar_hmr2_sesion.py`
   en `/etc/boxtwin.env`, el build de la web (`despliegue/README.md`, "Los pasos") y reiniciar.
   Volver es sacar las dos variables.
