# Registro de tareas (SDD) · BM, prueba académica con el modelo de BoxMind

Spec: `docs/specs/2026-10-22-boxmind-prueba/` (00 a 03). Estado: `hecha` (con evidencia),
`en curso`, `parcial` (lo que falta está dicho), `pendiente`, `bloqueada` (qué la bloquea),
`no realizada` (con el motivo). La evidencia es el comando que la verifica o el archivo que
produjo.

Trabajo: `~/boxtwin/boxmind-prueba-2026-10/` (`<bm>`). Nada de lo que hay ahí entra a git.
Intérpretes: `twinboxing_env` (por defecto), `boxtwin_mmaction` (solo `elevar.py`),
`boxmind_bm0` (solo `bm0.py`).

## Antes de lanzar (Lucas, a mano)

| Id | Tarea | Estado |
|---|---|---|
| P1 | Rama `exp/boxmind-prueba-22-10` en la estación, con el pre-registro commiteado | pendiente |
| P2 | Clon de BoxingWeb en `e06add2`, visible en `~/externos/BoxingWeb` (enlace simbólico si está en otro lado) | pendiente |
| P3 | Datos visibles en `~/externos/boxingweb-datos/boxingweb/` con `data_train/` y `data_test/` adentro (enlace simbólico si están en otro lado) | pendiente |
| P4 | Entorno `boxmind_bm0` (clon de `twinboxing_env` + opencv-python, scikit-learn, joblib, torchvision, tensorboard, matplotlib) | pendiente |
| P5 | Benchmark R-11 terminado o la GPU libre esa noche | pendiente |
| P6 | Datos de E4 en `~/boxtwin/clasificador-geo-2026-10/datos/` (`golpes.csv`, `series/`) y pesos de MotionBERT en `pesos/` | pendiente |

## Registro

| Id | Tarea | Depende de | Spec | Estado | Evidencia / notas |
|---|---|---|---|---|---|
| BM0 | Chequeo de entorno: rutas, commit del clon, GPU, espacio en disco, archivos de E4 | P1 a P6 | 03 §1 | pendiente | |
| BM1 | Pruebas en verde, con el clon presente y sin saltear ninguna | BM0 | 01 §8 | pendiente | `python -m pytest scripts/boxmind_prueba/tests -q -rs` |
| BM2 | Inspección de BoxingWeb: rondas por partición, archivos por ronda, términos o licencia en el paquete, qué es `pose_gt` contra `uve.pkl` | BM1 | 01 §7 | pendiente | `<bm>/inspeccion.json` |
| BM3 | BM-0: `train.py` original sobre su test | BM2 | 01 §4.3 | pendiente | `<bm>/bm0/bm0.json` |
| BM4 | BoxingWeb re-estimado con el preproceso de BoxTwin y emparejado de identidad | BM2 | 01 §4.1 B3 | pendiente | `<bm>/boxingweb/golpes.meta.json` |
| BM5 | Elevación 3D de BoxingWeb | BM4 | 01 §4.1 | pendiente | `<bm>/datos/pose3d_boxingweb.npz` |
| BM6 | Elevación 3D de BoxTwin (atacante y rival, ventana anotada) | BM1 | 01 §4.1 | pendiente | `<bm>/datos/pose3d_boxtwin.npz` |
| BM7 | Armado de los dos conjuntos | BM5, BM6 | 01 §4.1 | pendiente | `<bm>/datos/conjunto_{boxtwin,boxingweb}.npz`; BoxTwin con 1.047 golpes |
| BM8 | Entrenamiento BM-0b (V0) | BM7 | 01 §4.3 | pendiente | `<bm>/corridas/BM-0b/` |
| BM9 | Entrenamiento BM-1b, BM-1, BM-2, BM-2-50, BM-2-25, en ese orden | BM8 | 01 §4.3 | pendiente | `<bm>/corridas/<exp>/` |
| BM10 | Evaluación contra el pre-registro | BM9 (y BM3 si se hizo) | 01 §5, §6 | pendiente | `<bm>/resultados/veredictos.json`, `resumen.md` |
| BM11 | Documento de experimento y entrada del README raíz preparada (sin tocar el README) | BM10 | CLAUDE.md §7 | pendiente | `docs/experiments/2026-10-XX-boxmind-prueba.md` |

## Decisiones

Una línea por decisión tomada al implementar o al correr, con el motivo y antes de correr la
parte afectada. Las delegadas están en `docs/specs/2026-10-22-boxmind-prueba/03-instructivo-noche.md` §4.

1. 22-10: el clasificador de atributos se entrena con un bucle propio y no con `train.py`,
   porque `train.py` evalúa contra el test en cada época y solo lee BoxingWeb (pre-registro,
   B6). BM-0 usa `train.py` tal cual para separar los dos efectos.
2. 22-10: `evaluar.py` de esta carpeta se llama `evaluar_bm.py` para no tapar a
   `scripts/clasificador_geo/evaluar.py`, del que importa las reglas.

## Preguntas abiertas

(vacío)
