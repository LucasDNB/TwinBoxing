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
| P1 | Rama `exp/boxmind-prueba-22-10` en la estación, con el pre-registro commiteado | hecha |
| P2 | Clon de BoxingWeb en `e06add2`, visible en `~/externos/BoxingWeb` (enlace simbólico si está en otro lado) | hecha |
| P3 | Datos visibles en `~/externos/boxingweb-datos/boxingweb/` con `data_train/` y `data_test/` adentro (enlace simbólico si están en otro lado) | hecha |
| P4 | Entorno `boxmind_bm0` (clon de `twinboxing_env` + opencv-python, scikit-learn, joblib, torchvision, tensorboard, matplotlib) | hecha |
| P5 | Benchmark R-11 terminado o la GPU libre esa noche | hecha (GPU sin procesos de cómputo a las 18:05 del 23-10) |
| P6 | Datos de E4 en `~/boxtwin/clasificador-geo-2026-10/datos/` (`golpes.csv`, `series/`) y pesos de MotionBERT en `pesos/` | hecha |

## Registro

| Id | Tarea | Depende de | Spec | Estado | Evidencia / notas |
|---|---|---|---|---|---|
| BM0 | Chequeo de entorno: rutas, commit del clon, GPU, espacio en disco, archivos de E4 | P1 a P6 | 03 §1 | hecha | 23-10 18:05: rama y pre-registro ok, clon e06add2, 40/10 rondas, E4 y MotionBERT presentes, imports de boxmind_bm0 ok, GPU libre, 730 GB libres |
| BM1 | Pruebas en verde, con el clon presente y sin saltear ninguna | BM0 | 01 §8 | hecha | `python -m pytest scripts/boxmind_prueba/tests -q -rs`: 33 pasadas, 0 salteadas |
| BM2 | Inspección de BoxingWeb: rondas por partición, archivos por ronda, términos o licencia en el paquete, qué es `pose_gt` contra `uve.pkl` | BM1 | 01 §7 | hecha | `<bm>/inspeccion.json`: 40 rondas train y 10 test; sin licencia ni términos; 6910 eventos, 5307 + 1215 pasan el filtro de largo; solo `pose_gt.pkl` (45 articulaciones, 2D y 3D por esquina), no hay `uve.pkl`: la comparación gt contra uve no se puede hacer |
| BM3 | BM-0: `train.py` original sobre su test | BM2 | 01 §4.3 | hecha | `<bm>/bm0/bm0.json`: `train.py` sale con 0, 12 épocas en unos 17 min; última época F1 macro técnica 0,617, objetivo 0,712, distancia 0,657, efecto 0,602 (paper: 0,692 y 0,747). La lectura es de BM10 |
| BM4 | BoxingWeb re-estimado con el preproceso de BoxTwin y emparejado de identidad | BM2 | 01 §4.1 B3 | hecha | `<bm>/boxingweb/golpes.meta.json`: 5668 de 6522 golpes usables (86,9%), 854 descartados por identidad; train 4616/5307, cobertura por ronda y esquina 0.622 a 0.965; test 1052/1215, cobertura por ronda y esquina 0.713 a 0.947;  50 rondas en unas 2 h 50 min. D4 no aplica |
| BM5 | Elevación 3D de BoxingWeb | BM4 | 01 §4.1 | hecha | `<bm>/datos/pose3d_boxingweb.npz`: 5668 golpes, 11336 elevaciones; sin 3D: 0 atacantes, 294 rivales; 1308 s en cuda:0 |
| BM6 | Elevación 3D de BoxTwin (atacante y rival, ventana anotada) | BM1 | 01 §4.1 | hecha | `<bm>/datos/pose3d_boxtwin.npz`: 1047 golpes, 2094 elevaciones en 309 s (cuda:0); sin 3D: 0 atacantes, 34 rivales; ventana máxima 18 cuadros |
| BM7 | Armado de los dos conjuntos | BM5, BM6 | 01 §4.1 | hecha | `<bm>/datos/conjunto_{boxtwin,boxingweb}.npz`: BoxTwin 1047 golpes, largo máx. 18, sin 2D 0, sin 3D 0; BoxingWeb 5668, largo máx. 29, sin 2D 0, sin 3D 0 (decisión 6) |
| BM8 | Entrenamiento BM-0b (V0) | BM7 | 01 §4.3 | en curso | `<bm>/corridas/BM-0b/` |
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
3. 23-10 (BM2): el paquete descargado trae solo `pose_gt.pkl` por ronda (45 articulaciones de
   4D-Humans, 2D y 3D por esquina) y ningún `uve.pkl`. Según B3, la identidad roja y azul sale
   de `pose_gt`; la comparación contra el seguidor automático queda sin hacer. Son 6910 eventos
   en las 50 rondas, no los 10,9 mil del paper (que suman BoxingStudio).
4. 23-10 (BM4): `boxtwin.preprocess` busca `yolov8l-pose.pt` en el directorio de trabajo y el
   worktree no lo tiene. BM4 corre con cwd en `~/Proyectos/TwinBoxing`, cuyo checkpoint es el de
   producción (`BOXTWIN_MODELO_POSE`, sha256 `4be6ab6c…`, igual que las otras copias). Es de
   entorno, no cambia ninguna regla.
5. 23-10 (BM6): la espera de GPU de `elevar3d.elegir_dispositivo` cuenta como ocupada la GPU
   que usa BM4, que es de esta misma prueba (420 MB de 8 GB). BM6 y BM5 corren con
   `--dispositivo cuda:0`. El dispositivo no está pre-registrado.
6. 23-10 (BM7): `armar.py --dominio boxingweb` murió por memoria (15 GB): `leer_3d` releía y
   descomprimía `P3` entero por cada una de las 11.336 elevaciones. Se lee una vez
   (`test_armar.py`, falla sin el arreglo). Es de lectura: el conjunto de BoxTwin no cambia.

## Preguntas abiertas

(vacío)
