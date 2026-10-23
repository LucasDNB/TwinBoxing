# BM · Pasos de corrida en la estación

Pre-registro: `docs/specs/2026-10-22-boxmind-prueba/01-preregistro.md`. Nada de lo que
producen estos pasos entra a git; todo queda en `~/boxtwin/boxmind-prueba-2026-10/`.

| # | Paso | Entorno | Estado |
|---|---|---|---|
| 0 | `git clone https://github.com/gouba2333/BoxingWeb ~/externos/BoxingWeb && git -C ~/externos/BoxingWeb checkout e06add2` | — | pendiente |
| 1 | Descargar BoxingWeb de Tsinghua Cloud (enlace del README del clon) y descomprimir en `~/externos/boxingweb-datos/` hasta que exista `boxingweb/data_train/` y `boxingweb/data_test/`. Anotar en Decisiones si el paquete trae términos de uso o licencia | — | pendiente |
| 2 | `python -m pytest scripts/boxmind_prueba/tests -q` con el clon presente: las dos pruebas de equivalencia y las tres de torch tienen que pasar, no saltearse | twinboxing_env | pendiente |
| 3 | Revisar el contenido de un `pose_gt.pkl` y un `video_event.json` y confirmar lo que sugiere la historia del repo: `pose_gt` es 4D-Humans con identidad verificada y `uve.pkl` la del seguidor automático (comparar los dos en una ronda: si difieren solo en la asignación roja/azul, se confirma). Anotarlo en Decisiones | twinboxing_env | pendiente |
| 4 | `python scripts/boxmind_prueba/bm0.py` (BM-0, `train.py` original; puede pedir instalar opencv, scikit-learn, joblib, torchvision, tensorboard y matplotlib en un entorno aparte) | entorno aparte | pendiente |
| 5 | `python scripts/boxmind_prueba/boxingweb.py` (de noche, después de R-11) | twinboxing_env, GPU | pendiente |
| 6 | `python scripts/boxmind_prueba/elevar.py --dominio boxingweb` y `--dominio boxtwin` | boxtwin_mmaction, GPU | pendiente |
| 7 | `python scripts/boxmind_prueba/armar.py --dominio boxingweb` y `--dominio boxtwin` | twinboxing_env | pendiente |
| 8 | `python scripts/boxmind_prueba/entrenar.py --exp X` para BM-0b, BM-1, BM-1b, BM-2, BM-2-25 y BM-2-50 | twinboxing_env | pendiente |
| 9 | `python scripts/boxmind_prueba/evaluar_bm.py` y documento `docs/experiments/2026-10-XX-boxmind-prueba.md` con el commit del pre-registro, los números y la lectura punto por punto | twinboxing_env | pendiente |

Los pasos 6 y 7 usan los datos de E4 (`~/boxtwin/clasificador-geo-2026-10/datos/`) y los pesos
de MotionBERT de E5; si faltan, correr antes `scripts/clasificador_geo/datos.py`.

## Decisiones

Una línea por decisión tomada al implementar o al correr, con el motivo y antes de correr la
parte afectada.

1. 22-10: el clasificador de atributos se entrena con un bucle propio y no con `train.py`,
   porque `train.py` evalúa contra el test en cada época y solo lee BoxingWeb (pre-registro,
   B6). BM-0 usa `train.py` tal cual para separar los dos efectos.
2. 22-10: `evaluar.py` de esta carpeta se llama `evaluar_bm.py` para no tapar a
   `scripts/clasificador_geo/evaluar.py`, del que importa las reglas.
