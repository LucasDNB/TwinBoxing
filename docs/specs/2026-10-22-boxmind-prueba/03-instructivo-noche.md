# BM · Instructivo de ejecución desatendida (SDD)

22-10-2026 · lo ejecuta Claude Code en la estación con `/noche-boxmind` · registro de tareas en
`scripts/boxmind_prueba/TAREAS.md` · el pre-registro (`01-preregistro.md`) manda sobre este
documento

## 1. Chequeo de entorno (BM0)

Todo tiene que dar bien antes de usar la GPU. Si algo falla, BM0 queda `bloqueada` con el
motivo y la noche termina con el resumen: no se improvisa una ruta.

1. `git rev-parse --abbrev-ref HEAD` es `exp/boxmind-prueba-22-10` y `git log` muestra el
   commit del pre-registro. `git status` limpio.
2. `~/externos/BoxingWeb/model/mfusion.py` existe y `git -C ~/externos/BoxingWeb rev-parse HEAD`
   empieza con `e06add2`.
3. `~/externos/boxingweb-datos/boxingweb/data_train/` y `data_test/` existen y tienen
   subcarpetas.
4. `~/boxtwin/clasificador-geo-2026-10/datos/golpes.csv` y `series/` existen, y
   `~/boxtwin/clasificador-geo-2026-10/pesos/` tiene los pesos de MotionBERT.
5. `~/miniforge3/envs/boxmind_bm0/bin/python -c "import cv2, sklearn, joblib, torchvision,
   tensorboard, matplotlib"` no falla. Si falla, BM3 queda `bloqueada` (no BM0) y se sigue.
6. `nvidia-smi`: si hay procesos de cómputo, se espera de a 10 minutos, hasta 2 horas. Si
   sigue ocupada, la noche termina con el resumen: la GPU es del R-11 primero.
7. `df -h ~` con al menos 60 GB libres (el `train.py` de BM-0 escribe recortes dentro de la
   carpeta de datos). Con menos: D3.

## 2. El bucle

Repetí hasta que no quede ninguna tarea `pendiente` con sus dependencias `hechas`:

1. Elegí la primera `pendiente` con dependencias `hechas` y marcala `en curso`.
2. Corré lo que dice su fila con los comandos de la sección 3. Lo largo va en segundo plano
   con `nohup ... > <bm>/logs/<Id>.log 2>&1 &`; mientras corre, avanzá con otra tarea
   independiente (BM6 no depende de BM4; BM3 no depende de nada después de BM2). Para esperar,
   `sleep 300` entre chequeos del log.
3. Si un script nuevo o un cambio hace falta, primero la prueba, después el código, solo en
   `scripts/boxmind_prueba/`, y las pruebas en verde antes de seguir.
4. Marcá `hecha` con la evidencia (comando, cifra de control, archivo). Commit
   `<DD-MM> <Id> <qué>` con las líneas de coautoría que indique el sistema. Lo que se commitea
   es `TAREAS.md`, scripts o pruebas; nunca nada de `<bm>`.
5. Una línea en `<bm>/NOCHE.md`: hora, tarea, resultado, decisión delegada si hubo.

## 3. Comandos

`PY=~/miniforge3/envs/twinboxing_env/bin/python`, `MM=~/miniforge3/envs/boxtwin_mmaction/bin/python`,
`B0=~/miniforge3/envs/boxmind_bm0/bin/python`. Escribilos completos (los permisos los
reconocen así). Antes de todo: `mkdir -p ~/boxtwin/boxmind-prueba-2026-10/logs`.

| Id | Comando |
|---|---|
| BM1 | `$PY -m pytest scripts/boxmind_prueba/tests -q -rs` (33 pasadas, 0 salteadas) |
| BM2 | `$PY scripts/boxmind_prueba/inspeccionar.py` |
| BM3 | `$B0 scripts/boxmind_prueba/bm0.py` |
| BM4 | `$PY scripts/boxmind_prueba/boxingweb.py` |
| BM5 | `$MM scripts/boxmind_prueba/elevar.py --dominio boxingweb` |
| BM6 | `$MM scripts/boxmind_prueba/elevar.py --dominio boxtwin` |
| BM7 | `$PY scripts/boxmind_prueba/armar.py --dominio boxtwin` y `--dominio boxingweb` |
| BM8 | `$PY scripts/boxmind_prueba/entrenar.py --exp BM-0b` |
| BM9 | `$PY scripts/boxmind_prueba/entrenar.py --exp <exp>` para BM-1b, BM-1, BM-2, BM-2-50, BM-2-25 |
| BM10 | `$PY scripts/boxmind_prueba/evaluar_bm.py` |

Cifras de control que se anotan en la evidencia:

- BM2: rondas de train y de test (el paper dice 40 y 10), si hay términos o licencia, y la
  lectura de `pose_gt` contra `uve.pkl`.
- BM4: golpes usables sobre el total del filtro de largo y cobertura roja y azul por ronda.
- BM6 y BM7: 1.047 golpes de BoxTwin; golpes sin 2D y sin 3D.
- BM8: F1 macro de técnica de BM-0b por semilla, antes de seguir a BM9 (la lectura de V0 la
  hace BM10; acá solo se anota).

## 4. Decisiones delegadas

Lo que no está acá y obliga a decidir algo de fondo va a "Preguntas abiertas" de `TAREAS.md`,
y se sigue con otra tarea.

- **D1. Pruebas que no pasan.** Tres intentos de arreglo y sigue fallando: la tarea queda
  `bloqueada` con el error y se sigue con lo independiente. Una prueba de equivalencia contra
  BoxMind que falla no se arregla tocando la prueba: bloquea BM5 a BM10.
- **D2. Formato distinto del esperado en BoxingWeb** (otra estructura de carpetas, otras
  claves en el pkl, otro esquema de eventos): se adapta el lector con prueba nueva, siempre
  que la adaptación sea de lectura y no cambie una regla del pre-registro. Se anota en
  Decisiones con el ejemplo concreto. Si no se puede adaptar sin decidir algo de fondo,
  pregunta abierta y BM4 a BM9 quedan bloqueadas.
- **D3. Disco.** Con menos de 60 GB libres se corre BM3 al final y solo si quedan 40 GB
  después de BM4 a BM9. No se borra nada.
- **D4. Identidad pobre en BoxingWeb.** Si menos del 50% de los golpes pasan la cobertura, se
  sigue igual (el pre-registro lo cuenta como ruido), se informa y va a preguntas abiertas.
  No se cambian `TAU_ID` ni `COBERTURA_MIN`.
- **D5. Memoria de la GPU.** En la evaluación se baja el lote de predicción (256) hasta que
  entre. El lote de entrenamiento (64) está pre-registrado y no se toca: si no entra, la tarea
  queda bloqueada.
- **D6. BM-0 falla** por dependencias o por su código: BM3 queda `bloqueada` con el error y
  se sigue. No se modifica el clon.
- **D7. V0.** La lectura la hace `evaluar_bm.py`. Si V0 no es válida, BM9 se corre igual y
  todo se informa como exploratorio, como dice el pre-registro.
- **D8. Sospecha.** Si BM10 marca D9 de E4 (familia ≥ 0,90 u ocho clases ≥ 0,80), antes de
  escribir el documento se busca fuga: que ningún `frame_dir` de prueba esté en el
  entrenamiento de su fold, y que BoxingWeb no tenga material de las fuentes de BoxTwin
  (Pacquiao es profesional; BoxingWeb es olímpico de 2021 a 2024). Lo que se encuentre va al
  documento.
- **D9. Tiempo.** Si a las 8:00 no terminó BM9, se dejan corriendo las corridas en curso, se
  marca lo que falta como `pendiente` y se escribe el resumen.

## 5. Reglas de la noche

- El pre-registro manda: ningún umbral, hiperparámetro, filtro, semilla o regla de lectura de
  `01-preregistro.md` se cambia.
- Ningún número se mira para decidir algo antes de BM10, salvo las cifras de control.
- No se escribe prosa de tesis. El documento de BM11 copia números de
  `<bm>/resultados/veredictos.json` y los lee contra el pre-registro, punto por punto, en el
  formato de `docs/experiments/2026-10-16-clasificador-geometrico.md`.
- No se tocan: `README.md` raíz, el resto de `docs/specs/`, las anotaciones, el clon de
  BoxingWeb, los datos descargados (salvo lo que su `train.py` escribe solo en BM3), el
  trabajo de E4, el checkpoint de producción, `prototipo/`, `/etc/boxtwin.env`.
- No se instala nada ni se hace push.

## 6. Al terminar

Al final de `<bm>/NOCHE.md`, y como último mensaje, un resumen de no más de 25 líneas:

1. Tareas hechas, bloqueadas o no realizadas, y por qué.
2. V0 (con BM-0 al lado), P7, P8, P9, P10 y sub-hipótesis 1, con su número primario y su
   veredicto (`mejora`, `no demostrada`, `no concluyente`, `no realizado`).
3. El resultado del gatillo de `02-dataset-propio.md` y la curva en una línea.
4. Decisiones delegadas aplicadas y preguntas abiertas.
5. Qué tiene que hacer Lucas: revisar, abrir el PR, la entrada del README raíz, y si
   corresponde la nota para la tesis.
