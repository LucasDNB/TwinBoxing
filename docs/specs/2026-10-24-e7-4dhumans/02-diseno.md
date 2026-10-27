# E7 · Diseño

24-10-2026 · el pre-registro (`01-preregistro.md`) manda sobre este documento

## 1. Entorno

Conda nuevo `boxtwin_4dhumans`, separado de los dos que ya existen: Python 3.10, torch y
torchvision con CUDA para la 2080, y 4D-Humans instalado desde un clon en `~/externos/4D-Humans`
(commit anotado en `TAREAS.md`) con `pip install -e .`, sin el extra `all` (detectron2 no hace
falta: las cajas son nuestras). `chumpy` lo pide `smplx` para leer el `.pkl` de SMPL. Los pesos de
HMR2.0b se bajan solos a `~/.cache/4DHumans` en la primera corrida; su sha256 va a `TAREAS.md`.

Lo que no se sabe hasta probar y se mide en E7-1: si el `import` de `hmr2` arrastra detectron2,
cuánta VRAM usa el ViT-H con lote 8 y a qué velocidad corre. Si no entra en 8 GB con lote 1,
E7 queda no realizada.

## 2. Scripts (`scripts/clasificador_geo/`)

| Archivo | Entorno | Qué hace |
|---|---|---|
| `elevar_hmr2.py` | `boxtwin_4dhumans` | Por golpe, los cuadros de la ventana de E4 (inicio − 8 a fin + 8, en los índices nativos del video, con el paso de E5 por encima de 45 fps); caja del atacante desde `kp` con relleno de huecos (E2); HMR2 por cuadro; las 24 articulaciones de SMPL desde `pred_smpl_params` con `smplx`; mapeo a H36M (E3); remuestreo a 32 pasos con `elevar3d.a_ventana` y máscara con `elevar3d.mascara_ventana`. Escribe `~/boxtwin/e7-4dhumans/datos/pose3d.npz` con el formato del `pose3d.npz` de E5. Reanudable por fuente |
| `features3d.py` | `twinboxing_env` | Sin cambios de cálculo. Gana `--trabajo` para leer `pose3d.npz` y escribir `features3d.npz` en el trabajo de E7 |
| `entrenar.py` | `twinboxing_env` | Gana el experimento `E7` (columnas y 3D como E5, features desde el trabajo de E7). Mismo dispositivo que E5 (`cuda`) |
| `evaluar.py` | `twinboxing_env` | Gana P11a y P11b reutilizando `veredicto_mejora`, y la vibración del 3D |

Pruebas nuevas en `scripts/clasificador_geo/tests/`, antes del código:

1. Mapeo SMPL → H36M: con un esqueleto sintético, cada índice de §3 E3 del pre-registro cae
   donde dice, y `features3d.features_de` da lo mismo que con el H36M equivalente.
2. Caja: con keypoints de score bajo o faltantes, la caja usa solo los ≥ 0,3; sin ninguno, el
   cuadro queda con máscara 0.
3. Ventana: los índices de cuadro de un golpe a 29,97 y a 59,94 fps coinciden con los de E5.
4. `entrenar.py --exp E7` lee del trabajo de E7 y no del de E5.
5. Vibración: un 3D constante da 0 y uno con ruido conocido da lo esperado.

## 3. Pasos

| Id | Paso | Depende de |
|---|---|---|
| E7-0 | Commit del pre-registro aprobado | Q1 |
| E7-1 | Entorno y clon; prueba de humo: un golpe, VRAM, velocidad, claves de salida de HMR2 | Q2, Q3 |
| E7-2 | Pruebas en verde | E7-1 |
| E7-3 | `elevar_hmr2.py` sobre las siete fuentes. Control: 1047 golpes, fracción de pasos con 3D | E7-2, Q4 |
| E7-4 | `features3d.py --trabajo ~/boxtwin/e7-4dhumans` | E7-3 |
| E7-5 | `entrenar.py --exp E7`, siete folds, cinco semillas | E7-4 |
| E7-6 | `evaluar.py` contra el pre-registro | E7-5 |
| E7-7 | Documento `docs/experiments/2026-10-XX-e7-4dhumans.md` y entrada del README raíz preparada | E7-6 |

Costo estimado sin medir: unos 1047 × 32 = 33,5 mil recortes por HMR2 más la lectura de video;
el entrenamiento es el de E5. Se corrige con la prueba de humo.
