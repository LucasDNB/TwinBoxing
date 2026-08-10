# BoxTwin

  26-04 Instalacion de controladores y dependencias
  1. MiniForge3 conda
  2. CUDA toolkit
  3. ultralytics opencv numpy shapely matplotlib jupyterlab 

  12-05 Nombre cambiado a BoxTwin y testeos de dataset de terceros (Experimento 1)
  1. Entorno paralelo boxtwin_mmaction (PyTorch 2.1 + CUDA 11.8, stack OpenMMLab 2.1.0 pinneado)
  2. Fine-tuning del modelo sobre dataset Bhargav (270 train / 71 val, 6 clases de golpes)
  3. Mejor checkpoint: epoch 8, 84.51% top-1 (paper reporta 87.32%)
  4. Demo realizada sobre videos de terceros
  5. Se deben realizar mas testeos

10-08 Verificacion de anotaciones BoxingVI: placas de titulo y calidad de V2

  1. V1 tiene 209 clips sin persona (181 placas de titulo enteras, 28 sobre un
     corte), 11.2% del video, todos con etiqueta de golpe
  2. Es el unico video afectado: los otros ocho no tienen ningun clip por encima
     del corte de fraccion de negro
  3. Resultado negativo: un primer detector por brillo medio con umbral global
     marco 528 de 810 clips de V3 como placas, todos falsos positivos. V3 es
     metraje real de estudio con fondo oscuro. El brillo medio no distingue
     video oscuro de pantalla negra y no transfiere entre videos
  4. boxingvi_placas.py mide fraccion de negro y movimiento, y exige dos
     poblaciones separadas antes de contar
  5. V2 reanotado: 42% de los clips no contienen golpe, o sea la segmentacion
     temporal tambien esta rota, no solo las clases
  6. V2 acierta la etiqueta en 12.7% de los usables, por debajo del azar de 6
     clases (16.7%)

13-05 Experimento 2: Smoke test del pipeline base sobre video de entrenamiento
  1. Pipeline YOLOv8l-pose + BoT-SORT sobre 3 formatos (bolsa, sombra, sparring)
  2. Bolsa y sombra: 1 ID único, 0 falsos positivos, pipeline base suficiente
  3. Sparring: 46 IDs únicos confirma que la oclusión mutua entre peleadores es el problema central
  4. Performance >32 FPS en los tres formatos (target del proyecto: 25-30 FPS)
  5. Justifica experimentalmente la lógica de dominio (ROI + top-2 + Hungarian) declarada para Fase 1
