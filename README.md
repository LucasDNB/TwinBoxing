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

13-05 Experimento 2: Smoke test del pipeline base sobre video de entrenamiento
  1. Pipeline YOLOv8l-pose + BoT-SORT sobre 3 formatos (bolsa, sombra, sparring)
  2. Bolsa y sombra: 1 ID único, 0 falsos positivos, pipeline base suficiente
  3. Sparring: 46 IDs únicos confirma que la oclusión mutua entre peleadores es el problema central
  4. Performance >32 FPS en los tres formatos (target del proyecto: 25-30 FPS)
  5. Justifica experimentalmente la lógica de dominio (ROI + top-2 + Hungarian) declarada para Fase 1
