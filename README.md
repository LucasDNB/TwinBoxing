# BoxTwin

  26-04 Instalacion de controladores y dependencias
  1. MiniForge3 conda
  2. CUDA toolkit
  3. ultralytics opencv numpy shapely matplotlib jupyterlab 

  12-05 Nombre cambiado a BoxTwin y testeos de dataset de terceros
  1. Entorno paralelo boxtwin_mmaction (PyTorch 2.1 + CUDA 11.8, stack OpenMMLab 2.1.0 pinneado)
  2. Fine-tuning del modelo sobre dataset Bhargav (270 train / 71 val, 6 clases de golpes)
  3. Mejor checkpoint: epoch 8, 84.51% top-1 (paper reporta 87.32%)
  4. Demo realizada sobre videos de terceros
  5. Se deben realizar mas testeos
