# Experimento 01: Integración de PoseConv3D Boxing

**Fecha:** 13 mayo 2026
**Objetivo:** Validar PoseConv3D Boxing (Bhargav 2024) como baseline y punto de partida para transfer learning.

## Decisión estratégica
- Rol: baseline + warm-up para transfer learning a modelo propio.
- Manejo del mismatch HRNet vs YOLO-pose: aceptar en baseline, decidir más adelante.

## Stack pinneado (env: boxtwin_mmaction)
- Python 3.10
- torch 2.1.0+cu118 / torchvision 0.16.0+cu118
- numpy 1.26.4
- opencv-python 4.10.0.84
- mmengine 0.10.7
- mmcv 2.1.0
- mmdet 3.2.0
- mmpose 1.3.2
- chumpy 0.70

## Resultados del entrenamiento
- Dataset: Bhargav (270 train / 71 val, 6 clases de golpes)
- 10 épocas completadas (CosineAnnealingLR con T_max=24)
- **Mejor checkpoint: epoch 8, top-1 acc 84.51%, top-5 acc 100%**
- Comparación con paper: 87.32% top-1 reportado → diferencia dentro de ruido estadístico
- Ubicación checkpoint: NO en git (work_dirs/punch_recognition_checkpoints/best_acc_top1_epoch_8.pth, ~8.5MB)

## Demo end-to-end
- Video de prueba: combo personal (274 frames)
- Pipeline: Faster-RCNN → HRNet → PoseConv3D
- Performance: ~8-9 FPS pipeline acumulado (cuello: detección + pose en 2 pases)
- Limitación identificada: script demo clasifica video completo como UNA acción, no detecta golpes en secuencia. Lógica de ventanas deslizantes es trabajo nuestro.

## Implicancias para Fase 1
- Pipeline de inferencia continua: TO-DO propio (no se hereda de Bhargav)
- En producción NO usamos Faster-RCNN+HRNet (latencia incompatible con target 25-30 FPS): vamos con YOLO-pose
- PoseConv3D Boxing queda como: (1) baseline de comparación, (2) warm-up para transfer learning cuando tengamos dataset propio
