# Modelos entrenados

Los `.pth` no se versionan (están en el `.gitignore` de la raíz): son 8 MB cada uno y se
regeneran corriendo el entrenamiento con el config que sí está acá.

## `poseC3D_sparring3_indist`

PoseConv3D fine-tuneado sobre `sparring-3`, partición en distribución 285/96 estratificada,
semilla 42. Inicializado desde el checkpoint de Bhargav.

**62,5% top-1 contra 37,5% de línea de base**, exactitud balanceada 0,41.

Es el único modelo del proyecto con señal medida, y solo en distribución: los tres folds que
dejan una fuente afuera quedaron en o por debajo de su línea de base. Sobre un video que no
sea `sparring-3` sus predicciones son ruido.

Lo usa `boxtwin-annotator/tools/demo_vivo.py`.
