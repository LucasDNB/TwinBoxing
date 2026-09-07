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


## `poseC3D_7fuentes` y `poseC3D_7fuentes_pesos`

Reentrenamiento sobre las **siete fuentes**: 779 de entrenamiento y 268 de validación,
partición 75/25 estratificada por fuente y por clase, semilla 42. Contra las 285/96 de una
sola fuente del modelo anterior, y con las guardias ya corregidas —175 eventos tenían jab y
cross intercambiados—.

Las dos configuraciones son idénticas salvo por los pesos de clase.

| | top-1 (6 clases) | familia | straight | hook | uppercut |
|---|---|---|---|---|---|
| sin pesos | 0,534 | **0,746** | 0,946 | 0,487 | 0,200 |
| con pesos | 0,526 | 0,694 | 0,904 | **0,289** | **0,520** |

**Los pesos arreglan el uppercut y empeoran el hook.** Sirven para lo primero: 0,200 a 0,520
sin un dato nuevo. No sirven para lo segundo, y ese es el resultado del experimento.

El acierto por familia del modelo anterior era **0,745**; este da **0,746** con 3,7 veces más
datos. Ver
[`../boxtwin-detector/docs/experiments/2026-09-07-hook-no-es-prior.md`](../boxtwin-detector/docs/experiments/2026-09-07-hook-no-es-prior.md).

Los `.pth` no se versionan. Se regeneran con:

```bash
~/miniforge3/envs/boxtwin_mmaction/bin/python \
    test-poseConvBoxing/mmaction2/tools/train.py modelos/poseC3D_7fuentes.py \
    --cfg-options randomness.seed=42
```

Los `.pkl` de la partición se rearman con el script del experimento; los exports de mmaction
salen de `boxtwin-annotator` con `export --format mmaction --classes 6 --label-space lead-rear`.
