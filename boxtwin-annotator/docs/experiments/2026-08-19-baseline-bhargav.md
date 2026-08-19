# El baseline de 84,51% no mide generalización

**19-08-2026 · PoseConv3D fine-tuneado sobre Bhargav, evaluado sobre Sparring.mp4.**

## Qué se probó

Antes de invertir cinco horas más de anotación, la pregunta barata: ¿el modelo que ya está
entrenado funciona sobre material propio? Los 125 eventos anotados se exportaron en el
espacio de 6 clases lead-rear, que coincide con las seis de Bhargav, y se pasaron por
`best_acc_top1_epoch_8.pth` sin reentrenar nada.

## Resultado

| Corrida | top-1 |
|---|---|
| **Control**: val de Bhargav, mismo checkpoint | **0,8451** |
| Sparring, clips tal cual (mediana 10 cuadros) | **0,1053** |
| Sparring, con relleno a mediana 50 cuadros | 0,0526 |
| Sparring, con relleno a mediana 86 cuadros | 0,0263 |

Con 6 clases el azar es 0,167. **El modelo rinde por debajo del azar sobre material nuevo.**

El control reproduce el 84,51% al decimal, así que el arnés de evaluación es correcto: el
checkpoint, el config y el pipeline funcionan. El resultado sobre Sparring no es un error de
montaje.

Las predicciones no se distribuyen al azar, se concentran: 43 de 114 caen en `lead_UC`, una
clase que en los datos reales tiene 2 ejemplos. Es el colapso típico de un modelo ante
entrada fuera de distribución. La mejor permutación de etiquetas posible daría 32,5%, así que
tampoco es un problema de orden de clases.

## Por qué

El split de Bhargav es **por clip, no por sujeto**:

| | |
|---|---|
| Sujetos en train | 67 |
| Sujetos en val | 49 |
| **Sujetos en los dos** | **47, o sea el 96% de los de validación** |

No hay clips repetidos entre train y val, así que no es fuga trivial. Pero las mismas
personas, en el mismo gimnasio y con la misma cámara, están de los dos lados. El 84,51% mide
qué tan bien reconoce el modelo golpes de gente que ya vio. No mide generalización a una
persona nueva, y nunca lo midió.

Sparring.mp4 es la primera prueba genuinamente fuera de distribución: otras personas, otro
gimnasio, otra cámara, 640×360 contra 1080p, y dos peleadores en clinch en vez de uno.

## Diferencias de dominio medidas

| | Bhargav | Sparring |
|---|---|---|
| Cuadros por clip (mediana) | 86 | 10 |
| Resolución | 1920×1080 | 640×360 |
| Alto del esqueleto / alto de imagen | 0,30 | 0,56 |
| Keypoints con score > 0,3 | 1,00 | 0,84 |

La longitud del clip se descartó como causa: igualarla empeora el resultado. Las otras tres
quedan como candidatas y no se separaron entre sí.

## Qué se puede afirmar y qué no

**Se puede afirmar**: el 84,51% no es evidencia de un clasificador de golpes que generalice,
porque su validación comparte el 96% de los sujetos con el entrenamiento. Y el modelo, tal
como está, no sirve sobre material nuevo.

**No se puede afirmar** cuál de las diferencias de dominio pesa más, ni que el problema sea
irrecuperable. No se probó reentrenar con los datos propios, que es lo que corresponde
hacer.

## Consecuencia

Es el mismo error metodológico que ya se había encontrado en BoxingVI y que motivó hacer el
split por video: acá está en el propio baseline del proyecto. El número no se puede llevar al
capítulo como línea de base de un clasificador funcionando; se lleva como lo que es, una
medición en distribución sobre 71 clips con sujetos compartidos.

Y cambia el rol del dataset propio. Deja de ser un complemento del público y pasa a ser la
base: sin él no hay ningún número de generalización, ni propio ni ajeno.

## Reproducir

```bash
conda activate boxtwin_mmaction
cd ~/Proyectos/TwinBoxing/test-poseConvBoxing/mmaction2
python tools/test.py <config> work_dirs/punch_recognition_checkpoints/best_acc_top1_epoch_8.pth
```

El pkl de Sparring sale de:

```bash
boxtwin-annotator export videos/Sparring.mp4 --format mmaction \
  --label-space lead-rear --classes 6 --persons attacker
```

y hay que convertirlo de `{split, annotations}` a lista pelada, que es lo que espera el
config del entrenamiento.
