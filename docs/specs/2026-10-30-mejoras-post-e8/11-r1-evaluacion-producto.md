# R1 · El detector evaluado con la máscara del producto (pre-registro)

1-11-2026. El commit de este archivo es el pre-registro. Correcciones solo antes de la primera
corrida de R1-S2, anotadas como decisión en `10-pasos.md`.

## 1. El problema, medido

La auditoría de E12 (`docs/experiments/2026-11-01-e12-guantes-en-detector.md` §3.2) encontró que
la evaluación del detector (`evaluacion.barrer`, llamada desde `folds`) decodifica con `valido`
igual al tramo anotado, sin mirar si en cada cuadro hay pose. Entre el 13% y el 21% de los
cuadros del tramo no tienen pose usable y entran al modelo como un vector de ceros. El modelo
nunca recibe pérdida ahí, pero puede marcar golpes, y esas marcas cuentan como predicciones. Lo
que responde ante ese vector depende de la inicialización (0,015 a 1,0 entre semillas en un
mismo fold), y el ensamble lo promedia: si el promedio queda sobre el umbral, el ensamble marca
casi todos los tramos sin pose.

El producto no hace eso: `inferencia.carriles_de` y `detectar` decodifican con
`valido = usable`, donde `usable` es la máscara de `features_de` por la presencia del peleador.
La evaluación medía entonces un detector distinto del que corre, y las comparaciones de E8, E8b,
E10 y E12 quedaron sujetas a ese azar.

## 2. La corrección (fija desde ahora)

1. **La máscara de pose del producto.** `build --mascara-pose` guarda, por carril, `pose = ok &
   valid`: la máscara de `features_de` (con la regla de vecinos y la de extensión) por la
   identidad resuelta, en el timebase del modelo. Es exactamente la `usable` de
   `inferencia.carriles_de`. No saca los amagues, ni los cuadros fuera del tramo anotado, ni la
   pose interpolada: eso es de la pérdida y de la región evaluada, no de dónde opina el producto.
   Apagada, los tensores son los de antes.
2. **La evaluación.** Con la máscara presente, `barrer` decodifica cada carril con `valido =
   tramo anotado ∩ pose del carril`. Lo anotado, los golpes que cuentan y el emparejamiento no
   cambian: solo dónde puede abrir un segmento el decodificador. La heurística de extensión pasa
   por el mismo decodificador con la misma máscara.
3. **Los dos números con el mismo entrenamiento.** `folds` reporta, para cada modelo entrenado,
   la curva de siempre (`ensamble_curva`) y la nueva (`ensamble_curva_pose`). Así cada
   condición tiene un control contra lo que ya se había publicado.

## 3. Condiciones

Las seis condiciones del detector de la campaña, re-armadas con `--mascara-pose` y re-entrenadas
con la configuración de E8 (5 semillas, 7 folds, 40 épocas, alpha 0,5). `folds` no guarda los
modelos, así que no hay forma de re-evaluar sin re-entrenar.

| Condición | Exports | Opciones de build |
|---|---|---|
| base | `~/boxtwin-gemelo/e8/base/exports` | — |
| y26 | `~/boxtwin-gemelo/e8/y26/exports` | — |
| y26cal | ídem y26 | `--umbrales` (E8b) |
| v8r | ídem base | `--relleno` |
| y26calr | ídem y26 | `--umbrales --relleno` |
| v8g | ídem base | `--guantes` |

E4 no cambia: clasifica segmentos anotados y no decodifica.

## 4. Controles

| Id | Qué | Si falla |
|---|---|---|
| K-r1a | Con `--mascara-pose`, los tensores de siempre son idénticos a los de la condición sin la opción (K-reuso contra los de E8, E8b, E10 y E12, con `pose` como único arreglo de más) | R1 no se lee |
| K-r1b | La curva vieja de cada condición re-entrenada queda a ±0,02 del F1 en 0,80 que ya se publicó (media de los seis folds comparables) | se declara la diferencia; R1 se lee igual, porque compara dentro del mismo entrenamiento |
| K-r1c | La máscara nueva solo puede sacar cuadros marcados: en cada carril y umbral, los cuadros dentro de segmentos predichos con la máscara de pose están contenidos en los de sin ella. (Corregido antes de la primera corrida: la versión anterior pedía predichos ≤, y eso es falso en general, porque la máscara puede partir un segmento largo en dos) | la implementación está mal |

## 5. Reglas de lectura

Las mismas de cada experimento, aplicadas a `ensamble_curva_pose` en lugar de `ensamble_curva`,
con la base re-entrenada en las mismas condiciones:

- **P12a-R1** (E8): y26 contra base, la regla de P12a.
- **P13a-R1** (E8b): y26cal contra base, la regla de P13a (mejora, iguala, peor).
- **P14a-R1** (E10): v8r contra base, la regla de P14a; **P14c-R1**: y26calr menos y26cal.
- **P16a-R1** (E12): v8g contra base, la regla de P16a.

Al lado, sin decidir: la curva de 0,70 a 0,90 de cada una, y la diferencia entre la curva
vieja y la nueva por condición (cuántas predicciones sacaba la máscara).

## 6. Qué decide cada resultado

- Los veredictos de E8, E8b, E10 y E12 publicados con la regla vieja **no se borran**: quedan
  como lectura pre-registrada, con este aviso. Los que van a la tesis son los de R1, porque
  miden al detector como corre.
- Si un veredicto cambia de categoría, el documento del experimento lo dice en su primera línea
  y sus cambios de tesis se reescriben.
- El número de referencia del detector (F1 0,635 publicado) se vuelve a informar con la máscara
  del producto, al lado del de antes.
- Nada cambia A: el producto ya decodifica así.

## 7. Auditoría

1. Commit de este archivo anterior a la primera corrida de R1-S2.
2. K-r1a, K-r1b y K-r1c.
3. Recálculo con numpy, sin importar `dataset.py`, de la máscara `pose` de una fuente desde su
   export, comparada con la que escribió el build y con `inferencia.carriles_de` sobre los
   mismos keypoints.
4. Recálculo de los cuatro veredictos desde los `folds.json`.
5. Originales y cachés con el mismo sha256 (`~/boxtwin-mejoras/antes.txt`).

## 8. Pasos

| Id | Qué |
|---|---|
| R1-T1 | `--mascara-pose` en `build` y `pose` en el dataset, apagado por defecto |
| R1-T2 | `barrer` con la máscara por carril; `folds` con las dos curvas |
| R1-T3 | `lectura.py --curva pose` y la etapa R1 en `mejoras.sh` |
| R1-S1 | Pruebas en rojo y después en verde; suites del detector y de `scripts/mejoras` |
| R1-S2 | Build y `folds` de las seis condiciones (GPU, unos 70 minutos, bajo el candado) |
| R1-S3 | Lectura, auditoría, documento y avisos en E8, E8b, E10 y E12 |
