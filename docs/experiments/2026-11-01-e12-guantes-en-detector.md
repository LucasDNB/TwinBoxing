# E12 · El guante detectado como features del detector de golpes

1-11-2026. Pre-registro: `docs/specs/2026-10-30-mejoras-post-e8/04-e12-guantes-en-detector.md`
(commit `8741a8d`, 30-10), con la decisión 17 (K-mano sobre los cuadros juzgables). Código:
`547d58b`. Datos: `~/boxtwin-mejoras/e12/`. JSON:
`docs/experiments/datos/2026-11-01-e12-guantes-en-detector.json`. Lectura: `scripts/mejoras/lectura.py e12`.

**En una línea: con la regla pre-registrada, el guante no mejora el detector (F1 0,429 contra
0,623), pero esa caída no la causa el guante: viene de un defecto de la evaluación del detector
que E12 heredó y que también está en E8. Y E12, tal como se diseñó, no llegó a probar su hipótesis.**

> **Aviso del 2-11 (R1).** Re-medido con la máscara del producto (`2026-11-01-r1-evaluacion-producto.md`), el guante iguala a la
> base: 0,601 contra 0,600 (+0,001, 3 de 6), no demostrada. La caída a 0,429 era entera del
> defecto de la evaluación. Lo de §3.2 sigue: E12 tal como se diseñó no prueba su hipótesis.

## 1. Qué se mide (pre-registrado)

El guante detectado, cuadro por cuadro, entra al detector como dos keypoints más (17 y 18), con
cinco features por brazo, y la máscara deja usar un cuadro con la muñeca bajo umbral si hay
guante en esa mano (§3). Una sola variable declarada, "la fuente de la posición de la mano",
contra la base v8 de E8. P16a, F1 por evento en 0,80; P16b, recall en golpes con la muñeca bajo
umbral en el pico; P16c, cobertura del guante; P16d, costo.

## 2. Controles

| Id | Resultado |
|---|---|
| K-reuso | Pasa (el de S3): sin `--guantes`, los tensores son los de E8 |
| K-mano | Pasa con la decisión 17: 137 bien, 6 mal y 57 "no se puede saber" sobre 200; sobre los juzgables 0,958 (Wilson 0,912 a 0,981). Sobre los 200 sería 0,685 |
| K-golpes | Pasa: mismos eventos y mismas etiquetas que la base en las siete fuentes |

La decisión 17 se commiteó en el mismo segundo en que arrancó S14, y una corrección de
redacción (sacar "casi todos en clinch", una afirmación sin verificar) entró 5 s después. Las
respuestas de Lucas son anteriores (15:31 UTC).

## 3. Resultados

### 3.1 P16a, detector

F1 del ensamble en 0,80, seis folds comparables: **base 0,6227, v8+G 0,4289** (−0,194), 0 de 6
folds. Recall 0,515 contra 0,487; precisión **0,370 contra 0,869**. La curva de v8+G es plana:
0,437, 0,424, 0,429, 0,429 y 0,439 de 0,70 a 0,90.

### 3.2 Por qué cae (auditoría, no pre-registrado)

Las 20 features de siempre y la máscara de v8+G son idénticas a las de la base: lo único que
cambia son las cinco columnas de guante. La auditoría buscó un bug (alineación temporal,
espejado, dimensión de entrada, estandarizador, escala, etiquetas) y no lo encontró; reprodujo
en GPU el fold sin-03-sparring con los mismos números.

La caída viene de otro lado. Dentro del tramo anotado, entre el 13% y el 21% de los cuadros no
tienen pose usable y entran al modelo como un vector de ceros. El modelo nunca recibe pérdida
ahí, pero la decodificación lo deja marcar golpes en esos cuadros y la evaluación los cuenta. Lo
que el modelo responde ante ese vector depende de la inicialización: en el fold sin-03, la
probabilidad de golpe ante la entrada en cero va de 0,015 a 0,965 entre semillas en la base
(media 0,64) y de 0,57 a 1,0 con guante (media 0,89). Con el umbral en 0,80, el ensamble de la
base anula esas marcas y el de v8+G las deja: 127 de sus 204 golpes predichos caen sobre tramos
sin pose, contra 0 de 75 en la base. Con las columnas de guante en cero en validación, v8+G da lo
mismo (F1 0,414): el guante no está decidiendo.

Si no se deja al detector marcar donde la entrada es toda cero (diagnóstico de un fold, cinco
semillas, no pre-registrado): base 0,624, v8+G 0,637. Sin efecto apreciable, no peor.

### 3.3 La hipótesis de E12 no llegó a probarse

La máscara de §3 no agregó ningún cuadro usable (147788 en las dos condiciones). La asignación
de mano de §2.3 usa la posición esperada de cada mano (`derive_gloves`), que exige codo y muñeca
con score ≥ 0,3; si una mano no está disponible, el guante va a la otra o se descarta. En los
cuadros con la muñeca bajo umbral, que son justo los que E12 quería recuperar, el guante nunca
queda asignado a esa mano (verificado en las siete fuentes). E12 midió entonces "sumar cinco
features de guante donde ya había muñeca", no "el guante lleva la mano cuando la muñeca se
pierde". Por el mismo motivo P16b no puede moverse por el mecanismo previsto, y no se calculó.

### 3.4 P16c y P16d

Cobertura del guante sobre los cuadros con el peleador, por mano: de 0,16 a 0,24 en Sparring, de
0,29 a 0,37 en 01-sparring y de 0,46 a 0,73 en las demás (detalle en el JSON). Costo de la
extracción por cuadro con peleador en la 2080: mediana 12,6 ms, p95 15,4 ms. Sumado a la
proyección de C5 (1,74 a 1,86 veces el tiempo real) da 2,12 a 2,24: por §5, aunque mejorara, no
entra en el producto sin muestrear.

## 4. Lectura contra el pre-registro

- **P16a: no demostrada**, como dice la regla. No se puede escribir que el guante empeora el
  detector ni que lo confunde: la caída es del defecto de §3.2.
- **P16b:** no calculado; no puede probar el mecanismo (§3.3).
- **P16e:** no se corrió.

## 5. Lo que hay que declarar

1. **El defecto de la evaluación afecta a E8, E8b y E10**, no solo a E12: la precisión de 0,869 de
   la base depende de que, en cada fold, el promedio de las semillas ante la entrada en cero haya
   quedado debajo de 0,80, y eso es azar de inicialización. YOLO26 con 0,3 (E8) tiene más cuadros
   sin pose, así que su caída de 0,623 a 0,433 puede estar inflada por lo mismo. Antes de citar
   esas comparaciones hay que revisarlas.
2. La regla de asignación de mano de §2.3 hace inerte el cambio de máscara de §3.
3. El diagnóstico sin cuadros en cero es post hoc, de un fold y no reemplaza a P16a.
4. La semilla 42 es la mejor en los 7 folds de las dos condiciones, y es la que menos responde a
   la entrada en cero: el defecto también explica la dispersión entre semillas.

## 6. Auditoría

Auditor independiente, 1-11: los ítems de §8 pasan (pre-registro, K-mano y decisión 17,
asignación de mano recalculada con numpy en las siete fuentes, alineación, espejado, features,
K-golpes, configuración, P16a recalculado, 49 sha256). Encontró el defecto de §3.2 con una
corrida de diagnóstico de un fold y cinco semillas por condición, unos 4 minutos de GPU, más de lo
que se le había autorizado (una semilla), y lo declaró. Sus scripts quedaron en
`~/boxtwin-mejoras/e12/auditoria/`.

## 7. Entrada para el README raiz

```
01-11 E12: el guante detectado como features del detector de golpes

1. Guante por cuadro en las siete fuentes; K-mano 137 de 143 juzgables (0,958)
2. Detector, F1 en 0,80: 0,429 contra 0,623, 0 de 6 folds: no demostrada
3. La caida no es del guante: el detector marca golpes en cuadros sin pose (entrada en cero)
   y la evaluacion los cuenta; defecto heredado, tambien en E8
4. Sin esos cuadros (diagnostico de un fold): 0,637 contra 0,624, sin efecto
5. La regla de asignacion de mano dejo inerte el cambio de mascara: la hipotesis no se probo
6. Costo 12,6 ms por cuadro: con C5 da 2,12 a 2,24 veces, no entra sin muestrear
7. Documentado en docs/experiments/2026-11-01-e12-guantes-en-detector.md
```
