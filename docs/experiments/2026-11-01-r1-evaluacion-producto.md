# R1 · El detector evaluado con la máscara del producto

1-11-2026. Pre-registro: `docs/specs/2026-10-30-mejoras-post-e8/11-r1-evaluacion-producto.md`
(commit `b74c421`, 16:00 UTC; la primera corrida empezó 16:05), con la decisión 18 (K-r1c, antes de correr). Código: `fca0be8` (rama
`r1-evaluacion-producto`). Datos: `~/boxtwin-mejoras/r1/`. JSON:
`docs/experiments/datos/2026-11-01-r1-evaluacion-producto.json`. Lectura: `scripts/mejoras/lectura.py r1`.

**En una línea: medido como corre el producto, el detector base queda en F1 0,600 (no 0,623), el
guante deja de hundirlo (0,601 contra 0,600), el relleno es la mejor condición sin llegar a la
regla (+0,019, 5 de 6) y YOLO26 calibrado sigue igualando, ahora por debajo (−0,025).**

## 1. Qué se mide

La evaluación del detector dejaba al modelo marcar golpes en cuadros sin pose usable, que entran
como un vector de ceros; el producto decodifica solo sobre los usables. R1 re-arma las seis
condiciones de la campaña con `--mascara-pose`, las re-entrena con la configuración de E8 y saca
de cada entrenamiento las dos curvas: la de siempre y la que decodifica con `tramo ∩ pose`. Los
veredictos se leen sobre la segunda, con las reglas de cada experimento.

## 2. Controles

| Id | Resultado |
|---|---|
| K-r1a | Pasa en las seis: 7 archivos por condición, 0 diferencias fuera de `pose` |
| K-r1b | Pasa en las seis: la curva vieja re-entrenada repite el F1 publicado a menos de 0,0001 (base 0,6227, y26 0,4333, y26cal 0,6255, v8r 0,6276, y26calr 0,6438, v8g 0,4289). El entrenamiento es determinista |
| K-r1c | Pasa: la prueba de `test_evaluacion.py` y, en la auditoría, 7056 casos de carril por umbral sobre los datos reales con probabilidades sintéticas, 0 violaciones |

## 3. Cuánto sacaba la máscara

F1 en 0,80, media de los seis folds comparables, y predicciones sumadas en los seis:

| Condición | F1 viejo | F1 producto | Predichos | Emparejados | Precisión |
|---|---|---|---|---|---|
| base (v8) | 0,623 | 0,600 | 512 → 467 | 452 → 421 | 0,869 → 0,885 |
| y26 | 0,433 | 0,479 | 1106 → 352 | 447 → 314 | 0,404 → 0,878 |
| y26cal | 0,626 | 0,575 | 561 → 464 | 470 → 404 | 0,844 → 0,858 |
| v8r | 0,628 | 0,619 | 514 → 501 | 452 → 441 | 0,864 → 0,866 |
| y26calr | 0,644 | 0,616 | 564 → 509 | 483 → 445 | 0,856 → 0,868 |
| v8g | 0,429 | 0,601 | 1320 → 479 | 472 → 421 | 0,370 → 0,866 |

Dos comportamientos distintos. En y26 y en v8g el ensamble marcaba casi todos los tramos sin
pose (más de 700 y 800 predicciones de más, casi todas falsas): ahí la evaluación vieja medía el
azar de la inicialización, y la máscara les devuelve la precisión de las demás. En las otras
cuatro pasa lo contrario: buena parte de lo que la máscara saca estaba emparejado con un golpe
real (31 de 45 en la base). El golpe hace perder la pose, y el modelo, sin entrada, acertaba
igual por el contexto. El producto pierde esos aciertos, así que su recall es menor que el
informado: 0,456 contra 0,487 en la base.

Con la máscara, el mejor umbral baja en todas las condiciones: el F1 en 0,70 supera al de 0,80
en las seis (base 0,645 contra 0,600). No decide nada; queda para cuando se revise el umbral
del producto.

## 4. Veredictos (curva del producto, en 0,80)

| Regla | Comparación | Diferencia | Folds | Veredicto R1 | Veredicto publicado |
|---|---|---|---|---|---|
| P12a-R1 (E8) | y26 − base | −0,121 | 0 de 6 | no demostrada | no demostrada (−0,189) |
| P13a-R1 (E8b) | y26cal − base | −0,025 | 0 de 6 | iguala | iguala (+0,003) |
| P14a-R1 (E10) | v8r − base | +0,019 | 5 de 6 | no demostrada | no demostrada (+0,005, 3 de 6) |
| P14c-R1 (E10) | y26calr − y26cal | +0,041 | — | positiva | positiva (+0,018) |
| P16a-R1 (E12) | v8g − base | +0,001 | 3 de 6 | no demostrada | no demostrada (−0,194) |

Ninguno cambia de categoría. Lo que cambia es el contenido:

- **E8b.** Sigue en "iguala", pero el signo se dio vuelta: Y26-cal queda 0,025 debajo de v8 y no
  gana ningún fold. Está a 0,005 del borde de "peor". La frase "el umbral explicaba la mayor
  parte de la caída de E8" se sostiene (−0,121 sin calibrar, −0,025 calibrado); la de que
  "iguala" hay que decirla con el signo.
- **E10.** El relleno pasa a ser la condición que más se acerca a mejorar: +0,019 y 5 de 6
  folds, contra +0,005 y 3 de 6. Encaja con §3: rellenar hace usables cuadros que la máscara
  sacaba. No llega a los 0,03. Sobre YOLO26 calibrado la diferencia se duplica (+0,041).
- **E12.** La caída de −0,194 era entera del defecto: con la máscara, el guante iguala a la base
  (+0,001, 3 de 6). La hipótesis de E12 sigue sin probarse por lo que dice su §3.2 (la mano
  asignada pide muñeca ≥ 0,3).
- **E8.** YOLO26 sin calibrar sigue peor, por menos: −0,121 contra −0,189.

## 5. El número de referencia del detector

El F1 0,635 publicado (ensamble de cinco semillas, seis folds comparables) es de la evaluación
vieja y de otra corrida. En esta campaña la base reproduce 0,623 con la evaluación vieja y da
0,600 con la del producto (recall 0,456, precisión 0,885). Ese es el número que describe al
detector como corre. El 0,635 no se re-midió: se informa al lado, con la diferencia declarada.

## 6. Auditoría

Auditor independiente, 2-11, de solo lectura: **5 de 5 ítems pasan**.

1. El pre-registro (`b74c421`, 16:00:49) es anterior a la corrida (16:05:12). `fca0be8` cambió
   el texto de K-r1c 13 segundos antes de correr: entra como corrección previa, decisión 18.
2. K-r1a revisado con numpy en las seis condiciones: 0 diferencias de tipo, forma o valor fuera
   de `pose`. K-r1b recalculado desde emparejados, golpes y predichos: las seis a menos de
   0,0001. K-r1c sobre base, v8g y y26 con cuatro semillas de probabilidades sintéticas: 7056
   casos, 0 violaciones; la máscara saca un 33% de los cuadros marcados y en 2 casos parte un
   segmento. `barrer` con y sin máscara da los mismos golpes y el mismo emparejamiento.
3. La máscara recalculada en numpy, sin importar `dataset.py` ni `features.py`, en las siete
   fuentes de base (con los dos remuestreos, 59,94 y 29,643 fps): 0 cuadros distintos contra el
   build y 0 contra `inferencia.carriles_de`.
4. Los cinco veredictos recalculados desde los `folds.json` coinciden. P14c pide además que la
   diferencia sobre Y26-cal sea mayor que sobre v8: +0,041 contra +0,019, se cumple.
5. Originales y cachés: los 49 sha256 iguales a `antes.txt`.

Advertencias, que quedan en §7: P13a-R1 cerca del borde, y el alcance de "la máscara del producto"
en las condiciones con umbrales o relleno.

## 7. Lo que hay que declarar

1. R1 corrige la evaluación, no el producto: A ya decodifica así.
2. Los veredictos publicados de E8, E8b, E10 y E12 quedan como lectura pre-registrada, con aviso;
   los que van a la tesis son los de esta tabla.
3. P13a-R1 está a 0,0054 del borde de "peor" y Y26-cal pierde en los seis folds (de −0,015 a
   −0,046). La categoría es "iguala"; la tesis lo dice con el signo y esa consistencia.
4. "La máscara del producto" es literal en base, y26 y v8g. En y26cal y y26calr se calcula con
   los umbrales de E8b y en v8r y y26calr sobre los keypoints rellenados: es la máscara que
   tendría el producto con esa mejora puesta, que hoy `carriles_de` no acepta.
5. En v8g la máscara es idéntica a la de base en las siete fuentes: la relajación de E12 no
   rescata ningún cuadro (lo de su §3.2). E12 cambió solo las features.
6. La recuperación de golpes en cuadros sin pose (31 emparejados de la base) es una capacidad que
   el modelo tiene y el producto no usa. Usarla es otro experimento, con su pre-registro.

## 8. Entrada para el README raíz

```
01-11 R1: el detector evaluado con la mascara de pose del producto

1. La evaluacion dejaba marcar golpes en cuadros sin pose; el producto no
2. Seis condiciones re-entrenadas, curva vieja igual a la publicada en las seis
3. Base: F1 0,600 con la mascara del producto contra 0,623 con la vieja
4. E12: la caida a 0,429 era del defecto; con la mascara el guante iguala (+0,001)
5. E10: +0,019 y 5 de 6 folds, debajo de +0,03: sigue no demostrada
6. E8b: sigue iguala, ahora por debajo (-0,025)
7. Documentado en docs/experiments/2026-11-01-r1-evaluacion-producto.md
```
