# E4, E4a, E5 y F · Clasificador geométrico y 3D, dejando una fuente afuera

16-10-2026 · spec: `docs/specs/2026-10-16-clasificador-geometrico/` (00 a 03), pre-registrado en
`703fd11d8a3f9a74d871777357491ba8993cd9ba` · scripts `scripts/clasificador_geo/` (`datos.py`,
`features2d.py`, `modelo.py`, `entrenar.py`, `regla_c8.py`, `evaluar.py`, `elevar3d.py`,
`features3d.py`) · datos en `datos/2026-10-16-clasificador-geo-E4.json`, `-E4a.json`, `-E5.json`,
`-F.json`, `-regla_c8.json` y `-veredictos.json` · registro de tareas y decisiones en
`scripts/clasificador_geo/TAREAS.md`

E6 (rol por brazo) va aparte, en `2026-10-16-rol-por-brazo.md`.

## 1. Qué se mide (pre-registrado)

- P5 (E4): si un modelo chico sobre las features geométricas del detector generaliza mejor que
  PoseConv3D en familia, y si le gana en altura a la regla de C8.
- P6 (E5): si sumar features 3D de MotionBERT en un marco del cuerpo mejora la familia sobre
  E4, en particular el hook.
- Sub-hipótesis 1: exactitud de ocho clases, con la lectura a/b/c del 14-10 (`d359247`).
- F y E4a: sin umbral.

| | Entrada por cuadro | Modelo |
|---|---|---|
| E4 | 20 features del brazo que golpea (`features_de` del detector) + 6 del rival = 26 | `ClasificadorGolpe` |
| E4a | Solo las 20 del brazo | igual (ablación) |
| E5 | Las 26 de E4 + 16 features 3D = 42 (ver §2, punto 2) | igual |
| F | Promedio de probabilidades de E4 y E3 (PoseConv3D con rival, `pred.pkl` del 15-10) | sin entrenamiento |

Fijo para E4, E4a y E5: los mismos 1047 golpes completos y los mismos siete folds que E1 a E3
(mismo conjunto de `frame_dir`, verificado por `datos.py`); ventana del inicio menos 8 al fin
más 8 cuadros, remuestreada a 32 pasos; tres bloques convolucionales residuales dilatados con
pooling enmascarado y dos cabezas (tipo y altura); Adam, lr 1e-3, weight decay 1e-4, batch 64,
60 épocas, sin early stopping ni validación; pesos de clase inversos a la frecuencia del fold
de entrenamiento; estandarización ajustada en el fold de entrenamiento; semillas 0 a 4.
Número primario: ensamble de las cinco semillas, folds concatenados.

Lectura fijada en el pre-registro §3: P5 familia, E4 mejora a E1 si la supera en 0,03 o más y
gana en al menos 5 de 7 folds; P5 altura, E4 le gana a la regla de C8 recalculada sobre los
mismos 775 golpes si su balanceada la supera en 0,03 o más y gana en al menos 3 de las 4
fuentes, con C8b al lado; P6, la misma regla que P5 familia, E5 contra E4; si el desvío entre
semillas de la familia pasa de 0,03, una mejora se informa como no concluyente.

## 2. Lo que cambió respecto del diseño

Cada punto está en `TAREAS.md`, decisiones, con el motivo.

1. La elevación 3D no usa `inference_pose_lifter_model` de MMPose 1.3.2: esa función arma el
   objetivo con un solo cuadro y falla con secuencias de 243, y el demo eleva cuadro por cuadro,
   que le saca a MotionBERT el contexto temporal. Se usa el mismo pipeline del config y el mismo
   `test_step`, una pasada por golpe con 243 cuadros de contexto centrados en la ventana
   (decisión 7).
2. Desvío del texto literal del pre-registro: E5 entra con 42 features, no con las 44 del §2.
   El pre-registro define las features 3D por referencia al diseño §4, que dice "18 en total" pero
   nombra 16. Se implementaron las 16 nombradas; inventar dos para llegar al número era decidir
   algo de fondo después del pre-registro. Queda como pregunta abierta (decisión 9). El campo
   `que` de los `config.json` de E5 quedó escrito "18 features 3D"; es una etiqueta errónea,
   corregida en `entrenar.py` después de correr, y las features reales que lista cada
   `config.json` son 42.
3. 13 golpes no tienen ningún paso con pose usable. Quedan en la evaluación: el pre-registro
   fija 1047 golpes, y el modelo los clasifica solo con el sesgo (decisión 1).

Nada de esto se decidió mirando un resultado: los tres puntos se resolvieron al implementar,
antes de la primera corrida de la configuración afectada.

## 3. Resultados

Ensamble de cinco semillas, folds concatenados, 1047 golpes. Líneas de base del 15-10: E1
familia 0,656, siempre recto 0,632, altura siempre cabeza 0,50 de balanceada.

### 3.1 Total

| | Familia (Wilson 95%) | Familia balanceada | Recall recto | Recall hook (Wilson 95%) | Recall uppercut |
|---|---|---|---|---|---|
| E4 | 0,727 (0,699–0,753) | 0,643 | 0,804 | 0,620 (0,564–0,673) | 0,506 |
| E4a | 0,724 (0,696–0,750) | 0,635 | 0,816 | 0,583 (0,527–0,638) | 0,506 |
| E5 | 0,738 (0,711–0,764) | 0,623 | 0,844 | 0,590 (0,534–0,644) | 0,435 |

| | Altura exactitud | Altura balanceada | Recall cabeza | Recall cuerpo (Wilson 95%) |
|---|---|---|---|---|
| E4 | 0,832 | 0,702 | 0,896 | 0,509 (0,435–0,582) |
| E4a | 0,768 | 0,613 | 0,844 | 0,382 (0,312–0,456) |
| E5 | 0,828 | 0,679 | 0,902 | 0,457 (0,384–0,531) |

Dispersión entre semillas (media y desvío de cada semilla con sus siete folds):

| | Familia | Altura balanceada | Ocho clases |
|---|---|---|---|
| E4 | 0,706 ± 0,015 | 0,686 ± 0,005 | 0,597 ± 0,018 |
| E4a | 0,703 ± 0,005 | 0,619 ± 0,022 | 0,561 ± 0,010 |
| E5 | 0,711 ± 0,011 | 0,676 ± 0,020 | 0,599 ± 0,010 |

### 3.2 Familia por fold

| Fold (golpes) | E1 | E4 | E4a | E5 |
|---|---|---|---|---|
| Sparring (114) | 0,465 | 0,561 | 0,544 | 0,596 |
| 01-sparring (96) | 0,615 | 0,656 | 0,635 | 0,635 |
| 02-sparring (62) | 0,839 | 0,790 | 0,855 | 0,855 |
| 03-sparring (129) | 0,713 | 0,783 | 0,760 | 0,721 |
| 04-sparring (120) | 0,592 | 0,800 | 0,825 | 0,808 |
| sparring-3-rounds (381) | 0,638 | 0,722 | 0,706 | 0,748 |
| pacquiao_margarito (145) | 0,807 | 0,779 | 0,800 | 0,800 |

### 3.3 Altura en las cuatro fuentes de C8 (775 golpes completos, 128 al cuerpo)

La regla de C8 recalculada sobre estos mismos golpes (`altura_relativa_v1`, τ 0,20, ventana
anotada, sin dato = error): balanceada 0,729, recall de cuerpo 0,828 (Wilson 0,754–0,884),
recall de cabeza 0,629, 51 golpes sin dato.

| Fuente (golpes, al cuerpo) | Regla de C8 | E4 | E5 | E3 (15-10) |
|---|---|---|---|---|
| 03-sparring (129, 17) | 0,689 | 0,872 | 0,847 | 0,663 |
| 04-sparring (120, 23) | 0,775 | 0,768 | 0,752 | 0,693 |
| sparring-3-rounds (381, 70) | 0,727 | 0,675 | 0,674 | 0,551 |
| pacquiao_margarito (145, 18) | 0,712 | 0,675 | 0,683 | 0,748 |
| Las cuatro juntas | 0,729 | 0,719 | 0,713 | 0,612 |

Recall de cuerpo sobre las cuatro: regla 0,828, E4 0,539 (Wilson 0,453–0,623), E5 0,516
(0,430–0,601).

### 3.4 Ocho clases (sub-hipótesis 1)

Total, 1047 golpes: E4 0,623 (Wilson 0,593–0,652), balanceada 0,445; E5 0,632 (0,603–0,661),
balanceada 0,430. Sobre las cuatro fuentes de C8: E4 0,649, E5 0,666. Referencia del 15-10:
E3 0,522.

| Clase (n) | E4 recall | E5 recall |
|---|---|---|
| jab cabeza (427) | 0,808 | 0,845 |
| jab cuerpo (47) | 0,277 | 0,255 |
| cross cabeza (166) | 0,741 | 0,795 |
| cross cuerpo (22) | 0,273 | 0,227 |
| hook cabeza (223) | 0,489 | 0,448 |
| hook cuerpo (77) | 0,390 | 0,325 |
| uppercut cabeza (58) | 0,328 | 0,397 |
| uppercut cuerpo (27) | 0,259 | 0,148 |

### 3.5 Control de calidad del 3D

Coeficiente de variación del largo de los huesos del brazo que golpea a lo largo de cada
ventana (en un esqueleto rígido sería 0), mediana por fuente: antebrazo de 0,084 (04-sparring)
a 0,144 (Sparring), brazo de 0,049 (02-sparring) a 0,080 (03-sparring); p90 del antebrazo hasta
0,312. El 99,7% de los pasos tiene pose 3D. Sin umbral, como dice el diseño. Detalle en
`<trabajo>/datos/pose3d.qc.json`.

## 4. Lectura, punto por punto contra el pre-registro §3

1. P5, familia: mejora. E4 da 0,727 contra 0,656 de E1, una diferencia de 0,071, por encima del
   mínimo de 0,03. Gana en 5 de los 7 folds (Sparring, 01, 03, 04 y sparring-3-rounds), justo
   el mínimo; pierde en 02-sparring y en pacquiao_margarito. El desvío entre semillas de la
   familia es 0,015, debajo de 0,03, así que no es no concluyente.
2. P5, altura: no hay diferencia demostrada. Sobre los 775 golpes de C8, E4 da balanceada 0,719
   contra 0,729 de la regla, y gana solo en 03-sparring. C8b no se cumple: balanceada 0,719 y
   recall de cuerpo 0,539, contra 0,75 y 0,60. E4 queda cerca de la regla en balanceada con un
   perfil distinto: no deja golpes sin dato (la regla deja 51, que cuentan como error) y acierta
   muchos menos cuerpos (0,539 contra 0,828).
3. P6: no hay diferencia demostrada. E5 da 0,738 contra 0,727 de E4: diferencia 0,011, debajo
   del mínimo, aunque gana en 5 de 7 folds. La expectativa declarada era que, si algo mejoraba,
   fuera el hook, y no pasó: recall de hook 0,590 en E5 contra 0,620 en E4, con los intervalos
   solapados.
4. Sub-hipótesis 1, lectura a. Ocho clases queda por debajo de 0,80 en E4 (0,623) y en E5
   (0,632). La parte de golpes no alcanza el umbral. Lectura c: jab y cross salen del rol
   anotado, lo que supone un detector que acierta el brazo siempre.
5. D9 no se activó: ningún número llegó a 0,90 de familia ni a 0,80 de ocho clases.

## 5. F y E4a, sin umbral

- F, el promedio de E4 con E3, da familia 0,741 (Wilson 0,714–0,767), altura balanceada 0,671 y
  ocho clases 0,642. Como chequeo del cruce, E3 recalculado desde sus `pred.pkl` dentro de F da
  0,667 de familia y 0,605 de altura, iguales al 15-10.
- E4a, sin las 6 features del rival, da familia 0,724, casi lo mismo que E4, y altura
  balanceada 0,613 contra 0,702. Las features del rival son las que llevan la altura, no la
  familia.

## 6. Observación posterior, sin umbral

Lo que sigue apareció al mirar los números y no estaba en la lectura del pre-registro. No
decide nada.

1. Ensamble contra semilla. El número primario de E4 es el ensamble de cinco semillas; el de E1
   fue de una sola, por la decisión del 14-10. Ninguna semilla suelta de E4 llega al ensamble:
   van de 0,685 (s1) a 0,721 (s4), con media 0,706. Contra el 0,656 de E1, cada semilla suelta
   queda entre 0,029 y 0,065 arriba. La regla se aplicó como estaba escrita, con el ensamble;
   comparada semilla contra semilla, la mejora no es robusta: según el recálculo de la auditoría
   (§7), la regla completa (≥ 0,03 y 5 de 7) se cumple en 3 de las 5 semillas; s1 queda en
   +0,029 y s2 gana solo 4 de 7 folds.
2. El 3D no ayuda al hook. Era la razón de E5. Sin el umbral no se puede decir que lo empeore,
   pero no hay rastro de la mejora esperada.
3. La regla de C8 y E4 se equivocan en golpes distintos: la regla pierde la cabeza (51 sin dato,
   recall de cabeza 0,629) y E4 pierde el cuerpo (0,539). No se probó ninguna combinación de las
   dos; sería otro pre-registro.
4. MotionBERT se entrenó con H36M a 50 fps y acá recibe del orden de 30 (Pacquiao, con un
   cuadro de cada dos): los 243 cuadros cubren unos 8 s en vez de 4,9. Es lo que pide el diseño
   §4.1, pero es un corrimiento de dominio que puede explicar parte de que E5 no ayude. Los
   cuadros rellenados entran con visibilidad 1, igual que en MMPose.
5. Las ocho clases de E3 recalculadas dentro de F dan 0,526 y no el 0,522 del 15-10: acá tipo y
   altura salen de argmax separados (sumando probabilidades), allá del argmax sobre las clases
   juntas. Las cifras de F de ocho clases usan la primera lectura.
6. Escala del modelo. Las 70 corridas de E4 y E4a tardaron unos 3 s cada una en la 2080; las
   21 de E1 a E3 tardaron entre 3296 y 4703 s cada una.

## 7. Auditoría

Auditor independiente con las instrucciones de `auditor-medicion`, adaptadas a esta campaña,
16-10-2026, sobre los resultados (no sobre este texto). Once ítems, ninguna falla:

| # | Resultado | Evidencia |
|---|---|---|
| 1 | PASA | Pre-registro en un solo commit, `703fd11` (02:36:28 UTC, un archivo), sin cambios hasta HEAD; `golpes.npz` 02:39:53, primer `pred.npz` 02:42:02. |
| 2 | PASA | `huellas.py verificar`: anotaciones intactas. |
| 3 | PASA | 1047 golpes, conteos por fuente del pre-registro, mismo conjunto de `frame_dir` que los folds de E1, etiquetas consistentes. |
| 4 | PASA | 105 corridas; cada `pred.npz` trae solo los golpes de su fuente; `n_train` = 1047 − fuente; estandarización y pesos de clase solo con el fold de entrenamiento. |
| 5 | PASA | Hiperparámetros iguales al §2 en los 105 `config.json`; una corrida registra el commit anterior a `regla_c8.py`, mismo código de entrenamiento. 42 features en E5: desvío declarado (§2, punto 2). |
| 6 | PASA | Recálculo con numpy: E4 familia 0,7268 y por fold, altura C8 0,7185, cuerpo 0,5391, ocho clases 0,6227, desvío 0,0150; E5 0,7383. |
| 7 | PASA | Veredictos P5 familia, P5 altura, C8b, P6 y sub-hipótesis 1 iguales a `veredictos.json`. Por semilla, E4 cumple la regla en 3 de 5. |
| 8 | PASA | Regla de C8: 775, 128, τ 0,20, sin dato = error, balanceada 0,7286. |
| 10 | PASA | E3 recalculado dentro de F = 0,6667 y 0,6048. |
| 11 | PASA | Pesos con su sha256; `data_info` igual al de MMPose salvo la forma del objetivo, que no toca la predicción; el 3D correlaciona 0,976 con el 2D de la misma ventana; features 3D recalculadas con error máximo 3e-6. |

(El ítem 9 es E6 y está en su documento.)

## 8. Consecuencias

- E4 es el primer clasificador de tipo del proyecto que mejora a PoseConv3D fuera de
  distribución según una regla escrita antes de medir. No reemplaza nada en producción: el
  pre-registro §4.3 lo prohíbe en esta campaña, y el cambio de checkpoint en la Fight-Card sería
  otro spec.
- La altura en la Fight-Card sigue con la regla de C8: E4 no la supera.
- E5 no justifica sumar MotionBERT al pipeline.
- Propuesta de entrada para el README raíz (lo pide confirmado, no se tocó):

```
16-10 Clasificador geometrico y 3D (E4 a E6), una fuente afuera

1. E4, modelo chico sobre las 20 features del detector y 6 del rival: familia 0,727 contra
   0,656 de PoseConv3D (E1), 5 de 7 folds, mejora segun el pre-registro
2. Altura de E4 0,719 de balanceada contra 0,729 de la regla de C8: no la supera
3. E5 con 3D de MotionBERT: 0,738, no mejora a E4; el hook no sube
4. Ocho clases 0,623 (E4) y 0,632 (E5): debajo de 0,80
5. E6, rol por brazo del detector y guardia anotada: 0,985 (techo) contra 0,726 de E1
```
