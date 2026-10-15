# E1 a E3 · Clasificador por tipo y altura, dejando una fuente afuera

15-10-2026 · spec: `docs/specs/2026-10-14-clasificador-tipo-altura.md` §3, §5 a §7, pre-registrado
en `d359247e39a9184880b85a545d83c7bc3c228931` · scripts `scripts/clasificador/` (`exportar_fuentes.py`,
`armar_folds.py`, `generar_configs.py`, `correr_campana.sh`, `evaluar.py`) · datos en
`datos/2026-10-15-clasificador-E1.json`, `-E2.json` y `-E3.json`

## 1. Qué se mide (pre-registrado en el spec)

- P2: cuánto acierta el clasificador de producción sobre una fuente que no vio. El 0,746 de
  familia es en distribución.
- P3: si cambiar el espacio de clases a tipo por altura, sin lado, mejora la familia y además
  da la altura.
- P4: si sumar al rival en la entrada mejora la altura.
- Sub-hipótesis 1 (Cap 1, apartado 5): exactitud de ocho clases ≥ 0,80.

| | Espacio de clases | Entrada | Variante de export |
|---|---|---|---|
| E1 | 6: tipo por rol (lead-rear), el de producción | atacante | `v6lr_at` |
| E2 | 6: tipo por altura, sin lado | atacante | `v12s_at`, colapsado |
| E3 | 6: tipo por altura, sin lado | atacante y rival | `v12s_amb`, colapsado |

Fijo para las tres: una fuente afuera por vez (siete folds, 1047 golpes completos), 10 épocas
sobre `RepeatDataset` ×10, último checkpoint, sin validación, sin pesos de clase, inicio desde
`best_acc_top1_epoch_8.pth` (el `load_from` de producción, fine-tuning sobre Bhargav, sin
ninguna de las siete fuentes). El config es el de producción, `modelos/poseC3D_7fuentes.py`,
con los cambios de `generar_configs.py`: `ann_file` y `split` del fold, `work_dir`, `load_from`
absoluto, `num_classes` y `seed` (6 y 0, iguales a producción en la semilla 0), sin
`val_cfg`, `val_dataloader`, `val_evaluator`, `val_begin` ni `val_interval`, el hook de
checkpoint guardando solo el último (sin `save_best`), y sin `diff_rank_seed`, que con una GPU
no tiene efecto.

Lectura fijada en el spec §6: P2 con la familia de E1 al lado del 0,746; P3 familia, E2 mejora a
E1 si la diferencia es ≥ 0,03 y gana en al menos 5 de 7 folds; P3 altura, C8b (balanceada ≥ 0,75
y recall de cuerpo ≥ 0,60 sobre las cuatro fuentes de C8); P4, E3 mejora a E2 en altura con la
misma regla que P3 familia; sub-hipótesis 1 con la lectura a/b/c del §6.5.

## 2. Lo que cambió respecto del spec, por decisión de Lucas

1. Una sola semilla. La corrida cronometrada (E1/Sparring/s0) midió 4472 s: 4428 de
   entrenamiento y 44 de prueba. Extrapolado por tamaño de fold da 8,4 h por experimento y
   semilla, unas 75 h para las 63 corridas. Las dos alternativas del §9 también pasaban de
   14 h (25 h solo con la semilla 0, 50 h con tres semillas de dos configuraciones). El
   14-10 Lucas eligió correr solo la semilla 0 de E1, E2 y E3: 21 corridas. El ensamble del
   §6 queda en una semilla y no hay dispersión entre semillas. `evaluar.py` marca
   `completo: false` porque faltan s1 y s2; es la decisión, no un dato que se perdió.
2. Calendario. La campaña corrió de corrido, del 14-10 a las 16:01 ART (corrida cronometrada)
   al 15-10 a las 17:30 ART, y el benchmark de perfiles pasó a correrse después. El §9 lo
   ponía antes y sin la campaña en la GPU; el orden se invirtió, la regla de no compartir la
   GPU se mantuvo.

Transparencia sobre la decisión 1: al revisar la corrida cronometrada, la cola de su `test.log`
mostró el top-1 de ese fold (0,254) antes de que se decidiera el alcance. Ese número no se le
pasó a Lucas, y la decisión se apoya solo en el tiempo medido, como prevé el §9.

Las 21 corridas tardaron entre 3296 y 4703 s cada una (la de sparring-3-rounds, con el fold de
entrenamiento más chico, es la corta). Ninguna falló.

## 3. H-F (verificado antes de medir, spec §3)

Lectura de código del 14-10, detalle con archivo y líneas en `<trabajo>/hf.md`:

1. `Flip` no toca la etiqueta. `processing.py:783-784` resuelve `type='Flip'`; la etiqueta solo
   cambia con `flip_label_map` (`processing.py:890-892`) y el config no lo pasa
   (`modelos/poseC3D_7fuentes.py:217-238` y `271-292`). En prueba cada golpe también se mezcla
   con su espejo (`GeneratePoseTarget` con `double=True`, `:116` y `:164`, y
   `average_clips='prob'`, `:68`).
2. El espejo es anatómicamente consistente: `_flip_kps` (`processing.py:838-850`) invierte x y
   permuta izquierda y derecha. Un jab espejado sigue siendo un jab, el del zurdo. En el
   espacio lead-rear la etiqueta sobrevive al espejo; E2 y E3 colapsan el lado.
3. Producción: jab y cross de la Fight-Card salen del argmax del clasificador sobre sus 6
   clases (`boxtwin-detector/tools/clasificar_sesion.py:43-51`, `:69-87`, `:126-127`) y la
   Fight-Card copia `tipo` tal cual (`boxtwin-annotator/src/boxtwin/mvp/fightcard.py:136-141`
   y `:346-348`). Nada cruza el brazo del detector con la guardia.

El número que contesta si el espejo pesa por una pista absoluta de pantalla es la exactitud de
eje de E1 (§4.3 abajo).

## 4. Resultados

Ensamble de una semilla, folds concatenados, 1047 golpes. Líneas de base: familia siempre recto
0,632, balanceada al azar 1/3, altura siempre cabeza 0,50 de balanceada.

### 4.1 Total

| | Familia (Wilson 95%) | Familia balanceada | Recall recto | Recall hook | Recall uppercut |
|---|---|---|---|---|---|
| E1 | 0,656 (0,627–0,684) | 0,483 | 0,852 | 0,337 | 0,259 |
| E2 | 0,641 (0,611–0,669) | 0,469 | 0,786 | 0,457 | 0,165 |
| E3 | 0,667 (0,638–0,695) | 0,510 | 0,787 | 0,530 | 0,212 |

| | Altura exactitud | Altura balanceada | Recall cabeza | Recall cuerpo (Wilson 95%) |
|---|---|---|---|---|
| E2 | 0,810 | 0,557 | 0,935 | 0,179 (0,129–0,243) |
| E3 | 0,766 | 0,605 | 0,846 | 0,364 (0,296–0,438) |

### 4.2 Por fold

| Fold (golpes, al cuerpo) | E1 familia | E2 familia | E3 familia | E2 altura bal. | E3 altura bal. | E2 recall cuerpo | E3 recall cuerpo |
|---|---|---|---|---|---|---|---|
| Sparring (114, 15) | 0,465 | 0,579 | 0,474 | 0,590 | 0,572 | 0,200 | 0,467 |
| 01-sparring (96, 27) | 0,615 | 0,594 | 0,656 | 0,526 | 0,612 | 0,370 | 0,296 |
| 02-sparring (62, 3) | 0,839 | 0,806 | 0,790 | 0,667 | 0,500 | 0,333 | 0,000 |
| 03-sparring (129, 17) | 0,713 | 0,636 | 0,721 | 0,553 | 0,663 | 0,294 | 0,353 |
| 04-sparring (120, 23) | 0,592 | 0,625 | 0,583 | 0,543 | 0,693 | 0,087 | 0,696 |
| sparring-3-rounds (381, 70) | 0,638 | 0,651 | 0,654 | 0,540 | 0,551 | 0,086 | 0,129 |
| pacquiao_margarito (145, 18) | 0,807 | 0,641 | 0,828 | 0,572 | 0,748 | 0,222 | 0,944 |

### 4.3 Eje (E1)

Exactitud de lead contra rear: 0,656 (Wilson 0,627–0,684), balanceada 0,627. Recall de lead
0,714 (499/699), de rear 0,540 (188/348).

### 4.4 Cuatro fuentes de C8 (775 golpes completos, 128 al cuerpo)

| | Familia | Altura balanceada | Recall cuerpo (Wilson 95%) | Ocho clases (Wilson 95%) |
|---|---|---|---|---|
| E1 | 0,675 | — | — | — |
| E2 | 0,643 | 0,541 | 0,133 (0,085–0,202) | 0,533 (0,498–0,568) |
| E3 | 0,687 | 0,612 | 0,375 (0,296–0,461) | 0,539 (0,504–0,574) |

### 4.5 Ocho clases (sub-hipótesis 1)

Total, 1047 golpes: E2 0,526 (Wilson 0,496–0,556), balanceada 0,295; E3 0,522 (0,491–0,552),
balanceada 0,332. Referencia sin altura: E1 sobre sus 6 clases, top-1 0,467.

| Clase (n) | E2 recall | E3 recall |
|---|---|---|
| jab cabeza (427) | 0,775 | 0,742 |
| jab cuerpo (47) | 0,085 | 0,191 |
| cross cabeza (166) | 0,663 | 0,572 |
| cross cuerpo (22) | 0,182 | 0,364 |
| hook cabeza (223) | 0,363 | 0,404 |
| hook cuerpo (77) | 0,208 | 0,247 |
| uppercut cabeza (58) | 0,086 | 0,138 |
| uppercut cuerpo (27) | 0,000 | 0,000 |

Dispersión entre semillas: no hay, se corrió una sola (§2).

## 5. Lectura, punto por punto contra el §6

1. P2. Fuera de distribución, la familia de E1 es 0,656 (Wilson 0,627–0,684), contra 0,746 en
   distribución. Queda apenas por encima de decir siempre recto (0,632) y su balanceada es
   0,483. El 0,746 no transfiere a una fuente que el modelo no vio.
2. P3, familia. E2 da 0,641 contra 0,656 de E1: no mejora, y menos por el mínimo de 0,03. En la
   tabla 4.2, E2 supera a E1 solo en Sparring, 04-sparring y sparring-3-rounds. No hay
   diferencia demostrada.
3. P3, altura. C8b no pasa: sobre las cuatro fuentes de C8, E2 da balanceada 0,541 y recall
   de cuerpo 0,133, contra 0,75 y 0,60. Denominador: 775 golpes completos con 128 al cuerpo, no
   los 806 de C8, que contaban amagues, abortados y golpes sin dato como error. C8b no es C8 ni
   lo reemplaza. Al lado, E3 tampoco pasaría C8b (0,612 y 0,375).
4. P4. E3 mejora a E2 en altura según la regla: la balanceada total pasa de 0,557 a 0,605,
   más que el mínimo de 0,03, y en la tabla 4.2 E3 supera a E2 en 01-sparring, 03-sparring,
   04-sparring, sparring-3-rounds y pacquiao_margarito; pierde en Sparring y en 02-sparring,
   que tiene 3 golpes al cuerpo. Es una mejora sobre un número bajo, con una
   semilla, y ninguna de las dos configuraciones llega a C8b.
5. Sub-hipótesis 1, lectura a. La exactitud de ocho clases queda por debajo de 0,80 en las dos
   configuraciones (E2 0,526, E3 0,522). La parte de golpes de la sub-hipótesis no alcanza el
   umbral. El número ya supone el rol de jab y cross tomado de la anotación (lectura c): con
   el lado que da el detector sería igual o más bajo.
6. Eje (§6.3). La exactitud de lead contra rear es 0,656. El recall de rear (0,540) es el que
   la baja. Por cómo quedó H-F (§3), el espejo de entrenamiento no contradice la etiqueta en
   este espacio. El número no decide la conjetura de la pista absoluta de pantalla: muestra que
   el eje está lejos de resuelto, no por qué.

## 6. Observación posterior, sin umbral

Lo que sigue apareció al mirar los números y no estaba en la lectura del §6. No decide nada.

1. Eje contra mayoría. Lead son 699 de los 1047 golpes y E1 acierta el eje en 687 (499 lead y
   188 rear): decir siempre lead acertaría más. La balanceada de E1 (0,627) sí está por encima
   de 0,50.
2. El hook. El §1 declaraba de antemano que su recall no se movería en ninguna configuración.
   Se movió: 0,337 en E1, 0,457 en E2 y 0,530 en E3, con los intervalos de E1 (0,286–0,392) y
   de E3 (0,474–0,586) sin solaparse. Es una semilla, el espacio de clases de E1 no es el de
   E2 y E3, y el criterio de los intervalos se eligió después de ver: no contradice las cinco
   mediciones anteriores. Queda anotado como algo a medir, no como resultado.
3. E2 casi siempre dice cabeza. Acierta la altura en 848 golpes (817 a la cabeza y 31 al
   cuerpo), menos que los 874 que acertaría diciendo siempre cabeza, y su balanceada (0,557)
   queda cerca de 0,50.
4. La variación entre folds es grande. En E3 el recall de cuerpo va de 0,000 (02-sparring, 3
   golpes al cuerpo) a 0,944 (pacquiao_margarito, 18). Sin semillas, no se puede separar la
   fuente del azar del entrenamiento.

## 7. Auditoría

`auditor-medicion`, 15-10-2026, en dos pasadas. La primera marcó dos fallas menores en el
texto, ninguna en un número: cuentas hechas en la prosa que no estaban en los JSON, y una
conclusión del §6.2 que iba más allá de lo pre-registrado. Se corrigieron sin tocar la
lectura. La segunda pasada, sobre el texto actual:

| # | Resultado | Evidencia |
|---|---|---|
| 1 | PASA | Spec commiteado en d359247 el 14-10 a las 18:44:54 UTC, sin cambios después; folds 19:01 UTC, primer `pred.pkl` 20:16 UTC, resultados 15-10 21:46 UTC. |
| 2 | PASA | `huellas.py verificar`: anotaciones intactas. |
| 3 | PASA | Cada número del documento está en `resultados/E*.json` o es suma de conteos de ahí con los sumandos a la vista; las copias en `datos/` son idénticas. |
| 4 | PASA | C8b se presenta como distinto de C8, la sub-hipótesis 1 se lee con la lectura a, la observación posterior no concluye, nada se afirma de producción. |
| 8 | PASA | Ningún fold tiene un golpe de la fuente de prueba en entrenamiento; intersecciones 0; 1047 ids únicos por experimento; etiquetas consistentes entre E1, E2 y E3. |
| 9 | PASA | Contra `poseC3D_7fuentes.py` solo cambian fold, `work_dir`, validación quitada, hook de checkpoint y `diff_rank_seed`; `load_from` es el checkpoint de Bhargav del 13-05, sin ninguna de las siete fuentes. |
| 10 | PASA | Los 21 `train.log` cargan el checkpoint sin claves faltantes ni tamaños distintos, 0 líneas de validación, y cada prueba usa su propio `epoch_10.pth`. |
| 11 | PASA | Recálculo independiente desde `pred.pkl`: E1 familia 687/1047 = 0,6562; E2 0,6409, altura 0,5570, ocho clases 0,5263; E3 0,6667, 0,6048, 0,5215; coinciden fuentes de C8 y folds. |
| 12 | PASA | Las líneas citadas de `processing.py`, el config, `clasificar_sesion.py` y `fightcard.py` dicen lo que el §3 afirma. |

En E1 la familia y el eje dan los dos 0,6562 por coincidencia: golpe por golpe, 485 aciertan
las dos cosas, 202 solo la familia, 202 solo el eje y 158 ninguna. Por fold difieren.

## 8. Consecuencias (spec §7.4 y §10)

- No se entrena un modelo final. El §7.4 lo condiciona a que E2 o E3 pasen, y C8b no pasa en
  ninguna. El checkpoint de producción no se toca.
- El Cap 6 actualiza R-01 con el 0,656 de E1: es la primera vez que la generalización se mide
  dejando afuera cada una de las siete fuentes.
- El Cap 7 lleva E1 a E3 con sus números, lo que no pasó incluido. H-F va como hallazgo sobre
  el código, no como explicación de la distancia entre 0,534 y 0,746.
- La altura en la Fight-Card sigue como estaba: rotulada como estimación con la regla de C8.
  Esta campaña no le da un número mejor.
