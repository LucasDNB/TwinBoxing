# BM · Prueba académica con el modelo de BoxMind

23-10-2026 · spec: `docs/specs/2026-10-22-boxmind-prueba/` (00 a 03), pre-registrado en
`4dd1f2e` · scripts `scripts/boxmind_prueba/` · resultados en
`~/boxtwin/boxmind-prueba-2026-10/resultados/` (`veredictos.json`, `resumen.md`, un `.json` por
experimento), fuera de git por las condiciones de uso de BoxingWeb (pre-registro §2) · registro
de tareas y decisiones en `scripts/boxmind_prueba/TAREAS.md` · corrida del 23-10, de 18:05 a
21:41, en la estación

## 1. Qué se mide (pre-registrado)

- V0: si el modelo de atributos de BoxMind (`PoseOnly+TCN+MoE`, clon en `e06add2`) con las poses
  de BoxTwin se acerca en BoxingWeb a lo que reporta el paper. Válida si el F1 macro de técnica
  de BM-0b, mediana de cinco semillas, es ≥ 0,642. BM-0 (su `train.py` sin tocar) va al lado.
- P7: BM-1b (BoxMind entrenado con seis fuentes de BoxTwin) contra E4.
- P8: BM-2 (BoxTwin + BoxingWeb completo) contra BM-1b. Decide el gatillo de la propuesta.
- P9: BM-2 contra E4.
- P10: altura de BM-2 contra la regla de C8 sobre las cuatro fuentes de C8, con C8b al lado.
- Sub-hipótesis 1: ocho clases, lectura a/b/c del 14-10.
- Curva (sin umbral): familia y ocho clases con 0, 25%, 50% y 100% de BoxingWeb.

Regla de familia: mejora si supera a la base por 0,03 o más y gana al menos 5 de 7 folds; con
desvío entre semillas de la familia por encima de 0,03, no concluyente.

## 2. Lo que cambió respecto del diseño

Cada punto está en `TAREAS.md`, decisiones, con el motivo. Ninguno toca una regla del
pre-registro y los cuatro se resolvieron antes de correr la parte afectada.

1. BoxingWeb trae solo `pose_gt.pkl` por ronda, sin `uve.pkl`: la identidad roja y azul sale de
   `pose_gt` como dice B3, y la comparación contra el seguidor automático no se hizo
   (decisión 3). El paquete tiene 6910 eventos en 50 rondas; 6522 pasan el filtro de largo.
2. `boxtwin.preprocess` busca `yolov8l-pose.pt` en el directorio de trabajo: BM4 corrió con cwd
   en la raíz del repo, con el checkpoint de producción (decisión 4).
3. BM5 y BM6 corrieron con `--dispositivo cuda:0`, porque la espera de GPU del elevador contaba
   como ocupada la GPU que usaba BM4 (decisión 5).
4. `armar.py` agotaba la memoria con BoxingWeb por releer `P3` en cada golpe; se arregló la
   lectura con prueba (decisión 6). El conjunto de BoxTwin no cambia.

## 3. Cifras de control

| Paso | Cifra |
|---|---|
| BM2 | 40 rondas de train y 10 de test, como el paper; sin archivo de licencia ni términos |
| BM3 | BM-0 terminó las 12 épocas, código de salida 0, unos 17 min |
| BM4 | 5668 de 6522 golpes usables (86,9%): 4616 de train, 1052 de test; 854 descartados por identidad; cobertura por ronda y esquina de 0,622 a 0,965 |
| BM5 | 5668 golpes, 11.336 elevaciones; sin 3D: 0 atacantes, 294 rivales |
| BM6 | 1047 golpes, 2094 elevaciones; sin 3D: 0 atacantes, 34 rivales |
| BM7 | BoxTwin 1047 golpes (largo máx. 18), BoxingWeb 5668 (largo máx. 29); ninguno sin 2D ni sin 3D |
| BM8 | F1 de técnica de BM-0b por semilla: 0,511 0,478 0,513 0,454 0,471; unos 9 s por semilla |

## 4. Resultados

V0 no es válida, así que, como fija el pre-registro §5.1, todo lo que sigue se informa como
exploratorio, sin veredicto de mejora atribuible "al modelo de BoxMind".

### 4.1 V0, en BoxingWeb

| | F1 técnica | F1 objetivo |
|---|---|---|
| Paper, Tabla S3 (PoseOnly+TCN+MoE) | 0,692 | 0,747 |
| BM-0, su `train.py` y sus poses, última época, una corrida | 0,617 | 0,712 |
| BM-0b, poses de BoxTwin, mediana de cinco semillas | 0,478 | 0,641 |
| Piso de V0 | 0,642 | — |

### 4.2 Sobre los 1047 golpes de BoxTwin, dejando una fuente afuera

Ensamble de cinco semillas, folds concatenados. BM-1 no ve BoxTwin: entrena solo con BoxingWeb y
se prueba sobre los 1047.

| | Familia (Wilson 95%) | Familia balanceada | Recall recto | Recall hook (Wilson 95%) | Recall uppercut | Altura balanceada | Recall cuerpo | Ocho clases (Wilson 95%) |
|---|---|---|---|---|---|---|---|---|
| E4 | 0,727 (0,699–0,753) | 0,643 | 0,804 | 0,620 (0,564–0,673) | 0,506 | 0,702 | 0,509 | 0,623 (0,593–0,652) |
| BM-1b | 0,672 (0,643–0,700) | 0,454 | 0,864 | 0,417 (0,362–0,473) | 0,082 | 0,570 | 0,197 | 0,581 (0,551–0,610) |
| BM-1 | 0,682 (0,653–0,709) | 0,548 | 0,903 | 0,247 (0,201–0,298) | 0,494 | 0,618 | 0,295 | 0,589 (0,559–0,619) |
| BM-2 | 0,678 (0,649–0,706) | 0,559 | 0,885 | 0,263 (0,217–0,316) | 0,529 | 0,609 | 0,272 | 0,584 (0,553–0,613) |
| BM-2-50 | 0,670 (0,641–0,698) | 0,566 | 0,869 | 0,263 (0,217–0,316) | 0,565 | 0,597 | 0,254 | 0,575 (0,545–0,605) |
| BM-2-25 | 0,680 (0,651–0,708) | 0,545 | 0,863 | 0,350 (0,298–0,406) | 0,424 | 0,608 | 0,266 | 0,590 (0,560–0,620) |

Desvío máximo de familia entre semillas: 0,015 o menos en todos.

## 5. Lectura contra el pre-registro

1. **V0: no válida.** BM-0b da 0,478 contra el piso de 0,642. BM-0 tampoco llega: 0,617. La
   regla del §5.1 para ese caso dice que el problema es del código publicado y no de nuestras
   poses. Hay que declarar también lo que la regla no cubre: BM-0b queda 0,139 por debajo de
   BM-0, así que cambiar 4D-Humans por YOLOv8l-pose más MotionBERT también resta. Las dos cosas
   pasan a la vez.
2. **P7: no demostrada.** BM-1b contra E4: −0,054, gana 2 de 7 folds (02-sparring,
   pacquiao_margarito). Recall de hook 0,417 contra 0,620.
3. **P8: no demostrada.** BM-2 contra BM-1b: +0,006, gana 4 de 7 folds. Sumar 5668 golpes de
   BoxingWeb no movió la familia sobre fuentes no vistas, y el recall de hook bajó de 0,417 a
   0,263.
4. **P9: no demostrada.** BM-2 contra E4: −0,049, gana 3 de 7 folds.
5. **P10: no demostrada.** Altura de BM-2 contra la regla de C8 en las fuentes de C8: −0,086,
   gana 0 de 4. C8b no se cumple (balanceada 0,643, recall de cuerpo 0,344).
6. **Sub-hipótesis 1: lectura a.** Ocho clases entre 0,575 y 0,590 en todas las
   configuraciones de BM, por debajo de 0,80; E4 queda en 0,623.
7. **D9 de E4:** ninguna configuración llega a familia 0,90 ni a ocho clases 0,80; no hubo
   búsqueda de fuga.
8. **Curva, sin umbral.** Familia 0,672, 0,680, 0,670 y 0,678 con 0, 1609, 2966 y 5668 golpes de
   BoxingWeb. Es plana dentro del intervalo de Wilson de cualquiera de los puntos.

## 6. Gatillo de la propuesta de dataset propio

**No respaldada** (`evaluar_bm.gatillo`): V0 no es válida, y además P8 no da mejora ni BM-2
llega a 0,80 en ocho clases. Según el pre-registro §6, `02-dataset-propio.md` queda escrita como
alternativa evaluada y no recomendada, y la línea futura principal pasa a ser la segunda cámara.
Con la curva plana, la fase 1 de la propuesta (unos 10.000 golpes) tampoco se justifica con
estos datos.

Lo que esta prueba no separa, y que el pre-registro §7 ya anticipaba: si faltan datos o si
faltan datos de este dominio. BoxingWeb es transmisión olímpica; el dataset propio sería sparring
de gimnasio con celular. Eso solo lo respondería el piloto de la fase 0.

## 7. Lo que hay que declarar

- Nada de BoxingWeb ni de lo entrenado con él entró a git ni al producto. La carpeta de datos
  (`box-mind/`) está dentro del árbol del repo, excluida por `.git/info/exclude`, que es local.
- BM-0 es una sola corrida con la semilla 42 fija de su código: su 0,617 no tiene intervalo.
- No es la comparación del paper (pre-registro §7): mismo esquema de etiquetas y misma
  arquitectura, pero otra pose, otra partición y exactitud de familia dejando una fuente afuera.

## 8. Entrada para el README raíz (aplicada el 25-10, con el análisis de BoxingWeb y la ruta de este documento agregados)

```
23-10 BM: prueba académica con el modelo de BoxMind (exploratoria)

1. V0 no valida: BM-0b 0,478 de F1 de tecnica en BoxingWeb contra piso 0,642;
   el train.py publicado tampoco llega (0,617 contra 0,692 del paper)
2. BM-1b contra E4 en familia: 0,672 contra 0,727, no demostrada
3. Sumar 5668 golpes de BoxingWeb: 0,678 contra 0,672, no demostrada; curva plana
4. Propuesta de dataset propio: no respaldada, la linea futura principal es la segunda camara
```

## 9. Análisis posterior de BoxingWeb (24-10, sin umbral)

`scripts/boxmind_prueba/analisis_boxingweb.py` (BM12), salida en
`~/boxtwin/boxmind-prueba-2026-10/analisis_boxingweb.json`. Se hizo después de BM10 y no cambia
ninguna lectura de arriba.

1. 50 rondas, 151,5 min, 6872 golpes. Seis fps distintos: 30 (27 rondas), 60 (11), 24 (9), 25,
   29,91 y 29,97. 316 golpes tienen fin ≤ inicio, 311 de ellos de un solo cuadro.
2. Los índices están en los fps nativos. El último evento cae entre el 93% y el 100% del largo
   de la pose en toda tasa, y en los rectos a 60 fps la mano de la etiqueta coincide con la
   muñeca que se extiende en 0,818 con los índices tal cual y en 0,516 con los índices
   duplicados.
3. La ventana anotada mide lo mismo en cuadros y no en tiempo. La mediana es de 11 a 13 cuadros
   en toda tasa: 500 ms a 24 fps, 400 ms a 30 y 183 ms a 60. El filtro de 4 a 30 cuadros de
   BoxMind descarta el 12,9% a 24 fps y el 1,1% a 60.
4. La mano de la etiqueta coincide con la muñeca que se extiende en 0,724 de los rectos, 0,632
   de los hooks y 0,554 de los uppercuts. La misma heurística sobre BoxTwin, con el brazo
   anotado por el autor, da 0,707, 0,547 y 0,494. Con esta medida no hay evidencia de que la
   mano de BoxingWeb esté peor anotada que la de BoxTwin.
5. La partición de BoxingWeb es por ronda: ningún combate queda partido, pero 7 de los 18
   peleadores de test aparecen en train.
