# E8 · YOLO26l-pose contra YOLOv8l-pose, todo lo demás fijo (pre-registro)

27-10-2026. El commit de este archivo es el pre-registro. Se puede corregir hasta antes de H1,
con la corrección anotada como decisión en `05-pasos.md`; después de H1, no se toca.

## 1. Qué se mide

| Id | Pregunta | Tipo |
|---|---|---|
| P12a | ¿YOLO26 mejora el F1 por evento del detector de golpes sobre fuentes no vistas? | con regla |
| P12b | ¿YOLO26 mejora la exactitud de familia del clasificador E4 sobre fuentes no vistas? | con regla |
| P12c | ¿YOLO26 sigue mejor a cada peleador: menos cambios de track y menos tracks mixtos? | sin umbral, dirección declarada |
| P12d | ¿Cuánto da C1 (identidad con siembra simulada) sobre los cachés de YOLO26? | sin umbral |
| P12e | ¿Cuánto tarda y cuánta memoria usa la pose con tracking en la 2080? | sin umbral, dirección declarada |
| SH1 | Ocho clases de E4 sobre YOLO26, lectura a/b/c del 14-10 (`d359247`) | informativa: no cambia el veredicto del 31-10 |

P12c es la pregunta del trackeo: el 20-10, 18 de los 20 errores de C1 salieron de tracks
mixtos. P12a y P12b son las del reconocimiento.

## 2. Las dos condiciones

| | Base (v8) | Y26 |
|---|---|---|
| Pose | `yolov8l-pose.pt` | `yolo26l-pose.pt` |
| Cachés | los originales de los siete proyectos, solo lectura | `~/boxtwin-gemelo/raiz-y26/<dir>/cache/` (`01-percepcion.md` §4) |
| Identidad | las anotaciones originales de la copia principal, solo lectura (`BOXTWIN_RAIZ_FUENTES=~/Proyectos/TwinBoxing`) | las del puente (`BOXTWIN_RAIZ_FUENTES=~/boxtwin-gemelo/raiz-y26`, `01-percepcion.md` §5) |
| Exports, tensores y corridas | `~/boxtwin-gemelo/e8/base/` | `~/boxtwin-gemelo/e8/y26/` |

Las dos condiciones se exportan, se entrenan y se evalúan en la misma tanda, con el mismo commit
(el congelado de S5) y el mismo dispositivo (`cuda`). No se comparan contra corridas viejas: las
viejas sirven solo de control (K2).

Rama D-H2 (solo si H2 falla por el ReID, `01-percepcion.md` §3): la base pasa a ser un re-caché
de v8 con `botsort-sin-reid.yaml`, que también pasa por el puente, e Y26 usa el mismo yaml.

## 3. Controles (se leen antes que las preguntas)

| Id | Qué | Si falla |
|---|---|---|
| K1 | El puente del caché original contra sí mismo reproduce el track de cada peleador en el 100% de los cuadros observados | E8 no se lee hasta arreglar el puente |
| K2 | La base reproduce lo publicado: F1 medio del ensamble en los seis folds comparables a ±0,02 de 0,635 (`boxtwin-detector/README.md`, "El estado, en una línea"), y familia de E4 a ±0,02 de 0,727 (`docs/experiments/2026-10-16-clasificador-geometrico.md`) | Se frena la lectura hasta explicar la diferencia con evidencia, por ejemplo un cambio de anotación fechado. La explicación va al documento |
| K3 | Fracción de keypoints de muñeca con puntaje ≥ 0,3 (`features.MIN_SCORE`), por condición, en los cuadros emparejados | Sin umbral. Si las dos difieren en más de 0,10, se declara como mecanismo posible: cambia lo que el detector enmascara |
| K4 | Mediana del IoU del puente ≥ 0,70 por fuente | Esa fuente se informa con la marca "puente débil" |

## 4. Reglas de lectura

**P12a, detector.** Número primario: el F1 por evento (emparejador a IoU 0,3) del ensamble de
cinco semillas (42, 1, 2, 3, 4), promediado sobre los seis folds comparables, que es el mismo
estadístico del 0,635 publicado. Umbral de decodificación 0,80 en las dos condiciones, el de A;
la curva entre 0,70 y 0,90 se informa sin decidir nada con ella. **Mejora** si Y26 supera a la
base por 0,03 o más y gana en al menos 5 de los 6 folds comparables (la proporción de 5 de 7,
redondeada hacia arriba). El fold sin `Sparring` se informa aparte, como en el reporte
publicado. Al lado, sin umbral: recall, precisión, recall sobre pose medida y sobre pose
rellenada, y la media y el desvío de las corridas de una semilla.

No se usa la cláusula de dispersión de E7 (desvío entre semillas mayor que 0,03, no
concluyente): para el detector el desvío de una corrida sola ya es de 0,09 a 0,10 de F1
(`README.md` del detector), por eso el objeto medido es el ensamble y no la semilla. Si K2
falla, P12a queda no concluyente.

**P12b, clasificador.** La regla de E7 sin cambios: familia del ensamble de cinco semillas (0 a
4), folds concatenados sobre los 1047 golpes, ventanas anotadas (no las del detector). **Mejora**
si supera a la base por 0,03 o más y gana en al menos 5 de los 7 folds; con desvío entre semillas
de la familia por encima de 0,03, no concluyente. El conjunto de `frame_dir` tiene que ser
idéntico en las dos condiciones; si no, `datos.py` frena y E8 también. Al lado, sin umbral: recall
de hook y de uppercut, altura balanceada y recall de cuerpo.

**P12c, trackeo.** Por fuente y en total, sobre los cuadros donde las dos condiciones tienen
emparejado a ese peleador (el denominador común de `01-percepcion.md` §5): cambios de track por
peleador por minuto, tracks distintos por peleador, tracks mixtos (un mismo track asignado a A en
unos cuadros y a B en otros) y cuadros dentro de tracks mixtos. Aparte, sin verdad: tracks
totales del caché. Dirección declarada: menos cambios y menos mixtos con Y26. Sin umbral porque
no hay precedente y porque la verdad sale de v8.

**P12d, C1.** `boxtwin-annotator/scripts/medir_c1.py medir --raiz ~/boxtwin-gemelo/raiz-y26`
sobre las seis fuentes de gimnasio, con las reglas del pre-registro de C1 (`2026-10-20`). Se
informa al lado de 0,810 (85/105), sin umbral: con otro tracker cambia la cantidad de tracks y
el denominador no es el mismo.

**P12e, tiempo.** H3: mediana y p95 de ms por cuadro de pose y tracking, y pico de VRAM.
Dirección declarada: Y26 más rápido. No es C5: C5 mide la cadena completa de A.

**SH1.** Ocho clases de E4 sobre Y26 con la lectura a/b/c. Es informativa: la sub-hipótesis 1
se resuelve el 31-10 con lo medido hasta esa fecha.

## 5. Qué decide cada resultado

- Los modelos de la fase 2 son los de Y26, salga lo que salga. C muestra sus propios números.
- P12a y P12b mejoran: el documento recomienda evaluar YOLO26 para A después del 17-11, con C5
  medido otra vez, y eso va al Capítulo 8 como recomendación con evidencia.
- Ninguna mejora: el Capítulo 7 informa que actualizar el estimador de pose no mejoró el sistema
  sobre este material, y la Tabla 4.5 dice que YOLO26 se evaluó.
- Una mejora y la otra no, o alguna no concluyente: se informa tal cual, sin promediar.
- Nada de E8 cambia A, ni C1, C3 o C5, ni la sub-hipótesis 1.

## 6. Lo que E8 no separa

- Si una diferencia viene de la cabeza RLE de keypoints, de la arquitectura o del entrenamiento
  de YOLO26: cambian juntos.
- Otros tamaños: solo L. M y X quedan afuera.
- Otra resolución: solo 640. Los cuerpos chicos (`amateur_estatico`, 18 px de hombro) no tienen
  anotación de identidad y no entran.
- Lo que YOLO26 ve y YOLOv8 no veía: la verdad sale de v8 (`01-percepcion.md` §5).
- Pesos ajustados a boxeo: se usan los de COCO en las dos condiciones.

## 7. Auditoría

Un auditor independiente con las instrucciones de `.claude/agents/auditor-medicion.md`, adaptadas
a E8, sobre los resultados y antes del texto. Revisa como mínimo:

1. El commit de este archivo es anterior al primer archivo de resultados (H1).
2. Las anotaciones y los cachés originales con el mismo sha256 antes y después (S3 y S14).
3. El mismo conjunto de 1047 `frame_dir` en las dos condiciones de E4; los mismos eventos en las
   dos del detector.
4. Las configuraciones de las dos condiciones iguales salvo el modelo de pose (y el yaml, si
   entró D-H2).
5. Estandarización y pesos de clase ajustados solo con el fold de entrenamiento.
6. Recálculo independiente, con numpy, de los números primarios de P12a y P12b.
7. La regla de §4 aplicada como está escrita.
8. El log del candado de `e8.sh` sin ninguna etapa corrida con trabajos de A en cola o tomados.
9. K1 a K4.

## 8. Salidas

- `docs/experiments/<fecha de la corrida>-e8-yolo26.md`, con la estructura de E7: qué se mide,
  lo que cambió respecto del diseño, resultados, lectura punto por punto, lo que hay que declarar
  y auditoría.
- `docs/experiments/datos/<fecha>-e8-yolo26-{detector,E4,trackeo}.json`. El de E4 es el que lee
  `entrenar_final.py --exp E4-y26` (T4).
- La entrada terse del README raíz y `docs/tesis/<fecha>-e8-cambios.md` con lo que cambia en los
  Capítulos 4, 7 y 8, en formato ANTES / DESPUÉS / QUÉ CAMBIÓ.
- Los cachés, tensores y corridas quedan en `~/boxtwin-gemelo/e8/`, fuera de git.
