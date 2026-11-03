# E10 · Relleno temporal de keypoints bajo umbral (pre-registro)

30-10-2026. El commit de este archivo es el pre-registro. Correcciones solo antes de la primera
corrida de S8 (`10-pasos.md`), anotadas como decisión.

## 1. La pregunta

Hoy un keypoint con puntaje bajo el umbral saca el cuadro entero de la máscara, y los vecinos
también, porque las derivadas cruzan el agujero (`features.py`, `features_de`). En un golpe la
muñeca se borronea justo en la extensión: el cuadro que más importa es el que más se tira. E8
lo midió del lado de YOLO26 (K3), pero con v8 pasa lo mismo en menor medida (10% de muñecas
bajo 0,3).

**¿Rellenar los huecos cortos por interpolación, en vez de enmascararlos, mejora el detector y
E4 sobre la línea v8?**

| Id | Pregunta | Tipo |
|---|---|---|
| P14a | F1 por evento del detector, v8 con relleno contra v8 (base de E8) | con regla |
| P14b | Familia de E4, v8 con relleno contra v8 | con regla |
| P14c | Lo mismo sobre Y26-cal (E8b) con y sin relleno | sin umbral, dirección declarada |
| P14d | Distribución del largo de los huecos por articulación y condición | descriptiva, sin etiquetas |

## 2. La regla de relleno (fija desde ahora)

Por peleador, por articulación (las cinco de E8b §2), sobre la serie del video entero ya
asignada a A o a B:

1. Un **hueco** es una corrida de cuadros consecutivos con puntaje bajo el umbral de la
   condición, con un cuadro aceptado inmediatamente antes y otro inmediatamente después, y con
   la detección del peleador presente en todos los cuadros del hueco.
2. Si el hueco dura **≤ 0,10 s** (3 cuadros a 30 fps, 6 a 60 fps; se redondea hacia abajo), las
   coordenadas se interpolan linealmente entre los dos extremos aceptados y el puntaje queda en
   el umbral (se acepta). Se marca en una máscara nueva, `kp_rellenado`.
3. Huecos más largos, o abiertos (al principio o al final de una presencia), quedan como hoy.
4. El relleno corre **antes** de `features_de`: las features, las derivadas y la máscara se
   calculan sobre la serie rellenada sin ningún otro cambio.
5. Las coordenadas medidas nunca se tocan, aunque tengan puntaje bajo: solo se reemplazan las
   de los cuadros del hueco.

Por qué 0,10 s: un golpe de brazo extendido dura del orden de 0,3 a 0,5 s; interpolar un hueco
que tape la extensión entera borra el golpe (la muñeca iría en línea recta de guardia a
guardia). Con 0,10 s el relleno puede tapar a lo sumo un tercio de un golpe rápido. P14d mide
cuántos huecos caen debajo de ese tope **sin mirar etiquetas**; si menos del 30% de los
cuadros perdidos de muñeca cae en huecos de ≤ 0,10 s, se declara que el relleno actúa sobre la
minoría de la pérdida, pero el tope **no** se cambia.

Una variante informativa con 0,20 s se corre solo si sobra noche (`10-pasos.md` S8b) y no
decide nada.

## 3. Condiciones y controles

| | Base v8 | v8+R (nueva) | Y26-cal | Y26-cal+R (nueva) |
|---|---|---|---|---|
| Caché | `raiz-v8` | `raiz-v8` | `raiz-y26` | `raiz-y26` |
| Umbrales | 0,3 | 0,3 | E8b §2 | E8b §2 |
| Relleno | no | sí | no | sí |
| Corridas | E8, reusadas | nuevas | E8b, reusadas | nuevas (solo si E8b corrió) |

| Id | Qué | Si falla |
|---|---|---|
| K-reuso | Con el relleno apagado, sha256 de los tensores igual a los de la base de E8 | igual que en E8b |
| K-hueco | En un caso sintético con un hueco de 3 cuadros y otro de 4 a 30 fps, el primero se rellena y el segundo no (prueba de T4) | no se corre |
| K-golpes | El mismo conjunto de 1047 `frame_dir` y los mismos eventos del detector en las cuatro condiciones | E10 no se lee |

## 4. Reglas de lectura

**P14a** y **P14b**: las reglas de E8 (P12a y P12b), v8+R contra base v8. **Mejora** con +0,03
y 5 de 6 folds (detector) o 5 de 7 (E4, desvío ≤ 0,03). Al lado: recall sobre pose medida y
sobre pose rellenada por separado (el dataset ya separa la pose interpolada de la identidad;
`kp_rellenado` se suma como categoría aparte).

**P14c**: Y26-cal+R menos Y26-cal. Dirección declarada: positiva y mayor que la de v8, porque a
YOLO26 le faltan más keypoints.

**P14d**: histograma del largo de los huecos de muñeca y codo, por condición, en segundos.

## 5. Qué decide cada resultado

- **v8+R mejora**: candidato para A después del 17-11 (`09-integracion-y-tesis.md` §1). El
  relleno no cambia el modelo de pose ni el tiempo de forma medible; se mide C5 igual. Cap 7: el
  recall del detector sube por recuperar pose, con el número.
- **No mejora**: Cap 7 lo informa en una línea como hallazgo negativo: la pose que se pierde en
  el golpe no se recupera interpolando huecos cortos. Refuerza la lectura de E12 (otra fuente de
  la posición de la mano) y de E15.
- Nada cambia A antes del 17-11.

## 6. Lo que E10 no separa

- Relleno lineal contra un refinador aprendido (SmoothNet y similares): se elige el lineal
  porque no necesita entrenamiento ni datos de pose verdadera, y porque SmoothNet tiende a
  suavizar de más los movimientos rápidos y amplios, que es justo un golpe. Un refinador
  aprendido queda para el Cap 8.
- Relleno de huecos causados por el tracker (persona sin detección): eso ya lo hace la
  interpolación de identidad del anotador.

## 7. Auditoría

Como E8b §7, más: el auditor recalcula con numpy el relleno de una fuente entera y compara los
tensores; revisa que ninguna coordenada medida haya cambiado (regla 5).

## 8. Salidas

`docs/experiments/<fecha>-e10-relleno-temporal.md`, JSON en `docs/experiments/datos/`, entrada
del README y `docs/tesis/<fecha>-e10-cambios.md`.
