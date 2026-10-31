# E10 · Relleno temporal de keypoints bajo umbral

31-10-2026. Pre-registro: `docs/specs/2026-10-30-mejoras-post-e8/02-e10-relleno-temporal.md`
(commit `8741a8d`, 30-10), con la decisión 9 (`036c9a9`, antes de la corrida): el tope es
round(0,10 s × fps), no hacia abajo como dice §2. Código: `547d58b`. Datos: `~/boxtwin-mejoras/e10/`.
JSON: `docs/experiments/datos/2026-10-31-e10-relleno-temporal.json`. Lectura:
`scripts/mejoras/lectura.py e10`.

**En una línea: rellenar los huecos cortos recupera cuadros pero no mejora de forma demostrada
ni el detector ni E4; las dos diferencias son positivas y quedan debajo de 0,03.**

## 1. Qué se mide (pre-registrado)

Una sola variable respecto de la base v8 de E8: los huecos de keypoints bajo umbral de hasta
0,10 s (3 cuadros a 30 y a 29,97 fps, 6 a 59,94), cerrados y con la detección presente, se
interpolan en vez de enmascararse (§2). P14a, F1 por evento del detector en 0,80; P14b, familia
de E4; P14c, lo mismo sobre Y26-cal (E8b); P14d, cuánto de la pérdida cae en huecos cortos.

## 2. Controles

| Id | Resultado |
|---|---|
| K-reuso | Pasa (el de S3): con el relleno apagado, los tensores son los de E8 |
| K-hueco | Pasa: la prueba de T4 rellena un hueco de 3 a 30 fps y no uno de 4, y 6 y no 7 a 59,94 |
| K-golpes | Pasa: los mismos 1047 golpes de E4 (mismos `frame_dir`, mismo orden) y los mismos eventos del detector en las cuatro condiciones |

El relleno tocó 19450 cuadros-carril en v8+R, 15449 de ellos usables. Cuadros-carril usables:
158976 contra 147788 de la base (+7,6%). Pasos usables de E4: 0,914 contra 0,842.

## 3. Resultados

### 3.1 P14a, detector

F1 del ensamble en 0,80, seis folds comparables: **base 0,6227, v8+R 0,6276** (+0,005). Gana 3
de 6 (01, 03 y 04), empata 02, pierde pacquiao y sparring-3-rounds. Recall 0,496 contra 0,487;
precisión 0,864 contra 0,869.

| Umbral | Base | v8+R |
|---|---|---|
| 0,70 | 0,578 | 0,631 |
| 0,75 | 0,619 | 0,656 |
| 0,80 | 0,623 | 0,628 |
| 0,85 | 0,580 | 0,574 |
| 0,90 | 0,468 | 0,479 |

Como en E8b, la diferencia aparece en los umbrales bajos (+0,053 en 0,70) y no en el que decide.
En su mejor umbral, v8+R llega a 0,663 contra 0,638.

### 3.2 P14b, clasificador E4

Familia del ensamble: **base 0,7316, v8+R 0,7431** (+0,012). Gana 5 de 7 folds (01-sparring
empata), con desvío entre semillas 0,018. Tiene los folds pero no los 0,03.

### 3.3 P14c, relleno sobre Y26-cal

Y26-cal+R menos Y26-cal: **+0,018 de F1** en 0,80 (0,6255 a 0,6438, gana 5 de 6) y **+0,010 de
familia** (0,7125 a 0,7230, gana 4 de 7). Positivas las dos, como se declaró. Que el efecto sea
mayor sobre YOLO26 vale solo para el F1 (+0,018 contra +0,005 de v8); en familia es al revés
(+0,010 contra +0,012). Las cuatro diferencias están dentro de ±0,03.

### 3.4 P14d, cuánto de la pérdida cae en huecos cortos

Fracción de los cuadros perdidos de muñeca que están en huecos de hasta el tope, sin etiquetas:
v8 0,268, YOLO26 con 0,3 0,139, Y26-cal 0,224 (codo 0,300, 0,187 y 0,287). Las tres debajo de
0,30: por §2 se declara que **el relleno actúa sobre la minoría de la pérdida**. El tope no se
cambió.

## 4. Lectura contra el pre-registro (§4 y §5)

- **P14a: no demostrada** (+0,005, 3 de 6). Con las categorías de E8b sería "iguala".
- **P14b: no demostrada** (+0,012, 5 de 7, debajo de +0,03).
- **P14c: positiva** en las dos medidas.
- Por §5, el Capítulo 7 lo informa en una línea como hallazgo negativo: la pose que se pierde en
  el golpe no se recupera interpolando huecos cortos. Encaja con P14d: tres de cada cuatro
  cuadros perdidos de muñeca están en huecos más largos que 0,10 s. Refuerza E12 (otra fuente de
  la posición de la mano) y E15.

## 5. Lo que hay que declarar

1. El pre-registro dice "se redondea hacia abajo" (§2); rige la decisión 9 (al entero más
   cercano). En Sparring, a 29,643 fps, el tope son 3 cuadros, 0,101 s.
2. La regla de presencia no es la misma en el detector y en E4. El detector usa `valid` y no
   `interpolated` del export; las series de E4 no traen `interpolated`, así que ahí la pose
   interpolada por la identidad cuenta como presente (decisión 10). Eso rellena de más 105, 274
   y 143 cuadros-peleador en Sparring, pacquiao y sparring-3-rounds, y toca 56 de los 1047
   golpes de E4.
3. Falta la métrica que §4 pedía al lado: el recall sobre los golpes con pose rellenada por E10.
   El `recall_rellenados` de los reportes del detector es el de la pose interpolada por la
   identidad, no el del relleno de E10; no se cita como si lo fuera. No decide nada.
4. El detector tiene 1046 eventos y E4 1047 golpes; la diferencia viene de E8 y es la misma en
   las cuatro condiciones.
5. S8c, la variante a 0,20 s, no se corrió.

## 6. Auditoría

Auditor independiente, 31-10, de solo lectura: **7 de 8 ítems pasan**. Recalculó el relleno de
las siete fuentes con numpy, sin importar `relleno.py` ni `dataset.py`, en las dos condiciones:
0 diferencias en la máscara de los 28 carriles de cada una, y las features de su pose rellenada
coinciden bit a bit con las del build. Ninguna coordenada medida cambió (regla 5). Recalculó
P14a, P14b y P14c. Falló el total de P14d: `huecos.py` llevaba los largos a cuadros de 30 y
dejaba afuera los huecos de 3 de Sparring y los de 6 de Pacquiao (daba 0,225 para la muñeca de
v8). Se corrigió (`184bb75`, con prueba) y los números de §3.4 son los corregidos; la conclusión
no cambia. Sus advertencias sobre la redacción están en §3.3 y §5.

## 7. Entrada para el README raíz

```
31-10 E10: relleno de huecos cortos de keypoints en vez de enmascararlos

1. Huecos cerrados de hasta 0,10 s, por articulacion, interpolados antes de las features
2. Cuadros-carril usables 158976 contra 147788; pasos usables de E4 0,914 contra 0,842
3. Detector, F1 en 0,80: 0,628 contra 0,623, 3 de 6 folds: no demostrada
4. Familia de E4: 0,743 contra 0,732, 5 de 7 folds, debajo de +0,03: no demostrada
5. Sobre YOLO26 calibrado: +0,018 de F1 y +0,010 de familia, positivas
6. Solo 27% de los cuadros perdidos de muneca caen en huecos de hasta 0,10 s
7. Auditoria independiente: 7 de 8 items pasan; el total de P14d se corrigio
8. Documentado en docs/experiments/2026-10-31-e10-relleno-temporal.md
```
