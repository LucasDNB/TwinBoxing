# E8b · YOLO26 con el umbral de puntaje calibrado

31-10-2026. Pre-registro: `docs/specs/2026-10-30-mejoras-post-e8/01-e8b-umbral-calibrado.md`
(commit `8741a8d`, 30-10, sin cambios hasta la corrida). Código: `547d58b` (worktree fijo de S2).
Datos: `~/boxtwin-mejoras/e8b/`, fuera de git. JSON: `docs/experiments/datos/2026-10-31-e8b-umbral-calibrado.json`.
Lectura: `scripts/mejoras/lectura.py e8b`, que escribe `e8b/veredictos.json`.

**En una línea: con el umbral calibrado a su propia distribución de puntajes, YOLO26 iguala a
YOLOv8 en el detector y en E4 sobre este material; el umbral explicaba la mayor parte de la caída
de E8.**

## 1. Qué se mide (pre-registrado)

Una sola variable respecto de Y26 de E8: el umbral de puntaje de keypoint, que deja de ser 0,3 y
pasa a ser, por articulación, el que acepta la misma fracción de keypoints que v8 con 0,3 (§2).
La base es la condición v8 de E8 (sin ReID, D-H2), reusada. P13a, F1 por evento del ensamble del
detector en 0,80 sobre los seis folds comparables; P13b, familia de E4 con la regla de E7; P13c,
cuánto recupera respecto de Y26 con 0,3; P13d, el mecanismo por fuente.

## 2. Controles

| Id | Resultado |
|---|---|
| K-reuso | Pasa: con los umbrales en 0,3, el código nuevo arma los 7 `.det.npz` y `golpes.npz` de las dos condiciones de E8 con los mismos arreglos (0 diferencias) |
| K-cal | Pasa: la fracción aceptada de Y26-cal es la de v8 en las cinco articulaciones (diferencia < 0,00005). Pasa por construcción: se mide sobre la misma población que se calibró |
| K-mec | 139534 cuadros-carril usables en Y26-cal, contra 147788 de la base y 111472 de Y26: la calibración recupera el 78% de lo que YOLO26 perdía |

Los umbrales (sha256 de `umbrales.json` `feaa054c…`): nariz 0,134, hombro 0,101, codo 0,111,
muñeca 0,110, cadera 0,178. Todos debajo de 0,3: YOLO26 puntúa más bajo que v8 en todas las
articulaciones, no solo en la muñeca. Fracción de v8 con 0,3: nariz 0,688, hombro 0,998, codo
0,963, muñeca 0,896, cadera 0,978.

## 3. Resultados

### 3.1 P13a, detector

F1 del ensamble en 0,80, media de los seis folds comparables: **base 0,6227, Y26-cal 0,6255**
(+0,003). Y26-cal gana 3 de 6 folds (02, 03 y sparring-3-rounds), empata 01 y pierde 04 y
pacquiao. Recall 0,499 contra 0,487; precisión 0,844 contra 0,869.

| Umbral | Base | Y26-cal | Y26 (E8) |
|---|---|---|---|
| 0,70 | 0,578 | 0,609 | 0,347 |
| 0,75 | 0,619 | 0,652 | 0,355 |
| 0,80 | 0,623 | 0,626 | 0,433 |
| 0,85 | 0,580 | 0,588 | 0,529 |
| 0,90 | 0,468 | 0,489 | 0,425 |

En su mejor umbral, cada condición: base 0,638, Y26-cal 0,653. La curva de Y26-cal queda por
encima de la base en todo el rango, y donde más se separa es en 0,70 y 0,75 (+0,031 y +0,033).
Eso va al lado: la regla decide en 0,80.

### 3.2 P13b, clasificador E4

Familia del ensamble sobre los 1047 golpes: **base 0,7316, Y26-cal 0,7125** (−0,019). Y26-cal
gana 2 de 7 folds (02-sparring y pacquiao). Desvío entre semillas 0,018 y 0,011. Recall de gancho
0,627 contra 0,607; de uppercut 0,447 contra 0,541; altura balanceada 0,701 contra 0,700.

### 3.3 P13c, cuánto recupera la calibración

Y26-cal menos Y26 de E8: **+0,192 de F1 en 0,80** y **+0,037 de familia** (0,675 a 0,713), las
dos en la dirección declarada. El +0,192 exagera la recuperación porque Y26 tenía el pico corrido
a 0,85: comparando cada uno en su mejor umbral, la recuperación es +0,124 (de 0,529 a 0,653).

### 3.4 P13d, el mecanismo por fuente

| Fuente | Usables base / Y26 / Y26-cal | Pasos de E4 base / Y26 / Y26-cal | Familia base / Y26 / Y26-cal |
|---|---|---|---|
| 01-sparring | 15950 / 13086 / 15947 | 0,829 / 0,661 / 0,848 | 0,656 / 0,625 / 0,615 |
| 02-sparring | 16741 / 13559 / 16746 | 0,923 / 0,759 / 0,925 | 0,790 / 0,839 / 0,855 |
| 03-sparring | 15197 / 11383 / 14361 | 0,822 / 0,646 / 0,783 | 0,767 / 0,721 / 0,729 |
| 04-sparring | 16976 / 12099 / 16166 | 0,882 / 0,678 / 0,841 | 0,833 / 0,683 / 0,767 |
| Sparring | 13828 / 8571 / 11209 | 0,772 / 0,424 / 0,543 | 0,605 / 0,395 / 0,509 |
| pacquiao_margarito | 15878 / 11410 / 14917 | 0,825 / 0,629 / 0,794 | 0,752 / 0,752 / 0,793 |
| sparring-3-rounds | 53218 / 41364 / 50188 | 0,854 / 0,680 / 0,800 | 0,727 / 0,698 / 0,722 |

En seis fuentes la calibración cierra casi toda la brecha de cuadros usables. Sparring es la
excepción: recupera la mitad (pasos de E4 0,543 contra 0,772) y es donde más pierde Y26-cal en
E4 (−0,097). La correlación por fuente entre pasos recuperados y familia recuperada es −0,44
(siete fuentes): recuperar cuadros no predice recuperar familia fuente por fuente.

## 4. Lectura contra el pre-registro (§4 y §5)

- **P13a: iguala** (+0,003, menos de 0,03 en cualquier dirección).
- **P13b: iguala** (−0,019), del lado bajo: no se puede escribir que Y26-cal supera a la base en
  E4.
- **P13c: positiva**, como se declaró.
- Con P13a y P13b en "iguala", §5 dice: la frase del Capítulo 7 pasa de "actualizar el estimador
  de pose no mejoró el sistema" a "con el umbral calibrado, YOLO26 iguala a YOLOv8 sobre este
  material", y la regla de §2 queda como receta para cualquier estimador nuevo (E9 la hereda).
  Nada cambia A ni el 0,3 de A.

## 5. Lo que hay que declarar

1. La calibración es transductiva y sin etiquetas: usa los mismos cuadros que la evaluación,
   aunque no ve ningún golpe (§2). Un producto haría lo mismo sobre su propio material. K-cal
   prueba la implementación, no que la calibración generalice.
2. La base no es la línea publicada de A (rama D-H2, decisión 2): es v8 sin ReID con el puente.
3. El puente sigue sesgando a favor de v8 (E8 §5.2).
4. En Sparring la calibración no alcanza; queda sin explicar si es la pose o el trackeo de esa
   fuente, que es la que más tracks mixtos tiene.
5. Una calibración por fuente o por peleador, y recalibrar el 0,80 de decodificación, no se
   midieron (§6). La curva sugiere que Y26-cal tendría su mejor umbral en 0,75, y eso no decide.

## 6. Auditoría

Auditor independiente, 31-10, de solo lectura, con los seis ítems de §7: **6 de 6 pasan**.
Recalculó `umbrales.json` con numpy desde los cachés sin importar el código de T2 (iguales al
último dígito), comparó los arreglos de K-reuso contra E8 con `np.array_equal`, verificó que las
35 corridas de E4 difieren de las de E8 solo en el commit y que los `.det.json` registran el
sha256 de `umbrales.json`, recalculó P13a y P13b sin importar `lectura.py` (y la familia desde los
105 `pred.npz`), y comprobó los 49 sha256 de originales y cachés contra `antes.txt`. Sus cuatro
advertencias sobre la redacción están en §3 y §5.

## 7. Entrada para el README raíz

```
31-10 E8b: YOLO26 con el umbral de puntaje calibrado por articulacion

1. Umbral que iguala la fraccion aceptada de v8 con 0,3: 0,10 a 0,18 segun la articulacion
2. Detector, F1 en 0,80: 0,626 contra 0,623 de la base, 3 de 6 folds: iguala
3. Familia de E4: 0,713 contra 0,732, 2 de 7 folds: iguala, del lado bajo
4. Recupera +0,124 de F1 (mejor umbral de cada uno) y +0,037 de familia respecto de E8
5. Cuadros-carril usables: 139534 contra 147788 (base) y 111472 (YOLO26 con 0,3)
6. Auditoria independiente: 6 de 6 items pasan
7. Documentado en docs/experiments/2026-10-31-e8b-umbral-calibrado.md
```
