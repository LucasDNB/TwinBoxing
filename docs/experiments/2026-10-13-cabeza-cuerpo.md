# C8 · Altura del golpe por regla geométrica (cabeza / cuerpo)

13-10-2026 · spec: `prototipo/specs/ampliacion-13-10/03-cabeza-cuerpo.md` §2 y §4 ·
regla `altura_relativa_v1` en `boxtwin-annotator/src/boxtwin/mvp/objetivo.py` ·
script `boxtwin-annotator/scripts/medir_objetivo.py` (pruebas en `tests/test_script_medir_objetivo.py`)

## 1. Qué se mide (pre-registrado en el spec)

- Calibración: 01-sparring y 02-sparring (166 golpes, 30 al cuerpo). Ahí se elige τ en la
  grilla {−0,10; 0,00; 0,10; 0,20; 0,30; 0,40}.
- Medición, una sola vez, con ese τ: 03-sparring, 04-sparring, sparring-3 y pacquiao_margarito
  (806 golpes, 132 al cuerpo).
- Nivel 1 (regla sola): golpes anotados, identidad anotada, keypoints de la caché de cada
  proyecto de anotación.
- Criterio C8: exactitud balanceada ≥ 0,75 y recall de cuerpo ≥ 0,60, con Wilson al 95% del
  recall de cuerpo. Línea de base "todo es cabeza": balanceada 0,50.
- Si pasa: bandera `objetivo` (C8) y se construye la tanda 2. Si no: el bloque queda "No
  disponible" y el número va al Cap 7.

## 2. Lo que el spec no fijaba, fijado antes de correr

Escrito y commiteado antes de la primera corrida del script, y no se toca después:

1. Golpes que entran: todos los anotados de cada fuente, fintas y abortados incluidos, que es
   como el spec cuenta 166 y 806.
2. Un golpe sin dato (rival sin pose válida en el pico ni en ±2 cuadros, o el que pega sin
   ningún cuadro válido) cuenta como **error** en la métrica del criterio. Es la lectura
   estricta de "sobre 806 golpes" y no puede inflar el número. La métrica sobre los golpes con
   dato se reporta al lado, sin umbral, junto con cuántos quedaron sin dato.
3. Desempate de τ en la calibración: mayor exactitud balanceada; si empata, mayor recall de
   cuerpo; si empata, el τ menor.
4. Un keypoint cuenta si su score es ≥ 0,3 (el umbral del anotador). La línea de hombros y el
   torso del rival piden hombros y caderas por encima de ese score.
5. Mano: la muñeca. El spec pide el centro del guante cuando el detector de guantes dio caja en
   ese cuadro, pero ninguna caché guarda cajas de guante por cuadro (ni las de anotación ni las
   de las sesiones), así que se usa la muñeca en la medición y en el producto, igual en los dos.
6. Torso del rival: distancia euclídea de punto medio de hombros a punto medio de caderas.
7. Confianza: `2 / (1 + exp(−10 · |h − τ|)) − 1`. No entra en ninguna métrica.

## 3. Calibración

(se completa con la salida de `medir_objetivo.py calibrar`)

## 4. Medición

(se completa con la salida de `medir_objetivo.py medir --tau <elegido>`)
