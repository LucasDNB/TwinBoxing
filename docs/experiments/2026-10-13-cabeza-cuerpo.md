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

01-sparring + 02-sparring, 166 golpes (30 al cuerpo), 11 sin dato. Salida completa en
`datos/2026-10-13-c8-calibracion.json`.

| τ | Balanceada (estricta) | Recall cabeza | Recall cuerpo | Balanceada con dato |
|---|---|---|---|---|
| −0,10 | 0,641 | 0,382 | 0,900 | 0,687 |
| 0,00 | 0,645 | 0,456 | 0,833 | 0,691 |
| 0,10 | 0,689 | 0,544 | 0,833 | 0,738 |
| **0,20** | **0,698** | 0,596 | 0,800 | 0,748 |
| 0,30 | 0,664 | 0,662 | 0,667 | 0,712 |
| 0,40 | 0,583 | 0,699 | 0,467 | 0,624 |

τ elegido: 0,20, sin empate.

## 4. Medición (una sola corrida, τ = 0,20)

Salida completa en `datos/2026-10-13-c8-medicion.json`.

| Fuente | Golpes | Cuerpo | Sin dato | Balanceada | Recall cabeza | Recall cuerpo (Wilson 95%) | Balanceada con dato |
|---|---|---|---|---|---|---|---|
| 03-sparring | 131 | 17 | 21 | 0,693 | 0,561 | 0,824 (0,590–0,938) | 0,833 |
| 04-sparring | 124 | 23 | 6 | 0,776 | 0,683 | 0,870 (0,679–0,955) | 0,814 |
| sparring-3 | 400 | 74 | 13 | 0,722 | 0,620 | 0,824 (0,722–0,894) | 0,749 |
| pacquiao_margarito | 151 | 18 | 13 | 0,707 | 0,692 | 0,722 (0,491–0,875) | 0,807 |
| **Total** | **806** | **132** | **53** | **0,726** | **0,634** | **0,818 (0,744–0,875)** | **0,781** |

Línea de base "todo es cabeza": 0,50 en todas.

Matriz de confusión total (filas: anotado; columnas: regla):

| | cabeza | cuerpo | sin dato |
|---|---|---|---|
| cabeza (674) | 427 | 204 | 43 |
| cuerpo (132) | 14 | 108 | 10 |

## 5. Resultado

**C8 no pasa.** Exactitud balanceada 0,726 contra 0,75; el recall de cuerpo (0,818, Wilson
0,744–0,875) sí supera 0,60. La regla le gana con claridad a la línea de base (0,50) y casi no
se le escapan los golpes al cuerpo, pero llama "cuerpo" a 204 de 674 golpes a la cabeza: el
error dominante es de un solo lado.

Ninguna fuente queda por debajo de 0,60 de balanceada. La más baja es 03-sparring (0,693), que
es también la que más golpes sin dato tiene (21 de 131, 16%): ahí el rival pierde la pose más
seguido.

Lo que hay que decir con el número, sin cambiarlo:

- La decisión 2 de §2 (sin dato cuenta como error) se tomó antes de correr y es la que define
  el resultado: sobre los 753 golpes con dato la balanceada es 0,781, que pasaría. Esa lectura
  es posterior a ver el número y no reemplaza al criterio. Lo que sí muestra es que los 53
  golpes sin rival en cuadro (6,6%) explican la distancia entre 0,726 y 0,781.
- El resto del error, con dato, son los 204 golpes a la cabeza que la regla llama cuerpo. No
  está diagnosticado. La hipótesis, sin probar: con la cámara de costado, la muñeca en el
  cuadro de máxima extensión queda por debajo de la línea de hombros del rival más 0,2 torsos
  cuando el rival se inclina o cuando el golpe se mide antes de llegar. Probarla es trabajo de
  una regla v2, con su propia calibración y una partición de medición que no sea esta (esta
  ya se miró).

Consecuencias (spec `08` §1): el bloque "A la cabeza y al cuerpo" queda "No disponible" con
este número, la bandera `objetivo` no se enciende y la tanda 2 (W14, C9, B12, W15, B13, W16)
no se construye. El nivel 2 (punta a punta) no se corrió: necesita procesar las cuatro fuentes
con el sistema completo en la GPU.

## 6. Nivel 2 (punta a punta), fijado antes de correr

Sin umbral, como dice el spec. Lo que se fija antes de ver números:

1. Las cuatro fuentes de medición se procesan enteras por el producto actual (`procesar` y
   `completar`, que ya llama a `enriquecer`), en `~/boxtwin/mediciones/c8-nivel2/`, fuera de
   la base de producción.
2. Siembra: la pareja que propone el sistema (`pareja_sugerida`). Qué semilla es A y cuál B
   no importa: los roles se alinean con la anotación tomando la orientación que más empareja,
   como `medicion.py` el 22-09.
3. Detector: el de producción, `detector_7fuentes.ens.pt`, que se entrenó con estas cuatro
   fuentes. Eso hace que los golpes encontrados sean más y con fronteras mejores que en
   material nuevo; el número de nivel 2 es optimista por ese lado y se dice así.
4. Emparejamiento: `agreement.emparejar` a IoU 0,3 por carril. La altura se mide sobre los
   golpes anotados que quedaron emparejados, con la métrica estricta y la de con dato.
5. Se reporta además el recall del detector sobre golpes anotados a la cabeza y al cuerpo.
   Es la cuenta de C9, pero C9 no aplica (C8 no pasó): queda como dato.
