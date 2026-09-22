# Los negativos arreglan el detector, pero solo si la unidad es el recorte y no la etiqueta vacía

**20-09-2026 · seis fuentes anotadas · dos entrenamientos con el mismo material y resultados opuestos.**

## Por qué

La asignación automática de identidad le daba rol de peleador a 17 tracks que no lo son.
Revisados uno por uno en el video, son todos gente **afuera del ring**: espectadores y
entrenadores apoyados en las cuerdas o sentados, que por estar cerca de la cámara son
grandes y pasan el filtro de altura.

Mirando qué recorta el detector cuando se equivoca, la variedad es mayor de la esperada: el
**logo estampado** de una remera, una **tablet azul**, un **graffiti rojo** en una pared,
zapatillas, cabezales, cajas de cartón, una bolsa de boxeo y un bidón. Todos bultos
redondeados, lisos y saturados. El dataset público no puede enseñar a rechazarlos porque
tiene una sola clase y ningún ejemplo de qué **no** es un guante.

## El primer intento, que salió mal

Se tomaron los 371 recortes de persona de esos tracks y se les puso **etiqueta vacía**, que
es como YOLO representa un negativo. El razonamiento era: si el track no es peleador, sus
detecciones son falsas por construcción.

Es falso. Un entrenador puede estar **sosteniendo guantes de verdad**, y el detector los
encuadra bien. La etiqueta vacía afirma entonces que ahí no hay ningún guante.

| | Original | Etiquetas vacías |
|---|---|---|
| Fracción de guante en tracks de peleador | **0,868** | **0,344** |
| Tracks de peleador sobre el umbral | 119/125 | **45/125** |
| Fuentes que deciden algo | 6 de 6 | **2 de 6** |

El detector quedó inservible sobre material propio. En `04-sparring` la fracción cayó de
0,857 a 0,111.

**Y el mAP de validación casi no se movió: 0,895 a 0,876.** Mirando solo ese número el
experimento parecía neutro. Es la demostración más clara de por qué esa métrica no sirve
acá: se mide sobre fotografía de producto.

Se intentó antes separar los negativos por geometría, quedándose con las detecciones lejos
de las manos. Sobre 70 detecciones de tracks equivocados, **69 caen cerca de las manos y una
sola lejos**: no hay nada que separar automáticamente.

## El segundo, con la unidad correcta

El problema era el tamaño de la unidad. Un recorte de persona entero puede tener un guante
real adentro; un **parche ajustado a cada detección** afirma solo sobre esa detección, y se
revisa de un vistazo.

Se generaron 389 parches, se revisaron, y quedaron **62 que sí son guante y 327 que no**. Con
ese veredicto se rehicieron los 229 recortes de persona correspondientes, esta vez con las
etiquetas **corregidas**: las detecciones falsas no se etiquetan, los guantes confirmados sí.
Ninguna etiqueta afirma de más.

## Resultado

| Modelo | Umbral | Partición | Falsos positivos | Peleadores perdidos | Fuentes vivas |
|---|---|---|---|---|---|
| original | 0,45 | 88/103 | 17 | 8/125 | 6 |
| original | 0,25 | 90/105 | **30** | 5/125 | 6 |
| etiquetas vacías | 0,45 | 21/23 | 1 | **82/125** | **2** |
| **corregido** | **0,30** | **86/103** | **8** | 9/125 | 6 |

**Los falsos positivos bajan de 17 a 8 con la partición casi intacta**, 86 aciertos contra 88
sobre los mismos 103 tracks. Dos errores de partición a cambio de nueve falsos menos, que son
el error invisible: los keypoints de un espectador exportados como peleador no se ven en el
overlay.

Bajarle el umbral al modelo **original** no logra lo mismo: los falsos se disparan a 30. La
mejora viene del detector.

## Lo que se aprendió sobre el umbral

`umbral_guante` no es una constante, es **una propiedad del detector**, y tratarlo como
constante casi hace descartar el modelo bueno. Con 0,45 —calibrado para el original— el
detector corregido dejaba a `Sparring` con tres tracks en el núcleo, ninguna pareja que
coexista, y la fuente no decidía nada. Movido a 0,30 vuelve a andar.

Si se cambia el modelo, hay que recalibrar.

## Reproducir

```bash
conda activate twinboxing_env
cd ~/Proyectos/TwinBoxing/boxtwin-guantes
python3 tools/parches.py --evidencia <dir> --out data/parches
python3 tools/parches.py --evidencia <dir> --out data/parches --son-guante <indices>
boxtwin-guantes train data/recortes-corregidos --out modelos-corr
```
