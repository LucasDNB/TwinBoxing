# E15 · Segunda cámara lateral para el hook (protocolo y pre-registro, fase 3)

30-10-2026. Fase 3: necesita filmar. Si no se filma antes del 17-11, entra al Capítulo 8 como
diseño rotulado con este pre-registro. Correcciones hasta antes de la primera filmación.

## 1. Por qué

El hook confundido con el recto es el hallazgo más firme del proyecto: cinco mediciones
convergentes, reentrenar con más fuentes y etiquetas corregidas movió la familia 0,001, y la 3D
monocular (MotionBERT en E5, 4D-Humans en E7) no lo arregló. La explicación en pie es geométrica:
un arco alrededor del eje vertical, visto de frente, se proyecta como una recta. La literatura
que reconstruye boxeo de élite en 3D lo hace con varias cámaras, triangulación y restricciones
epipolares (arXiv 2504.08175). Antes de triangular, la pregunta barata es si **una segunda vista
2D, a 90°**, ya separa el hook.

## 2. Protocolo de filmación (fijo desde ahora)

- **Material**: bolsa y sombra, una sola persona (Lucas u otro boxeador adulto con
  consentimiento escrito, Ley 25.326). Sin sparring: sin rival no hay oclusión ni identidad,
  que es lo que se quiere aislar.
- **Cámaras**: dos celulares en trípode, a la altura del pecho, a 3 m del boxeador; cámara F de
  frente (la vista del producto) y cámara L a 90° del lado de la mano adelantada. 1080p, la
  mayor tasa de cuadros que den los dos por igual (60 fps si se puede), exposición fija.
- **Sincronización**: un aplauso fuerte al inicio y al final de cada toma; el desfase se calcula
  por correlación cruzada del audio y se verifica con el cuadro del aplauso.
- **Guion**: el boxeador sigue una secuencia leída de una lista que se arma antes de filmar con
  semilla fija: bloques de 10 golpes de un mismo tipo (jab, cross, hook de adelante, hook de
  atrás, uppercut de adelante, uppercut de atrás) en orden aleatorio, cabeza y cuerpo, con 1 s
  entre golpes. **Las etiquetas salen del guion**: Lucas solo marca el inicio de cada bloque y
  corrige en el anotador los golpes que no siguieron el guion.
- **Volumen**: 6 tipos × 2 alturas × 30 golpes = 360 golpes por persona; dos personas si se
  consigue, para poder dejar una afuera.

## 3. La pregunta y la regla

| Id | Pregunta | Regla |
|---|---|---|
| P20a | Recall del hook con E4a (sin rival: hay una sola persona) sobre la vista F sola contra F+L (las 20 features de cada vista, concatenadas, 20 + 20), sobre las ventanas del guion | **Mejora** si F+L supera a F por 0,10 o más en recall del hook, con cinco semillas, en el fold de la persona afuera (si hay dos) o en validación cruzada por bloques de guion (si hay una) |
| P20b | Familia y ocho clases, F contra F+L | sin umbral |
| P20c | L sola | sin umbral; si L sola ya separa el hook, una cámara lateral reemplazaría a la frontal para ese análisis |

El 0,10 (no el 0,03 de los demás) es porque el material es de guion, más fácil que un sparring:
una mejora chica acá no sobreviviría al material real.

## 4. Qué decide cada resultado

- **Mejora**: Cap 8 con evidencia: el hook necesita una segunda vista, y alcanza con una lateral
  2D sin triangular. Consecuencias para el producto: S-01 del Cap 6 (una cámara fija) pasa a
  "dos cámaras para el análisis de técnica"; el modo de captura cambia y eso es otra versión.
- **No mejora**: el siguiente paso es triangular (calibración con tablero ChArUco filmado por
  las dos cámaras), con otro pre-registro.

## 5. Lo que no separa

- Guion contra sparring real: el material es de guion a propósito.
- Ángulos distintos de 90°: uno solo.
