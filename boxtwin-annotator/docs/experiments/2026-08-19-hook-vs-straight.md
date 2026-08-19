# Hook contra straight no es separable con descriptores 2D

**19-08-2026 · Sparring.mp4, 101 golpes completos · resultado negativo, con consecuencias.**

## Por qué se buscó

De los 8 desacuerdos de `punch_type` en la reanotación ciega, **6 son golpes anotados
`hook` y reanotados `straight`**, contra 1 al revés. La confusión es direccional, no
simétrica: no es ruido, es una frontera que se aplica distinto según el día.

El objetivo era darle al anotador un criterio medible en vez de uno de apreciación.

## Lo que se probó

Cinco descriptores sobre los golpes completos, cada uno con su AUC contra la etiqueta:

| Descriptor | AUC |
|---|---|
| Rectitud de la trayectoria de ida | 0,638 |
| Ángulo del codo en la máxima extensión | 0,608 |
| Apertura del codo durante el golpe (pico − mínimo previo) | 0,535 |
| **Desplazamiento del codo respecto de la recta hombro-puño, con signo** | **0,534** |
| Barrido angular alrededor del hombro | 0,510 |
| Rango total del ángulo del codo | 0,482 |

El mejor umbral posible, sobre el mejor descriptor, acierta 62% contra 59% de predecir
siempre la clase mayoritaria. **Tres puntos sobre adivinar.**

Los dos últimos merecen mención porque codifican directamente la definición del anotador:
que el codo se mantenga flexionado en ángulo constante (hook) contra que se extienda
siguiendo al puño (straight), y que el codo quede por debajo de la recta hombro-puño en el
straight. Ninguno separa.

Sobre los seis casos ambiguos, el descriptor del codo bajo la recta da "straight" a los
seis, con valores entre +0,025 y +1,160. No es que se equivoque en el borde: no está
midiendo lo que el ojo mide.

## Interpretación

La explicación más probable es la **proyección monocular**. Un hook es un arco alrededor
del eje vertical del cuerpo; si ese arco ocurre en un plano que contiene al eje óptico de
la cámara, en la imagen se ve como una línea. Toda la geometría que distingue las dos
familias vive en la dimensión que la cámara colapsa.

**Lo que esto NO prueba**: que la información no esté en los keypoints. Un modelo
espaciotemporal ve la secuencia completa de las 17 articulaciones, incluida la rotación de
cadera y hombro en el tiempo, y puede combinar señales que cinco escalares hechos a mano no
capturan.

**Lo que sí sugiere**: que la familia del golpe es el eje difícil de este problema, y que
conviene declararlo antes de entrenar. Converge con tres cosas medidas por separado:

- El acuerdo intra-anotador: `side` kappa 1,000 contra `punch_type` 0,755.
- BoxingVI: V4 se equivocaba en familia conservando la lateralidad, V10 al revés.
- El baseline de Bhargav colapsa fuera de distribución, y su matriz de confusión también se
  concentra por familia.

## Qué se hizo en consecuencia

El panel de definiciones operacionales **no definía el tipo de golpe**. Solo tenía las
fronteras temporales. Y se nota en los números: lo definido da error de 1,12 cuadros, lo no
definido da kappa 0,755 con sesgo direccional.

Se agregó la definición, escrita por el anotador, que es quien pone el criterio de dominio.
Queda visible mientras se anota, junto a las fronteras.

`boundary_definitions_version` pasa de 1 a 2. Los eventos anotados antes y después no se
produjeron bajo el mismo criterio y tienen que poder distinguirse: los 125 de Sparring.mp4
son versión 1.

## Qué queda abierto

Si al entrenar el clasificador la confusión hook/straight domina la matriz, hay dos lecturas
y no se pueden separar con los datos actuales: que el modelo no aprendió, o que la
distinción no está en la pose monocular. Distinguirlas necesitaría material con dos cámaras,
o etiquetas de un experto sobre los mismos clips.
