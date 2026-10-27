# 06 · Avatar 3D

Pedido (13-10): el avatar 3D del perfil táctico (pantalla 06).

Tareas: B14 (etapa `elevar_3d` del worker y API), W17 (visor en la web). Esfuerzo: 5 a 6 días.
Es el pedido más caro y el de más riesgo; va detrás de una bandera propia (`avatar_3d`).

---

## 1. Qué es el avatar, en esta versión

El sistema tiene una cámara y keypoints en 2D. La profundidad no se mide: la infiere un modelo
entrenado para "elevar" poses 2D a 3D a partir de cómo se mueve el cuerpo. El avatar es, entonces,
una reconstrucción estimada, y se presenta así.

Esto toca un hallazgo firme del proyecto: el hook falla por proyección monocular (un arco
alrededor del eje vertical se ve como recta en el plano de la imagen), y la conclusión fue que
la solución requiere geometría 3D o varias cámaras. Un modelo de elevación no es geometría medida:
pone la profundidad que aprendió de otros cuerpos. Por eso el avatar no se usa como evidencia de
nada, y en particular no se dice que resuelve el hook. Lo que sí abre es un experimento barato que
hoy no se puede hacer (§6).

DA-6 · Qué muestra. Tomada: postura promedio; la repetición del golpe queda en P2.

- Postura promedio (tomada como alcance del 03-11): la guardia del peleador en reposo,
  promediada sobre los cuadros sin golpe, por contexto. Se puede rotar. Es lo que el diseño de la
  pantalla 06 dibuja en el lugar del avatar.
- Postura y repetición del golpe: además, desde el análisis de sesión, "ver en 3D" un golpe
  elegido (la secuencia de cuadros de ese golpe). Más vistoso y más expuesto: es justo donde la
  profundidad inventada se nota, y un hook que se ve como recto en 3D es peor que no mostrarlo. Se
  especifica como P2.

## 2. Modelo de elevación

DA-7 · Modelo de elevación. Tomada: MotionBERT vía MMPose, con la reconstrucción propia como plan B
de la postura.

- MotionBERT vía MMPose (tomada). MMPose 1.3.2, la versión ya fijada en `boxtwin_mmaction`,
  trae MotionBERT y VideoPose3D entre sus modelos de keypoints 3D de cuerpo, con su demo de
  elevación. No hay que instalar ni actualizar nada en ese entorno, que es la regla dura del
  repositorio. Entrada: 17 puntos en formato Human3.6M; la conversión desde COCO-17 (pelvis como
  punto medio de caderas, columna, tórax y cuello interpolados, cabeza desde nariz y orejas) es la
  que hace la demo de MMPose.
- Reconstrucción propia por largos de huesos: con los largos de cada segmento estimados sobre la
  sesión, la profundidad relativa de cada articulación sale de `dz = √(L² − l²)`, con `l` el largo
  proyectado. No necesita modelo ni licencia, pero tiene el signo de `dz` indefinido en cada hueso
  (adelante o atrás), que es exactamente la ambigüedad del hook. Sirve como plan B para la postura
  promedio, que es casi estática; no sirve para golpes.

### Riesgo de licencia (a sumar al Cap 6 como R-21)

Todos los modelos de elevación de MMPose están entrenados sobre Human3.6M. La licencia de
Human3.6M da uso gratuito solo académico y deriva el uso comercial a una licencia aparte; no dice
nada explícito sobre modelos entrenados con los datos. Para el PFI (uso académico) no hay
problema. Para el producto del Cap 5 es una situación parecida a la de Ultralytics: o se licencia,
o se reemplaza por un modelo entrenado sobre datos de licencia permisiva, o se usa la
reconstrucción propia para la postura. Se declara en el Cap 6 con esas tres respuestas, y el
avatar queda fuera del flujo de fondos del Cap 5 hasta resolverlo.

## 3. B14 · Worker

Etapa nueva `elevar_3d`, en el entorno `boxtwin_mmaction`, llamada por subproceso como el
clasificador:

1. Entrada: `poses.npz` de la sesión.
2. Por peleador: tramos continuos de cuadros válidos; huecos de hasta 0,5 s se interpolan, más
   largos cortan el tramo. Tramos de menos de 1 s se descartan.
3. COCO-17 → H36M-17, normalización por tamaño de imagen como espera el modelo, elevación con
   MotionBERT.
4. Salida: `poses3d.npz` con `xyz (2 × T × 17 × 3)` en float16, relativo a la pelvis, y `valid3d`.
   Unas decenas de MB por hora de video: se guarda como `poses.npz`.
5. Escala: si el peleador tiene `altura_m` cargada, se escala para que la altura de pie de la
   postura promedio coincida; si no, unidades del modelo, y el visor no muestra medidas.
6. Postura promedio: media robusta (mediana por coordenada) sobre cuadros sin golpe, excluyendo
   ±0,3 s alrededor de cada golpe detectado. Por sesión, en la Fight-Card como
   `peleadores.X.postura_3d` (17 × 3 números, unos pocos KB).
7. Tiempo registrado en `sesion.json`. Cuenta para C5 (≤ 2× la duración). Si lo rompe, la etapa
   pasa a correr en diferido, después de marcar la sesión `listo`, y el avatar aparece cuando
   termina.

El perfil (`GET /boxeadores/{id}/perfil`) suma `postura_3d` por contexto: mediana de las posturas
de sus sesiones, con la cantidad de sesiones que la forman, y guardia para espejar a los zurdos de
forma consistente.

En P2: `GET /fightcards/{id}/golpes/{golpe}/3d` devuelve la secuencia de ese golpe con 0,3 s antes y
después.

## 4. Validación (C10), pre-registrada

No hay verdad 3D contra la cual medir: nadie filmó estas sesiones con captura de movimiento. Lo que
sí se puede medir es si la reconstrucción es coherente y si a un boxeador le parece plausible.

C10, sobre las cuatro fuentes de medición de `03` y las sesiones de prueba:

1. Coherencia de huesos: coeficiente de variación del largo de cada segmento (antebrazo, brazo,
   muslo, pierna, torso) a lo largo de la sesión ≤ 0,10 en la mediana de los segmentos.
2. Cambios de lado: fracción de cuadros en que la reconstrucción invierte izquierda y derecha
   respecto del 2D ≤ 2%.
3. Plausibilidad: Lucas mira 20 posturas promedio (de peleadores y contextos distintos) y 20 golpes
   (para P2), y marca cada uno como plausible o no con una regla escrita antes: plausible si la
   guardia, el lado adelantado y la altura de las manos coinciden con lo que se ve en el video.
   Umbral: 16 de 20 (80%) en posturas. Los golpes se reportan sin umbral.

Si 1 y 2 pasan y 3 pasa en posturas, la bandera `avatar_3d` se enciende para la postura. Si no, el
lugar del avatar lo ocupa la postura 2D de B9 (P2 de `01-producto.md`) y se declara en el Cap 7.

## 5. W17 · Web

- Componente `Avatar3D` con Three.js. No entra en el JavaScript inicial: se carga con import
  dinámico cuando el bloque entra en pantalla, para no romper el presupuesto de 380 KB por ruta
  (`npm run presupuesto` lo verifica). Three.js 0.186 entero, minificado y comprimido, pesa
  unos 185 KB (medido el 13-10); `/peleador/` está hoy en 310 KB, así que cargado de entrada la
  ruta quedaría cerca de 495 KB. Con import dinámico el inicial no cambia; el bloque muestra un
  esqueleto de carga mientras baja.
- Esqueleto de 17 puntos como cápsulas y esferas, sin malla de cuerpo ni cara: un maniquí. Color A
  o B según el lado que tenga el peleador en la sesión de origen; neutro en el perfil.
- Arrastrar rota sobre el eje vertical; dos botones fijos "De frente" y "De costado". Sin zoom libre
  en celular, para que el gesto no pelee con el desplazamiento de la página.
- Rótulo fijo: "Reconstrucción estimada desde una sola cámara. La profundidad la pone un modelo, no
  se mide." Y la cantidad de sesiones que forman la postura.
- Sin WebGL, o con `prefers-reduced-motion`, se muestra la postura en 2D de frente, sin rotación.
- Alternativa accesible: texto con guardia, altura de las manos respecto de la barbilla y apertura
  de pies en anchos de hombro, calculados de la misma postura.

## 6. Experimento que el avatar habilita (opcional, no comprometido)

Con `poses3d.npz` sobre las fuentes anotadas, se puede probar algo que hoy no se puede: si la
profundidad inferida separa hook de recto mejor que el 2D. Una medida simple: la trayectoria de la
muñeca respecto del hombro en el plano horizontal, curvatura sobre hooks y rectos anotados. Si
separa, es un argumento para la sub-hipótesis 1 que el Cap 8 puede mencionar como línea abierta; si
no separa, confirma el hallazgo del hook con otra herramienta. Las dos salidas son resultado. No
entra al plan antes del 03-11.

## 7. Criterios de aceptación

- CA-17.1. Con la bandera apagada el bloque sigue "No disponible" y Three.js no se descarga.
- CA-17.2. El JavaScript inicial de `/peleador/` sigue por debajo de 380 KB con el avatar.
- CA-17.3. El rótulo de reconstrucción estimada aparece siempre que aparece el avatar.
- CA-17.4. La etapa `elevar_3d` no cambia ningún número de la Fight-Card.
