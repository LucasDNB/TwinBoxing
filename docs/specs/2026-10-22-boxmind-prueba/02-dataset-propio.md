# Propuesta · Dataset propio para un BoxTwin comercial

22-10-2026 · condicional: se presenta según el gatillo de `01-preregistro.md` §6 · alcance:
el producto hipotético del Capítulo 5, no el PFI

## 1. Para qué y cuándo

BM pregunta si diez veces más golpes con el esquema de BoxMind mueven el clasificador sobre
fuentes no vistas. Si la respuesta es sí (gatillo **respaldada**), el cuello de botella son
los datos y un producto comercial no puede depender de BoxingWeb: no tiene licencia, es otro
dominio (transmisión olímpica) y nada entrenado con él puede venderse. Esta propuesta describe
cómo construir un conjunto propio con las mismas características, en el dominio del producto.

Si el gatillo da **no respaldada**, la propuesta queda escrita como alternativa evaluada y no
recomendada: el dinero va antes a la segunda cámara (sección 4.3), que es la única ruta con
evidencia de resolver el hook.

## 2. Qué se copia de BoxingWeb y qué no

Se copia el esquema de evento atómico de BoxMind, porque es el que permite comparar y porque
calza con el tipo v2:

| Campo | BoxingWeb | Dataset propio |
|---|---|---|
| Inicio | la mano sale de la guardia | igual |
| Fin | retracción completa | igual |
| Mano | física (izquierda / derecha) | física, y rol lead / rear con la guardia anotada |
| Técnica | recto, hook, uppercut | igual; jab y cross salen del rol |
| Objetivo | cabeza, pecho, abdomen | igual (tres niveles, se pueden colapsar a dos) |
| Distancia | larga, media, corta | igual |
| Efecto | efectivo / no | **no**: con una cámara no se puede afirmar contacto, y el producto no lo muestra |
| Identidad | rojo / azul por ronda | A / B por sesión, con la siembra del producto |

Lo que no se copia: el dominio. El conjunto propio es sparring de gimnasio filmado con
celular en cámara fija (el supuesto S-01 del Capítulo 6), que es lo que el producto va a ver.

## 3. Tamaño

Referencias medidas: BoxMind entrenó con 10,9 mil eventos en 80 rondas (unos 45 golpes por
minuto); `sparring-3-rounds` tiene 381 golpes completos en 9,7 minutos (unos 39 por minuto).

- **Fase 0, piloto de costo:** un gimnasio, unos 500 golpes. Mide el tiempo real de
  anotación por golpe con el flujo del producto, que es el número que falta para presupuestar.
- **Fase 1, conjunto base:** del orden de BoxingWeb, unos 10.000 golpes, que son unas 4 horas de sparring
  (entre 3,7 y 4,3). El número definitivo sale de la curva de BM (25%, 50% y 100% de
  BoxingWeb): si la familia sube con la cantidad, se extrapola la cantidad para la meta; si la
  curva es plana, la fase 1 no se justifica.
- **Fase 2, crecimiento:** las correcciones de tipo que hacen los usuarios en la web (F14)
  entran como etiquetas sobre material no visto, con consentimiento (la oportunidad O-01 del
  Capítulo 6). El producto alimenta su propio conjunto.

Diversidad mínima de la fase 1, porque la lección de BoxTwin es que el clasificador no
generalizaba a fuentes nuevas: al menos cinco gimnasios, treinta boxeadores, ortodoxos y
zurdos, varones y mujeres, distintas categorías de peso y nivel, y distintos celulares,
resoluciones, luz y ángulos. Ningún boxeador ni gimnasio en más de una partición.

## 4. Protocolo

### 4.1 Captura

- Celular fijo en trípode, los dos peleadores en cuadro, guantes de distinto color (S-01).
- Rondas completas de sparring; manopla como segmento aparte (Suárez la marcó como el trabajo
  técnico más importante, Apéndice A), con su propio rótulo.
- Metadatos por sesión: gimnasio, cámara, resolución, fps, guardia de cada boxeador.

### 4.2 Anotación

- Herramienta: el anotador de BoxTwin. El detector del producto propone los segmentos y el
  clasificador propone tipo y altura; la persona corrige. Es la diferencia entre segmentar
  desde cero (cerca de un minuto por golpe en BoxingVI) y corregir sobre una segmentación
  existente (2,5 segundos por clip, mismo registro). El piloto de costo mide dónde cae el
  flujo real entre esos dos extremos.
- Guía de anotación escrita con las definiciones de la tabla de la sección 2 y ejemplos.
- Doble anotación del 10% por un entrenador, con kappa por campo; un campo con kappa bajo se
  redefine en la guía antes de seguir. Thomson et al. (2013) miden variabilidad entre
  observadores de 33% a 100%: el techo humano es parte del resultado.
- El tiempo por golpe se registra en cada sesión de anotación.

### 4.3 Segunda cámara en una parte

En el 20% de las sesiones, un segundo celular a unos 90 grados, sincronizado por una señal
visible y sonora al inicio. Con dos vistas se triangula un 3D de referencia, que sirve para
dos cosas: medir cuánto se equivoca MotionBERT en este dominio y entrenar o ajustar la
elevación 3D con datos propios. Es lo que BoxMind hizo con BoxingStudio (cuatro vistas, 60
fps) y es la única vía con evidencia contra el problema del hook, que en BoxTwin es de
proyección monocular.

### 4.4 Particiones y evaluación

Particiones por gimnasio (dejar uno afuera) y no por golpe. Las métricas son las de E4 y BM:
familia, altura balanceada, ocho clases, con Wilson, ensamble de cinco semillas y la regla de
0,03 y folds ganados. Un test fijo de dos gimnasios queda sellado hasta la evaluación final.

## 5. Datos personales y consentimiento

Video de personas identificables es dato personal bajo la Ley 25.326. No soy abogado; esto es
lo que el proyecto ya asumió en el Capítulo 6 y lo que habría que confirmar con asesoramiento:

- Consentimiento informado por escrito de cada boxeador, con la finalidad explícita
  (entrenar y evaluar modelos), el plazo de conservación y cómo pedir acceso o supresión.
- Solo mayores de edad, como en el piloto del PFI.
- Inscripción de la base de datos ante el organismo de control (AAIP) antes de recolectar.
- Si un boxeador retira el consentimiento, sus golpes salen del conjunto y del siguiente
  entrenamiento; los modelos ya entregados se documentan con la versión del conjunto que
  usaron.
- Licencia del conjunto: todos los derechos reservados, como el resto del proyecto.

## 6. Costo, para el Capítulo 5

El costo es casi todo horas de anotación y se expresa así hasta tener el piloto:

horas = golpes × minutos por golpe ÷ 60 × 1,1 (doble anotación del 10%)

Con 10.000 golpes, entre unas 8 horas (corrección a 2,5 s por golpe, cota inferior poco
realista) y unas 180 horas (segmentación desde cero a un minuto). El piloto de la fase 0
cierra ese rango. A eso se suman las horas de captura en los gimnasios, el segundo celular
del 20% de las sesiones y el honorario del entrenador que hace la doble anotación. En el flujo
de fondos va como inversión del año 0, a la tarifa horaria del Capítulo 5.

## 7. Riesgos nuevos, para el Capítulo 6

- Debido a que la calidad depende de un solo anotador, podría ocurrir que el conjunto herede
  su criterio, lo que provocaría un modelo que generaliza a los gimnasios pero no a otros
  entrenadores. Respuesta: mitigar con la doble anotación y la guía.
- Debido a que los gimnasios ceden material de sus alumnos, podría ocurrir que no se consigan
  cinco gimnasios con consentimiento, lo que provocaría un conjunto con poca diversidad.
  Respuesta: mitigar con el plan gratuito como canal (Capítulo 5) y la federación vía Suárez.
- Debido a que el consentimiento se puede retirar, podría ocurrir que un conjunto ya usado
  pierda golpes, lo que provocaría modelos no reproducibles. Respuesta: aceptar, versionando
  el conjunto.
