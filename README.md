# BoxTwin

  26-04 Instalacion de controladores y dependencias
  1. MiniForge3 conda
  2. CUDA toolkit
  3. ultralytics opencv numpy shapely matplotlib jupyterlab 

  12-05 Nombre cambiado a BoxTwin y testeos de dataset de terceros (Experimento 1)
  1. Entorno paralelo boxtwin_mmaction (PyTorch 2.1 + CUDA 11.8, stack OpenMMLab 2.1.0 pinneado)
  2. Fine-tuning del modelo sobre dataset Bhargav (270 train / 71 val, 6 clases de golpes)
  3. Mejor checkpoint: epoch 8, 84.51% top-1 (paper reporta 87.32%)
  4. Demo realizada sobre videos de terceros
  5. Se deben realizar mas testeos

13-05 Experimento 2: Smoke test del pipeline base sobre video de entrenamiento
  1. Pipeline YOLOv8l-pose + BoT-SORT sobre 3 formatos (bolsa, sombra, sparring)
  2. Bolsa y sombra: 1 ID único, 0 falsos positivos, pipeline base suficiente
  3. Sparring: 46 IDs únicos confirma que la oclusión mutua entre peleadores es el problema central
  4. Performance >32 FPS en los tres formatos (target del proyecto: 25-30 FPS)
  5. Justifica experimentalmente la lógica de dominio (ROI + top-2 + Hungarian) declarada para Fase 1
Stack YOLOv8l-pose + BoT-SORT sin logica de dominio, confidence 0.5, entorno twinboxing_env, RTX 2080 Super
Bolsa: 716 frames, 32.7 fps, 1 ID unico, 0.96 detecciones promedio por frame, max 1
Sombra: 435 frames, 34.9 fps, 1 ID unico, 0.97 detecciones promedio, max 1
Sparring: 5531 frames, 36.7 fps, 46 IDs unicos, 1.88 detecciones promedio, max 4
Bolsa y sombra no requieren logica adicional: cero falsos positivos, el tracker no se confunde con la bolsa, el espejo ni el fondo del gimnasio
Sparring da ~23 IDs por peleador, comparable a los ~25 por boxeador del smoke test de Fase 0 sobre pelea profesional
Hipotesis principal refutada para sparring: el problema no es el ruido del entorno profesional (publico, arbitro, camara dinamica) sino la dinamica de dos personas en interaccion cercana
Causa principal de ID switch: oclusion mutua entre peleadores
Tres tipos de falso positivo observados: persona en la esquina, peaton de fondo, poster en la pared
Todos los formatos superan el objetivo de 25-30 fps
Documentado en docs/experiments/02_smoke_test_training_video.md, resultados en 02_artifacts/results.json

09-06 Fase 1 Bloque A componente 1: Ring ROI

ring_roi.py: clase RingROI que filtra detecciones de YOLOv8-pose por test de punto en poligono contra un cuadrilatero definido manualmente
define_roi.py: utilidad interactiva para marcar el poligono del ring sobre el primer frame y guardarlo en JSON
El punto de apoyo se deriva de los keypoints de tobillo (COCO-17 indices 15 y 16), no del centro del bbox, siguiendo el principio de geometria derivada de pose
Fallback al centro inferior del bbox cuando la confianza de tobillo esta por debajo del umbral
El filtro se aplica aguas arriba de BoT-SORT para que el tracker nunca asigne IDs a falsos positivos, reduciendo la carga sobre el matcher
Descartada la deteccion automatica de ring: agrega un componente que requiere validacion independiente, el alcance ya estaba cerrado en Cap 1, y con camara fija la posicion del ring es constante durante la sesion

23-06 Arquitectura dual LIVE/VIDEO y schema de anotacion v0.3

Abandonado el supuesto de camara fija como unico caso. Modo LIVE (camara fija en gimnasio) conserva el ROI manual
Modo VIDEO (broadcast, camara dinamica) reemplaza el ROI por un scorer top-2 agnostico a camara: area de bbox, centralidad, confianza de keypoints y persistencia temporal
Deteccion de guantes evaluada como señal blanda, no como filtro duro. Se agrega solo si el scorer top-2 deja falsos positivos residuales medidos
Schema de anotacion congelado en v0.3 con 13 clases: 8 golpes (jab, cruzado, hook, uppercut x cabeza y cuerpo), 3 defensas (bloqueo, slip, parry), feint generico y background
Onset de golpe definido como inicio de extension del brazo con validacion de completitud, operacionalizado geometricamente sobre keypoints, para distinguir feints de golpes reales
Campo responds_to obligatorio en anotaciones defensivas, con occluded como valvula de escape
Footwork diferido a v2
Subdivision cabeza/cuerpo se anota desde el inicio pero es colapsable en tiempo de entrenamiento si el volumen no alcanza

02-07 Fase 1 Bloque A componente 2: selector de peleadores

fighter_selector.py: arquitectura de dos capas sobre BoT-SORT
BoT-SORT asigna track_ids crudos, el selector puntua todos los tracks activos y elige los 2 mejores candidatos por frame, y el matching hungaro resuelve la asignacion entre esos candidatos y el estado previo de Fighter A / Fighter B
El hungaro no trackea personas: mantiene estable la identidad de rol A/B a traves de los ID switches de BoT-SORT confirmados en Experimento 02
Costo del hungaro puramente espacial en v1: distancia de punto de apoyo mas IoU inverso, sin descriptor de apariencia
Smoke test: cuando BoT-SORT cambia el ID del peleador derecho de 2 a 99 simulando un switch por oclusion, la asignacion mantiene la continuidad de rol
Persona de fondo con bbox chica, posicion de esquina y baja confianza queda excluida de la seleccion top-2
Pesos de scoring (area 0.35, centralidad 0.25, confianza 0.20, persistencia 0.20) son un punto de partida razonado, sin calibrar
Pendiente: calibrar pesos contra footage real, decidir si se agrega termino de apariencia al costo, y fijar un valor fundamentado de max_frames_lost

15-07 Descarga e integracion del dataset BoxingVI

9 de 10 videos descargados desde YouTube. V6 con link muerto, no recuperable
Anotaciones distribuidas en 10 planillas Excel con formatos incompatibles entre si: algunas con headers usables y variantes sucias, otras sin header (columnas Unnamed) o con el header pegoteado
Loader multi-estrategia: deteccion de columnas Start/End/Class por nombre, con fallback posicional a las tres primeras columnas
Validacion por fila (frames enteros, end mayor que start, clase canonica) con reporte de descartes por planilla
Resolucion de anotacion a archivo de video en tres estrategias: match directo por nombre, via Meta_data.ods extrayendo el id de YouTube del link, y match difuso
5442 anotaciones validas de las cuales 685 corresponden a V6 y no tienen video

30-07 Pipeline de dataset BoxingVI: descarga, corte y split

Fix de fps en boxingvi_clip.py: la v0.1 reencodeaba a 30 fps constante segun declara el paper, verificacion empirica muestra fps mixtos (V1 24, V10 25, V2/V3/V4/V9 23.976, V5/V7/V8 29.97)
El reencoding corria cada indice de frame por el cociente fps_nativo/30 de forma acumulativa, error silencioso
Test discriminante: duracion mediana de golpe usando los videos a 29.97 como control (ahi las dos hipotesis coinciden). Bajo fps nativo el grupo de 24 fps da 0.359 s promedio contra 0.356 s del control; bajo 30 fps da 0.286 s, exactamente 20% mas corto (relacion 24/30)
v0.2: sin reencoding, sondeo con ffprobe, fps como Fraction exacta, seek en (start-0.5)/fps, deteccion de VFR, validacion de rangos
Corte completo: 4757 de 4757 clips, cero fallos, largos verificados exactos sobre muestra de 20
V6 confirmado perdido (link muerto): 685 anotaciones sin video, 12.6% del total nominal
Distribucion: Cross 1371, Jab 1282, Lead Hook 907, Rear Uppercut 473, Lead Uppercut 428, Rear Hook 296. Desbalance 4.6x
boxingvi_split.py: descarta 6 clips de largo no plausible (3 de dos frames, 3 de mas de 30 incluido un uppercut de 111 frames)
Split por video para evitar fuga de dominio, no aleatorio. Val = V5, V9, V10 (893 clips, 18.8%), elegidos por tener las 6 clases. V2 y V8 sin ningun Rear Hook quedan en train, V4 con distribucion invertida tambien
Train 3858 clips. V1 concentra 48.3%, se deja y se mide antes de submuestrear

30-07 Extraccion de poses y seleccion del atacante

Problema: las anotaciones no indican quien tiro el golpe, alimentar al clasificador con el defensor desalinea la etiqueta
Heuristica en boxingvi_pose.py: desplazamiento maximo de muñeca en la ventana, normalizado por largo de torso para invariancia a escala
Asociacion greedy por centroide entre frames, sin BoT-SORT, para no arrastrar decisiones sin calibrar del fighter_selector
Verificada sobre casos sinteticos: elige al atacante con margen 0.944, da score identico ante mismo gesto a distinta escala, y en intercambio simultaneo devuelve margen 0.063 que queda marcado como ambiguo
Validacion sobre 30 clips reales: en V3, V4, V5, V7 y V8 el estimador detecta exactamente una persona. En V1 detecta entre cero y ocho
Causa en V1: hay boxeadores pintados en la pared del gimnasio. El estimador los detecta como personas y en un clip donde el boxeador real queda tapado por la bolsa la heuristica eligio una pintura
La normalizacion por torso amplifica el ruido de deteccion de figuras chicas: pocos pixeles de jitter sobre torso pequeño dan score alto
Pendiente: mascara de personas estaticas por video fuente, sirve tambien para LIVE por los posters de gimnasio
Cuatro de los 30 clips dan cero detecciones, son placas de texto insertadas en el video

30-07 Verificacion manual de calidad de anotacion del dataset

Revision a ojo de clips por video: V4 5/5 correctos, V7 3/3, V5 3/4, V3 1/4, V1 1/6, V2 0/3
Los errores no son corrimiento temporal, las ventanas caen sobre golpes reales y bien encuadrados. Falla la clase
Tampoco es un mapeo recuperable, los errores de V1 son variados (Rear Hook que es uppercut, otro que es lead hook, otro sin golpe)
V2 falla 0 de 3 siendo el video mas chico, cae la hipotesis de que el problema viene del volumen anotado
boxingvi_annot.py: herramienta web de reanotacion, servidor local, loop a 0.25x, teclado, ciega por defecto, reanudable, registra tiempo por clip
V2 reanotado completo: 232 clips en 10.3 minutos, mediana 2.2 s por clip, p90 4.2 s
Coincidencia con la anotacion original 7.3%, por debajo del azar (16.7% con 6 clases)
42.2% de los clips de V2 no contienen ningun golpe. Tramo 9000-9999 es 100% sin golpe (20 de 20), tramo 12000-12999 es 84.6%
Cero de 116 uppercuts eran uppercuts: los 58 Lead Uppercut y 58 Rear Uppercut resultaron jabs (36), crosses (17), hooks (19) o nada (41)
Lo que realmente hay en V2: 54 jabs, 42 crosses, 29 hooks, 7 rear hooks, 2 uppercuts en total
V2 y V1 descartados. V1 tiene segmentacion rota ademas de clase, recuperarlo implicaria resegmentar
V3 pendiente de reanotacion, sus errores son entre clases sobre golpes reales, caso recuperable, 811 clips a unos 36 minutos
V8, V9 y V10 sin verificar (549 clips). V9 y V10 estan en el split de validacion
Verificados confiables: V4, V5, V7, total 1349 clips
La validacion automatica no detecta errores de etiqueta: todas las metricas daban perfectas mientras tres videos estaban cerca del azar
Reanotar sobre segmentacion existente cuesta 2.5 s por clip, segmentar desde cero cerca de un minuto por golpe

10-08 Verificacion de anotaciones BoxingVI: placas de titulo y calidad de V2

  1. V1 tiene 209 clips sin persona (181 placas de titulo enteras, 28 sobre un
     corte), 11.2% del video, todos con etiqueta de golpe
  2. Es el unico video afectado: los otros ocho no tienen ningun clip por encima
     del corte de fraccion de negro
  3. Resultado negativo: un primer detector por brillo medio con umbral global
     marco 528 de 810 clips de V3 como placas, todos falsos positivos. V3 es
     metraje real de estudio con fondo oscuro. El brillo medio no distingue
     video oscuro de pantalla negra y no transfiere entre videos
  4. boxingvi_placas.py mide fraccion de negro y movimiento, y exige dos
     poblaciones separadas antes de contar
  5. Los "cuatro clips placas de texto" del 30-07 eran cuatro de los 30 clips
     mirados, no un total. El barrido completo da 209 en V1
  6. Precision sobre la coincidencia de V2: 7.3% es 17/232 sobre el video
     entero, 12.7% es 17/134 restringido a los clips que si contienen un golpe.
     Los dos numeros son correctos, hay que decir cual denominador se usa
  7. Correccion de la distribucion del 30-07: Lead Hook 908 y Rear Uppercut 472
     sobre manifest.csv, no 907 y 473. Sumaban igual, por eso no se veia
  8. El pipeline corre sobre manifest_filtrado.csv (4751), no sobre el total del
     corte (4757). Todo numero que vaya a la tesis tiene que decir cual usa
  9. Las 209 placas caen todas en train, ninguna en validacion. Sacarlas mueve
     los pesos de clase 1.9% como maximo

11-08 Verificacion ciega por muestreo estratificado de V4, V5, V8, V9 y V10

  1. Criterio pre-registrado, 18 clips por video, minimo 2 por clase: 16 o mas
     aciertos se usa tal cual, 11 a 15 se reanota, 10 o menos se descarta
  2. V5 18/18, V8 17/18, V9 17/18 se usan tal cual. V10 15/18 y V4 12/18 se
     reanotan completos
  3. V4 figuraba como confiable con 5 de 5 clips mirados a ojo. Una muestra de 5
     no es evidencia: su piso Wilson es 56.6% y el 66.7% real cae adentro. La
     revision vieja no estaba mal, no informaba nada
  4. Se reporta piso Wilson 95% y no el porcentaje pelado: 18/18 no es calidad
     del 100%, es un piso de 82.4% con n=18
  5. Cero clips sin golpe en los 90 de muestra, contra 42% en V2. En estos cinco
     videos la segmentacion temporal esta intacta y el dano es solo de clase
  6. Cada video falla en un eje distinto y sistematico: V10 se equivoca en
     lateralidad conservando la familia (Rear Hook que es Lead Hook), V4 en
     familia conservando la lateralidad (Rear Hook que es Rear Uppercut)
  7. Rear Hook, la clase mas rara, sale 2 de 2 mal en V4 y en V10, y 2 de 2 bien
     en V5 y V9. V4 aporta 75 de los 136 que sobreviven al descarte de V1 y V2.
     Con n=2 por video no alcanza para afirmar nada, se sabe al reanotar V4
  8. La validacion aguanta: 83% de sus 893 clips son V5 y V9, que pasaron. El
     dano esta en train, donde tras descartar V1 y V2 el unico bloque limpio son
     los 199 clips de V8
  9. boxingvi_verifica.py aplica la tabla y se escribio con el CSV de salida
     vacio, asi que la regla de conteo tampoco se eligio viendo los datos. Se
     niega a aplicar la tabla si el n no es 18 y no redondea los dudosos
 10. V7 sigue pasando como confiable con 3 clips mirados, mismo error que se
     acaba de pagar con V4. Le falta su muestra de 18

19-08 Dataset propio: instrumento medido y el baseline que no medía generalización

  1. boxtwin-annotator terminado y usado sobre material real. Sparring.mp4 anotado
     completo: 125 eventos, 66 asignaciones de identidad, 243 interpolaciones
  2. Reanotacion ciega sobre muestra de 35 ventanas, semilla 7: side kappa 1.000,
     punch_type 0.755, target 0.635, completeness 0.651 sobre 51 golpes emparejados.
     Deteccion recall 0.911, precision 0.903
  3. Error de fronteras 1.12 cuadros al inicio y 1.55 al final, sobre golpes que
     duran 9.9. Es el techo contra el que hay que reportar el error del modelo, no
     cero
  4. El primer protocolo de reanotacion media su propia ambiguedad: pedia reanotar
     un evento sin decir cual, y 26 de 35 ventanas contienen mas de un golpe del
     mismo peleador. 15 de 34 intentos reanotaron el golpe de al lado y el reporte
     los conto como desacuerdo. Rehecho pidiendo todos los golpes de la ventana y
     emparejando por solapamiento, como en deteccion temporal de acciones
  5. La reanotacion encontro 7 golpes que faltaban en la anotacion, un subconteo del
     5.6%. Se agregaron, pero el numero de acuerdo NO se recalcula sobre la version
     corregida: seria circular. Queda congelado declarando que se midio sobre 118
     eventos
  6. El 84.51% de PoseConv3D sobre Bhargav no mide generalizacion. Su validacion
     comparte 47 de 49 sujetos con el entrenamiento, el 96%: el split es por clip y
     no por sujeto. Evaluado sobre Sparring.mp4 da 10.53% top-1, por debajo del
     16.7% de azar con 6 clases. El control reproduce el 84.51% al decimal, asi que
     el arnes es correcto
  7. Los 9 videos de BoxingVI son contenido de fitness: shadowboxing y bolsa, una
     sola persona, con cronometros en pantalla. Ninguno es sparring. Sin oponente no
     existen landed, blocked ni slipped, y el subsistema de identidad es irrelevante
  8. Medido que el ReID de BoT-SORT no reduce los cambios de identidad a 640x360:
     31 con ReID contra 29 sin el, y 4.4% mas de tiempo. Falta rehacerlo sobre 1080p
  9. Hook contra straight no es separable con descriptores 2D. Cinco probados, mejor
     AUC 0.638, y el mejor umbral acierta 62% contra 59% de predecir la clase
     mayoritaria. Dos de ellos codifican la definicion del anotador. Explicacion
     probable: proyeccion monocular. El eje dificil de este problema es la familia
     del golpe, no el brazo
 10. Definicion operacional del tipo de golpe escrita y agregada al panel siempre
     visible. Hasta ahora solo estaban definidas las fronteras temporales, y se nota:
     lo definido da 1.12 cuadros de error, lo no definido kappa 0.755 con sesgo
     direccional. boundary_definitions_version pasa a 2
 11. Pelea profesional Pacquiao vs Margarito preprocesada, 12 rounds netos: 136364
     cuadros a 1080p60, 87 minutos con FP16, 1.11 millones de detecciones, 11100
     tracks. FP16 mide 1.8x mas rapido con 0.02% de diferencia en detecciones
 12. 181 cortes de camara, uno cada 12.6 s. El tracker se reinicia en cada uno: sin
     eso la identidad se arrastra entre planos y le pone a un peleador el cuerpo del
     otro, sin que se vea en el overlay
 13. Dos agujeros cerrados por los que casi se pierde trabajo: project_paths seguia
     los symlinks y un proyecto de prueba escribia sobre el original, y el preproceso
     no miraba si existia una anotacion antes de rehacer el cache y renumerar los
     track_id

20-08 Identidad por click, y el validador de colisiones que no validaba

  1. Medido sobre el primer round de Pacquiao: 151 eventos en 2,9 minutos de video
     costaron 70,7 minutos de trabajo, y solo el 22% se fue en clasificar golpes. El
     78% es identidad y navegacion, a 23 s por asignacion. La pelea entera son 15
     horas, no las 5 estimadas
  2. Asignar ahora es click sobre el peleador y Ctrl+1 o Ctrl+2: el track se busca
     solo, tomando la caja mas chica que contiene el punto. Antes habia que buscar el
     track_id en una lista lateral, que es un numero sin significado para quien anota,
     y no habia ningun atajo de teclado de identidad
  3. Ctrl+4 ignora todo lo que en ese plano no sea peleador, acotado al plano entre
     dos cortes de camara: en el siguiente los track_id ya son otros
  4. Las 1165 colisiones de rol eran del validador y no de los datos. Comparaba
     rangos sin mirar si los dos tracks existen ahi; con el cache quedan 10 en
     Pacquiao y 0 en Sparring
  5. Truncar al ocupante anterior al asignar elimina todas las colisiones y empeora
     el dato: los cuadros sin pose dentro de eventos suben de 51 a 218. Los
     solapamientos hacen de respaldo cuando el tracker fragmenta a un peleador y un
     track viejo reaparece. Revertido, con un test que lo fija
  6. IgnoreSmallTracks agregaba su asignacion sin recortar lo que el track ya tuviera,
     dejando dos intervalos del mismo track superpuestos y un resolver que elige por
     orden de lista. 30 casos, todos ignore contra ignore

23-08 Pacquiao round 1 cerrado

  1. 151 eventos. Reparados los 280 solapamientos que dejo la version vieja de
     IgnoreSmallTracks: asignaciones 9910 -> 9658 y la validacion baja a 9
     colisiones, que son las reales
  2. Rellenados los huecos internos, 415 interpolaciones sobre 1089 cuadros. Los
     cuadros sin pose dentro de eventos bajan de 51 a 1, y los eventos afectados de
     21 a 1
  3. 11 de 12 clases con ejemplos y desbalance 63:1, con straight-left-head y
     straight-right-head como el grueso, que sobre Pacquiao es lo esperable. Ritmo
     5,2 s por golpe contra 9,0 en el sparring

25-08 sparring-3, tercera fuente y la primera donde el ritmo mejora

  1. Sparring de gimnasio a 1080p30 con camara casi fija: 11 cortes en 9,7 minutos
     contra los 181 de la transmision profesional, y 257 tracks contra 11.100
  2. Dos rounds, 146 eventos. Aislando el segundo, ya con el click-para-asignar
     rodado: 11,8 minutos de trabajo por minuto de video contra 24,4 en Pacquiao
  3. El tiempo de identidad sigue siendo el 73% pese a hacer 3,7 veces menos
     asignaciones por minuto. Ese tiempo no es de asignar sino de mirar, o sea
     recorrer el video y confirmar que la asignacion se sostiene, y no hay mejora de
     interfaz que lo devuelva
  4. Calidad muy superior: 2 avisos y ningun error de validacion contra 9 colisiones
     reales en Pacquiao, y 4 cuadros sin pose dentro de eventos contra 51
  5. El reparto de clases es genuinamente distinto y se sostiene entre los dos rounds:
     hook 33% contra 15% en Pacquiao. Sumar material de este tipo ayuda a las clases
     raras por composicion y no solo por volumen

01-09 Dataset propio cerrado en 676 eventos, y el primer entrenamiento

  1. sparring-3 anotado entero, 9,6 de 9,7 minutos: 400 eventos, 101 asignaciones y
     314 interpolaciones que bajan los cuadros sin pose dentro de eventos de 27 a 9.
     Ritmo 9,5 minutos de trabajo por minuto de video, tres veces mas eficiente que
     la transmision profesional
  2. Corrige una extrapolacion equivocada. Con 222 eventos la composicion daba
     uppercut 5% y de ahi se dijo que con 360 eventos llegaria a unos 18; con el
     video entero salieron 39, porque los rounds 4 y 5 tuvieron el doble en
     proporcion. Es el mismo error ya documentado dos veces en este proyecto: la
     muestra chica no estaba mal, no informaba nada
  3. El dataset queda en 676 eventos sobre tres fuentes, 418 straight, 209 hook y 49
     uppercut, con las 12 clases pobladas
  4. Primer entrenamiento propio: PoseConv3D sobre las 640 muestras que caen en el
     espacio de 6 clases lead-rear, inicializado del checkpoint de Bhargav y con los
     hiperparametros de aquella corrida sin tocar, a proposito. Cambiar datos e
     hiperparametros a la vez deja sin saber cual movio el numero
  5. En distribucion 62,5% top-1 contra 37,5% de linea de base. Dejando una fuente
     afuera: 33,3 contra 41,2, 58,6 contra 60,7 y 34,1 contra 37,8. Aprende en
     distribucion y no generaliza a una fuente nueva
  6. Descartada la longitud de clip. Los clips tienen 7 cuadros de mediana y el
     pipeline los estira a 48, casi siete repeticiones por cuadro, asi que parecia un
     candidato fuerte. Con clip_len 12, 24 y 48 sale 55,2%, 58,3% y 59,4%: los tres
     dentro del intervalo de +-10 puntos que corresponde a 96 muestras de validacion,
     y si algo hay, favorece al clip largo
  7. Hook contra straight tiene ahora tres mediciones independientes que coinciden.
     De los 13 hooks que el modelo falla, 11 los llama straight; el anotador los
     distingue con kappa 0,881; y ningun descriptor 2D paso de AUC 0,64, dos de ellos
     codificando la definicion escrita por el anotador. La distincion existe y un
     humano la hace, pero no esta accesible en la pose monocular
  8. Bhargav saca 84,51% con 275 muestras de entrenamiento y este trabajo 62,5% con
     285. Mismo volumen, veintidos puntos menos: su material es una persona sola
     haciendo shadowboxing en plano fijo y el de aca son dos boxeadores ocluyendose
     en sparring real. El numero publicado no es comparable
  9. tools/demo_vivo.py corre pose, identidad y clasificacion juntas sobre video, que
     es la primera vez que las tres piezas van juntas, y hace visible el agujero que
     los numeros escondian: encuentra 14 de 21 golpes reales pero dispara 66 veces,
     21% de precision. El clasificador no tiene clase "no hay golpe" porque se
     entreno sobre ventanas que siempre contienen uno
 10. No sigue tunear: la brecha entre en distribucion y cruzado es de mas de 20
     puntos y los hiperparametros mueven cinco. Sigue sumar fuentes, y construir el
     detector de secuencia con carriles BIO, que no esta hecho. El 21% de precision
     del disparador heuristico es la linea de base contra la que se mide
 11. Medido que el disparador por extension de muneca NO supera al azar. Con 42
     golpes por minuto, el 46% de la linea de tiempo esta a menos de medio segundo
     de un golpe por construccion, y esa es la precision de poner paradas al azar;
     el disparador saca 0,52 en sparring-3, 0,49 en Sparring y 0,70 en Pacquiao
     contra 0,46, 0,47 y 0,65. En Sparring es indistinguible del azar
 12. La causa es que la guardia vive en 1,0-1,5 anchos de hombro, o sea donde viven
     los golpes: AUC 0,61 a 0,67 por cuadro, el mismo orden que el 0,64 de hook
     contra straight. Se descarto construir el salto a candidatos, que iba a usarlo
     para navegar. Corrige ademas la lectura del demo: sus 66 disparos no son un
     detector rudimentario sino ruido, y el 21% no habla del clasificador
 13. Construido el detector temporal, en boxtwin-detector: una TCN de convoluciones
     dilatadas con campo receptivo de 63 cuadros que decide por cuadro y por brazo si
     hay golpe. Cuatro decisiones de dataset que no se ven en los tensores y arruinan
     el resultado en silencio: remuestrear las tres fuentes a 30 fps porque Pacquiao
     va a 60 y el mismo golpe dura el doble de cuadros, enmascarar los amagues en vez
     de darlos por fondo, acotar al tramo anotado, y normalizar por
     max(ancho de hombros, largo del torso)
 14. Normalizar por el ancho de hombros solo estaba roto y se descubrio mirando un
     golpe real: la guardia de boxeo es de perfil y ahi los hombros se superponen, de
     16 px en el percentil 1 contra 142 en la mediana. Entre el 19% y el 37% de los
     cuadros entraban al modelo con extensiones fisicamente imposibles, hasta 680
     anchos de hombro. Con el torso en la escala, ademas, la feature separa mejor:
     AUC 0,657 a 0,727 en sparring-3 y 0,639 a 0,744 en Pacquiao
 15. Medido por evento con el mismo emparejador del acuerdo intra-anotador, cinco
     semillas por particion:

                                 golpes    F1     recall  precision   F1 heuristica
       en distribucion               81   0,445    0,674     0,359          0,248
       sin Sparring                 114   0,348    0,582     0,280          0,243
       sin Pacquiao                 145   0,506    0,583     0,493          0,303
       sin sparring-3               380   0,416    0,589     0,349          0,274
       techo humano                        0,907   0,911     0,903

 16. EL DETECTOR NO SE DERRUMBA AL CAMBIAR DE FUENTE, y el clasificador si lo hacia.
     En distribucion 0,445 contra 0,348, 0,506 y 0,416 en los cruzados, con uno de
     ellos por encima del de distribucion; el clasificador tenia 25 puntos de ventaja
     en distribucion y caia a o por debajo de su linea de base en los tres cruzados.
     No son numeros comparables entre si, pero la forma si: la lectura mas probable es
     que la representacion era el problema, y que geometria normalizada transfiere
     donde los heatmaps en pixeles no
 17. El ruido entre semillas es mas grande que casi todo lo demas: 0,24 de F1 entre la
     mejor y la peor de una misma configuracion. El primer barrido de alpha, con una
     semilla, eligio 0,25 por un 0,634 cuya media real es 0,445; con cinco semillas el
     barrido no tiene ganador y quedo el default. Y dos corridas con la misma semilla
     daban distinto porque el modelo se construia antes de sembrar, con los pesos
     inicializados al azar
 18. Lo que falla es la precision, 0,28 a 0,49 contra 0,903 del humano; el recall se
     sostiene entre 0,58 y 0,67. El umbral no lo arregla porque el modelo esta
     saturado: solo el 9% de los cuadros cae en la zona indecisa. Con precision 0,35
     hay dos marcas falsas por cada tres golpes reales, asi que todavia no es un
     sistema que cuenta golpes

03-09 Cuarta fuente, y la varianza que estaba en el entrenamiento

  1. 02-sparring anotado a ciegas: 67 golpes en 2,8 minutos, 438 asignaciones de
     identidad y UNA sola colision en la validacion, contra 18 avisos y 15 colisiones
     de sparring-3. La anotacion mas limpia del dataset
  2. Tiene la mitad de golpes por minuto que el resto, 23,9 contra 41,7 y 42,5.
     Verificado que no es anotacion incompleta: la tasa se sostiene entre la primera y
     la segunda mitad del video. Es un sparring mas medido, y eso lo hace mas dificil
     en precision porque hay mas fondo verdadero por golpe
  3. Los tres folds que ya existian pasan de entrenar con dos fuentes a entrenar con
     tres, y el cambio medio es -0,002. Sumar una fuente no movio nada medible. Con
     desvio 0,10 sobre cinco semillas el piso de deteccion es 0,06 de F1, asi que lo
     que se puede afirmar es "no se detecto mejora", no "sumar fuentes no sirve"
  4. El desvio entre semillas es el MISMO con 62 golpes de validacion que con 380. Si
     viniera del muestreo de la evaluacion tendria que caer como la raiz del tamano.
     No cae: la varianza esta en el entrenamiento, no en la medicion
  5. Promediar las probabilidades de cinco semillas gana en los cuatro folds cruzados,
     entre +0,05 y +0,10, y pierde -0,04 en distribucion. Es coherente: la ganancia
     viene de suprimir detecciones espurias, que son idiosincrasia de cada corrida, y
     sobre la misma fuente en que se entreno esas manias encajan

       particion            golpes   una corrida   ensamble   recall  precision
       en distribucion          81         0,445      0,409    0,691      0,290
       sin 02-sparring          62         0,458      0,505    0,435      0,600
       sin Sparring            114         0,356      0,429    0,307      0,714
       sin Pacquiao            145         0,479      0,580    0,490      0,710
       sin sparring-3          380         0,430      0,527    0,382      0,853
       techo humano                        0,907               0,911      0,903

  6. LA PRECISION DEJO DE SER EL PROBLEMA: de 0,28-0,43 pasa a 0,60-0,85, a un paso
     del 0,903 humano. Cuatro de cada cinco marcas son un golpe real, que es el umbral
     que se habia estimado necesario para que la preanotacion fuera rentable. Ahora
     manda el recall, que bajo a 0,31-0,49
  7. Una teoria equivocada costo una medicion entera. Parecia que promediar n modelos
     saturados daba una salida cuantizada en pasos de 1/n, o sea que solo habria n
     puntos de operacion; sobre eso se barrieron cinco umbrales y el ensamble salio
     PEOR que una corrida sola en las cinco particiones. El reparto medido de la
     probabilidad promedio es un continuo, y con grilla fina aparece un acantilado
     angosto: entre umbral 0,75 y 0,80 se caen 360 marcas y la precision salta de
     0,307 a 0,853. La grilla gruesa se lo saltaba entero. Se elimino la abstraccion y
     quedo un test que fija que el promedio no cae sobre los escalones
  8. Anotada la quinta fuente, 03-sparring: 131 golpes en 2,9 minutos, 45,9 por minuto,
     la mas intensa del dataset, con 28% de hooks contra 13% de 02-sparring. Una
     colision en la validacion
  9. Y ESTA VEZ SUMAR LA FUENTE SI MOVIO LA AGUJA. Pasando de tres a cuatro fuentes de
     entrenamiento, el cambio medio sobre los folds cruzados es +0,046 en una corrida y
     +0,048 en ensamble, contra el -0,002 de cuando se sumo la cuarta. El control lo
     respalda: la particion en distribucion no toca las fuentes nuevas y da numeros
     identicos bit a bit entre las dos rondas
 10. No se puede separar "mas fuentes" de "mas datos" con dos incrementos: 03 es el
     doble de grande que 02 y bastante mas diverso. Lo que queda establecido es que el
     techo no estaba donde parecia despues de la primera medicion
 11. El fold mas confiable es el mejor: sin sparring-3, con 380 golpes en validacion,
     da F1 0,615 con recall 0,518 y precision 0,755. Y la precision de sin Pacquiao es
     0,905 contra 0,903 del humano, sobre transmision profesional que el modelo nunca
     vio. El recall sigue siendo la frontera, entre 0,39 y 0,52 contra 0,911

04-09 Sexta fuente, la guardia por fin en la interfaz, y sumar fuentes que si sirve

  1. 04-sparring anotado: 124 golpes en 3,0 minutos y dos planos, y trae la clase que
     faltaba: 24 uppercuts, el 19% de sus golpes, cuando en las cinco fuentes
     anteriores eran la clase marginal con 2, 5 y 6 ejemplos. El dataset queda en 950
     golpes sobre seis fuentes
  2. SUMAR FUENTES SIRVE, y la progresion es monotona sobre los tres folds presentes en
     las tres rondas, entrenando sobre 3, 4 y 5 fuentes:

                          3fte   4fte   5fte  |  ens 3  ens 4  ens 5
       sin Sparring      0,356  0,397  0,407  |  0,429  0,470  0,535
       sin Pacquiao      0,479  0,494  0,540  |  0,580  0,548  0,618
       sin sparring-3    0,430  0,496  0,530  |  0,527  0,615  0,664
       media             0,422  0,462  0,492  |  0,512  0,544  0,605

  3. La primera medicion no lo vio, y no fue error sino falta de potencia: con desvio
     +-0,10 entre semillas el piso de deteccion ronda 0,06 de F1 y el primer incremento
     valia menos que eso. Lo hicieron visible el ensamble, que es determinista, y
     incrementos mas grandes: 02 sumo 67 golpes, 03 sumo 131 y 04 sumo 124
  4. Sigue sin poder separarse "mas fuentes" de "mas datos" de "mas diversidad": los
     tres cambiaron juntos en cada incremento. Lo honesto es decir que el conjunto se
     paga, no cual de los tres factores
  5. La precision de sin Pacquiao llega a 0,907 contra 0,903 del humano: sobre metraje
     de transmision profesional que el modelo nunca vio, empata el techo. El recall ahi
     es 0,469, y el recall es lo que queda como frontera en todos los folds
  6. LA GUARDIA NO TENIA INTERFAZ. state.py la fijaba en orthodox al crear el documento
     con un pendiente que nunca se hizo, y como arm_role se deriva de ella, una guardia
     equivocada intercambia jab y cross sin que nada lo delate. Costo tres correcciones
     a mano sobre 175 eventos: fighter_B de Pacquiao y fighter_A de 02 y 04
  7. El selector va en el panel de identidad, no en configuracion: la guardia es una
     propiedad del peleador y se descubre mirando el video. Al cambiarla la GUI pregunta
     que hacer con lo ya anotado, porque cambiar la base significa dos cosas distintas:
     se anoto mal, y hay que reescribir las instantaneas invirtiendo lead/rear; o el
     peleador cambio de guardia de verdad, y reescribir destruiria trabajo correcto
  8. Corregidas las dos guardias, el dataset queda con 175 eventos de zurdo sobre 998,
     el 17,5%, en tres peleadores de doce. Verificado que al detector no lo toca: los
     .det.npz salen identicos byte a byte, porque exporta con label-space side y
     colapsa a O/B/I. Lo que cambia es lead-rear, o sea el clasificador
  9. Septima y ultima fuente, 01-sparring: 99 golpes, la unica con camara en mano
     dentro del ring. Y la mas barata de anotar de todo el dataset pese a ser la mas
     caotica visualmente: 162 asignaciones contra 518 de 03 y 438 de 02, porque la
     camara adentro del ring recorta el publico (149 tracks contra 375 a 473) y quedo
     en un plano unico. El dataset cierra en 1046 golpes y 147.848 cuadros-carril
 10. LA CAMARA EN MANO NO ERA EL PROBLEMA. Se esperaba que fuera el fold mas dificil y
     es el segundo mejor: entrenando sobre las otras seis, sin-01-sparring da precision
     0,924, la mas alta de los siete y por encima del 0,903 humano. Es lo que el diseno
     de las features predecia: centradas en el punto medio de los hombros y escaladas
     por max(ancho de hombros, largo del torso), un paneo mueve la imagen y no mueve
     nada en el espacio de features
 11. Y LA CURVA SE APLANO. El ensamble sobre los folds comunes venia 0,512, 0,544 y
     0,605 con los tres primeros incrementos; el cuarto da 0,604, o sea +0,000, y el
     efecto medio sobre todos los folds es +0,013, dentro del ruido. Un punto no
     alcanza para declarar saturacion, pero si para decir que sumar fuentes de este
     tipo ya no es la palanca que era
 12. El fold dificil es otro: sin-Sparring, peor por margen amplio en las cuatro rondas,
     0,519 con precision 0,561 cuando el resto esta entre 0,70 y 0,92. Sparring es la
     unica fuente a 640x360 contra 1080p de las otras seis, y la hipotesis es
     comprobable: reprocesar una de 1080p a 640x360 y ver si su fold cae al mismo lugar
 13. Queda el recall como frontera: entre 0,47 y 0,59 contra 0,911 del humano, mientras
     la precision ya esta entre 0,56 y 0,92. Ahi queda casi todo el error, y el ensamble
     lo empeora porque compra precision sacrificando recall
