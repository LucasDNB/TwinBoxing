# BoxTwin

13-05 Experimento 02: smoke test del pipeline base sobre video de entrenamiento

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
