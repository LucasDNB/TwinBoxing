# Cambios del 14-10 · Perfil táctico y Rival Fantasma

Pedido de Lucas (14-10), después de abrir combinaciones y mostrar cabeza/cuerpo como estimación
(commit `d6223cb`):

Peleadores, perfil táctico (pantalla 06):

1. Distribución de golpes entre izquierda/derecha y cabeza/cuerpo.
2. Volumen por round: el round dura 3 minutos. Un video más largo que no esté fraccionado se
   divide en segmentos de 3 minutos y se evalúa sobre esos segmentos.
3. Mapa de exposición a partir de los golpes recibidos, por izquierda/derecha y cabeza/cuerpo.
4. Avatar 3D: por ahora, un avatar estático.
5. Distancia de trabajo cargada a mano (larga, media o corta) y estilo (fajador o estilista).

Rival Fantasma (pantalla 07):

1. Dimensiones reales: avatares 3D en guardia en lugar de los de palitos, con los brazos que
   se estiran y se recogen.
2. "Dónde ataca uno y dónde recibe el otro" y "Oportunidades y amenazas" pasan a ser un solo
   bloque.
3. Ese bloque se completa con los mapas de exposición y la distribución de golpes del punto 1.

Specs: `01-perfil.md` (B15 a B17, W20 a W24) y `02-rival.md` (B18, W25 y W26).

## Decisiones (14-10)

Las dos primeras son de Lucas; el resto sigue a decisiones que ya estaban tomadas en
`ampliacion-13-10`, o son menores y se toman acá.

DC-1 · Rounds sin C2. Tomada por Lucas: se muestran, rotulados. C2 (el detector detecta parejo
en todos los rounds) sigue pendiente y no se toca. La bandera `comparacion_rounds` pasa a
encenderse también por decisión de producto (`BOXTWIN_ROUNDS_ESTIMADO`, por defecto 1), igual
que la altura el 14-10. Donde se compara entre rounds, la pantalla lo dice: "comparación entre
rounds sin validar".

DC-2 · El lado del mapa de exposición. Tomada por Lucas: por mano del rival (DA-4 de
`ampliacion-13-10/04` §2). Las cuatro zonas son cabeza o cuerpo, cada una por la mano
izquierda o la derecha del rival. No se espeja a un lado del cuerpo: un recto va al centro, y
no hay anotación para validar el espejo.

DC-3 · "Recibidos" quiere decir "dirigidos". Una cámara no establece contacto (CLAUDE.md §2), así
que el mapa cuenta los golpes del rival dirigidos a cada zona, no los que entraron. Es lo que ya
fijaba `ampliacion-13-10/04` §1. La pantalla nunca dice "recibió", "conectó" ni "le pegaron"
(CA-15.2 se mantiene).

DC-4 · Corrección por sesgo del detector. La regla de C9 está pre-registrada en
`ampliacion-13-10/04` §3: si la diferencia de recall del detector entre cabeza y cuerpo pasa de
0,10, el mapa se muestra corregido. El nivel 2 de C8 (14-10) midió 0,512 a la cabeza y 0,348
al cuerpo: diferencia 0,16. Se aplica la regla tal cual: cada zona dividida por el recall de su
altura y renormalizada, con los dos recalls a la vista. C9 queda medido con ese dato.

DC-5 · Segmentos de 3 minutos. Con rounds declarados se usan los declarados. Sin rounds
declarados se corta en segmentos de 180 s sin descanso, desde el segundo 0. Un último segmento de
menos de 60 s se suma al anterior: si no, un "round" de 10 s da un ritmo por minuto que no dice
nada. La Fight-Card lo marca en `video.rounds_origen` (`declarados` o `segmentos_180s`), y la
pantalla dice "segmento" en lugar de "round" cuando corresponde.

DC-6 · Avatar estático. No se corre ningún modelo de elevación (eso es B14, que sigue bloqueado).
El avatar es un maniquí procedural en guardia, a escala de la altura y la envergadura cargadas,
con la guardia del peleador (un zurdo con el pie derecho adelante). Se rotula "ilustrativo, no
reconstruido del video". Sin medidas cargadas se dibuja en proporciones estándar y lo dice.
Three.js se carga aparte, solo cuando el bloque se ve, para no pasar el presupuesto de 380 KB de
JS inicial.

DC-7 · Distancia y estilo son datos cargados, no medidos. Se guardan en el boxeador, como la
altura. La pantalla los muestra como "cargado a mano"; la distancia medida (B9, P2) sigue sin
existir.

DC-8 · Bloque unificado del Rival Fantasma. Se llama "Ataque contra exposición". Cruza, zona por
zona, la distribución de ataque de uno con el mapa de exposición del otro, en los dos sentidos.
Las lecturas son las tres reglas de zona de `ampliacion-13-10/05` §2, con su mínimo de 50
golpes por lado: una regla que se dispara a favor de A es una oportunidad para A; una a favor de
B es una amenaza para A, y al revés. Las cinco reglas de métricas de W12 (`lib/rival.ts`), que
estaban "en construcción", salen de la pantalla: el bloque nuevo las reemplaza y no se basan en
zonas.

DC-9 · Los brazos del Rival Fantasma. Cada avatar tiene un control de 0 a 100% de extensión por
brazo (adelantado y atrasado), con el alcance en metros escrito al lado. No se anima un
intercambio: la pantalla no simula una pelea (`ampliacion-13-10/07` §2). La distancia entre los
dos es de presentación, no medida, y lo dice.

DC-10 · Sin circularidad en el cruce (14-10, Lucas, después de medir sobre producción). La
exposición del receptor se calcula sin las sesiones contra el atacante. Detalle en `02-rival.md` §1.

DC-11 · Umbral de la regla de zona coincidente: 35% (14-10, Lucas). Con 25% se disparaba con 26%
y 27%, apenas sobre el reparto parejo.

Abierto, sin decidir: la exposición y el ataque pueden tener el cuerpo inflado. La regla de altura
manda golpes a la cabeza al cuerpo (nivel 2: 108 de 345) y la corrección de DC-4 multiplica el
cuerpo por 1,47; la corrección compensa lo que el detector no ve, no lo que la regla clasifica mal.
En producción Lucas da 47% al cuerpo contra 17% en lo anotado. No se toca sin medir.

## Orden de trabajo

1. B15 rounds de 3 minutos (anotador y API) y W21.
2. B16 distribución por mano y altura, y exposición dirigida (API) y W20, W22.
3. B17 distancia y estilo (API) y W24.
4. W23 avatar estático (perfil).
5. B18 cruce de ataque y exposición (API) y W26 bloque unificado.
6. W25 avatares del Rival Fantasma.

Cada tarea se cierra con sus pruebas en verde y su fila en `TAREAS.md`.
