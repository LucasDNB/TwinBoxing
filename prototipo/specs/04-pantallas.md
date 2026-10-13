# 04 · Pantallas

Una sección por pantalla, en el orden del diseño, más las tres que el diseño no dibuja y el
flujo necesita (ingreso, lista de sesiones, lista de peleadores). Cada sección dice de dónde
salen los datos, cómo se ve primero en el celular y después en escritorio, qué estados tiene y
cómo se acepta.

Convenciones:

- "Celular" es la base de 360 a 767 px; "escritorio" es `lg` (≥ 1024 px). `md` toma el layout
  del celular con más aire, salvo donde se indica.
- Los estados de bloque (medido, estimación, condicionado, no disponible) son los de `01` §4 y
  se ven como dice `03` §6.
- Toda pantalla tiene tres estados de carga: esqueleto mientras llega el dato, error con botón
  "Reintentar" y el mensaje de la API, y vacío con la acción que lo resuelve.
- Los criterios de aceptación (CA) se prueban en `06-pruebas-y-aceptacion.md`.

---

## 00 · Ingreso

Ruta `/ingresar/`. Datos: `POST /auth/login`, `POST /auth/registro`, `GET /producto`
(para saber si el registro está abierto o pide invitación).

- Celular y escritorio: una columna de 400 px máximo, centrada. Marca arriba, email, clave,
  botón "Entrar". Enlace "Tengo una invitación" que despliega el campo de código y cambia el
  botón a "Crear cuenta".
- Errores: 401 "Email o clave incorrectos"; 403 en registro "El registro está cerrado: pedí una
  invitación". Ninguno dice cuál de los dos campos falló.
- Después de entrar va a `?siguiente=` o al panel.

CA-00.1. Con credenciales válidas, entra y llega al panel en menos de 2 s en 4G simulada.
CA-00.2. Un 401 en cualquier pantalla devuelve a ingreso y, al entrar, vuelve a esa pantalla.

---

## 01 · Panel del gimnasio

Ruta `/`. Datos: `GET /panel` (B5), `GET /jobs`, `GET /boxeadores`, `GET /producto`.

Celular, de arriba hacia abajo:

1. Encabezado: fecha (rótulo mono) y "Panel".
2. Aviso de pendientes: una tarjeta ámbar por cada sesión en `espera_siembra` ("Falta un paso:
   marcá una vez a cada peleador del sparring de T. Ríos vs N. Ferreyra"), que lleva a
   `/sesion/identificar/?id=`. Si hay más de dos, se muestran dos y "Ver las N pendientes". Si
   alguna está en `fallo`, una tarjeta roja con el error legible y "Reintentar".
3. Indicadores en grilla de 2×2: peleadores activos, sesiones del mes (con desglose por estado),
   rounds analizados (con desglose por contexto), golpes detectados (con la frase del margen:
   "el sistema encuentra cerca de 5 de cada 10 golpes: el total real es mayor", con el número
   salido del recall del documento).
4. Sesiones recientes: lista de tarjetas (no tabla). Cada tarjeta: fecha, A vs B con sus marcas,
   contexto, duración, insignia de estado y acción. En curso: barra de progreso con la etapa
   ("Procesando · pose · 64%").
5. Mis peleadores: lista con nombre, categoría y guardia, cantidad de sesiones (B6) y tendencia.
   La tendencia sale de `caida_ultimo_round` del perfil (B7) y es un bloque condicionado a C2:
   apagado, o mientras B7 no esté, muestra "última sesión 21/09".
6. Botón flotante no hace falta: "Subir" está en la barra inferior.

Escritorio: como el diseño. Indicadores en una fila de cuatro; debajo, sesiones recientes como
tabla (1,6 fr) y peleadores (1 fr). Sin el bloque del plan.

Estados:

- Usuario nuevo sin sesiones: en lugar de los indicadores, una tarjeta "Subí tu primer sparring"
  con la checklist de filmación resumida y el botón.
- Sin peleadores cargados: "Cargá a tus peleadores para armar sus perfiles" con "Agregar".

Interacción: tocar una sesión lista abre `/sesion/?id=`; una en `espera_siembra` abre la
identificación; una en curso abre la sesión en su estado de espera (§04, estado "procesando").

CA-01.1. Una sesión que pasa de `procesando` a `espera_siembra` mientras el panel está abierto
genera el aviso de pendiente sin recargar, en menos de 15 s.
CA-01.2. Ningún número del panel aparece sin una fuente en la API (no hay constantes de
demostración en el build: una prueba busca los nombres del diseño en `out/`).
CA-01.3. Con C2 apagado no aparece ninguna frase de tendencia entre rounds.

---

## 02 · Nueva sesión (escritorio) y 08 · Subir desde el celular

Ruta `/sesiones/nueva/`. La pantalla 08 del diseño es esta misma ruta en la base; la 02 es su
layout de escritorio. Datos: `GET /boxeadores`, `GET /producto`, `POST /videos` (con los campos
de B1), `PUT /sesiones/{id}/boxeadores`.

Formulario en pasos dentro de la misma página, cada uno plegable cuando está completo:

1. **Video.** Dos botones grandes en el celular: "Filmar ahora" (`<input type="file"
   accept="video/*" capture="environment">`) y "Elegir video" (sin `capture`). En escritorio, zona
   para soltar el archivo y botón. Al elegir: nombre, duración, resolución y peso leídos en el
   navegador; validación de formato, tamaño y duración (`max_minutos`, 30 por defecto)
   contra `/producto`. Si el video está en vertical o
   mide menos de 720 px de alto, aviso (no bloqueo): "El sistema rinde mejor con el celular
   horizontal y en 1080p".
2. **Qué se filmó.** Grupo de cuatro opciones (bolsa, sombra, sparring, combate) con su
   descripción corta, como el diseño. Sombra agrega "Qué se practicó" (técnica libre,
   combinaciones, defensa). Bolsa y sombra quedan deshabilitadas con "Próximamente" mientras B8
   no esté en `/producto`.
3. **Peleadores.** Selector A (rincón rojo) y, si el contexto es de dos, B (rincón azul). Opciones:
   los peleadores cargados, "Rival sin perfil (no se guarda)" y "+ Agregar peleador" (hoja con
   nombre, categoría, guardia). No permite el mismo peleador en los dos lados.
4. **Rounds.** Cantidad, duración (mm:ss) y descanso. Opción "Sin rounds (video continuo)".
   Valores por defecto: 3, 3:00, 1:00.
5. **Para que el análisis salga bien.** Las cuatro condiciones del diseño (cámara fija, cuerpo
   entero, guantes de distinto color, luz de frente) como casillas que el usuario marca. No
   bloquean: si queda alguna sin marcar, el resumen dice "Puede que el análisis pierda tramos".
6. **Consentimiento** (D9). Casilla obligatoria: "Las personas que aparecen en el video dieron su
   consentimiento por escrito y ninguna es menor de 18 años." Sin marcarla, el botón queda
   deshabilitado y explica por qué.
7. **Resumen y envío.** Estimación de tiempo calculada con la duración leída:
   `duración × 1,13` a `duración × 2,62`, redondeado a minutos, con el texto "estimación con lo
   medido en la estación; avisamos cuando esté listo". Botón "Subir y procesar".

Subida: barra de progreso con porcentaje y megabytes, botón "Cancelar" (confirmación con
alert-dialog). La pantalla no se puede abandonar con el botón atrás sin confirmar mientras sube
(`beforeunload`). Al terminar: aviso "Subido. El análisis sigue aunque cierres la página" y
navegación a la sesión en estado de espera.

Celular (08): una sola columna, el botón "Subir y procesar" fijo abajo, encima de la barra de
navegación, visible siempre. Escritorio (02): como el diseño, formulario a la izquierda y la
columna derecha con checklist, consentimiento y resumen.

CA-02.1. Un `.avi` o un archivo mayor que `max_bytes` se rechaza antes de empezar a subir, con
el motivo.
CA-02.2. Sin la casilla de consentimiento no se puede enviar; la API también rechaza un
`POST /videos` sin `consentimiento=true` (B1).
CA-02.3. Cancelar una subida al 40% deja la sesión sin crear o la borra; no queda una sesión
fantasma en el panel.
CA-02.4. En un celular real (Android con Chrome e iPhone con Safari), "Filmar ahora" abre la
cámara trasera.

---

## 03 · Identificación de peleadores

Ruta `/sesion/identificar/?id=`. Datos: `GET /jobs/{id}` (candidatos, `pareja_sugerida`,
avisos), recortes por `/jobs/{id}/candidatos/{track}?t=`, `POST /jobs/{id}/siembra`,
`GET /fightcards/{id}` (identidad, después de completar), `GET /jobs/{id}/tramos` y
`PUT /jobs/{id}/tramos` (B3).

La pantalla tiene dos fases, porque la propagación tarda: el diseño las dibuja juntas, la web las
separa en el tiempo.

**Fase 1, marcar (estado `espera_siembra`).**

- Encabezado: contexto, fecha, nombres asignados ("Sparring · 22/09 · T. Ríos vs N. Ferreyra"),
  título "¿Quién es quién?" y la indicación en una línea: "Marcá una vez a cada peleador. El
  resto del video se sigue por altura y color de guante."
- Personas encontradas, una tarjeta por candidato: recorte (con proporción reservada),
  "Persona N", color del guante con su muestra (B2; si falta, no se muestra), tramo en cuadro
  ("00:04 a 03:10", B2), altura relativa, y tres botones "Es A", "Es B", "Ninguno". La pareja
  sugerida va primero y con el rótulo "sugerido".
- Celular: tarjetas a lo ancho en una columna, recorte a la izquierda (96 × 136), datos a la
  derecha, botones abajo. Escritorio: grilla de tres columnas como el diseño.
- Regla: un solo A y un solo B. Marcar un segundo A le saca la marca al anterior, con aviso.
- "Completar análisis" (fijo abajo en celular) se habilita con A y B marcados; abre un
  alert-dialog con los dos recortes lado a lado: "¿Rojo es Ríos y azul es Ferreyra?". Al confirmar,
  `POST /jobs/{id}/siembra` y la pantalla pasa a "Completando".
- Si los candidatos no coexisten (`pareja_sugerida` nula), aviso: "No encontramos un momento
  con los dos en cuadro a la vez. Fijate bien que no sea la misma persona dos veces."

**Fase 2, revisar (estado `listo`).**

- Resultado de la propagación (D6): "13 de 20 tramos asignados solos" y "separación de color
  1,38 (mínimo 0,55)". Si `identidad.sin_asignar` es mayor que 0, la frase: "Los golpes que
  caen en tramos sin asignar no se cuentan. Revisalos para no perder conteo."
- Revisar tramos dudosos: lista de `GET /jobs/{id}/tramos` con rango de tiempo, motivo (cuerpo a
  cuerpo, salió de cuadro, guante tapado, cruce frente a cámara), recorte, y A / B / Ignorar.
  Tocar el rango abre un reproductor en hoja (celular) o en línea (escritorio) en ese instante.
- "Aplicar" manda `PUT /jobs/{id}/tramos` con todas las decisiones y vuelve a "Completando"; al
  terminar la Fight-Card se actualiza. "Ver resultados" va a la sesión sin revisar.
- Si `sin_asignar` es 0, la fase 2 se saltea y la pantalla redirige a la Fight-Card.

CA-03.1. Con A y B marcados y confirmados, la sesión llega a `listo` sin otra intervención
(C4).
CA-03.2. Asignar a mano un tramo que caía sin asignar suma sus golpes al peleador elegido en la
Fight-Card siguiente, y la cobertura sube.
CA-03.3. Los recortes cargan con ticket (sin 401) en Safari y Chrome de celular.

---

## 04 · Análisis de sesión

Ruta `/sesion/?id=&vista=video`. Datos: `GET /jobs/{id}`, `GET /fightcards/{id}`,
`GET /videos/{id}/ticket`, `HEAD /videos/{id}/procesado`, `/videos/{id}/stream` o
`/procesado`, `PATCH /fightcards/{id}/golpes/{golpe}`, `GET /producto`.

Encabezado común a 04 y 05: contexto, fecha, rounds, A vs B con sus marcas, y pestañas "Video y
golpes", "Fight-Card", "Perfiles de la sesión" (la tercera lleva a los perfiles de A y B).

**Si la sesión no está lista**, la vista muestra el estado en lugar del contenido: en cola
(posición si la API la da), procesando con etapa y porcentaje, espera de siembra con el botón a
la identificación, completando, o fallo con el error y "Reintentar".

**Lista**, celular:

1. Video arriba, pegado al tope al desplazar (sticky), con controles nativos, velocidad
   (0,25×, 0,5×, 1×) y el interruptor "Esqueleto" (procesado u original). Si no hay procesado:
   "Armar video con esqueleto" (`POST /jobs/{id}/render`).
2. Línea de tiempo de golpes detectados: dos carriles (A arriba, B abajo), un tick por golpe,
   divisiones por round, cursor del tiempo actual. Tocar un tick lleva el video a
   `t_inicio − 0,5 s` y selecciona el golpe. En celular la línea es desplazable con zoom de dos
   niveles (sesión, round).
3. Selector A / B (toggle) y debajo la lista de golpes de ese peleador cerca del tiempo actual:
   tiempo, tipo estimado con confianza (baja si < 0,5, en ámbar), brazo, guardia (condicionado a
   C3), y "Corregir". La lista sigue al video mientras corre.
4. Corregir abre una hoja desde abajo con los nombres de `fightcard.nomenclatura` y "No fue un
   golpe", y "Guardar". Una corrección guardada muestra el tipo nuevo en verde con "corregido" y
   el original tachado.
5. Nota fija al pie: "El tipo de golpe es una estimación del modelo [checkpoint] y mejora con cada
   versión. Tu corrección se guarda para entrenar y no cambia el conteo de golpes detectados."
   "No fue un golpe" tampoco cambia el conteo de detectados: se cuenta aparte como "descartados
   por el entrenador".

Escritorio: como el diseño. Video y línea de tiempo a la izquierda, dos consolas (A y B) a la
derecha, corrección en popover.

Teclado (escritorio): espacio reproduce y pausa; J y L van al golpe anterior y siguiente del
peleador seleccionado; los atajos solo actúan con el foco dentro del reproductor o de la línea.

CA-04.1. Tocar un golpe a los 05:47 deja el video en 05:46,5 ± 0,1 s y reproduciendo, con el
video original de 1 GB, sin descargarlo entero (pedidos con `Range`).
CA-04.2. Una corrección sobrevive a recargar la página y aparece en la Fight-Card y en el
perfil.
CA-04.3. Con C3 apagado no aparece "guardia arriba / caída" en ningún golpe.
CA-04.4. La línea de tiempo se puede recorrer solo con teclado y con lector de pantalla.

---

## 05 · Fight-Card y 09 · Fight-Card en el celular

Ruta `/sesion/?id=&vista=fightcard&round=`. La 09 es esta vista en la base. Datos:
`GET /fightcards/{id}`, `GET /producto`, `GET /fightcards/{id}/export`.

Contenido, el mismo en celular y escritorio (D5):

1. Margen arriba de todo, en una línea: "Detectados: acierta 9 de cada 10 que marca y se le
   escapa la mitad. El volumen real es mayor." Enlace "Cómo leer esta tarjeta" (acordeón).
2. Selector Sesión / R1 / R2 / R3 (tantos como rounds). Con C2 apagado, solo "Sesión" y un
   rótulo "comparación entre rounds en validación".
3. Cabecera A y B con nombre, categoría y guardia.
4. Filas medidas, con barras horizontales A contra B: golpes detectados (grande), izquierda y
   derecha, golpes por minuto.
5. Filas estimadas, en estilo de estimación: jab, gancho (cross), hook, uppercut, contados con
   el tipo corregido cuando existe, con "tipo menos confiable" en hook. Rótulo con el checkpoint y
   su exactitud medida o "sin medir".
6. Guardia opuesta arriba: bloque condicionado a C3.
7. Golpes por round (gráfico de barras), condicionado a C2, con la lectura en una frase calculada
   ("De R2 a R3, Ríos cae de 44 a 31, −30%").
8. Descartados por el entrenador (si hay) y porcentaje del video sin identidad resuelta
   (`identidad.sin_asignar`), con enlace a revisar tramos.
9. No disponible en esta versión, plegado: "Conectados, bloqueados, esquivados y recibidos" y
   "A la cabeza y al cuerpo", cada uno con su motivo en una línea.
10. Exportar: menú con "Para imprimir (PDF)" (abre `export?formato=pdf` en pestaña nueva; la
    página se imprime desde el navegador, como hoy), "Lista de golpes (CSV)" y "Datos completos
    (JSON)" (descarga del documento tal cual).

Celular (09): selector de rounds como pestañas de 44 px; filas en una columna con A a la izquierda
y B a la derecha; al pie, "Ver video" (vuelve a la vista 04) y "Compartir", que usa
`navigator.share` con el CSV como archivo cuando el navegador lo permite y, si no, lo descarga.

CA-05.1. La suma de golpes detectados de los rounds es igual al total de la sesión, y ambos
coinciden con `peleadores.X.total` del documento.
CA-05.2. Cambiar de round no pide nada a la API (todo sale del documento ya cargado).
CA-05.3. El texto del margen cambia si cambia `recall_medido` en el documento, sin tocar la web.
CA-05.4. Ninguna versión de la tarjeta muestra "conectados", "puntos" ni "ganador".

---

## Lista de sesiones

Ruta `/sesiones/?estado=`. Datos: `GET /jobs`. La misma tarjeta del panel, con filtro por estado
y por contexto (toggle-group desplazable) y búsqueda por nombre de peleador. Orden: más reciente
primero.

---

## Lista de peleadores

Ruta `/peleadores/`. Datos: `GET /boxeadores`, `POST /boxeadores`, `PATCH /boxeadores/{id}`
(con los campos de B6). Lista con nombre, categoría, guardia y sesiones; "Agregar" abre una hoja
con nombre, categoría, guardia, altura (m), envergadura (m) y notas. Tocar uno abre su perfil.

---

## 06 · Perfil táctico

Ruta `/peleador/?id=&contexto=`. Prioridad P1. Datos: `GET /boxeadores/{id}/perfil` extendido
(B7), `GET /producto`.

Encabezado: rótulo "Perfil táctico", nombre, categoría y guardia, altura y envergadura si están,
"N sesiones · M rounds", y "Comparar en Rival Fantasma". El rótulo del diseño "gemelo digital" no
se usa mientras no haya modelo que simule: se llama "perfil".

Filtro de contexto: Todas, Bolsa, Sombra, Sparring, Combate, con la cantidad de sesiones de cada
uno (toggle-group desplazable en celular).

Bloques, en este orden en el celular (en escritorio, grilla de tres columnas como el diseño):

1. Distribución de golpes (estimación): barras por familia con porcentaje y cantidad; aviso en
   hook; "N detectados, M sin clasificar". La división cabeza/cuerpo del diseño no está
   (no disponible).
2. Evolución entre sesiones (medido): línea de golpes por minuto por sesión, con fecha; cada punto
   lleva a su sesión. Si cambió el checkpoint del detector entre sesiones, la línea se corta y lo
   dice (el perfil ya devuelve con qué detector se midió cada una).
3. Mismo peleador en distintos contextos (medido): tabla con golpes por minuto, jab sobre el total
   (estimación), guardia opuesta arriba (condicionado a C3) y caída de volumen en R3 (condicionado
   a C2). Lectura fija debajo: "Cada contexto mide algo distinto: se compara cómo cambia él, no
   contra una línea de base común."
4. Volumen por round promedio (condicionado a C2).
5. Combinaciones frecuentes (estimación): secuencias de golpes detectados del mismo peleador
   separados por no más de la pausa de `GET /producto` (`pausa_combinacion_s`, valor inicial
   0,5 s a confirmar por Lucas). Se nombran con la numeración de gimnasio (1 jab, 2 recto de
   atrás, 3 y 4 hook de adelante y de atrás, 5 y 6 uppercut de adelante y de atrás) cuando la
   guardia del peleador está cargada, porque adelante y atrás salen de la guardia y del brazo;
   sin guardia cargada, con el nombre de la familia.
6. No disponible en esta versión, plegado: mapa de exposición, avatar 3D, distancia de trabajo.
   El lugar del avatar en escritorio lo ocupa, en P2, la postura promedio en 2D (W-F17).

Estado vacío: con menos de 2 sesiones, "Perfil preliminar: con 5 sesiones o más las tendencias
se leen mejor" (umbral del diseño) y los bloques que se pueden calcular.

CA-06.1. Un peleador que aparece como A en una sesión y como B en otra suma las dos.
CA-06.2. Corregir un tipo en una sesión cambia la distribución del perfil al volver (el perfil no
se cachea, como ya decide la API).

---

## 07 · Rival Fantasma

Ruta `/rival/?a=&b=`. Prioridad P1. Datos: `GET /boxeadores`, dos `GET /boxeadores/{id}/perfil`
(B7), `GET /producto`.

1. Selectores de perfil A y B (select nativo), con cantidad de sesiones de cada uno. La frase fija
   del diseño debajo: "Lectura cualitativa entre perfiles ya construidos. No es un pronóstico ni
   una probabilidad de victoria."
2. Dimensiones reales: siluetas a escala por altura y envergadura cargadas, superpuestas o lado a
   lado (toggle). Si falta algún dato, la silueta no se dibuja y dice qué falta.
3. Frente a frente: tabla de dos columnas con altura, envergadura, guardia, golpes por round
   (condicionado a C2; si no, golpes por minuto), jab sobre el total (estimación), guardia
   opuesta arriba (C3), caída de volumen en R3 (C2). Solo métricas de sparring y combate de cada
   perfil, como dice el diseño.
4. Oportunidades y amenazas para cada uno, generadas por reglas fijas sobre las filas medidas.
   Cada lectura cita la métrica de la que sale. Reglas iniciales, cuyo texto valida Lucas como
   criterio de dominio antes de publicarlas:
   - Envergadura: diferencia de 3 cm o más → oportunidad para el de mayor envergadura (trabajo a
     distancia con el jab).
   - Guardia: ortodoxo contra zurdo → amenaza para los dos (la mano de atrás del rival entra por el
     lado abierto), solo como hecho de guardia, sin porcentajes de zona.
   - Ritmo: diferencia de golpes por minuto del 15% o más → oportunidad para el de mayor ritmo.
   - Caída en R3 (solo con C2): el que cae 15% o más → amenaza para él, oportunidad para el otro.
   - Guardia opuesta (solo con C3): diferencia de 10 puntos o más → amenaza para el más bajo.
   Si ninguna regla aplica: "Con los datos medidos no hay diferencias marcadas entre estos dos
   perfiles."
5. No disponible en esta versión, plegado: "Dónde ataca uno y dónde recibe el otro" (requiere el
   mapa de exposición).

CA-07.1. Con C2 y C3 apagados ninguna lectura cita caída en R3 ni guardia opuesta.
CA-07.2. Elegir el mismo peleador en A y B no está permitido.

---

## Mensajes que se repiten

Se definen una vez en `lib/textos.ts` y se revisan con la regla de lenguaje de gimnasio de
`03` §8. Los que llevan números los toman del documento o de `/producto`.

| Clave | Texto |
|---|---|
| margen.detector | "Acierta {p} de cada 10 que marca y se le escapa {r_falta}. El volumen real es mayor." |
| estimacion.tipo | "Tipo estimado por el modelo {checkpoint}. Exactitud sobre material nuevo: {exactitud o 'sin medir'}." |
| condicionado.c2 | "La comparación entre rounds se muestra cuando se confirme que el sistema detecta parejo en todos los rounds." |
| condicionado.c3 | "La guardia se muestra cuando el indicador pase la validación contra marcado a mano." |
| nodisp.contacto | "Una cámara sola no ve si el golpe tocó. Para estimarlo hace falta una pieza que todavía no está." |
| nodisp.altura | "Esta versión del modelo dice el tipo de golpe, no si fue a la cabeza o al cuerpo." |
| nodisp.3d | "El giro de un golpe alrededor del cuerpo no se ve con una cámara de frente. El avatar necesita más de una cámara." |
| sinasignar | "El {x}% del video quedó sin saber quién es quién. Los golpes de esos tramos no se cuentan." |
