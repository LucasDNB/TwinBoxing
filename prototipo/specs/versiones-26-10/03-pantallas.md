# 03 · Pantallas por versión

Formato del Prompt 10 de la Clase 5 (ingreso, panel con tres tarjetas, detalle con edición,
perfil; mobile first, WCAG AA) aplicado sobre la web que ya existe. No hay diseño visual nuevo:
todo usa los tokens, los componentes y las reglas de `prototipo/specs/03-diseno-y-componentes.md`,
y las pantallas que no se nombran acá se ven igual que en `prototipo/specs/04-pantallas.md`.

Convenciones, las mismas de `prototipo/specs/04`: celular de 360 a 767 px, escritorio desde 1024;
tres estados de carga en toda pantalla (esqueleto, error con "Reintentar", vacío con la acción que
lo resuelve); los criterios de aceptación (CA) se prueban en `05`.

Lenguaje: en Individual la web le habla al boxeador, en segunda persona y con voseo ("tus golpes",
"marcate"). En Entrenador y Gimnasio, como hoy. La regla de lenguaje de gimnasio de
`prototipo/specs/03` §8 sigue: "tramo" y no "track", "marcá" y no "sembrá".

---

## 1. Cómo sabe la web qué versión es (W27)

- `lib/cuenta.ts`: `useCuenta()` con TanStack Query (clave `["cuenta"]`), esquema zod de
  `GET /cuenta`, y `tiene(funcion)`. Se pide una vez después de entrar y se invalida al cambiar
  algo de la cuenta.
- La guarda de `(app)/layout.tsx` espera a la cuenta además del token. Mientras tanto, esqueleto.
- `lib/banderas.ts` suma un segundo eje: `estadoDe(bloque, producto, cuenta)`. Un bloque que no es
  de la versión vuelve `fuera` y no se dibuja; no es "no disponible", porque en esa versión no
  existe. `01.plan` y `02.consumo`, que hoy están en `fuera` por D4, pasan a `medido` cuando
  `cuenta.horas` no es nulo.
- Navegación (`components/boxtwin/Navegacion.tsx`), con los destinos que salen de la cuenta:

| Versión y rol | Barra inferior del celular | Barra lateral del escritorio |
|---|---|---|
| Individual | Inicio · Sesiones · Subir · Mi perfil | Lo mismo |
| Entrenador | Panel · Sesiones · Subir · Peleadores · Rival, como hoy | Como hoy |
| Gimnasio, entrenador | Panel · Sesiones · Subir · Plantel · Rival | Lo mismo |
| Gimnasio, titular | Panel · Sesiones · Subir · Plantel · Equipo | Los seis, con Rival |

  El titular de Gimnasio llega al Rival Fantasma desde el perfil ("Comparar en Rival Fantasma"):
  la barra del celular no admite seis destinos sin achicar los toques por debajo de 44 px.
- Una página que no es de la versión, abierta por URL (por ejemplo `/rival/` en Individual),
  muestra "Esta función no está en la versión Individual" y un enlace al inicio. La API también
  la rechaza (B22).

CA-27.1. Con cada una de las cuatro combinaciones de versión y rol, la navegación muestra
exactamente los destinos de la tabla.
CA-27.2. Ningún bloque que la matriz de `01` §4 deja afuera de una versión aparece en el DOM de esa
versión, ni siquiera rotulado.

---

## 2. 00 · Ingreso

### En la versión de prueba (W38)

Ruta `/ingresar/`. Datos: `GET /producto` (`demo`, B29) y `POST /auth/demo` (B29).

Con `producto.demo.activo` en verdadero, la página no tiene email, clave, código de invitación
ni "Crear cuenta". En la misma columna de 400 px, debajo de la marca:

1. Rótulo "Versión de prueba" y una línea: "Elegí cómo querés entrar. Son cuentas de prueba
   compartidas: lo que hagas lo ve el que entre después."
2. Cuatro botones grandes, uno debajo del otro, cada uno con su nombre, una línea de descripción y
   las horas del mes, que salen de `producto.versiones`:
   - "Individual": "Un boxeador con su bolsa, su sombra y sus sparrings."
   - "Entrenador": "Un entrenador con sus peleadores."
   - "Gimnasio · titular": "Administra el gimnasio: plantel, equipo y horas."
   - "Gimnasio · entrenador": "Un entrenador del mismo gimnasio."
   El orden y los textos de los perfiles vienen de `producto.demo.perfiles`; la web no los fija.
3. Al tocar uno: el botón muestra que está entrando, la web guarda el token como hoy y va al inicio
   de esa versión. Un error se muestra en la misma página con "Reintentar".

Dentro de la aplicación, el encabezado lleva el rótulo "Prueba · Gimnasio · entrenador" y un
"Salir" que vuelve a esta página, para cambiar de versión sin borrar nada a mano. Si
`producto.demo.subir` es falso, "Subir" sigue en la navegación y la pantalla dice "En la versión de
prueba la subida está apagada en este momento".

Con `producto.demo.activo` en falso, la página es la de hoy (`prototipo/specs/04` §00), sin
cambios.

CA-00.7. Con la versión de prueba encendida, la página de ingreso no tiene ningún campo de
texto: ni email, ni clave, ni código.
CA-00.8. Cada uno de los cuatro botones lleva al inicio y a la navegación de su versión y rol
(CA-27.1), en celular y en escritorio.
CA-00.9. Con la versión de prueba apagada, la página es la de hoy y los botones no están en el DOM.
CA-00.10. "Salir" desde cualquier versión vuelve a los cuatro botones, y entrar con otro cambia de
cuenta sin restos de la anterior en la caché de TanStack Query.

### Registro e invitación (W28, postergada por la versión de prueba)

Queda especificada así para cuando vuelvan el email y las invitaciones. Ruta `/ingresar/`. Datos:
`POST /auth/login`, `POST /auth/registro` (B21), `GET /producto`, `GET /invitaciones/{token}` (B23).

Crear una cuenta, en pasos dentro de la misma columna de 400 px:

1. "¿Cómo vas a usar BoxTwin?", grupo de radio con tres tarjetas:
   - "Soy boxeador": "Analizá tu bolsa, tu sombra y tus sparrings. 4 horas de video por mes."
   - "Soy entrenador": "Tus peleadores y sus sesiones. 15 horas de video por mes."
   - "Represento a un gimnasio": "Varios entrenadores y un plantel compartido. 50 horas de video
     por mes. El análisis en vivo no está incluido en esta versión."
   Sin precios: no hay pagos.
2. Los datos de esa versión. Boxeador: "Tu nombre" (dice "así se va a llamar tu perfil") y la
   casilla "Tengo 18 años o más". Gimnasio: "Nombre del gimnasio" y "Tu nombre". Entrenador:
   "Tu nombre", opcional.
3. Email, clave y, si la instancia pide invitación, el código, como hoy.

Después de crear la cuenta va al inicio de su versión.

Entrada por invitación, `/ingresar/?invitacion=<token>`:

- La web lee el token, lo saca de la barra de direcciones (`history.replaceState`) para que no
  quede en el historial ni en el Referer, y pide `GET /invitaciones/{token}`.
- Válida: "Te invitaron a Gimnasio Norte como entrenador", el email de la invitación sin editar,
  "Tu nombre", clave, "Crear cuenta y entrar". No pregunta la versión ni el código de instancia.
- 410: "Esta invitación venció o ya se usó. Pedile una nueva a quien te invitó." 404: "Esta
  invitación no existe. Revisá que el enlace esté completo."

CA-00.3. Cada una de las tres versiones se crea desde el celular y llega a su inicio.
CA-00.4. En Individual, sin la casilla de mayoría de edad no se puede enviar, y la API responde 422
si se manda igual.
CA-00.5. Después de abrir un enlace de invitación, la URL ya no contiene el token.
CA-00.6. Una invitación usada muestra el mensaje y no el formulario.

---

## 3. Versión Individual

### I-01 · Inicio: Mi progreso (W29)

Ruta `/`, la misma del panel: la página dibuja Mi progreso o el panel según la versión. Datos:
`GET /cuenta`, `GET /panel`, `GET /jobs`, `GET /boxeadores/{propio}/perfil` (`evolucion`).

Celular, de arriba hacia abajo:

1. Encabezado: fecha (rótulo mono) y "Mi progreso".
2. Pendientes: una tarjeta ámbar por sesión en `espera_siembra` ("Falta un paso: marcate en la
   sesión de bolsa del 26/10"), que lleva a la identificación. Las de `fallo`, en rojo con
   "Reintentar", como en el panel.
3. Tres tarjetas:
   - Última sesión: contexto, duración, golpes detectados, golpes por minuto y la frase del margen.
     En bolsa y sombra la frase dice que el margen se midió en sparring.
   - Cómo venís: golpes por minuto de cada sesión en los últimos 30 días, una línea por contexto,
     de `perfil.evolucion`. Las líneas se cortan donde cambia el detector, como en el perfil. Con
     menos de dos sesiones de un contexto, esa línea no se dibuja y el texto dice "Con dos sesiones
     de bolsa ya se puede comparar".
   - Tus sesiones: las últimas cinco, con la tarjeta de sesión de hoy.
4. Horas del mes: "1,2 de 4 h usadas en octubre", con una barra. Si se pasó con `informar`: el
   texto `cupo.excedido.informar` de §5.
5. Cuenta sin sesiones: en lugar de las tarjetas, "Subí tu primera sesión de bolsa" con la
   checklist para una persona (cámara fija, cuerpo entero, guantes a la vista, luz de frente; en
   sombra, el espejo fuera de cuadro si se puede) y el botón.

Escritorio: las tres tarjetas en una fila, horas del mes arriba a la derecha.

CA-I01.1. Ningún número de la pantalla aparece sin una fuente en la API (la misma prueba de
CA-01.2 sobre el build).
CA-I01.2. La tendencia nunca pone en la misma línea sesiones de contextos distintos.
CA-I01.3. Con una sola sesión de un contexto no hay línea de ese contexto, y aparece el texto.

### I-02 · Subir (W30)

Ruta `/sesiones/nueva/`, la misma pantalla 02 y 08, con estos cambios:

- El paso "Peleadores" no está. En su lugar, una línea: "Vas a aparecer como Lucas".
- Bolsa y sombra llevan el rótulo de función sin validar hasta que pase CV4, con el texto
  `individual.sin_validar`.
- Checklist: en bolsa y sombra, la de una persona; en sparring y combate, la de hoy, con los guantes
  de distinto color.
- Consentimiento, obligatorio, en dos casillas según el contexto:
  - Siempre: "Soy yo quien aparece en el video y tengo 18 años o más."
  - En sparring y combate, además: "La otra persona que aparece aceptó ser filmada y analizada, y
    tiene 18 años o más."
- Horas: "Esta sesión usa unos 3 min de las 2,8 h que te quedan", con la duración que ya lee la
  pantalla y `cuenta.horas`. Si no alcanza: con `informar`, aviso que no bloquea; con `bloquear`,
  el botón deshabilitado y el motivo.
- Manda `duracion_declarada_s`.

CA-I02.1. No aparece el selector de peleadores, y la sesión creada tiene como A al peleador propio
(verificado en la API).
CA-I02.2. En sparring, sin la segunda casilla no se puede enviar.
CA-I02.3. Con `BOXTWIN_CUPO=bloquear` y sin horas, el botón queda deshabilitado y la API rechaza
el pedido directo con 409.

### I-03 · Marcarse (W30)

Ruta `/sesion/identificar/?id=`, la pantalla 03.

- Bolsa y sombra: W19 tal cual, con el texto en segunda persona: "¿Cuál sos vos?", "Tocá una vez tu
  figura. El resto del video se sigue por el color de tus guantes." El aviso de espejo se queda.
- Sparring y combate, dos pasos en la misma pantalla:
  1. "¿Cuál sos vos?" Cada tarjeta de candidato con un solo botón, "Soy yo".
  2. "¿Y tu compañero?" Las tarjetas que quedan, con "Es mi compañero".
  Confirmación con los dos recortes: "¿Vos sos el de guantes rojos y tu compañero el de azules?".
  Manda `track_a` el elegido como "soy yo" y `track_b` el compañero.
- Fase 2, revisión de tramos: "Soy yo", "Es mi compañero", "Ignorar".

CA-I03.1. En Individual la pantalla no muestra "Es A", "Es B", "Peleador B" ni la palabra
"rincón".
CA-I03.2. La siembra manda como `track_a` el candidato marcado "Soy yo".

### I-04 y I-05 · Sesión y Fight-Card (W31)

Las pantallas 04 y 05 sin cambios de estructura:

- Las columnas se llaman "Vos" y "Compañero". La del compañero lleva el texto
  `individual.compañero`.
- Las correcciones de tipo se pueden hacer sobre los dos.
- El export usa el nombre del perfil propio y "Compañero".

CA-I05.1. En Individual el texto "Peleador B" no aparece en ninguna pantalla ni en el export.

### I-06 · Mi perfil (W31)

Ruta `/peleador/?id=<propio>`, la pantalla 06:

- Sin "Comparar en Rival Fantasma".
- Los datos propios (altura, envergadura, guardia, categoría, distancia, estilo) se editan desde el
  encabezado con la hoja de W24.
- El bloque de exposición se titula "Lo que te tiraron tus compañeros" y agrupa como "sin perfil".

CA-I06.1. No hay enlace al Rival Fantasma, y `/rival/` muestra el aviso de fuera de versión.

---

## 4. Versión Gimnasio

### G-01 · Panel del gimnasio (W34)

La pantalla 01 con estos cambios:

- Encabezado con el nombre del gimnasio.
- Pendientes de todos los entrenadores, cada uno con "Subió Ana". Un toggle "Todas / Mías", en
  "Todas" por defecto: el plantel es compartido (DV-3).
- Un quinto indicador, "Horas del mes" (12,4 de 50). En el celular, la grilla de 2×2 y debajo la
  barra a lo ancho.
- Solo el titular: "El equipo este mes", con sesiones, horas y pendientes por entrenador (B24).

CA-G01.1. Un entrenador no ve el bloque por entrenador, y la API se lo manda nulo.
CA-G01.2. Un pendiente que subió otro entrenador aparece con su nombre y se puede resolver desde
este usuario.

### G-02 · Subir en el gimnasio (W33)

La pantalla 02 y 08 con estos cambios:

- En los selectores A y B, cada atleta del plantel lleva su estado de consentimiento. Con
  consentimiento vigente se elige normalmente. Sin él, la opción aparece deshabilitada con "Falta
  el consentimiento" y un enlace "Registrarlo" que abre la hoja de G-08 sin salir de la subida.
- "Rival sin perfil (no se guarda)" se queda, para quien no está en el plantel. Lo cubre la casilla
  de consentimiento de la sesión, como hoy.
- Horas que le quedan al gimnasio en el mes, como en I-02.
- Con B27, la espera de la sesión muestra "Hay 3 sesiones antes que la tuya. Unos 12 a 28 minutos
  para empezar."

CA-G02.1. No se puede elegir un atleta sin consentimiento vigente, y la API rechaza la asignación
directa con 422.

### G-07 · Plantel (W33)

Ruta `/peleadores/`, que en Gimnasio se titula "Plantel".

- Cada atleta suma el entrenador a cargo y una insignia de consentimiento: vigente, falta o
  revocado.
- Filtros: "A mi cargo" y "Falta consentimiento".
- La hoja de alta suma "Entrenador a cargo" (los miembros activos) y "Registrar el consentimiento
  ahora", que abre G-08 al guardar.

CA-G07.1. Un atleta recién cargado aparece como "Falta consentimiento" hasta que se registra.

### G-08 · Consentimiento del atleta (W33)

Un bloque en el encabezado del perfil y en la hoja del atleta.

- Estado, y debajo la historia: "Registrado por Ana el 26/10", "Revocado por Juan el 30/10".
- "Registrar": muestra el texto vigente entero (`GET /cuenta/consentimiento`) en un área con
  scroll, las casillas "El atleta leyó y aceptó este texto" y "Tiene 18 años o más", y una nota
  opcional ("firmado en papel el 26/10"). Confirmar manda el `POST`.
- "Revocar": alert-dialog "Desde ahora no se lo va a poder elegir en sesiones nuevas. Sus sesiones
  anteriores quedan; borrarlas es la supresión, que esta versión todavía no hace."

CA-G08.1. Revocar no borra la constancia: la historia sigue mostrando quién lo registró y cuándo.

### G-09 · Equipo (W32)

Ruta `/equipo/`, solo el titular. Un entrenador que la abre ve "Solo el titular del gimnasio
administra el equipo".

En la versión de prueba (W32):

- Tu versión: "Gimnasio · 50 h de video por mes", las horas usadas y "El análisis en vivo no está
  incluido en esta versión".
- Miembros: nombre, rol y desde cuándo, de `GET /cuenta/miembros` (B24). Sin email y sin "Dar de
  baja".
- En lugar de invitar, una línea: "En la versión de prueba el equipo viene cargado."
- Consentimientos: "Descargar el registro (CSV)", con B25 en P1.

Cuando vuelvan las invitaciones (W32, parte postergada con B23):

- Miembros con email y "Dar de baja", con alert-dialog: "Ana deja de poder entrar. Las sesiones
  que subió quedan en el gimnasio."
- Invitar: un campo de email y "Crear invitación". La respuesta muestra el enlace una vez, con
  "Copiar" y, en el celular, "Compartir" (Web Share API), y el texto `invitacion.una_vez`.
- Invitaciones: pendientes, usadas, vencidas y revocadas, con "Revocar" en las pendientes.

CA-G09.0. En la versión de prueba, Equipo no muestra emails ni ofrece invitar o dar de baja.
CA-G09.1 (postergado). Al salir de la pantalla y volver, el enlace de una invitación ya no se ve
(la API no lo devuelve nunca más).
CA-G09.2 (postergado). El pedido siguiente de un miembro dado de baja lo lleva a ingreso con el
mensaje `miembro.baja`.

### Sesión, Fight-Card y perfil en Gimnasio (W34)

- El encabezado de la sesión suma "Subió Ana · 26/10".
- En la lista de golpes, un golpe corregido dice "corregido por Juan" (de `sesion.autoria`).
- El perfil de un atleta muestra todas sus sesiones del gimnasio, sin importar quién las subió.

CA-G04.1. Un entrenador abre el perfil de un atleta que filmó otro y ve las sesiones de los dos.

---

## 5. Mensajes nuevos

En `lib/textos.ts`, con la misma regla que los de `prototipo/specs/04` §"Mensajes que se repiten".

| Clave | Texto |
|---|---|
| version.individual | "Analizá tu bolsa, tu sombra y tus sparrings. {horas} horas de video por mes." |
| version.entrenador | "Tus peleadores y sus sesiones. {horas} horas de video por mes." |
| version.gimnasio | "Varios entrenadores y un plantel compartido. {horas} horas de video por mes." |
| envivo.no_incluido | "El análisis en vivo no está incluido en esta versión." |
| fuera_de_version | "Esta función no está en la versión {version}." |
| cupo.uso | "{usadas} de {incluidas} h usadas en {mes}." |
| cupo.excedido.informar | "Te pasaste de las {incluidas} h del mes. En este prototipo la sesión se procesa igual." |
| cupo.excedido.bloquear | "Esta sesión usaría {h} h y quedan {r} h este mes." |
| individual.sin_validar | "El sistema todavía no se midió en bolsa y sombra: los números pueden fallar más que en sparring." |
| individual.compañero | "Tu compañero no tiene perfil. Sus golpes sirven para separarlos a los dos y para ver qué te tiraron." |
| consentimiento.falta | "Falta el consentimiento de {nombre}." |
| invitacion.una_vez | "Este enlace se ve una sola vez y sirve para una persona. Vence el {fecha}." |
| invitacion.vencida | "Esta invitación venció o ya se usó. Pedile una nueva a quien te invitó." |
| miembro.baja | "Tu acceso a {cuenta} fue dado de baja." |
| demo.ingreso | "Elegí cómo querés entrar. Son cuentas de prueba compartidas: lo que hagas lo ve el que entre después." |
| demo.rotulo | "Prueba · {perfil}" |
| demo.subir_apagado | "En la versión de prueba la subida está apagada en este momento." |
| demo.equipo | "En la versión de prueba el equipo viene cargado." |

Los números de horas no se escriben en el texto: salen de `cuenta.funciones.horas_mes` con la
sesión iniciada, y de `GET /producto` (`versiones`, B29) en la pantalla de ingreso.
