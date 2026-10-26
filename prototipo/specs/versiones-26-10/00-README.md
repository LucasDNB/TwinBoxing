# Specs de las versiones Individual y Gimnasio · BoxTwin

Estado: propuesta · 26-10-2026 · base: `main` en `709c22a` (21-10)
Pedido de Lucas (26-10): "los specs necesarios para desarrollar las versiones individual y gimnasio".
Segundo pedido (26-10): por ser una versión de prueba, un botón en el ingreso para entrar como cada
una de las versiones, y por ahora sin email ni código de invitación (§"Versión de prueba").
Fuente de las versiones: Cap 5, Tabla 5.7 (estructura de precios por segmento) y Tabla E.1.
Congelamiento de funciones (Cap 6, R-03): 03-11-2026 · Entrega final: 17-11-2026

## Qué es una versión

El Cap 5 vende el mismo producto a cuatro segmentos. Hoy existe uno solo construido: la web
entrenador de `prototipo/specs/`, que en el Cap 5 es el plan Entrenador. Estos specs agregan dos.

| Versión | Segmento del Cap 5 | Modo | Horas incluidas | Quién la usa | Estado |
|---|---|---|---|---|---|
| Individual | Boxeador individual | Diferido | 4 por mes | Un boxeador adulto que filma su propio trabajo | Este spec |
| Entrenador | Entrenador | Diferido | 15 por mes | Un entrenador con sus peleadores | Construida (`prototipo/specs/`) |
| Gimnasio | Gimnasio o club | Diferido y en vivo | 50 por mes | Un gimnasio con varios entrenadores y un plantel | Este spec, solo diferido |
| Federación | Federación | Diferido y en vivo | Por contrato | | Fuera: R-08, se ofrece sujeta a validación |

La decisión que ordena todo lo demás: **una versión es el plan de la cuenta, no otra
aplicación**. Hay una sola web, una sola API y una sola base. La API dice de qué versión es la
cuenta y qué funciones tiene, y la web dibuja lo que le dicen, igual que hoy con las banderas de
`GET /producto`. La justificación está en `02-arquitectura.md` §1.

Con esa decisión, la versión Entrenador es el caso particular que ya funciona: una cuenta con un
solo miembro y su propio plantel. Toda la base actual queda migrada como cuentas Entrenador y
nada de lo que hoy anda cambia de comportamiento (compuerta de paridad, `05` §1).

## Versión de prueba

Con `BOXTWIN_DEMO=1`, la página de ingreso deja de pedir email, clave y código, y muestra un botón
por versión y rol:

| Botón | Entra a | Rol |
|---|---|---|
| Individual | La cuenta de prueba Individual | Titular |
| Entrenador | La cuenta de prueba Entrenador | Titular |
| Gimnasio · titular | La cuenta de prueba Gimnasio | Titular |
| Gimnasio · entrenador | La misma cuenta de prueba Gimnasio | Entrenador |

Gimnasio lleva dos botones porque titular y entrenador ven pantallas distintas (Equipo, la
actividad por entrenador), y con un solo botón la mitad de la versión no se podría mostrar.

Cuatro reglas que no se negocian, porque la estación sale a internet por un túnel y quien tenga la
URL puede tocar un botón:

1. Los botones entran solo a cuentas marcadas como de prueba (`cuentas.demo`). Las cuentas reales,
   con las sesiones de prueba de esta semana y el material de los atletas, no se alcanzan desde ahí.
2. La bandera viene apagada. Apagada, la ruta de ingreso de prueba responde 404 y la página de
   ingreso es la de hoy, con email y clave: así siguen entrando las cuentas reales.
3. En las cuentas de prueba solo se carga material cuyas personas aceptaron que se muestre, con
   seudónimos salvo que hayan aceptado su nombre (DV-9). Y el cupo siempre bloquea.
4. Un comando vuelve las cuentas de prueba a su estado inicial, porque todos los que entran por el
   mismo botón comparten la cuenta.

Lo que se posterga mientras dure la versión de prueba, especificado y sin construir: el registro
por versión (B21, W28) y las invitaciones y bajas de miembros (B23 y esa parte de W32). En la
cuenta de prueba Gimnasio el equipo viene cargado. Diseño en `02` §11, ruta en `04` B29, pantalla
en `03` §2.

## Archivos

| Archivo | Contenido | Formato que sigue |
|---|---|---|
| `01-producto.md` | Usuario y propuesta de cada versión, funcionalidades ranqueadas, flujos, matriz de funciones por versión, requisitos, desvíos de lo aprobado, tres riesgos | Prompt 9 de la Clase 5; prototipo vs MVP de la Clase 8 |
| `02-arquitectura.md` | Plan por cuenta, cuentas y miembros, control de acceso, cupo de horas, invitaciones, migración, seguridad, versión de prueba | Clase 8, arquitectura en capas y seguridad |
| `03-pantallas.md` | Navegación por versión, pantallas nuevas y cambiadas, estados, criterios de aceptación | Prompt 10 de la Clase 5; diseño de `prototipo/specs/03` |
| `04-api-y-datos.md` | Tareas B19 a B29: endpoints, payloads, tablas, archivos que se tocan | Prompt 11 de la Clase 5, pasos 1 y 2 |
| `05-pruebas-y-aceptacion.md` | Compuerta de paridad, cinco pruebas de integración críticas, criterios de aceptación pre-registrados, e2e y CI | Prompt 11, paso 3 |
| `06-plan.md` | Tandas con fechas contra el congelamiento, orden de recorte, qué va a cada capítulo de la tesis | Plan del MVP y Cap 6 |

## Cómo usarlos con Claude Code

Igual que los specs del 12-10. Cada tarea tiene identificador: B19 a B29 para la API y W27 a W38
para la web, a continuación de los que ya usa `prototipo/TAREAS.md`. Las P2 (B26, B28, W36, W37)
y las postergadas por la versión de prueba (B21, B23, W28) quedan especificadas y no se
construyen. Una sesión toma una tarea,
lee los apartados que esa tarea cita y nada más, y cierra con sus pruebas en verde y una fila en
`prototipo/TAREAS.md`.

Regla del Prompt 11: para cada tarea B, primero el plan (endpoint, payload, tabla, prueba),
después el código. B19 toca el control de acceso de todas las rutas: su plan se revisa antes que
cualquier otro.

## Lo que estos specs no cambian

- El pipeline de visión, los modelos y los criterios C1 a C11. Ninguna versión mejora un número
  del sistema: cambian quién ve qué y cuánto se puede subir.
- El contrato `fightcard.json`: queda en 0.3, sin campos nuevos.
- La línea epistemológica: golpes detectados y no lanzados, sin conectados, sin puntuación, sin
  veredicto (RF6 y RF8 de `docs/specs/MVP.md`).
- El despliegue de un nodo en la estación con túnel (RNF4). Una GPU para todas las cuentas.
- El modo en vivo sigue fuera (F12 del MVP), aunque el Cap 5 lo incluya en el plan Gimnasio. La
  sub-hipótesis 3 no se midió (R-11) y la alternativa B del Cap 5, que el cierre del 20-10 tomó
  como producto, lo excluye. La versión Gimnasio lo muestra como "no incluido en esta versión".
- Pagos, pasarela, facturación de horas excedentes y plan gratuito: fuera.

## Lo que sí cambian, y hay que declarar

- F13 del MVP ("planes, pagos, multi-organización"), aprobado como fuera el 21-09, entra en
  parte: planes y multi-organización sí, pagos no.
- RNF3 del MVP ("los videos y resultados de un usuario solo son visibles para ese usuario") pasa
  a decir "de una cuenta" y "para los miembros de esa cuenta" (`02` §4).
- D4 de `prototipo/specs/01` (se quitaron la barra del plan y las horas consumidas) se revierte
  en parte: el consumo vuelve, calculado con la duración real de los videos subidos.
- El registro cerrado por invitación (`_puede_registrarse`, "cerrado por defecto y a propósito")
  sigue igual para las cuentas reales. La versión de prueba agrega una entrada sin credenciales,
  pero solo a cuentas de prueba y solo con la bandera encendida.

> Nota para la tesis: es la segunda ampliación de alcance después de la del 13-10, a ocho días del
> congelamiento del 03-11 que el Cap 6 comprometió como contramedida de R-03. Si se construye, se
> declara en la entrega final con fecha y motivo, y lo que no llegue al 03-11 queda especificado y
> sin construir. `06-plan.md` §4 dice qué va a cada capítulo.

## Decisiones tomadas (26-10)

Lucas delegó el 26-10: "tomá todas las decisiones según tu criterio de recomendaciones". DV-1 a
DV-11 quedan con la recomendación de la tabla de abajo, sin cambios. Lo que eso significó al
construir:

- DV-1: se construyeron la tanda 1, la tanda 2 y la P1 en la rama `versiones-26-10`. Nada se
  desplegó: la regla de `06` §1 sigue, y el despliegue de cada tanda espera a su compuerta en la
  estación y a que terminen las sesiones de prueba.
- DV-5: el texto del 08-10 no estaba en el repo, así que B25 se construyó sin texto y la API
  contesta 409 hasta que se cargue en `boxtwin_api/textos/`. No se inventó uno.
- DV-6 y DV-9 dependen de Lucas y de la GPU: CV4 sin medir, y `~/boxtwin/demo.yaml` sin armar.

## Decisiones pendientes de Lucas

Cada una tiene una recomendación. Si no se decide, los specs se construyen con la recomendación.

| # | Decisión | Recomendación | Por qué | De qué depende |
|---|---|---|---|---|
| DV-1 | ¿Se construye antes del 03-11? | Tanda 1 (cuentas e Individual) del 28 al 30-10; tanda 2 (Gimnasio) solo si la 1 pasa su compuerta el 30-10. Si no, Gimnasio queda como diseño especificado | Individual reutiliza casi todo lo construido; Gimnasio es multiusuario de verdad y toca la seguridad | Cómo salgan las sesiones de prueba del 27-10, que van primero |
| DV-2 | Cupo de horas: ¿informar o bloquear? | Informar en la instancia del PFI (`BOXTWIN_CUPO=informar`); bloquear queda especificado y probado | Un bloqueo en medio de una sesión de prueba o de la demo cuesta más de lo que demuestra | Nada |
| DV-3 | En Gimnasio, ¿cada entrenador ve solo lo suyo o todos ven el plantel? | Plantel compartido: todos los entrenadores ven a todos los atletas y sus sesiones | Es el argumento institucional del Cap 5 §5.2: un registro "transferible entre entrenadores" para que la evaluación no dependa de quién miró | Criterio de dominio de Lucas sobre cómo se trabaja en un gimnasio |
| DV-4 | En Individual, ¿qué pasa con el compañero de sparring? | Se procesa y se marca, porque la identidad lo necesita, pero aparece como "Compañero", sin perfil ni historial | Sin la segunda persona la siembra no separa a los dos; guardarle un perfil a quien no tiene cuenta es tratar datos de un tercero sin necesidad | Nada |
| DV-5 | Consentimiento en Gimnasio | Una vez por atleta, registrado con fecha y quién lo cargó, revocable; la casilla por sesión sigue para quien no está en el plantel | Cap 5 §5.2 pide el consentimiento de los atletas y no solo de quien contrata; en un gimnasio se filma a los mismos todas las semanas | Texto del consentimiento: el del 08-10 para C6, si Lucas lo confirma |
| DV-6 | ¿Se mide B8 para Individual? | Sí: dos videos propios de 3 minutos (uno de bolsa, uno de sombra) y la revisión pre-registrada de `ampliacion-13-10/02` §2 | La versión Individual vive de bolsa y sombra, que desde el 21-10 son función sin validar. Es un criterio ya escrito y Lucas puede filmarlo en su gimnasio | Una hora de Lucas y la GPU fuera de las sesiones de prueba |
| DV-7 | Plazo de conservación de videos en Gimnasio | Sin borrado automático en el PFI; la supresión de un atleta queda especificada (B26) y sin construir | Un borrado automático mal configurado se lleva material de prueba del Cap 7 | Decisión de producto y legal que no tiene que salir antes del 17-11 |
| DV-8 | Nombre de las versiones en pantalla | "Individual", "Entrenador", "Gimnasio" | Son las palabras del pedido; "Gimnasio o club" no entra en un botón | Nada |
| DV-9 | ¿Qué material se carga en las cuentas de prueba? | Individual: tus propios videos de bolsa y sombra, los mismos de CV4. Entrenador y Gimnasio: solo sesiones en que todos los que aparecen firmaron que se puede mostrar, con seudónimos ("Atleta 1") salvo que hayan aceptado su nombre. Si no alcanza, las cuentas arrancan con menos sesiones, no con material sin ese permiso | Con la bandera encendida, cualquiera con la URL ve esas sesiones. El consentimiento de las sesiones de prueba cubre el análisis, y hay que revisar si cubre mostrarlas | Lo que diga el texto del consentimiento que firmaron |
| DV-10 | ¿Cuándo se enciende la versión de prueba? | Solo en las ventanas de ensayo y de presentación (03-11, 10-11 y 17-11), y se apaga después | Es la manera más simple de que el túnel no quede abierto sin credenciales; encender y apagar es una línea del env y un reinicio | Nada |
| DV-11 | ¿Se puede subir video desde las cuentas de prueba? | Sí, con el cupo en bloquear: el tribunal puede ver el flujo entero. Si en un ensayo alguien llena la cola, se apaga la subida con `BOXTWIN_DEMO_SUBIR=0` | Subir en vivo es lo que demuestra C4; dejarlo sin tope es regalar la GPU | Nada |
