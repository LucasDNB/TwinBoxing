# 01 · Producto: qué es cada versión

Formato: Prompt 9 de la Clase 5 (funcionalidades ranqueadas por valor y esfuerzo, cuáles entran,
flujo de 5 a 7 pasos, stack, tres riesgos) y la distinción de la Clase 8: esto sigue siendo un
MVP. Ninguna versión simula funciones con datos inventados, y ninguna muestra un número que el
sistema no mide.

---

## 1. De dónde salen las versiones

Del Cap 5, apartado 2.3 y Tabla 5.7. Lo que este spec toma de ahí, sin cambiar ningún número:

| Segmento | Modo | Mensual | Horas incluidas | Lo que el Cap 5 dice del segmento |
|---|---|---|---|---|
| Boxeador individual | Diferido | USD 15 | 4 por mes | "El más débil del conjunto": se sostiene como puerta de entrada hacia el plan de entrenador, no por su rentabilidad |
| Entrenador | Diferido | USD 45 | 15 por mes | Es la versión construida |
| Gimnasio o club | Diferido y en vivo | USD 160 | 50 por mes | Beneficiario del registro objetivo y trazable (Tabla 5.10) y de la justificación institucional (§5.2): un registro transferible entre entrenadores |

De ahí salen dos consecuencias de diseño. La versión Individual tiene que ser barata de construir
y llevar naturalmente a la de Entrenador, así que reutiliza casi todo y no agrega pantallas que
no tengan uso propio. La versión Gimnasio vale por lo que comparte: el plantel, el cupo y el
registro de quién hizo qué. Lo que la distingue no es un análisis nuevo sino la organización.

Los precios no aparecen en el producto: no hay pagos (F13 parcial, `00`). Las horas sí, porque son
consumo medido.

---

## 2. Versión Individual

### 2.1 Usuario y propuesta

Usuario objetivo: un boxeador amateur adulto que entrena sin entrenador fijo, o que lo tiene y
quiere su propio registro. Filma su bolsa y su sombra con el celular apoyado en el gimnasio, y
algún sparring si un compañero acepta. Lo revisa después, en el celular.

Lo que la versión le da, en orden de valor:

1. Subir el video y ver cuántos golpes encontró el sistema y de qué tipo, con el margen al lado.
2. Ver si cambia de una sesión a otra del mismo tipo de trabajo: golpes por minuto, mezcla de
   golpes, combinaciones.
3. En sparring, marcarse a sí mismo y al compañero en un toque cada uno y leer lo suyo.
4. Llevarle el resultado a un entrenador.

El punto 3 ya estaba en el MVP aprobado: F3 de `docs/specs/MVP.md` es "este soy yo / este es el
rival".

### 2.2 Funcionalidades ranqueadas

Prioridad: P0 entra en la tanda 1 de `06`. P1 entra si la tanda cierra con tiempo. P2 queda
especificada y no se compromete. "Hecho" quiere decir que ya está construido y solo se reutiliza.

| # | Funcionalidad | Pantalla | Valor | Esfuerzo | Prioridad | Depende de |
|---|---|---|---|---|---|---|
| I-F0 | Entrar con el botón "Individual" del ingreso (versión de prueba) | 00 | Alto | Bajo | P0 | B29, W38 |
| I-F1 | Crear la cuenta como boxeador, con declaración de mayoría de edad | 00 | Alto (legal) | Bajo | Postergada por la versión de prueba | B21 |
| I-F2 | El perfil propio se crea solo: la cuenta tiene un único peleador, el titular | — | Alto | Bajo | P0 | B22 |
| I-F3 | Subir bolsa, sombra, sparring o combate sin elegir peleadores | I-02 | Alto | Bajo | P0 | B22 |
| I-F4 | Marcarse en un toque en bolsa y sombra, con aviso de espejo | 03 | Alto | Hecho (W19) | P0 | B8 |
| I-F5 | En sparring y combate: "¿cuál sos vos?" y después "¿y tu compañero?" | I-03 | Alto | Bajo | P0 | B22 |
| I-F6 | Análisis de sesión y Fight-Card con textos en segunda persona | 04, 05 | Alto | Bajo | P0 | W31 |
| I-F7 | Corregir el tipo y "no fue un golpe" | 04 | Medio | Hecho (B4, W7) | P0 | |
| I-F8 | Mi progreso: última sesión, evolución por contexto, pendientes | I-01 | Alto | Medio | P0 | B22 |
| I-F9 | Mi perfil: el perfil táctico propio, sin Rival Fantasma | 06 | Medio | Bajo | P0 | B7 |
| I-F10 | Exportar para llevarle al entrenador | 05 | Medio | Hecho (W8) | P0 | |
| I-F11 | Horas usadas del mes contra las 4 incluidas | I-01, I-02 | Medio | Bajo | P1 | B20 |
| I-F12 | Pasar a la versión Entrenador conservando el historial | Ajustes | Medio (puerta de entrada del Cap 5) | Medio | P2 | B28 |
| — | Invitar a un entrenador a ver mis sesiones | | Alto para la conversión | Alto | Fuera | Comparte datos entre cuentas; post-PFI |
| — | Rival Fantasma | 07 | Bajo | | Fuera de la versión | Hay un solo perfil; compararse contra uno mismo es otra función |
| — | Lista y alta de peleadores | | | | Fuera de la versión | La cuenta tiene un peleador |

### 2.3 Flujo del usuario (6 pasos, celular)

1. En la versión de prueba toca "Individual" en el ingreso y entra. Cuando vuelva el registro, en
   lugar de eso crea la cuenta eligiendo "Soy boxeador" y declara que es mayor de edad. Si quiere,
   carga altura y guardia.
2. Apoya el celular y filma tres minutos de bolsa. Toca "Subir", elige "Bolsa", confirma que es él
   quien aparece y toca "Subir y procesar".
3. Cuando termina la primera parte, Mi progreso le muestra "Falta un paso". Toca una vez su figura.
   Si hace sombra frente al espejo, la pantalla le avisa que no marque el reflejo.
4. Ve su Fight-Card: golpes detectados por round y por brazo y la mezcla estimada. El margen dice
   que se midió en sparring y que en bolsa todavía no.
5. Toca un golpe, lo ve en el video y corrige el tipo si está mal.
6. En Mi progreso compara con sus sesiones anteriores de bolsa. Exporta la Fight-Card para
   mostrársela a un entrenador.

### 2.4 Estado de cada bloque

Lo que no está en esta tabla se ve igual que en la versión Entrenador (`prototipo/specs/01` §4 y
`lib/banderas.ts`).

| Bloque | Estado en Individual | Motivo |
|---|---|---|
| Contextos bolsa y sombra | Función sin validar, rotulada, hasta que pase CV4 | B8 encendido el 21-10 por decisión de producto, sin medir |
| Margen del detector en bolsa y sombra | "Medido en sparring; en bolsa todavía no se midió" | B8b, ya construido |
| Cabeza o cuerpo en bolsa y sombra | No aplica | La regla de altura usa el torso del rival |
| Exposición en Mi perfil | Estimación, igual que en Entrenador | Sale de los golpes que le dirigieron sus compañeros de sparring |
| Comparación entre rounds | Estimación rotulada | DC-1 del 14-10; C2 sigue sin medir |
| Guardia | Apagado | C3 rechazado el 20-10 |
| Horas del mes | Medido | B20 |
| Rival Fantasma, peleadores | Fuera de la versión | Matriz del §4 |

---

## 3. Versión Gimnasio

### 3.1 Usuarios y propuesta

Hay tres personas y solo dos tienen usuario:

- Titular: el dueño o el jefe de entrenadores. Crea la cuenta del gimnasio, invita a los
  entrenadores, ve todo lo del gimnasio y el consumo del mes, registra consentimientos y quita
  miembros.
- Entrenador: hace lo mismo que en la versión Entrenador, pero sobre el plantel del gimnasio, que
  comparte con los demás entrenadores.
- Atleta del plantel: no tiene usuario en esta versión. Es la persona medida, y su consentimiento
  se registra a su nombre.

Lo que la versión le da al gimnasio, en orden de valor:

1. Un plantel compartido: la historia de un atleta no depende de qué entrenador lo filmó (DV-3).
2. Varios entrenadores subiendo al mismo lugar, con un solo cupo de horas.
3. El consentimiento de cada atleta registrado una vez, con fecha y quién lo cargó (DV-5).
4. Saber qué se hizo en el mes y quién lo hizo.

El modo en vivo, que el Cap 5 incluye en este plan, no está (`00`).

### 3.2 Funcionalidades ranqueadas

| # | Funcionalidad | Pantalla | Valor | Esfuerzo | Prioridad | Depende de |
|---|---|---|---|---|---|---|
| G-F0 | Entrar con "Gimnasio · titular" o "Gimnasio · entrenador" (versión de prueba); el equipo viene cargado | 00 | Alto | Bajo | P0 | B29, W38 |
| G-F1 | Crear la cuenta del gimnasio con su nombre; quien la crea es el titular | 00 | Alto | Bajo | Postergada por la versión de prueba | B21 |
| G-F2 | Invitar entrenadores con un enlace de un solo uso | G-09 | Alto | Medio | Postergada por la versión de prueba | B23 |
| G-F3 | Entrar por invitación | 00 | Alto | Bajo | Postergada por la versión de prueba | B23 |
| G-F4 | Plantel compartido: sesiones, peleadores y perfiles visibles para todos los miembros | todas | Alto | Medio | P0 | B19, B24 |
| G-F5 | Quién subió cada sesión y quién corrigió cada golpe | Sesiones, 04 | Medio | Bajo | P0 | B24 |
| G-F6 | Consentimiento por atleta: registrar, ver, revocar; no se asigna un atleta sin consentimiento vigente | G-07, G-08, 02 | Alto (legal) | Medio | P0 | B25 |
| G-F7 | Quitar a un entrenador sin perder sus sesiones | G-09 | Medio | Bajo | Postergada por la versión de prueba | B23 |
| G-F8 | Panel del gimnasio con las horas del mes y la actividad por entrenador | G-01 | Medio | Medio | P1 | B20, B24 |
| G-F9 | Entrenador a cargo de cada atleta y filtro "a mi cargo" | G-07 | Medio | Bajo | P1 | B24 |
| G-F10 | Posición en la cola y espera estimada | 04 en espera | Medio | Bajo | P1 | B27 |
| G-F11 | Exportar el registro de consentimientos | G-09 | Medio (legal) | Bajo | P1 | B25 |
| G-F12 | Supresión de un atleta | G-08 | Alto (legal) | Alto | P2 | B26 |
| — | Acceso del atleta a su propio perfil | | Medio | Alto | Fuera | Un tercer rol y datos que salen de la cuenta |
| — | Modo en vivo | | Alto en el Cap 5 | Muy alto | Fuera | Sub-hipótesis 3 sin medir (R-11) |
| — | Varias sedes, o un usuario en varios gimnasios | | Bajo | Medio | Fuera | Un usuario pertenece a una sola cuenta (`02` §3) |
| — | Mandar la invitación por mail | | Bajo | Medio | Fuera | Sin servidor de correo en el PFI: el titular copia el enlace |

### 3.3 Flujos

En la versión de prueba el gimnasio ya existe, con su titular, un entrenador y el plantel cargados.

Titular (4 pasos, escritorio o celular):

1. Toca "Gimnasio · titular" en el ingreso y entra al panel del gimnasio de prueba.
2. En Equipo ve a los miembros y la versión: 50 h por mes, sin análisis en vivo.
3. Carga un atleta nuevo y registra su consentimiento.
4. Mira el panel: sesiones del mes, horas usadas de las 50, quién subió qué.

Entrenador (6 pasos, celular):

1. Toca "Gimnasio · entrenador" en el ingreso y entra al mismo gimnasio.
2. Sube un sparring y elige A y B del plantel. Solo puede elegir atletas con consentimiento
   vigente; si falta uno, la pantalla le dice a quién y cómo registrarlo.
3. Marca a cada peleador.
4. Lee la Fight-Card.
5. Abre el perfil de un atleta que filmó otro entrenador y ve toda su historia.
6. Corrige un golpe. La corrección queda con su nombre.

Cuando vuelvan el registro y las invitaciones, el titular crea el gimnasio, invita a cada
entrenador con un enlace que copia y manda por donde quiera, y da de baja al que se va; el
entrenador entra por ese enlace y crea su clave. Está especificado en B21 y B23.

### 3.4 Estado de cada bloque

| Bloque | Estado en Gimnasio | Motivo |
|---|---|---|
| Panel: horas del mes | Medido | B20; es el "01.plan" que D4 había quitado |
| Panel: actividad por entrenador | Medido | B24 |
| Sesiones: quién subió | Medido | B24 |
| Plantel: consentimiento de cada atleta | Medido | B25 |
| Rival Fantasma | Igual que en Entrenador, entre atletas del plantel | |
| Modo en vivo | Fuera: "no incluido en esta versión" | `00` |
| El resto | Igual que en Entrenador | |

---

## 4. Matriz de funciones por versión

Es la tabla que implementa `planes.py` (`02` §5). La web no la copia: la recibe de `GET /cuenta`.

| Función | Individual | Entrenador | Gimnasio |
|---|---|---|---|
| Horas de video incluidas por mes | 4 | 15 | 50 |
| Miembros de la cuenta | 1 | 1 | Sin tope |
| Peleadores | Solo el propio | Sin tope | Sin tope, compartidos |
| Alta y edición de peleadores | Solo los datos propios | Sí | Sí, cualquier miembro |
| Contextos | Los cuatro | Los cuatro | Los cuatro |
| Marca en sparring y combate | Vos y compañero | A y B | A y B |
| Rival Fantasma | No | Sí | Sí |
| Invitaciones | No | No | Solo el titular; postergadas en la versión de prueba |
| Entrada en la versión de prueba | Un botón | Un botón | Dos botones: titular y entrenador |
| Consentimiento | Por sesión: "soy yo" y, en sparring, el del compañero | Por sesión, como hoy | Por atleta, y por sesión para quien no está en el plantel |
| Entrenador a cargo | | | Sí |
| Panel con actividad por entrenador | | | Solo el titular |
| Corrección de tipo | Sí | Sí | Sí, con nombre de quien corrige |
| Exportar | Sí | Sí | Sí |
| Modo en vivo | No | No | No incluido en esta versión |

La versión Entrenador queda exactamente como está: la matriz no le agrega ni le quita nada. El
consentimiento por atleta podría servirle también, pero agregárselo cambiaría su comportamiento y
rompería la compuerta de paridad; queda como línea futura.

---

## 5. Requisitos

En el formato de `docs/specs/MVP.md` §6, para que el Cap 7 los pueda citar tal cual.

Funcionales, comunes:

- RF-V1. CUANDO alguien crea una cuenta, EL SISTEMA DEBE pedir la versión (Individual,
  Entrenador o Gimnasio) y crear la cuenta con las horas y funciones de esa versión.
- RF-V2. EL SISTEMA DEBE ofrecer solo las funciones de la versión de la cuenta y rechazar en la
  API las que no le corresponden, aunque se pidan directamente.
- RF-V3. EL SISTEMA DEBE calcular las horas usadas en el mes con la duración de los videos
  subidos y mostrarlas contra las incluidas.
- RF-V4. SI una subida supera las horas que quedan, EL SISTEMA DEBE avisarlo antes de subir y,
  en modo bloquear, rechazarla sin crear la sesión.

Individual:

- RF-I1. CUANDO se crea una cuenta Individual, EL SISTEMA DEBE pedir la declaración de mayoría de
  edad y crear el perfil del titular como único peleador de la cuenta.
- RF-I2. CUANDO un usuario Individual sube un video, EL SISTEMA DEBE asignarlo como peleador A sin
  pedirle que lo elija.
- RF-I3. En sparring y combate, EL SISTEMA DEBE pedir que se marque a sí mismo y al compañero, y
  NO DEBE crear un perfil para el compañero.

Gimnasio:

- RF-G1. CUANDO el titular invita a alguien, EL SISTEMA DEBE generar un enlace de un solo uso, con
  el rol fijado, que vence a los 7 días.
- RF-G2. EL SISTEMA DEBE mostrar a todos los miembros del gimnasio todas sus sesiones y todos sus
  peleadores.
- RF-G3. EL SISTEMA DEBE registrar qué miembro subió cada sesión, corrigió cada golpe y asignó
  cada tramo.
- RF-G4. SI un atleta del plantel no tiene consentimiento vigente, EL SISTEMA NO DEBE permitir
  asignarlo a una sesión.
- RF-G5. CUANDO el titular quita a un miembro, EL SISTEMA DEBE conservar las sesiones que subió y
  cerrarle el acceso en el pedido siguiente.

No funcionales:

- RNF-V1 (reemplaza a RNF3). Los videos y resultados de una cuenta solo son visibles para los
  miembros de esa cuenta. Un identificador de otra cuenta responde 404, nunca 403.
- RNF-V2. Ninguna ruta decide qué se ve comparando con el usuario: todas pasan por la cuenta. Se
  verifica con una prueba estructural (`05` §3).
- RNF-V3. La migración no cambia el comportamiento de ninguna cuenta existente (compuerta de
  paridad, `05` §1).
- RNF-V4. Las invitaciones se guardan como hash, y el enlace se muestra una sola vez.

Versión de prueba (pedido del 26-10). Mientras esté vigente, RF-V1, RF-I1 (la declaración la
cubre la carga de las cuentas de prueba), RF-G1, RF-G5 y RNF-V4 quedan postergados con sus tareas:

- RF-D1. CON `BOXTWIN_DEMO` encendida, la página de ingreso DEBE ofrecer un botón por versión y
  rol (Individual, Entrenador, Gimnasio como titular y Gimnasio como entrenador) y NO DEBE pedir
  email, clave ni código de invitación.
- RF-D2. CUANDO alguien toca un botón, EL SISTEMA DEBE abrir la cuenta de prueba de esa versión
  con ese rol.
- RF-D3. EL SISTEMA DEBE poder volver todas las cuentas de prueba a su estado inicial con un
  comando, sin tocar ninguna otra cuenta.
- RNF-D1. Los botones NO DEBEN dar acceso a ninguna cuenta que no esté marcada como de prueba.
- RNF-D2. Con `BOXTWIN_DEMO` apagada, que es como viene, la ruta de ingreso de prueba responde 404
  y la página de ingreso es la de hoy.
- RNF-D3. En una cuenta de prueba el cupo bloquea siempre, cualquiera sea `BOXTWIN_CUPO`.

---

## 6. Desvíos respecto de lo aprobado

| Qué estaba aprobado | Dónde | Qué cambia | Por qué |
|---|---|---|---|
| F13 "planes, pagos, multi-organización" fuera | `docs/specs/MVP.md` §3 | Planes y multi-organización entran; pagos no | Es lo que distingue una versión de otra |
| RNF3 "visibles para ese usuario" | `docs/specs/MVP.md` §6 | "para los miembros de esa cuenta" | En Gimnasio el dueño de una sesión es el gimnasio |
| D4: sin barra del plan ni horas | `prototipo/specs/01` §5 | Las horas vuelven, medidas | Son el límite de cada versión |
| "Lo post-MVP del diseño queda como diseño rotulado" | Cap 6 §8 y ficha de R-03 | Dos versiones nuevas a ocho días del congelamiento | Pedido del 26-10; se declara como ampliación con fecha |
| Registro cerrado por invitación | `app.py`, `_puede_registrarse`; `prototipo/specs/02` §8 | Sigue igual para las cuentas reales; se agrega una entrada sin credenciales a cuentas de prueba, detrás de una bandera | Segundo pedido del 26-10: es una versión de prueba |

---

## 7. Stack

Sin cambios respecto de `prototipo/specs/01` §6. No se agrega ninguna dependencia: ni librería de
correo (el enlace se copia), ni de pagos, ni de permisos. Los roles son dos y caben en una función.

---

## 8. Tres riesgos de ejecución

1. Tocar el control de acceso de todas las rutas a una semana del congelamiento, con las sesiones
   de prueba encima. Un error ahí no se ve como un error sino como una sesión que alguien no
   debería ver. Mitigación: nada se despliega en la estación antes de que cierren las sesiones de
   prueba de la semana del 27-10; compuerta de paridad con la batería existente sin tocar;
   migración solo aditiva, de modo que el código anterior corre sobre la base migrada y volver
   atrás es volver de commit; respaldo de la base antes de migrar (R-12); prueba de aislamiento
   sobre todas las rutas y prueba estructural (`05`).
2. La versión Individual descansa sobre bolsa y sombra, que son función sin validar desde el
   21-10. Si la identidad de una sola persona falla, la versión cuenta mal justo en su caso
   principal. Mitigación: DV-6, medir B8 con el criterio ya pre-registrado (CV4); hasta entonces,
   rótulo de función sin validar y margen "medido en sparring".
3. La versión de prueba deja entrar sin credenciales por un túnel público. Quien tenga la URL puede
   ver el material cargado en las cuentas de prueba y subir video a la única GPU, justo cuando hay
   que mostrar algo. Mitigación: bandera apagada por defecto y encendida solo en las ventanas de
   ensayo y presentación (DV-10); los botones no alcanzan cuentas reales (RNF-D1, CV6); solo
   material con permiso para mostrarse y con seudónimos (DV-9); cupo que siempre bloquea en las
   cuentas de prueba y `BOXTWIN_DEMO_SUBIR=0` para cortar la subida si hace falta (DV-11).

Queda fuera de los tres, y en el registro de riesgos (`06` §4, R-26), que la versión Gimnasio no
se prueba con un gimnasio: C6 se hace con un entrenador, y el plantel compartido no va a tener uso
real antes del 17-11.
