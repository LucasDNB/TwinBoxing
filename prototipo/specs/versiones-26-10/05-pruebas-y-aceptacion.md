# 05 · Pruebas y aceptación

Paso 3 del Prompt 11 (cinco pruebas de integración críticas y CI) y lo que la Clase 8 pide del
prototipo: cómo se probó y qué dio. A diferencia de los specs del 12-10, el riesgo grande de estas
versiones está del lado del servidor (quién ve qué), así que las cinco pruebas críticas son de la
API. Lo que estas pruebas den va al Cap 7 con fecha y commit.

---

## 1. Compuerta de paridad (RNF-V3)

Antes de desplegar cualquier tanda, las baterías que ya existen pasan sin tocar ninguna prueba:

- `boxtwin-api`: `python -m pytest`. El número de pruebas al empezar B19 se anota en
  `prototipo/TAREAS.md` y es la referencia.
- Web: `npm test` y `E2E_SIN_WEBKIT=1 npm run e2e` (6 de 6 en Chromium).

Si una prueba existente necesita un cambio para pasar, es una regresión del comportamiento y no una
prueba para arreglar. Hay una sola excepción: una prueba que compara una respuesta campo por campo
y falla porque la respuesta ganó un campo nuevo (`subida_por`, `cupo`, `horas`). Ahí se permite
agregar el campo a lo esperado, y el cambio se anota con su motivo en la fila de la tarea.

## 2. Cinco pruebas de integración críticas (pytest)

Con los fixtures de `boxtwin-api/tests/conftest.py` (SQLite nueva por prueba,
`BOXTWIN_CMD=echo`, sesiones escritas en disco por la propia prueba), sin GPU. El fixture
`registrado` de hoy no manda `version`, así que sigue creando una cuenta Entrenador: es parte de la
paridad.

| # | Recorrido | Qué tiene que pasar | Cubre |
|---|---|---|---|
| I1 | Aislamiento entre cuentas | Dos cuentas con sesiones y peleadores. La prueba recorre `app.routes` y, para cada ruta con `{sesion_id}` o `{boxeador_id}`, pide con la otra cuenta: siempre 404. Las listas (`/jobs`, `/boxeadores`, `/panel`) no traen nada ajeno. Un ticket de video de una cuenta no sirve para una sesión de la otra. Como las rutas se enumeran solas, una ruta nueva no se puede olvidar | RNF-V1, CV1 |
| I2 | Plantel compartido | Un gimnasio con titular T y entrenadores X e Y. X sube y marca; Y ve la sesión, la corrige, y la corrección queda con su usuario; `autoria` lo dice. T da de baja a X: el pedido siguiente de X da 401 con el mensaje de baja, y sus sesiones siguen visibles para Y y para T | RF-G2, RF-G3, RF-G5 |
| I3 | Ingreso de prueba | Con `BOXTWIN_DEMO` apagada, `POST /auth/demo` da 404 y `/producto` no lista perfiles. Encendida, cada perfil recibe un token de su cuenta y su rol, y con ese token la cuenta real de la misma base da 404 en todas las rutas de I1. Un perfil que no existe da 422. Si alguien cambia a mano `demo` a falso en una cuenta de prueba, la ruta deja de emitir. En la cuenta de prueba, una subida que pasa el cupo da 409 aunque `BOXTWIN_CUPO` diga informar; con `BOXTWIN_DEMO_SUBIR=0`, 403. `reiniciar` deja las cuentas reales idénticas fila por fila y carpeta por carpeta | RF-D1 a RF-D3, RNF-D1 a RNF-D3, CV6 |
| I4 | Individual de punta a punta | Registro Individual: cuenta, miembro y peleador propio. Subir bolsa: A es el propio. Siembra de una persona, Fight-Card, y el perfil propio la suma a `evolucion`. `POST /boxeadores`: 403. `PUT /sesiones/{id}/boxeadores` con B: 403. `GET /rival`: 403 | RF-I1 a RF-I3, RF-V2 |
| I5 | Horas del mes | Una sesión creada el 31-10 a las 23:30 de Buenos Aires cuenta en octubre y una del 01-11 a las 00:10, en noviembre. Cuenta la duración real si está y la declarada si no. Una sesión en `fallo` no cuenta; un reproceso tampoco. Con `informar`, subir de más da 202 y `cupo.excedido`; con `bloquear`, 409, y no queda ni sesión, ni archivo, ni trabajo en cola | RF-V3, RF-V4 |

La prueba de invitaciones que ocupaba este lugar queda escrita para cuando vuelva B23: un uso (la
segunda vez, 410), vencida y revocada (410), email distinto del invitado (422), email con cuenta
(409), un `rol: "titular"` en el pedido no cambia nada, un entrenador no invita (403) y en la base
no aparece el token, solo su hash.

Además, una prueba por tarea como dice cada tarea en `04`. Las de B25, que no entran en las cinco:
en Gimnasio, asignar un atleta sin consentimiento vigente da 422; revocar deja la constancia; el CSV
lo baja solo el titular; y en una cuenta Entrenador nada de esto cambia el comportamiento de hoy.

## 3. Prueba estructural (RNF-V2)

`boxtwin-api/tests/test_acceso_estructural.py`, dos partes:

1. Lee el código de `app.py`, `panel.py`, `perfil.py`, `rival.py` y `cuentas.py` y falla, con
   archivo y línea, si encuentra una comparación de `usuario_id` con `==` o `!=`. Se permiten las
   asignaciones (`usuario_id=` al crear una sesión, una corrección o una asignación de tramo),
   porque ahí `usuario_id` dice quién hizo algo y no quién puede verlo.
2. Recorre las rutas de FastAPI y falla si una ruta con identificador en el camino no depende de
   `acceso_actual` o del lector de tickets que verifica la cuenta. Excepciones escritas en la
   prueba: `/auth/*`, `/producto`, `/salud`, `/invitaciones/{token}`.
3. Busca en `app.py` y `demo.py` toda emisión de token y falla si alguna, fuera de `/auth/login` y
   `/auth/registro`, no pasa antes por la verificación de `cuenta.demo`.

Es la prueba que impide que una tarea futura vuelva a filtrar por usuario sin que nadie lo note.

## 4. Web

Integración (Vitest con MSW), con una cuenta de prueba por versión y rol en
`tests/fixtures/cuentas/`:

- Navegación por versión y rol (CA-27.1, CA-27.2).
- Ingreso de prueba (CA-00.7 a CA-00.10). Registro e invitación (CA-00.3 a CA-00.6) cuando vuelvan.
- Pantallas de Individual (CA-I01.1 a CA-I06.1) y de Gimnasio (CA-G01.1 a CA-G09.2).

De punta a punta (Playwright, Chromium; WebKit sigue pendiente por `libevent` en la estación), tres
recorridos nuevos al lado de los seis de hoy:

| # | Dispositivo | Recorrido |
|---|---|---|
| E1 | Pixel 7 | Botón "Individual": subir bolsa, marcarse, Fight-Card, y Mi progreso muestra la sesión |
| E2 | Escritorio 1440 × 900 | Botón "Gimnasio · titular": ver Equipo, cargar dos atletas y registrar su consentimiento |
| E3 | Pixel 7 | Botón "Gimnasio · entrenador": subir un sparring con los dos atletas de E2, marcar, Fight-Card, abrir el perfil de un atleta; "Salir" y entrar como "Entrenador" no muestra nada del gimnasio |

La base de los e2e se siembra con `demo.py` y un `demo.yaml` de prueba, y lleva además una cuenta
real con sesiones, para que E3 verifique que no aparece.

Axe sin violaciones serias ni críticas en las pantallas nuevas.

## 5. Criterios de aceptación de las versiones

Pre-registrados el 26-10, antes de construir, con la misma regla que C1 a C11: no se tocan después
de ver el resultado.

| Id | Qué | Cómo se mide | Umbral | Si no pasa |
|---|---|---|---|---|
| CV1 | Aislamiento | I1 sobre todas las rutas y la prueba estructural | Cero fugas | No se despliega. No hay versión parcial de esto |
| CV2 | Paridad | §1 | 100% en verde sin cambiar pruebas, salvo los campos agregados y registrados | No se despliega |
| CV3 | Individual de punta a punta | Lucas entra con el botón "Individual" desde su celular, por el túnel, sube una sesión real de bolsa, se marca una vez y llega a la Fight-Card | Sin ninguna otra intervención (C4 aplicado a Individual) | La versión no se muestra en la demo y se declara |
| CV4 | Bolsa y sombra con una persona (B8) | El criterio de `prototipo/specs/ampliacion-13-10/02-un-solo-boxeador.md` §2, sin cambios: dos videos propios de al menos 3 minutos, uno de bolsa y uno de sombra, recorridos en saltos de 2 s | Al menos 95% de los saltos con la persona en cuadro tienen su marca, en cada video | Individual se presenta con bolsa y sombra como función sin validar, como quedaron el 21-10, y se declara en el Cap 7 |
| CV5 | Gimnasio entre titular y entrenador | E2 y E3 en la base de prueba, y después una vez en la estación con los dos botones de Gimnasio, en dos dispositivos | El recorrido completo sin tocar la base a mano | Gimnasio queda especificado y no se presenta |
| CV6 | Las cuentas reales quedan fuera de la versión de prueba | I3 y la tercera parte de la prueba estructural; en la estación, con la bandera encendida, entrar por cada botón y buscar por id una sesión real conocida | Cero accesos | La bandera no se enciende en la estación |

Ninguna de las dos versiones entra en C6: la prueba con usuarios se hace con la versión Entrenador y
el guion que ya está. Que Individual y Gimnasio no tengan prueba con usuarios se declara tal cual.

CV3 y CV4 usan la GPU de la estación: van fuera de las sesiones de prueba y nunca en paralelo con
una medición de C5.

## 6. CI

`.github/workflows/web-entrenador.yml` ya corre el pytest de la API y las pruebas de la web, así
que las nuevas entran solas. El workflow sigue sin haber corrido en GitHub; mientras tanto, las
compuertas se corren a mano en la estación y se anotan en `prototipo/TAREAS.md` con el commit.
