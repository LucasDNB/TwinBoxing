# 04 · API y datos

Formato del Prompt 11 de la Clase 5: paso 1, el diseño (endpoints con método, ruta, payload y
respuesta; modelo de datos); paso 2, los archivos que se tocan con una línea de qué hace cada uno.
El paso 3, las pruebas, está en `05`. Ninguna tarea B se programa sin que su plan esté aprobado, y
B19 se revisa antes que ninguna porque cambia el control de acceso de todas las rutas.

---

## 1. Rutas que no cambian de forma y sí de alcance

Todas las que hoy reciben `usuario_actual` y tocan datos pasan a recibir `acceso_actual` (`02` §4).
El payload y la respuesta no cambian salvo donde se dice. Lo que cambia es de quién son los datos.

| Ruta | Hoy ve | Pasa a ver | Cambia la respuesta |
|---|---|---|---|
| `GET /jobs`, `GET /jobs/{id}` | Las sesiones del usuario | Las de la cuenta | Suma `subida_por` (B24) y `cola` (B27) |
| `POST /videos` | | Crea la sesión en la cuenta | Suma `cupo` (B20) |
| `POST /jobs/{id}/siembra`, `GET/PUT /jobs/{id}/tramos`, candidatos, recortes | Sesión del usuario | Sesión de la cuenta | No |
| `GET /fightcards/{id}`, export, `PATCH .../golpes/{golpe}` | Sesión del usuario | Sesión de la cuenta | El bloque `sesion` suma `autoria` (B24) |
| `GET /videos/{id}/ticket`, stream, procesado, render | Sesión del usuario | Sesión de la cuenta | No |
| `GET/POST/PATCH /boxeadores`, perfil | Peleadores del usuario | Peleadores de la cuenta | Suma `entrenador` (B24) y `consentimiento` (B25) |
| `PUT /sesiones/{id}/boxeadores` | | | Rechaza según versión (B22) y consentimiento (B25) |
| `GET /rival` | | | 403 en Individual (B22) |
| `GET /panel` | Lo del usuario | Lo de la cuenta | Suma `horas` (B20) y `por_entrenador` (B24) |
| `/auth/*`, `/producto`, `/salud` | | Sin cambio de alcance | `/auth/demo` es nueva (B29) y `GET /producto` suma `demo` y `versiones` (B29). `/auth/registro` cambiaría en B21, postergada |

Prefijos nuevos reservados para la API, que ninguna página de la web puede usar
(`prototipo/specs/02` §4): `/cuenta`, `/invitaciones`. Por eso la pantalla de miembros es
`/equipo/` y la entrada por invitación es `/ingresar/?invitacion=`.

---

## 2. Cambios de API

### B19 · Cuentas, miembros y acceso

Es la base de todo lo demás. No agrega ninguna función visible: después de B19 todo se comporta
igual que antes (compuerta de paridad, `05` §1).

Tablas nuevas:

| Tabla | Columnas |
|---|---|
| `cuentas` | `id`, `nombre` (String 120), `version` (String 16, índice), `boxeador_propio_id` (FK `boxeadores`, nula), `creada` |
| `miembros` | `id`, `cuenta_id` (FK), `usuario_id` (FK, único), `rol` (`titular\|entrenador`), `desde`, `baja_en` (nula), `baja_por` (nula) |

Columnas nuevas, nulas: `sesiones.cuenta_id` (índice), `boxeadores.cuenta_id` (índice). Índice
único nuevo `uq_boxeador_cuenta_nombre (cuenta_id, nombre)`.

`acceso.py`:

- `Acceso` y `acceso_actual` (`02` §4). Si el usuario no tiene ningún miembro, corre
  `completar_cuentas` para él y sigue. Si tiene un miembro con `baja_en`, 401 "tu acceso a
  <cuenta> fue dado de baja": un entrenador quitado no recibe una cuenta nueva sin que nadie lo
  decida.
- `sesion_de_la_cuenta`, `boxeador_de_la_cuenta`: 404 si no es de la cuenta.
- `completar_cuentas(db, usuario=None)`: el relleno idempotente de `02` §8.

`GET /cuenta`:

```json
{
  "id": "…", "nombre": "Gimnasio Norte", "version": "gimnasio", "rol": "titular",
  "usuario": {"email": "ana@…", "nombre": "Ana"},
  "boxeador_propio": null,
  "funciones": {"horas_mes": 50, "miembros_max": null, "peleadores": "libre", "rival": true,
                "invitaciones": true, "plantel_compartido": true, "consentimiento": "atleta",
                "en_vivo": false},
  "horas": null
}
```

`horas` es nulo hasta B20. `boxeador_propio` es `{"id", "nombre"}` en Individual.

- `POST /videos` guarda `cuenta_id` y deja `usuario_id` como quién la subió. `POST /boxeadores`
  guarda `cuenta_id` y valida el nombre único por cuenta (409, el mismo mensaje de hoy).
- Archivos: `acceso.py` (nuevo), `planes.py` (nuevo), `modelos.py`, `app.py` (las nueve
  comparaciones de `02` §4, `GET /cuenta`), `panel.py`, `db.py` (llama a `completar_cuentas`),
  `despliegue/migraciones/2026-10-versiones.sql` (nuevo).
- Pruebas: I1 y la estructural (`05`); `completar_cuentas` dos veces seguidas no duplica nada; un
  usuario sin miembro creado a mano entra y recibe una cuenta Entrenador; un miembro dado de baja
  recibe 401.

### B20 · Horas del mes

- `worker.py`: al terminar `procesar` con la sesión en `espera_siembra`, copia
  `video.duracion_s` y `video.fps` de `sesion.json` a `sesiones.duracion_s` y `sesiones.fps`.
- `POST /videos` suma el campo de formulario `duracion_declarada_s` (número ≥ 0, opcional), que se
  guarda en la columna nueva `sesiones.duracion_declarada_s`.
- `cupo.py`, `horas_del_mes(cuenta, db, ahora) -> dict` con la regla de `02` §6.
- `GET /cuenta` llena `horas`:

```json
"horas": {"mes": "2026-10", "incluidas": 50, "usadas": 12.4, "restantes": 37.6,
          "sin_duracion": 0, "modo": "informar"}
```

- `BOXTWIN_CUPO=bloquear` y lo declarado supera lo que queda: 409 "esta sesión usaría 0,8 h y
  quedan 0,3 h en el mes", antes de leer el archivo. Con `informar`, la respuesta de `POST /videos`
  suma `"cupo": {"excedido": true, "restantes_h": 0.3}` y la sesión se crea igual.
- `GET /panel` suma el mismo bloque `horas`. `GET /producto` suma `"cupo": "informar"`, que no
  expone nada de nadie.
- Archivos: `cupo.py` (nuevo), `worker.py`, `app.py`, `panel.py`, `config.py` (`BOXTWIN_CUPO`),
  `modelos.py`.
- Pruebas: I5.

### B21 · Registro por versión (postergada por la versión de prueba)

Mientras dure la versión de prueba, las cuentas de prueba las crea B29 y `POST /auth/registro`
queda como hoy: un registro sin versión, que B19 convierte en una cuenta Entrenador. Se especifica
para cuando vuelva el email.

`POST /auth/registro` recibe `Registro`, que extiende `Credenciales`:

| Campo | Tipo | Regla |
|---|---|---|
| `version` | `individual\|entrenador\|gimnasio` | Por defecto `entrenador`: la web de hoy no lo manda y sigue creando lo mismo |
| `nombre` | texto 1 a 80 | Obligatorio en Individual (es el nombre del peleador propio); opcional en los demás |
| `nombre_cuenta` | texto 2 a 80 | Obligatorio en Gimnasio |
| `mayor_de_edad` | bool | Tiene que ser `true` en Individual; 422 "BoxTwin Individual es para mayores de 18" |
| `invitacion_cuenta` | texto | Si viene, es el camino de B23 y los demás campos de versión se ignoran |

- Sin `invitacion_cuenta`: el chequeo de instancia de hoy (`_puede_registrarse`), después usuario,
  cuenta, miembro titular y, en Individual, el peleador propio, todo en un commit.
- Guarda el nombre en `usuarios.nombre`, la columna que agrega B29.
- `POST /auth/login` no cambia.
- La pantalla de registro lee las horas de `producto.versiones`, que agrega B29.
- Archivos: `esquemas.py` (`Registro`), `app.py`.
- Pruebas: cada versión crea lo que le corresponde; Individual sin `mayor_de_edad` da 422;
  Gimnasio sin `nombre_cuenta` da 422; un registro sin `version` crea una cuenta Entrenador.

### B22 · Reglas de la versión Individual

- `POST /videos` en Individual pone `boxeador_a_id = cuenta.boxeador_propio_id` al crear la sesión.
  El campo `consentimiento` sigue siendo uno: la web lo manda en `true` solo si el usuario marcó
  "soy yo quien aparece" y, en sparring y combate, "la otra persona aceptó ser filmada y
  analizada" (`03` §I-02).
- `PUT /sesiones/{id}/boxeadores` en Individual: `boxeador_a` solo puede ser el propio o nulo;
  `boxeador_b` tiene que ser nulo, si no 403 "en la versión Individual el compañero no tiene
  perfil".
- La siembra no cambia. La web pregunta primero "¿cuál sos vos?" y eso es `track_a`.
- `POST /boxeadores` da 403 en Individual. `PATCH /boxeadores/{id}` solo sobre el propio, que es
  el único que la cuenta ve.
- `GET /rival` da 403 en Individual.
- Mi progreso no necesita ruta nueva: usa `GET /panel`, `GET /jobs` y
  `GET /boxeadores/{propio}/perfil`, que ya trae `evolucion`.
- Archivos: `app.py`, `planes.py`.
- Pruebas: I4.

### B23 · Invitaciones y bajas (Gimnasio, postergada por la versión de prueba)

En la cuenta de prueba Gimnasio el equipo lo carga B29, y la lista de miembros de solo lectura está
en B24. Lo que sigue se especifica para cuando vuelvan el email y las invitaciones.

Tabla nueva `invitaciones`: `id`, `cuenta_id`, `email`, `rol`, `token_sha256` (único), `creada`,
`creada_por`, `vence`, `usada_en`, `usada_por`, `revocada_en`.

| Método y ruta | Quién | Cuerpo | Respuesta |
|---|---|---|---|
| `POST /cuenta/invitaciones` | Titular | `{"email"}` | 201 `{"id", "email", "rol": "entrenador", "vence", "enlace": "/ingresar/?invitacion=…"}`. El enlace solo aparece acá |
| `GET /cuenta/invitaciones` | Titular | | `[{"id", "email", "estado": "pendiente\|usada\|vencida\|revocada", "vence"}]`, sin token |
| `DELETE /cuenta/invitaciones/{id}` | Titular | | 204 |
| `GET /invitaciones/{token}` | Cualquiera | | 200 `{"cuenta", "rol", "email", "vence"}`; 410 si venció, se usó o se revocó; 404 si no existe |
| `POST /auth/registro` con `invitacion_cuenta` | Cualquiera | `email`, `clave`, `nombre`, `invitacion_cuenta` | Token, como hoy. El email tiene que ser el de la invitación (422 si no); un email con cuenta da 409 |
| `DELETE /cuenta/miembros/{usuario_id}` | Titular | | 204. Marca `baja_en` y `baja_por`; no borra al usuario ni sus sesiones. El titular no se puede dar de baja a sí mismo (422) |

- Solo en Gimnasio (403 en las otras versiones). `miembros_max` del plan se respeta al crear la
  invitación.
- Archivos: `cuentas.py` (nuevo: rutas de `/cuenta/*` e `/invitaciones/*`), `esquemas.py`,
  `modelos.py`, `app.py` (registro e inclusión del router).
- Pruebas: I3.

### B24 · Plantel compartido y autoría

La visibilidad compartida ya sale de B19. Esta tarea agrega lo que hace falta para trabajar entre
varios:

- `GET /cuenta/miembros` (titular, solo Gimnasio):
  `[{"usuario_id", "nombre", "email", "rol", "desde", "baja_en"}]`. En una cuenta de prueba no trae
  `email`, que es interno. Es de solo lectura: las bajas son de B23.
- Columna `boxeadores.entrenador_id` (FK `usuarios`, nula): el entrenador a cargo. Se edita con
  `PATCH /boxeadores/{id}` (`{"entrenador_id"}`, que tiene que ser un miembro activo de la cuenta).
  `GET /boxeadores?a_cargo=yo` filtra. `GET /boxeadores` suma `"entrenador": {"id", "nombre"} | null`.
- Columna `correcciones.usuario_id` (nula; las filas anteriores quedan nulas, "antes de las
  cuentas"). `PATCH /fightcards/{id}/golpes/{golpe}` la llena.
- `sesion_a_dict` suma `"subida_por": {"id", "nombre"}`.
- `GET /fightcards/{id}` suma al bloque `sesion`, que no se guarda en el archivo:
  `"autoria": {"subida_por": "Ana", "correcciones": {"<golpe>": "Ana"}}`. El contrato 0.3 no cambia.
- `GET /panel` del titular de un Gimnasio suma
  `"por_entrenador": [{"usuario_id", "nombre", "sesiones_mes", "horas_mes", "pendientes"}]`.
  Para el entrenador ese campo es nulo.
- Archivos: `modelos.py`, `esquemas.py`, `app.py`, `panel.py`.
- Pruebas: I2.

### B25 · Consentimiento por atleta

Tabla nueva `consentimientos`: `id`, `cuenta_id`, `boxeador_id`, `texto_version`, `mayor_de_edad`,
`nota` (nula, por ejemplo "firmado en papel"), `registrado_por`, `registrado_en`, `revocado_en`
(nula), `revocado_por` (nula). Un consentimiento revocado no se borra: es la constancia.

| Método y ruta | Quién | Cuerpo | Respuesta |
|---|---|---|---|
| `GET /cuenta/consentimiento` | Miembro | | `{"version": "…", "texto": "…"}`: el texto vigente, que vive en `boxtwin_api/textos/` |
| `POST /boxeadores/{id}/consentimiento` | Miembro | `{"texto_version", "mayor_de_edad": true, "nota"}` | 201; 409 si ya hay uno vigente; 422 si `mayor_de_edad` no es `true` o la versión no es la vigente |
| `DELETE /boxeadores/{id}/consentimiento` | Miembro | | 204. Marca `revocado_en` y `revocado_por` |
| `GET /cuenta/consentimientos.csv` | Titular | | Atleta, estado, versión, registrado por y cuándo, revocado por y cuándo |

- Solo en Gimnasio. `GET /boxeadores` suma `"consentimiento": {"estado": "vigente|falta|revocado",
  "desde"}`, nulo en las otras versiones.
- `PUT /sesiones/{id}/boxeadores` en Gimnasio: asignar un atleta sin consentimiento vigente da 422
  "Falta el consentimiento de Tomás Ríos". Desasignar siempre se puede.
- Revocar frena las asignaciones nuevas. Las sesiones anteriores se quedan: sacarlas es la
  supresión (B26), y la ficha del atleta lo dice.
- El consentimiento por sesión de `POST /videos` sigue igual, para quien aparece y no está en el
  plantel.
- El texto es decisión de Lucas (DV-5). Si el del 08-10 para C6 no está a mano, B25 no arranca:
  no se inventa un texto legal.
- Archivos: `cuentas.py`, `modelos.py`, `app.py`, `boxtwin_api/textos/consentimiento-<version>.md`.
- Pruebas: `05` §2.

### B26 · Supresión de un atleta (P2, se especifica y no se construye)

- `GET /boxeadores/{id}/supresion` (titular): simulacro. `{"sesiones": 9, "sesiones_con_otros": 4,
  "otros_atletas": ["N. Ferreyra"], "correcciones": 31}`.
- `DELETE /boxeadores/{id}?confirmar=<nombre exacto>` (titular): borra de disco las sesiones en que
  aparece (video, caché de pose, Fight-Card, video procesado), y de la base esas sesiones con sus
  trabajos, correcciones y asignaciones, el atleta y sus consentimientos. Deja una fila en una
  tabla `supresiones` con fecha, quién y cuántas sesiones, sin el nombre.
- Lo que hay que decir en la pantalla y en la tesis: una sesión de sparring tiene a dos personas, y
  suprimir a una se lleva también los datos del otro en esa sesión. Y las correcciones de esas
  sesiones son etiquetas del dataset que se pierden.

### B27 · Posición en la cola (P1)

- `GET /jobs/{id}` de una sesión en `en_cola` suma `"cola": {"delante": 3, "espera_min": [12, 28]}`.
- `delante` cuenta los trabajos en cola creados antes y los tomados, de todas las cuentas, porque
  la GPU es una. `espera_min` multiplica la duración real o declarada de esos trabajos por el
  `factor_procesamiento` mínimo y máximo de `GET /producto`. No dice de quién son.
- Archivos: `cola.py` (función de lectura, sin tocar la toma de trabajos), `app.py`.
- Pruebas: con trabajos de dos cuentas, la posición es la correcta y la respuesta no trae ningún
  dato de la otra cuenta.

### B28 · Cambio de versión (P2, se especifica y no se construye)

- `PATCH /cuenta {"version": "entrenador"}`, titular, solo de Individual a Entrenador: el peleador
  propio queda como un peleador más y `boxeador_propio_id` pasa a nulo. Es el camino de "puerta
  de entrada" del Cap 5.
- Cualquier otro cambio da 422 en el PFI. Sin pagos, cambiar de versión es gratis, y por eso no se
  abre más que el caso que el Cap 5 nombra.

### B29 · Ingreso de prueba

Segundo pedido del 26-10. Depende de B19 y reemplaza, mientras dure la versión de prueba, a B21 y
a B23. Diseño en `02` §11.

Columnas nuevas: `cuentas.demo` (bool, por defecto falso, índice) y `usuarios.nombre` (nula), que
usa la autoría de B24 y, cuando vuelva, el registro de B21.

`planes.py` suma `PERFILES_DEMO`, con los cuatro perfiles de `02` §11, su nombre en pantalla, su
descripción y si están disponibles.

`POST /auth/demo`:

- Cuerpo `{"perfil": "individual|entrenador|gimnasio_titular|gimnasio_entrenador"}`. Respuesta:
  el `Token` de siempre más `"perfil"`, con 8 horas de vida.
- Con `BOXTWIN_DEMO` apagada la ruta no se registra, así que responde el 404 de una ruta que no
  existe. Encender o apagar es cambiar el env y reiniciar, como las demás banderas.
- Perfil que no existe: 422. Perfil no disponible todavía: 409 "en construcción". Cuentas de prueba
  sin sembrar: 409 "faltan las cuentas de prueba: correr `python -m boxtwin_api.demo sembrar`".
- La ruta busca el usuario del perfil y verifica que su cuenta tenga `demo = true` antes de emitir.
  Si no la tiene, 500 y un registro en el log: es un error de integridad, no algo que el pedido
  pueda provocar.

`GET /producto` suma, sin exponer nada de nadie:

```json
"demo": {"activo": true, "subir": true,
         "perfiles": [{"perfil": "individual", "nombre": "Individual", "version": "individual",
                       "rol": "titular", "descripcion": "…", "disponible": true}]},
"versiones": {"individual": {"horas_mes": 4}, "entrenador": {"horas_mes": 15},
              "gimnasio": {"horas_mes": 50}}
```

Con la bandera apagada, `"demo": {"activo": false}` y nada más.

`GET /cuenta` suma `"demo": true|false`, para el rótulo y el "Salir" de la web.

Reglas en las cuentas de prueba:

- El cupo bloquea siempre (RNF-D3): `cupo.py` lee `cuenta.demo` antes que `BOXTWIN_CUPO`.
- `BOXTWIN_DEMO_SUBIR=0`: `POST /videos` da 403 "la subida está apagada en la versión de prueba" y
  `producto.demo.subir` es falso.
- Con B25: columna `consentimientos.semilla` (bool); un consentimiento de la semilla no se revoca
  (403).
- Los dos perfiles de Gimnasio quedan con `disponible: false` hasta que se despliegue la tanda 2;
  la web los muestra deshabilitados con "En construcción", el mismo estado que usaron bolsa y
  sombra desde el 13-10.

`demo.py`, con línea de comando:

- `python -m boxtwin_api.demo sembrar [--yaml ~/boxtwin/demo.yaml]`: crea las tres cuentas, los
  cuatro usuarios con clave aleatoria, los peleadores con su seudónimo, los miembros y las sesiones
  copiadas como dice `02` §11. Si las cuentas ya existen no hace nada y lo dice.
- `python -m boxtwin_api.demo reiniciar [--yaml …]`: en una transacción, junta todo lo que pertenece
  a cuentas con `demo = true`, verifica que cada sesión y cada carpeta sea de una de esas cuentas,
  borra las filas, hace commit, recién ahí borra las carpetas, y vuelve a sembrar. Si una
  verificación falla, no borra nada.
- `despliegue/demo.yaml.ejemplo`, versionado, con la forma del archivo:

```yaml
individual:
  cuenta: "Individual de prueba"
  peleador: {nombre: "Boxeador de prueba", guardia: ortodoxa, altura_m: 1.75}
  sesiones: ["<id de una sesión real de bolsa>", "<id de una de sombra>"]
entrenador:
  cuenta: "Entrenador de prueba"
  peleadores: [{nombre: "Atleta 1", guardia: ortodoxa}, {nombre: "Atleta 2", guardia: zurda}]
  sesiones: [{id: "<id>", a: "Atleta 1", b: "Atleta 2"}]
gimnasio:
  cuenta: "Gimnasio de prueba"
  miembros: {gimnasio_titular: "Titular de prueba", gimnasio_entrenador: "Entrenador de prueba"}
  peleadores: [{nombre: "Atleta 3", entrenador: gimnasio_entrenador}]
  sesiones: [{id: "<id>", a: "Atleta 3", b: null, subida_por: gimnasio_entrenador}]
```

  El archivo real vive en la estación (`~/boxtwin/demo.yaml`), como `/etc/boxtwin.env`. Qué
  sesiones entran lo decide Lucas (DV-9).

- Archivos: `demo.py` (nuevo), `planes.py`, `modelos.py`, `app.py`, `config.py` (`BOXTWIN_DEMO`,
  `BOXTWIN_DEMO_SUBIR`), `cupo.py`, `despliegue/demo.yaml.ejemplo` (nuevo).
- Pruebas: I3, y que `reiniciar` sobre una base con cuentas reales y de prueba deje las reales
  idénticas fila por fila y carpeta por carpeta.

---

## 3. Modelo de datos, resumen de cambios

| Tabla | Cambio | Tarea |
|---|---|---|
| `cuentas` | Nueva; `demo` | B19, B29 |
| `miembros` | Nueva | B19 |
| `sesiones` | `cuenta_id`; `duracion_declarada_s`; `duracion_s` y `fps` pasan a escribirse | B19, B20 |
| `boxeadores` | `cuenta_id`, índice único por cuenta; `entrenador_id` | B19, B24 |
| `usuarios` | `nombre` | B29 |
| `invitaciones` | Nueva, cuando vuelva B23 | B23 |
| `correcciones` | `usuario_id` | B24 |
| `consentimientos` | Nueva; `semilla` | B25, B29 |
| `supresiones` | Nueva, solo si se construye B26 | B26 |

Todas las columnas nuevas son nulas, así que `_columnas_faltantes()` de `db.py` las agrega sola. El
índice único y el relleno van en `completar_cuentas` y en el SQL de PostgreSQL.

## 4. Archivos que se tocan

API (`boxtwin-api/src/boxtwin_api/`):

| Archivo | Qué hace |
|---|---|
| `planes.py` (nuevo) | La matriz de funciones por versión y los perfiles de prueba |
| `demo.py` (nuevo) | Sembrar y reiniciar las cuentas de prueba, con línea de comando |
| `acceso.py` (nuevo) | `Acceso`, `acceso_actual`, recursos de la cuenta, relleno de cuentas |
| `cupo.py` (nuevo) | Horas del mes de una cuenta |
| `cuentas.py` (nuevo) | Rutas de `/cuenta` y consentimientos; las de `/invitaciones`, cuando vuelva B23 |
| `textos/consentimiento-<version>.md` (nuevo) | El texto que firma el atleta, versionado |
| `modelos.py` | Las tablas y columnas de §3 |
| `esquemas.py` | `Registro`, entradas de invitación y consentimiento, `subida_por` |
| `app.py` | `acceso_actual` en las rutas, reglas de versión, registro |
| `panel.py` | Filtro por cuenta, `horas`, `por_entrenador` |
| `worker.py` | Copia duración y fps a la base después de `procesar` |
| `cola.py` | Lectura de la posición en cola |
| `config.py` | `BOXTWIN_CUPO`, `BOXTWIN_DEMO`, `BOXTWIN_DEMO_SUBIR` |
| `db.py` | Llama a `completar_cuentas` |

Despliegue: `despliegue/migraciones/2026-10-versiones.sql` (nuevo); `despliegue/demo.yaml.ejemplo`
(nuevo); `despliegue/README.md`, las líneas de `BOXTWIN_CUPO` y `BOXTWIN_DEMO`, cómo sembrar y
reiniciar las cuentas de prueba, y el respaldo previo.

Web (`prototipo/boxtwin-web-entrenador/`): el detalle está en `03`. Lo estructural:
`lib/api/esquemas.ts` (zod de `Cuenta`), `lib/cuenta.ts` (nuevo: `useCuenta`, `tiene(funcion)`),
`components/boxtwin/Navegacion.tsx` (destinos por versión), `lib/textos.ts` (textos por versión),
`app/(app)/equipo/page.tsx` (nuevo), `components/pantallas/progreso/` (nuevo),
`components/pantallas/equipo/` (nuevo), y cambios en `nueva/`, `identificar/`, `peleadores/`,
`panel/`, `sesion/` e `ingresar`.
