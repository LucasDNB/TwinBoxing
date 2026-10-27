# 02 · Arquitectura: una versión es el plan de la cuenta

Sigue la vista en capas de la Clase 8 y su apartado de seguridad (autenticación, control de
accesos, riesgos que cubre cada medida). Parte de la arquitectura de `prototipo/specs/02`, que no
cambia: Next exportado servido por la FastAPI, un origen, un túnel, una cola en tabla, un worker,
una GPU.

---

## 1. Decisión principal

| Opción | A favor | En contra | Decisión |
|---|---|---|---|
| A. La versión es un campo de la cuenta; una web, una API, una base | Un build, un despliegue, una batería de pruebas. Entrenador queda como caso particular de lo que ya anda. Cambiar de versión es cambiar un campo | Cada ruta tiene que saber de qué cuenta es el pedido (se resuelve con una sola dependencia, §4). La web tiene ramas por versión | **Elegida** |
| B. Tres builds de la web sobre la misma API | Cada web es más chica | Las pantallas de análisis, Fight-Card y perfil son las mismas en las tres: se triplican o se extraen a un paquete. `BOXTWIN_WEB` es una sola carpeta | Descartada |
| C. Una instancia de la API por versión | Aislamiento total | Tres bases y tres colas compitiendo por una GPU sin coordinarse, tres túneles. No tiene sentido en un nodo (RNF4) | Descartada |

Consecuencia de A: la web nunca decide qué versión es ni qué funciones tiene. Lo pregunta a
`GET /cuenta` y dibuja, igual que con `GET /producto` (`prototipo/specs/01` §4). La API rechaza
lo que la versión no tiene aunque la web lo pida (RF-V2).

---

## 2. Capas

```
celular / escritorio
   │ HTTPS, un origen, un túnel
   ▼
presentación   web entrenador (Next exportado)
               GET /cuenta → versión, rol, funciones, horas del mes
               navegación y pantallas según funciones (W27)
   │
API            FastAPI. Cada ruta autenticada recibe Acceso(usuario, cuenta, rol, plan)
               /auth /cuenta /invitaciones /jobs /videos /fightcards /boxeadores /rival ...
   │
lógica         planes.py         qué tiene cada versión (la matriz de 01 §4)
               acceso.py         de qué cuenta es el pedido, con qué rol; 404 si no es suyo
               cupo.py           horas usadas en el mes
               cuentas.py        miembros, invitaciones, consentimientos
   │
datos          cuentas · miembros · invitaciones · consentimientos          tablas nuevas
               sesiones.cuenta_id · boxeadores.cuenta_id
               boxeadores.entrenador_id · correcciones.usuario_id          columnas nuevas
   │
worker         sin cambios: una cola, una GPU, todas las cuentas
```

El worker no sabe de cuentas. Procesa sesiones por id, y la sesión ya dice de quién es.

---

## 3. Cuentas y miembros

- **Cuenta**: quien tiene el plan. `id`, `nombre`, `version` (`individual|entrenador|gimnasio`),
  `boxeador_propio_id` (solo Individual), `creada`.
- **Miembro**: un usuario en una cuenta con un rol, `titular` o `entrenador`. En Individual y en
  Entrenador hay un solo miembro y es el titular. En Gimnasio hay un titular y cualquier cantidad
  de entrenadores.
- **Un usuario pertenece a una sola cuenta.** Restricción deliberada del PFI: con ella la cuenta
  sale del usuario y no hace falta elegirla en cada pedido ni en el token. Levantarla (un
  entrenador que trabaja en dos gimnasios) es post-PFI y se declara.
- `sesiones.usuario_id` se queda y pasa a significar "quién la subió". El dueño es
  `sesiones.cuenta_id`.
- `boxeadores.usuario_id` se queda como "quién lo cargó". El dueño es `boxeadores.cuenta_id`.
- En Individual, el peleador propio se crea al registrarse, con el nombre que el usuario escribe,
  y queda en `cuentas.boxeador_propio_id`. No se puede borrar ni crear otro.

---

## 4. Control de acceso

RNF-V1 reemplaza a RNF3: lo que es de una cuenta solo lo ven sus miembros, y un identificador
ajeno responde 404, como hoy (`sesion_del_usuario` explica por qué no 403).

Una sola dependencia reemplaza a `usuario_actual` en las rutas que tocan datos:

```python
@dataclass(frozen=True)
class Acceso:
    usuario: Usuario
    cuenta: Cuenta
    rol: str          # "titular" | "entrenador"
    plan: Plan        # de planes.PLANES[cuenta.version]

def acceso_actual(authorization: str = Header(""), db = Depends(obtener_sesion)) -> Acceso
```

- `sesion_del_usuario` pasa a `sesion_de_la_cuenta(sesion_id, acceso, db)`: 404 si
  `s.cuenta_id != acceso.cuenta.id`. `_boxeador_del_usuario` pasa a `boxeador_de_la_cuenta`.
- Todo filtro `Sesion.usuario_id == usuario.id` y `Boxeador.usuario_id == usuario.id` pasa a
  `cuenta_id == acceso.cuenta.id`. Hoy son nueve lugares: siete filtros (cinco en `app.py`, dos
  en `panel.py`) y las dos comparaciones `!= usuario.id` de `sesion_del_usuario` y
  `_boxeador_del_usuario`. La prueba estructural de `05` §3 falla si queda alguno.
- El ticket de video sigue llevando usuario y sesión. Al leerlo se verifica además que el usuario
  sea miembro de la cuenta dueña de esa sesión.
- La membresía se lee en cada pedido, junto con el usuario, que ya se lee hoy. Un miembro quitado
  queda afuera en el pedido siguiente aunque su token no haya vencido (RF-G5).
- Orden de los chequeos: primero si el recurso es de la cuenta (404), después si la versión tiene
  la función (403 "esta función no está en la versión Individual"), después el rol (403). El 403
  no filtra nada porque solo se da sobre recursos propios o sobre funciones.

| Acción | Titular de Individual | Titular de Entrenador | Titular de Gimnasio | Entrenador de Gimnasio |
|---|---|---|---|---|
| Ver sesiones, Fight-Cards y perfiles de la cuenta | Sí | Sí | Sí | Sí |
| Subir, marcar, corregir, asignar tramos | Sí | Sí | Sí | Sí |
| Alta de peleadores | No | Sí | Sí | Sí |
| Editar peleadores | Solo el propio | Sí | Sí | Sí |
| Rival Fantasma | No | Sí | Sí | Sí |
| Registrar o revocar un consentimiento | | | Sí | Sí |
| Exportar el registro de consentimientos | | | Sí | No |
| Invitar y quitar miembros (postergado con B23) | | | Sí | No |
| Ver horas del mes | Sí | Sí | Sí | Sí, las restantes |
| Ver actividad por entrenador | | | Sí | No |
| Suprimir un atleta (P2) | | | Sí | No |

Que un entrenador pueda registrar consentimientos es deliberado: es quien está en el gimnasio con
el atleta. Queda registrado quién lo cargó.

---

## 5. Funciones por versión

`boxtwin_api/planes.py`, una tabla escrita una vez:

```python
@dataclass(frozen=True)
class Plan:
    version: str
    horas_mes: float            # Cap 5, Tabla 5.7
    miembros_max: int | None
    peleadores: str             # "propio" | "libre"
    rival: bool
    invitaciones: bool
    plantel_compartido: bool
    consentimiento: str         # "sesion" | "atleta"
    en_vivo: bool               # False en las tres: el Cap 5 lo preve para Gimnasio y no existe

PLANES = {
    "individual": Plan("individual", 4, 1, "propio", False, False, False, "sesion", False),
    "entrenador": Plan("entrenador", 15, 1, "libre", True, False, False, "sesion", False),
    "gimnasio":   Plan("gimnasio", 50, None, "libre", True, True, True, "atleta", False),
}
```

Las horas citan la Tabla 5.7 en un comentario: si el Cap 5 cambia, cambia acá y en ningún otro
lado. `GET /cuenta` devuelve el plan entero como `funciones` para que la web no lo copie.

---

## 6. Cupo de horas

- Unidad: la hora de video subida, que es la "hora de video analizada" del Cap 5.
- Cuenta toda sesión de la cuenta creada en el mes calendario, con el mes cortado en hora de
  Buenos Aires. No cuentan las sesiones en `fallo`, porque no produjeron análisis. No cuentan los
  reprocesos (asignación de tramos, render, reclasificación): no suben video nuevo.
- Duración: `sesiones.duracion_s`. La columna existe desde el MVP pero hoy nadie la escribe: la
  duración real vive en `sesion.json` (`video.duracion_s`, que escribe `procesar`). B20 hace que
  el worker la copie a la base al terminar `procesar`, junto con `fps`, y la migración la rellena
  para las sesiones existentes leyendo su `sesion.json`. Mientras una sesión no tiene duración
  real, cuenta la que declaró la web al subir (campo nuevo `duracion_declarada_s`, que la web ya
  lee del archivo antes de subir, `prototipo/specs/02` §7). Si no hay ninguna de las dos, la
  sesión cuenta cero y `GET /cuenta` lo dice (`horas.sin_duracion`).
- `BOXTWIN_CUPO=informar` (por defecto, DV-2): la subida pasa siempre y la respuesta lleva
  `cupo.excedido`. `BOXTWIN_CUPO=bloquear`: si lo declarado supera lo que queda, 409 antes de leer
  el archivo, como hoy con el consentimiento. Lo que ya pasó la puerta se procesa aunque la
  duración real resulte mayor, y queda como excedente registrado. Es la misma lógica que el Cap 5
  usa para facturar horas excedentes, sin facturarlas.

---

## 7. Invitaciones

Postergadas mientras dure la versión de prueba (§11): en la cuenta de prueba Gimnasio el equipo
viene cargado. Quedan especificadas así para cuando vuelvan:

- Solo el titular de un Gimnasio invita, y solo con rol `entrenador`. No hay transferencia de
  titularidad en el PFI.
- Token de 192 bits (`secrets.token_urlsafe(24)`). Se guarda el SHA-256, nunca el token. El enlace
  (`/ingresar/?invitacion=<token>`) se devuelve una sola vez, en la respuesta que lo crea.
- Un solo uso, vence a los 7 días, atado al email que escribió el titular: el registro con esa
  invitación tiene que usar ese email. No prueba que la persona sea dueña del email (no hay correo
  en el PFI), pero evita que un enlace reenviado por error sirva para otra persona.
- Registrarse con una invitación de cuenta no pide el código de invitación de la instancia: el
  titular ya fue admitido. Con `BOXTWIN_INVITACION=cerrado`, las invitaciones de cuenta siguen
  funcionando y las cuentas nuevas no.
- Un email que ya tiene cuenta no puede aceptar una invitación (un usuario, una cuenta): 409.

---

## 8. Migración

Aditiva, como las tres anteriores (`despliegue/migraciones/`):

1. Tablas nuevas: las crea `create_all`.
2. Columnas nuevas, todas nulas: las agrega `_columnas_faltantes()` de `db.py`.
3. Índice único `(cuenta_id, nombre)` en `boxeadores`: `CREATE UNIQUE INDEX IF NOT EXISTS`, que
   SQLite y PostgreSQL aceptan sin reconstruir la tabla. El índice viejo `(usuario_id, nombre)` se
   queda: con una cuenta por usuario no choca con nada.
4. Relleno, `completar_cuentas(db)`, idempotente, llamado desde `crear_tablas()`: a cada usuario
   sin miembro le crea una cuenta Entrenador con su email como nombre y un miembro titular, y le
   pone esa `cuenta_id` a sus sesiones y peleadores. A cada sesión sin `duracion_s` le copia la
   de su `sesion.json`, si existe. Registra cuántas filas tocó.
5. La misma función corre desde `acceso_actual` si un usuario llega sin ningún miembro. Es un solo
   camino para la base migrada, para la base de pruebas y para un usuario creado a mano. Un
   usuario con un miembro dado de baja no entra en ese camino: recibe 401 y no una cuenta nueva.

`despliegue/migraciones/2026-10-versiones.sql` repite los pasos para PostgreSQL.

Antes de migrar la estación: copia de `~/boxtwin/datos/boxtwin.db` a
`boxtwin.db.antes-de-versiones.bak` y prueba de que la copia abre (R-12). Volver atrás es volver
al commit anterior y reiniciar: el código anterior ignora las tablas y columnas nuevas. Lo único
que se pierde al volver es que las sesiones subidas por un entrenador de Gimnasio pasan a verse
solo para quien las subió.

Variables nuevas en `/etc/boxtwin.env`: `BOXTWIN_CUPO` (`informar|bloquear`), `BOXTWIN_DEMO` y
`BOXTWIN_DEMO_SUBIR` (§11). Ninguna otra. La columna `cuentas.demo` entra con la misma migración.

---

## 9. Seguridad

| Superficie nueva | Qué podría pasar | Qué lo cubre |
|---|---|---|
| Invitación | Un enlace filtrado mete a alguien en el gimnasio | Un uso, 7 días, atado al email; el titular ve que se usó y puede quitar al miembro |
| Roles | Un entrenador se hace titular | El rol lo fija el servidor al crear la invitación; no hay ruta para cambiar roles |
| Rutas | Una ruta olvidada sigue filtrando por usuario y muestra o esconde de más | Prueba de aislamiento sobre todas las rutas con id (I1) y prueba estructural (`05` §3) |
| Ticket de video | Un ticket sirve para una sesión de otra cuenta | El ticket es por sesión y por usuario; al leerlo se verifica la cuenta |
| Miembro quitado | Su token sigue valiendo hasta 72 h | Membresía verificada en cada pedido |
| Compañero en Individual | Se guardan datos de un tercero sin cuenta | No se le crea perfil; el consentimiento de la sesión lo declara el titular (DV-4) |
| Menores | Un menor se registra solo o es filmado | Declaración de mayoría de edad en Individual y en cada consentimiento de atleta. Sin menores en el PFI (R-09). El consentimiento de un representante queda post-PFI |
| Ingreso de prueba | Alguien con la URL entra a una cuenta real | La ruta solo emite tokens para cuentas con `demo = true`, resueltas en el servidor por un nombre de perfil fijo; no recibe ids ni emails (§11) |
| Ingreso de prueba | Alguien con la URL usa la GPU o ve el material cargado | Bandera apagada por defecto y encendida solo en ventanas (DV-10); cupo que siempre bloquea; `BOXTWIN_DEMO_SUBIR=0`; solo material con permiso para mostrarse (DV-9) |
| Ingreso de prueba | Un visitante deja la cuenta de prueba rota para el siguiente | Reinicio con un comando que solo toca cuentas de prueba (§11) |

---

## 10. Qué cambia en la tesis

El Cap 4 describe hoy una aplicación de un usuario por cuenta. Con esto cambian la arquitectura
(modelo de cuentas y control de acceso), la parte de seguridad y el Anexo I. Detalle en `06` §4.

---

## 11. Versión de prueba

Segundo pedido del 26-10: un botón por versión en el ingreso, sin email ni código de invitación.
Las tareas son B29 (API) y W38 (web).

### Qué es una cuenta de prueba

- Una cuenta con `cuentas.demo = true` (columna nueva, por defecto falso). Son cuatro perfiles
  sobre tres cuentas, fijados en `planes.py` al lado de las versiones:

  | Perfil | Cuenta | Rol |
  |---|---|---|
  | `individual` | Individual de prueba | titular |
  | `entrenador` | Entrenador de prueba | titular |
  | `gimnasio_titular` | Gimnasio de prueba | titular |
  | `gimnasio_entrenador` | Gimnasio de prueba | entrenador |

- Cada perfil es un usuario real de la tabla `usuarios`, con un email interno que no se muestra
  (`individual@prueba.boxtwin`) y una clave aleatoria que nadie conoce. No hay forma de entrar a
  esos usuarios por `POST /auth/login`, y el ingreso de prueba no sirve para ningún otro.
- El control de acceso es el mismo de §4: una cuenta de prueba ve lo suyo y nada más. El
  aislamiento con las cuentas reales lo da B19, no una regla aparte.

### Cómo se entra

- `POST /auth/demo {"perfil": "gimnasio_entrenador"}` devuelve el mismo token que el ingreso
  normal, con 8 horas de vida en lugar de 72. La ruta busca el usuario por el perfil, verifica que
  su cuenta tenga `demo = true` y recién ahí emite. No acepta ids ni emails: no hay nada que
  cambiar en el pedido para llegar a otra cuenta.
- Con `BOXTWIN_DEMO` apagada, que es el valor por defecto, la ruta responde 404, igual que una ruta
  que no existe, y `GET /producto` no la anuncia.
- Con la bandera encendida, la web no muestra email, clave ni código (RF-D1). Las cuentas reales
  vuelven a entrar cuando la bandera se apaga. Mientras está encendida, la API sigue aceptando
  `POST /auth/login`, pero ninguna pantalla lo ofrece: es lo que se pidió, y la consecuencia es que
  durante la ventana de prueba Lucas no entra a su cuenta real desde la web.

### Qué tiene adentro y cómo vuelve a cero

- `despliegue/demo.yaml` dice qué se carga: el nombre de cada cuenta, los atletas con su seudónimo
  y sus datos, y la lista de sesiones reales que se copian a cada cuenta (DV-9). En la Individual,
  el peleador propio y sus sesiones de bolsa y sombra; en la de Gimnasio, también el equipo y un
  consentimiento por atleta con la nota "cargado para la versión de prueba".
- `python -m boxtwin_api.demo sembrar` crea las cuentas y copia las sesiones: filas nuevas con ids
  nuevos y una carpeta nueva por sesión, con enlaces duros para lo pesado (video, caché de pose,
  video procesado) y copia de los JSON, así no se duplican gigabytes. La Fight-Card no se
  reprocesa: es la misma, con los nombres reemplazados por los seudónimos.
- `python -m boxtwin_api.demo reiniciar` borra todo lo que pertenece a cuentas con `demo = true`
  (filas y carpetas) y vuelve a sembrar. Antes de borrar verifica que cada sesión que va a tocar
  sea de una cuenta de prueba, y falla sin borrar nada si encuentra una que no.
- Todos los que tocan el mismo botón comparten la cuenta: ven las correcciones y las subidas de los
  demás. Para una versión de prueba alcanza, y el reinicio lo arregla.

### Límites de las cuentas de prueba

- El cupo bloquea siempre (RNF-D3), con las horas de su versión.
- `BOXTWIN_DEMO_SUBIR=0` apaga la subida en las cuentas de prueba (403 en `POST /videos`, botón
  "Subir" con el aviso) sin apagar el resto. Por defecto, encendida (DV-11).
- Los consentimientos que carga la semilla no se pueden revocar (403 "en la versión de prueba
  este consentimiento viene cargado"): si un visitante los revoca, el siguiente no puede asignar
  atletas hasta el reinicio. Los que carga un visitante sí se revocan.
