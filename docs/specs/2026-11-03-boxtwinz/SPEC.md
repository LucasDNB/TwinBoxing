# BoxTwinZ — Spec de construcción para Claude Code

- **Fecha:** 3-11-2026
- **Autor de las decisiones:** Lucas Benítez
- **Fuente de verdad del producto:** "BoxTwinZ — Definición de producto" (https://claude.ai/artifact/BS26BLoc8VMR3dSGNAhvDg)
- **Ubicación en el repo:** `docs/specs/2026-11-03-boxtwinz/SPEC.md`, con el registro de tareas en `docs/specs/2026-11-03-boxtwinz/TAREAS.md`
- **Plazo:** 4 días de construcción. Entrega final de la tesis: 17-11-2026.

> Las diferencias entre esta spec y el repo están en `TAREAS.md`, sección "Desvíos". Esta
> spec se guarda tal como la escribió Lucas; manda el repo.

---

## 0. Cómo usar esta spec (instrucciones para Claude Code)

1. **Leé antes de escribir código:**
   - `CLAUDE.md`
   - `despliegue/README.md`
   - `prototipo/contrato/fightcard-0.3.schema.json`
   - el código de `boxtwin-api/`: la cola, el worker y la forma en que llama a `boxtwin-annotator procesar` y `completar`
   - el flujo de sesiones individuales de bolsa y sombra (B8/W19, bandera `BOXTWIN_SESIONES_INDIVIDUALES`)
   - los tokens visuales y componentes de `prototipo/boxtwin-web-entrenador/`

   Si algo de esta spec no coincide con el repo, **manda el repo**. Anotá la diferencia en `TAREAS.md` como "Desvío" y seguí.
2. **Trabajá en la rama `boxtwinz`, en un worktree propio** (`~/Proyectos/TwinBoxing-boxtwinz`). Abrí un PR por día de trabajo con `gh api` REST (GraphQL no está disponible).
3. **No toques A (producción).** No se modifican `boxtwin-api/`, `prototipo/boxtwin-web-entrenador/`, `/etc/boxtwin.env` ni sus servicios systemd. Tampoco se cambian modelos ni checkpoints. El motor se **importa o se invoca**, no se edita. Si hace falta un cambio en el motor, se escribe como propuesta en `TAREAS.md` y no se aplica.
4. **Entornos conda:** el worker corre en `twinboxing_env`. `boxtwin_mmaction` no se usa en BoxTwinZ, porque no hay clasificador de tipo.
5. **Registro SDD:** cada tarea Z* de la sección 12 se registra en `TAREAS.md` con estado, commit y evidencia (test que pasa o captura). Una tarea no está hecha sin evidencia.
6. **Idioma:** la interfaz, los mensajes de error y los textos van en español rioplatense neutro. El código y los identificadores, en inglés o español, siguiendo la convención del paquete del que se copie el patrón.
7. **Valores provisionales:** los números marcados **[PROVISIONAL]** van a configuración (`boxtwinz.toml` o variables de entorno), nunca hardcodeados. Lucas los valida después.

---

## 1. Qué es BoxTwinZ

BoxTwinZ es una PWA mobile first que procesa video pregrabado de sombra, bolsa y sparring, y cuenta los golpes detectados de izquierda y derecha. Funciona como puente de entrenamiento hacia el BoxTwin completo: cada sesión valorada por el usuario se vuelve material para reentrenar.

**No hace** clasificación de tipo de golpe, cabeza/cuerpo medido, detección de conexión, modo en vivo, avatar 3D ni consejos con LLM.

**Números del motor que hay que respetar en la interfaz:**
- **Detector como corre el producto:** F1 0,600, precisión 0,885, recall 0,456.
- **Identidad en sparring con siembra (C1):** 0,810.
- **Bolsa y sombra:** sin medir (B8).

Por eso la interfaz dice siempre **"golpes detectados"**, nunca "golpes lanzados".

---

## 2. Arquitectura

```
boxtwinz/
├── api/                 # FastAPI: cuentas, perfiles, sesiones, social, reglas, admin
│   ├── boxtwinz_api/
│   │   ├── main.py
│   │   ├── config.py        # lee boxtwinz.toml + entorno
│   │   ├── db.py            # SQLAlchemy 2, PostgreSQL (SQLite en tests)
│   │   ├── models/          # tablas de la sección 4
│   │   ├── routers/         # endpoints de la sección 5
│   │   ├── services/
│   │   │   ├── metrics.py       # sección 6
│   │   │   ├── rules.py         # consejos, sección 8
│   │   │   ├── archetypes.py    # sección 7
│   │   │   ├── ranking.py
│   │   │   ├── capacity.py      # sección 10
│   │   │   └── engine_adapter.py # traduce la salida del motor a golpes izq/der
│   │   └── worker.py        # cola + subproceso del motor
│   ├── alembic/
│   └── tests/
├── web/                 # Next.js (misma versión y stack que la web entrenador), PWA
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── public/manifest.webmanifest, sw.js
├── seeds/arquetipos.yaml
├── boxtwinz.toml.example
└── despliegue/
    ├── boxtwinz-api.service
    ├── boxtwinz-worker.service
    └── README.md
```

**Decisiones:**
- **Back:** FastAPI con Pydantic 2, como `boxtwin-api`. Copiá los patrones de la cola: tabla de trabajos con `SELECT … FOR UPDATE SKIP LOCKED` en PostgreSQL. Nada de Redis ni Celery.
- **Front:** Next.js con TS estricto, Tailwind 4, shadcn y `@animate-ui`, con las mismas versiones que la web entrenador. Exportación estática servida por la API, como hace A con `BOXTWIN_WEB`.
- **Estética:** misma paleta, tipografía, radios y componentes base que `prototipo/boxtwin-web-entrenador/`. Se copian los tokens; no se importa nada de esa carpeta en tiempo de build.
- **Base:** PostgreSQL con base `boxtwinz`, separada de A. Para los tests, SQLite.
- **Datos:** videos y artefactos en `~/boxtwinz-datos/` (`videos/`, `trabajos/<id>/`).
- **Puerto:** 8003 propuesto. Verificá que no lo usen A ni el gemelo (8002). Para la demo se expone por el mismo mecanismo de túnel que A, con una ruta o subdominio propios; ver `despliegue/README.md` de A.
- **GPU:** un solo trabajo de GPU a la vez. Si A y BoxTwinZ comparten la GPU, el worker de BoxTwinZ toma un lock de archivo (`/tmp/boxtwin-gpu.lock`) que A no conoce. Documentá ese riesgo en `TAREAS.md`.

---

## 3. Usuarios, autenticación y roles

- **Registro:** email, contraseña (argon2id), nombre visible y guardia del perfil 1 (ortodoxa o zurda).
- **Checks al registrarse:**
  - **`consentimiento_uso` (obligatorio, sin marcar por defecto):** "Tengo 18 años o más y cuento con la autorización de las personas que aparecen en los videos que cargo. No cargo videos con menores de edad." Se guardan la fecha, la versión del texto y la IP.
  - **`consentimiento_reentrenamiento` (opcional, revocable desde ajustes):** "Autorizo a usar mis videos y valoraciones para reentrenar el modelo. Lo que ya se usó en un modelo entrenado no puede retirarse de ese modelo." Se guarda con fecha y versión.
- **Sesión:** JWT en cookie httpOnly, SameSite=Lax, de 30 días.
- **Roles:** `usuario` y `admin`. Los admins son los emails listados en `BOXTWINZ_ADMINS`.
- **Plan:** `free` o `premium`. En esta versión, premium se activa con un botón "Pasar a premium (5 USD/mes)" que **no cobra**: registra la intención y cambia el plan, con el rótulo "Pago simulado — versión de demostración".
- **Baja de cuenta:** borra los videos, los perfiles y las sesiones. Las valoraciones ya usadas quedan anonimizadas.

---

## 4. Modelo de datos

Todas las tablas llevan `id` (UUID), `creado` y `actualizado`.

| Tabla | Campos clave | Notas |
|---|---|---|
| `usuarios` | email único, hash, nombre, plan (`free`/`premium`), rol, `consentimiento_uso_at`, `consentimiento_uso_version`, `consentimiento_reent_at` (nullable), `suspendido` | |
| `perfiles` | usuario_id, `ranura` (1–4), nombre, `estilo` (`propio`/`fajador`/`estilista`/`hibrido`), `guardia` (`ortodoxa`/`zurda`, solo ranura 1), `publico` (bool, solo ranura 1), `arquetipo_id` (nullable), `customizado` (bool) | Único (usuario_id, ranura). Ranura 1 = propio. Ver sección 7 |
| `arquetipos` | clave, nombre ficticio, estilo, `variante` (nullable), `solo_premium`, `metricas` (JSON: medias y desvíos por modalidad), `exposicion` (JSON), `curva_ritmo` (JSON) | Se carga de `seeds/arquetipos.yaml` |
| `sesiones` | usuario_id, perfil_id, `modalidad` (`sombra`/`bolsa`/`sparring`), `rival_perfil_id` (nullable), video (ruta, duración, fps, resolución), `estado` (`subida`/`procesando`/`esperando_siembra`/`completando`/`lista`/`error`), `error_etapa`, `rounds_declarados` (nullable), `apta_ranking` (bool), `calidad_pose` (float) | |
| `golpes` | sesion_id, `quien` (`usuario`/`rival`), `mano` (`izquierda`/`derecha`), `t_inicio`, `t_fin`, `confianza`, `altura_estimada` (nullable, solo para la capa opcional) | Solo golpes con identidad resuelta |
| `metricas_sesion` | sesion_id, `quien`, `segmento`, volumen, pct_izquierda, pct_mano_adelantada, combinaciones_min, sosten_ritmo, iniciativa (nullable), `curva` (JSON: golpes por bloque de 10 s) | `segmento` = null para el total |
| `exposicion_sesion` | sesion_id, golpes del rival por mano y por lado del usuario | Solo sparring |
| `valoraciones` | sesion_id, estrellas (1–5), comentario (nullable) | Obligatoria para cerrar la vista de resultados la primera vez |
| `correcciones` | sesion_id, golpe_id (nullable), `tipo` (`no_fue_golpe`/`falto_golpe`/`otra_mano`), `t` (para `falto_golpe`) | |
| `amistades` | solicitante_id, destinatario_id, `estado` (`pendiente`/`aceptada`/`rechazada`) | |
| `denuncias` | denunciante_id, perfil_id, `motivo` (`no_es_la_persona`/`video_ajeno`/`contenido_inapropiado`/`menor_en_video`/`otro`), detalle, `estado` (`abierta`/`resuelta`), `tanda_id` (nullable) | |
| `acciones_moderacion` | denuncia_id, admin_id, `accion` (`desestimar`/`sacar_ranking`/`suspender`), nota | |
| `tandas_moderacion` | admin_id, inicio, fin, cantidad revisada | |
| `trabajos` | sesion_id, `etapa` (`procesar`/`completar`), estado, intento, `inicio_gpu`, `fin_gpu`, log | Cola |
| `uso_gpu_diario` | fecha, segundos_ocupados, utilizacion (0–1) | Sección 10 |
| `notificaciones_admin` | tipo (`denuncia_nueva`/`capacidad_70_15`/`capacidad_90_5`/`fallo_trabajo`), payload, leida | |
| `eventos_publicidad` | usuario_id, ubicacion, `tipo` (`impresion`) | Espacio reservado, sin red real |
| `uso_minutos` | usuario_id, mes (AAAA-MM), segundos_procesados | Para los cupos |

---

## 5. API (prefijo `/api`)

**Cuentas:**
- `POST /auth/registro`, `POST /auth/ingreso`, `POST /auth/salida`, `GET /yo`
- `PATCH /yo`: nombre, guardia, consentimiento de reentrenamiento
- `POST /yo/premium`: pago simulado
- `DELETE /yo`

**Perfiles:**
- `GET /perfiles`: las 4 ranuras.
- `PATCH /perfiles/{id}`:
  - en la ranura 1: nombre, guardia y público;
  - en las ranuras 2 a 4, **solo premium**: nombre, estilo y `customizado`. En free responde 403 `plan_requerido`.
- `GET /perfiles/{id}/stats?modalidad=`: stat general (sección 6.3).

**Sesiones:**
- `POST /sesiones`: multipart con video, modalidad, perfil_id y rounds_declarados opcional. Antes de encolar valida el cupo de minutos del mes (sección 9); si no alcanza, responde 402 `cupo_agotado`.
- `GET /sesiones?perfil_id=&modalidad=`, `GET /sesiones/{id}`
- `GET /sesiones/{id}/candidatos`: recortes candidatos para la siembra (solo sparring en `esperando_siembra`).
- `POST /sesiones/{id}/siembra`: `{usuario: track_id, rival: track_id, rival_cajon_perfil_id?}`. El cajón solo se acepta si el usuario es premium; si no, se ignora.
- `GET /sesiones/{id}/resultado`: golpes, métricas, curva, exposición y consejos.
- `GET /sesiones/{id}/video`: con soporte de rangos HTTP.
- `POST /sesiones/{id}/valoracion`, `POST /sesiones/{id}/correcciones`
- `GET /comparar?a={sesion}&b={sesion}`: misma modalidad; si no, 422.
- `DELETE /sesiones/{id}`

**Social:**
- `GET /ranking?modalidad=&metrica=`: perfiles 1 públicos con sesiones aptas. Responde top 100 más la posición propia.
- `GET /publicos?q=`, `GET /publicos/{perfil_id}`
- `POST /amigos/solicitud` `{email}`, `POST /amigos/{id}/aceptar|rechazar`, `GET /amigos`
- `GET /amigos/{usuario_id}/perfil`: perfil 1 del amigo y sus sesiones de las tres modalidades. Los cajones 2 a 4 nunca se exponen.
- `POST /denuncias`

**Rival fantasma:**
- `GET /rival?perfil_id=&modalidad=&rival_tipo=arquetipo|publico|amigo|cajon&rival_id=`: comparación, curvas, exposición y consejos (secciones 7 y 8). Con `rival_tipo=cajon` solo vale para premium y cajones propios.

**Arquetipos:**
- `GET /arquetipos`: free ve los 3 base; premium ve también las variantes.
- `POST /arquetipos/custom` (solo premium): crea una copia editable de un arquetipo en una ranura 2 a 4.

**Admin** (rol admin), detallado en la sección 11:
- `GET /admin/resumen`, `/admin/capacidad`, `/admin/rendimiento`, `/admin/calidad`, `/admin/ingresos`
- `GET /admin/denuncias`, `POST /admin/tandas`, `POST /admin/denuncias/{id}/accion`
- `GET /admin/notificaciones`, `POST /admin/notificaciones/{id}/leida`

**Salud:**
- `GET /salud`: commit, arranque, estado de la cola y del worker.

Cada endpoint tiene un test de API: el camino feliz y el de permiso denegado cuando corresponde.

---

## 6. Procesamiento y métricas

### 6.1 Pipeline por modalidad

El worker procesa los trabajos de a uno en `twinboxing_env` e invoca el motor **igual que lo hace hoy el worker de A**. Copiá la invocación exacta, las variables y las rutas desde `boxtwin-api`; no las reinventes.

| Modalidad | Etapas | Siembra | Clasificador |
|---|---|---|---|
| Sombra | Ruta de sesión individual del MVP (W19, un peleador) | No | **Apagado** (sin tipo) |
| Bolsa | Ídem | No | **Apagado** |
| Sparring | `procesar`, luego `esperando_siembra`, luego `completar --semilla-a --semilla-b` | Sí, la marca el usuario | **Apagado** |

- **Sin clasificador:** se procesa sin la etapa de clasificación (`BOXTWIN_CMD_CLASIFICADOR` vacío, o la bandera equivalente que exista). Si el motor no permite saltearla, se corre y se ignora el tipo. Registralo como desvío.
- **`engine_adapter.py`:** lee la Fight-Card (contrato 0.3) y produce `golpes` con `mano` = brazo anatómico del detector, `quien` según la identidad y `t_inicio`/`t_fin`/`confianza`. Los golpes en cuadros sin identidad resuelta **no se guardan**; se cuentan aparte para el panel admin.
- **Calidad de pose:** `calidad_pose` = fracción de cuadros de la sesión con pose del boxeador del perfil. Si el contrato no lo trae, calculalo desde los artefactos intermedios que deje `procesar`.
- **Tiempo de GPU:** el worker registra `inicio_gpu` y `fin_gpu` por trabajo, y los suma en `uso_gpu_diario` (sección 10).
- **Rótulo en bolsa y sombra:** los resultados de estas modalidades llevan en la interfaz "Reconocimiento en evaluación".

### 6.2 Métricas por sesión (`services/metrics.py`, puras y con tests)

Sea `G` la lista de golpes de un boxeador en la sesión, ordenada por `t_inicio`, y `D` la duración analizada en minutos.

- **Segmentos:** si hay `rounds_declarados`, se reparte la duración en rounds iguales. Si no, segmentos de 180 s; el último, si dura menos de 60 s, se suma al anterior (misma regla que A).
- **Volumen:** `len(G) / D`, en golpes detectados por minuto.
- **Balance:** `pct_izquierda = izq / len(G)`. Además, `pct_mano_adelantada` según la guardia del perfil 1: la mano adelantada es la izquierda en ortodoxa y la derecha en zurda. En los arquetipos y cajones se supone ortodoxa, con el aviso "guardia supuesta".
- **Combinaciones:** recorrer `G` y agrupar golpes consecutivos con `t_inicio[i+1] - t_inicio[i] < 0,5 s`. Cada grupo de 2 o más golpes es **una** combinación, aunque la serie sea larga. Se reporta `combinaciones_min` y `pct_golpes_en_combinacion`.
- **Sostén del ritmo, por segmento:** `volumen(resto del segmento) / volumen(primer minuto del segmento)`. Si el segmento dura menos de 90 s, no se calcula. Se reporta el promedio de los segmentos.
- **Iniciativa (solo sparring):** `len(G_usuario) / (len(G_usuario) + len(G_rival))`.
- **Curva:** golpes por bloque de 10 s, para las curvas superpuestas.
- **Exposición por lado (solo sparring):** cuenta golpes **dirigidos** del rival. Enfrentados, la mano izquierda del rival llega por el lado **derecho** del usuario y la derecha del rival por el lado izquierdo. Se guardan los conteos por mano del rival y por lado del usuario.
- **Capa cabeza/cuerpo (opcional):** si el motor expone la altura por la regla C8 sin el clasificador, se guarda en `altura_estimada` y la interfaz la muestra como "estimación". Si no la expone, no se implementa. No se agrega código al motor para esto.

Ninguna métrica dice "recibió" ni "conectó".

### 6.3 Stat general por perfil y modalidad

Promedio y tendencia de las últimas N sesiones (N = 10 **[PROVISIONAL]**). La tendencia es la pendiente de una regresión lineal simple sobre el índice de sesión y se muestra como "subiendo", "estable" o "bajando", con una banda de ±5% **[PROVISIONAL]**. La comparación entre rounds lleva el rótulo "sin validar".

### 6.4 Apta para ranking

Una sesión es apta para el ranking si cumple todo esto:
- es del perfil 1;
- el perfil es público;
- dura 180 s o más;
- `calidad_pose` ≥ 0,70 **[PROVISIONAL]**;
- la cuenta no está suspendida y el perfil no fue sacado del ranking.

---

## 7. Perfiles, arquetipos y cajones

- **Al registrarse se crean 4 ranuras:**
  - ranura 1: `estilo=propio`;
  - ranuras 2, 3 y 4: vinculadas a los arquetipos base `fajador`, `estilista` e `hibrido`, con `customizado=false`.
- **Free:** las ranuras 2 a 4 son de solo lectura. Se pueden usar como rival fantasma, pero no reciben sesiones. En sparring, el rival se analiza en esa sesión (métricas, exposición, iniciativa) y queda en la tabla `golpes` con `quien=rival`, pero no se asocia a ningún perfil.
- **Premium:** una ranura 2 a 4 se puede pasar a `customizado=true`, con nombre y estilo elegidos por el usuario. A partir de ahí recibe sesiones de sparring: en la siembra, el rival va al cajón elegido. Su stat general sale de esas sesiones, vistas desde el rival.
- **Arquetipos (`seeds/arquetipos.yaml`):** nombres ficticios y el rótulo "Perfil simulado" siempre visible. Sus métricas se generan con valores **[PROVISIONAL]** que Lucas valida:

| Clave | Estilo | Volumen (g/min) | % mano adelantada | Comb./min | Sostén | Exposición (izq/der del usuario) |
|---|---|---|---|---|---|---|
| fajador | Fajador | 42 | 55% | 9 | 0,85 | 40/60 |
| estilista | Estilista | 26 | 72% | 5 | 0,95 | 55/45 |
| hibrido | Híbrido | 34 | 63% | 7 | 0,90 | 50/50 |
| fajador_volumen (premium) | Fajador | 50 | 52% | 12 | 0,80 | 35/65 |
| estilista_contra (premium) | Estilista | 22 | 75% | 4 | 0,97 | 60/40 |
| hibrido_presion (premium) | Híbrido | 38 | 60% | 8 | 0,88 | 45/55 |

Cada arquetipo trae medias y desvíos por modalidad (sombra y bolsa con un 10% más de volumen que sparring **[PROVISIONAL]**) y una curva de ritmo sintética coherente con su sostén. Los nombres son inventados, sin parecido con boxeadores reales; por ejemplo "El Toro Simulado" o "Zurda de Laboratorio".

---

## 8. Rival fantasma y consejos (`services/rules.py`)

La respuesta de `GET /rival` trae:
- las métricas lado a lado de una modalidad;
- las curvas de ritmo superpuestas;
- el mapa de exposición por lado: de los sparrings del perfil 1 contra ese estilo o rival, o simulado si el rival es un arquetipo (con rótulo);
- la capa cabeza/cuerpo, si existe;
- de 1 a 3 consejos.

**Reglas.** Cada regla tiene un disparador, una prioridad y una lista de frases; la app rota las frases con un índice determinístico por sesión: `hash(sesion_id) % len(frases)`. Los umbrales son **[PROVISIONAL]** y van a configuración.

| Clave | Disparador | Frases (las aprobadas por Lucas van primero; el resto, pendientes de validación) |
|---|---|---|
| `balance_manos` | `pct_mano_adelantada > 0,70` y al menos 30 golpes | "Estás trabajando mucho más la izquierda que la derecha, tratemos de alternar las manos." (en zurdo, invertir izquierda y derecha) |
| `caida_ritmo` | `sosten_ritmo < 0,80` | "Se detectó una caída del volumen del golpeo después del primer minuto, tratemos de mantener el ritmo constante todo el round." |
| `exposicion_lado` | En sparrings contra un estilo, al menos 20 golpes del rival y uno de los lados ≥ 65% | "Contra los {estilo_plural} te entran por la {lado}. Cerrá ese lado y salí por el costado." |
| `golpes_sueltos` | `pct_golpes_en_combinacion < 0,30` y al menos 30 golpes | "Tirás de a uno. Soltá las manos de a dos o tres y salí." |
| `volumen_bajo_vs_rival` (solo rival fantasma) | volumen del usuario < 0,8 × volumen del rival | PENDIENTE: frase a escribir por Lucas. Mientras tanto, mostrar "Tu volumen está por debajo del de tu rival en esta modalidad." |

- **Plantillas:** `{estilo_plural}` = fajadores, estilistas o híbridos; `{lado}` = derecha o izquierda.
- **Sin disparos:** si no se dispara ninguna regla, la frase es "Sin alertas en esta sesión. Seguí así."
- **Pie de los consejos:** "Los consejos salen de los golpes detectados; no son una predicción."

---

## 9. Planes y cupos

| | Free | Premium (5 USD/mes, simulado) |
|---|---|---|
| Publicidad | Espacio reservado en la lista de sesiones y bajo los resultados | No |
| Minutos de video por mes | 60 **[PROVISIONAL]** | 300 **[PROVISIONAL]** |
| Ranuras 2 a 4 | Arquetipos fijos | Customizables como cajones |
| Arquetipos | 3 base | Base, variantes y custom |
| Rival de sparring | Se analiza, no se guarda en un perfil | Se guarda en el cajón elegido |
| Ranking, amigos, rival fantasma, valoración | Sí | Sí |

- **Cupo:** se descuenta la duración del video al encolar. Si el trabajo falla por un error del sistema, se devuelve.
- **Publicidad:** es un componente `<EspacioPublicitario ubicacion="…"/>` con un recuadro rotulado "Publicidad" y una impresión registrada en `eventos_publicidad`. Sin red de anuncios real.

---

## 10. Capacidad y disparadores

- **Utilización diaria:** `utilizacion = segundos_ocupados_gpu / 86.400`. La calcula un job a las 00:05, con un timer de systemd o un loop en el worker.
- **Disparadores:** se evalúan después de cada cálculo y generan una notificación de admin, una vez por episodio:
  - `capacidad_70_15`: utilización > 0,70 en cada uno de los últimos 15 días;
  - `capacidad_90_5`: utilización > 0,90 en cada uno de los últimos 5 días.
- **Al cruzar un disparador:** el texto de la notificación dice "Evaluar pase a servidor externo". El pase en sí no se automatiza.
- **El panel muestra además:** cola actual, espera p50/p95 de los últimos 7 días y la proyección de horas de video por mes (`horas_video_procesadas / horas_gpu_ocupadas`, extrapolado al 70% y al 100%). Esto alimenta el Cap 5.
- **Costo por modalidad:** se registra el cociente tiempo de GPU / duración del video por sesión y por modalidad. Es la medición de costo que necesita el Cap 5 y que hoy no existe para bolsa y sombra.

---

## 11. Panel de administración (`/admin`, solo rol admin)

| Bloque | Contenido |
|---|---|
| Uso | Usuarios free y premium, altas por día, sesiones por modalidad, minutos consumidos por plan |
| Capacidad | Gráfico de utilización diaria (30 días) con las líneas 70% y 90%, cola, espera p95, estado de los disparadores |
| Rendimiento | Tiempo de GPU / duración del video por modalidad (mediana y p95), fallos por etapa, sesiones de sparring con golpes sin identidad (porcentaje) |
| Calidad del reconocimiento | Valoración promedio por modalidad y por semana, distribución de estrellas, correcciones por tipo, cantidad de sesiones con consentimiento de reentrenamiento |
| Moderación | Cola de denuncias abiertas, botón "Empezar tanda" (agrupa las abiertas), acción por denuncia (desestimar, sacar del ranking, suspender), historial de tandas |
| Ingresos | Usuarios premium (simulado), impresiones de publicidad por día |
| Notificaciones | Campana con las no leídas: denuncia nueva, capacidad, fallo de trabajo |

- **Exportación:** CSV de cada bloque, para volcar al Cap 5 y al Cap 7.

---

## 12. Interfaz (rutas de `web/`)

Mobile first, instalable como PWA (manifest, íconos y un service worker que cachea el shell sin cachear video). La estética es la de la web entrenador.

| Ruta | Pantalla |
|---|---|
| `/` | Bienvenida, ingresar o registrarse |
| `/registro` | Formulario con los dos checks (sección 3) |
| `/inicio` | Las 4 ranuras como tarjetas; botón grande "Subir sesión"; últimas sesiones; espacio publicitario en free |
| `/subir` | 1) modalidad, 2) perfil (en free, sparring solo con el perfil 1), 3) guía de filmación (cámara fija, cuerpo entero en cuadro, buena luz, guantes de distinto color en sparring), 4) video y rounds opcional, 5) progreso de subida y cupo restante |
| `/sesiones/[id]/siembra` | Sparring: elegir el recorte del usuario y el del rival; en premium, elegir el cajón del rival |
| `/sesiones/[id]` | Resultado: video con línea de tiempo de golpes coloreados por mano; métricas; curva; exposición (sparring); consejos; rótulo "Reconocimiento en evaluación" en bolsa y sombra; valoración obligatoria la primera vez; corrección golpe a golpe ("no fue un golpe", "faltó uno acá", "era la otra mano") |
| `/perfiles/[id]` | Pestañas Sombra, Bolsa y Sparring: stat general, tendencia, historial, "Comparar dos sesiones" |
| `/comparar` | Dos sesiones lado a lado, curvas superpuestas |
| `/rival` | Elegir rival (arquetipo, público, amigo o cajón premium), modalidad; ver comparación, mapa y consejos |
| `/ranking` | Selector de modalidad y métrica, top 100 y la posición propia; pie "Ranking de actividad: cuenta golpes detectados, no golpes que conectaron" |
| `/amigos` | Solicitudes, lista y perfil del amigo |
| `/publicos/[id]` | Perfil público: stats, "Usar como rival fantasma", "Denunciar" |
| `/ajustes` | Plan, pasar a premium (simulado), guardia, perfil público, consentimiento de reentrenamiento, baja |
| `/admin/*` | Sección 11 |

**Textos obligatorios en la interfaz:**
- "golpes detectados" en lugar de "golpes lanzados";
- "Perfil simulado" en los arquetipos;
- "Pago simulado — versión de demostración";
- "estimación" en la capa cabeza/cuerpo;
- "Comparación entre rounds sin validar".

---

## 13. Tareas (registro en `TAREAS.md`)

| Día | Tarea | Hecho cuando |
|---|---|---|
| 1 | Z1 Esqueleto: carpetas, `boxtwinz.toml`, FastAPI, Alembic, Next.js con tokens copiados, CI local (pytest, tsc, lint) | `GET /api/salud` responde y la web exporta |
| 1 | Z2 Cuentas: registro con checks, ingreso, roles, plan, baja | Tests de auth en verde |
| 1 | Z3 Perfiles y arquetipos: 4 ranuras al registrarse, seed de arquetipos, reglas free/premium | Tests de permisos de ranuras |
| 1 | Z4 Worker y adaptador: cola, lock de GPU, invocación del motor copiada de A, `engine_adapter`, registro de tiempo de GPU | Una sesión de sombra y una de sparring reales de punta a punta en la estación |
| 2 | Z5 Subida, siembra y cupos | e2e: subir sparring, sembrar, ver resultado |
| 2 | Z6 Métricas (sección 6) | Tests unitarios con casos borde: combinaciones encadenadas, segmento corto, zurdo |
| 2 | Z7 Pantalla de resultado, valoración y correcciones | e2e con valoración obligatoria |
| 2 | Z8 Historial, stat general y comparación | e2e de comparación |
| 3 | Z9 Ranking y perfiles públicos | Tests de aptitud (sección 6.4) |
| 3 | Z10 Amigos | Tests: un amigo ve el perfil 1 y nunca los cajones |
| 3 | Z11 Rival fantasma y reglas | Tests de cada disparador y de la rotación de frases |
| 3 | Z12 Denuncias | Tests de denuncia y acciones |
| 4 | Z13 Panel admin y disparadores de capacidad | Test de los disparadores con datos sintéticos de 15 y 5 días |
| 4 | Z14 Planes: premium simulado, espacios publicitarios, cupos | Tests de 402 y 403 |
| 4 | Z15 PWA: manifest, service worker, Lighthouse PWA | Instalable en Android Chrome |
| 4 | Z16 Accesibilidad y e2e: Playwright + axe en Chromium y WebKit | e2e en verde |
| 4 | Z17 Despliegue en la estación: servicios systemd, túnel, `despliegue/README.md` con vuelta atrás | Accesible desde un celular por el túnel; A sigue respondiendo igual |
| 4 | Z18 Datos de demo: 1 cuenta free y 1 premium, con sesiones de las tres modalidades | Recorrido completo de demo en menos de 5 minutos |

**Orden de recorte si no llega:** lo que no esté al cierre del día 4 queda como diseño rotulado en la interfaz ("Próximamente"), en este orden:
1. Corrección golpe a golpe
2. Capa cabeza/cuerpo
3. Arquetipos custom
4. Exportación CSV del admin
5. WebKit en e2e

El circuito subir → procesar → resultado → valoración no se recorta.

---

## 14. Criterios de aceptación de la construcción

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CZ1 | Punta a punta en las tres modalidades sin intervención más allá de la siembra en sparring | Una sesión real por modalidad en la estación |
| CZ2 | A no cambia | Hash de `/etc/boxtwin.env` y de los servicios de A igual antes y después; `/salud` de A con el mismo commit |
| CZ3 | Reglas de plan respetadas | Tests de 402 y 403 |
| CZ4 | Amigos y públicos nunca exponen los cajones 2 a 4 | Tests |
| CZ5 | Métricas correctas | Tests unitarios contra casos calculados a mano |
| CZ6 | Disparadores de capacidad correctos | Tests con series sintéticas |
| CZ7 | Textos obligatorios presentes | Test e2e que busca cada rótulo |
| CZ8 | PWA instalable y usable a 375 px de ancho | Lighthouse y captura |

**Mediciones para la tesis, a registrar con BoxTwinZ andando (no son criterios de aceptación):**
- B8, bolsa y sombra: valoración promedio y, si Lucas anota, recall contra conteo manual.
- Costo de GPU por modalidad.
- Proyección de capacidad de la estación.

Cada una va en un documento de `docs/experiments/` con fecha.

---

## 15. Pendientes de Lucas (no bloquean la construcción)

- Validar los valores **[PROVISIONAL]**: cupos, umbrales de reglas, arquetipos, N de tendencia y calidad de pose.
- Escribir o validar el resto del catálogo de frases.
- Nombres ficticios definitivos de los arquetipos.
- Videos propios para los datos de demo: sombra, bolsa y sparring.
