# BoxTwinZ — Registro de tareas

Spec: `SPEC.md` (3-11). Código en `boxtwinz/`. Rama `boxtwinz`. CI local: `bash boxtwinz/ci.sh`
(pytest de la API, tsc, lint, vitest y build exportado de la web).

Una tarea no está hecha sin evidencia. "Hecha (código)" quiere decir que el código y sus tests
están, pero falta la evidencia que pide la columna "Hecho cuando" de la spec.

## Estado

| Tarea | Estado | Commit | Evidencia |
|---|---|---|---|
| Z1 Esqueleto | Hecha | día 1 | `test_z1_salud.py` (4); `ci.sh` en verde; humo: `GET /api/salud` 200 y `/`, `/registro/`, `/sw.js` 200 servidos por la API en 8003 desde `web/out` |
| Z2 Cuentas | Hecha | día 1 | `test_z2_cuentas.py` (12): checks, cookie httpOnly/Lax/30 días, admin por lista, premium simulado, baja con anonimización, suspendido |
| Z3 Perfiles y arquetipos | Hecha | día 1 | `test_z3_perfiles.py` (10): 4 ranuras, 403 `plan_requerido` en free, premium edita, custom solo premium, seed con los valores de la spec |
| Z4 Worker y adaptador | Hecha | día 1 | `test_z4_worker.py` (13) con motor falso, y corrida real del 3-11 en la estación (ver "Corrida real de Z4") |
| Z5 Subida, siembra y cupos | Hecha | día 2 | API: `test_z5_circuito.py` (7). Web: `/subir` (5 pasos, progreso por XHR, cupo restante) y `/siembra`. e2e `circuito.spec.ts` › sparring: subir, sembrar, resultado |
| Z6 Métricas | Hecha | día 1 | `test_z6_metricas.py` (14): combinaciones encadenadas, segmento corto, resto < 60 s, zurdo, sostén a mano, exposición cruzada, tendencia |
| Z7 Resultado, valoración y correcciones | Hecha | día 2 | `/sesion`: video con línea de tiempo por mano, métricas, rounds rotulados sin validar, curva, exposición, altura como estimación, consejos. Salir sin valorar abre el diálogo y no navega. e2e `circuito.spec.ts` (2) |
| Z8 Historial, stat general y comparación | Hecha | día 2 | `/perfil` con pestañas, stat, tendencia, historial y cajón premium; `/comparar` con curvas superpuestas. e2e `comparar.spec.ts` (2) |
| Z9 Ranking y perfiles públicos | Hecha | día 3 | `test_z9_ranking.py` (7): aptitud en los bordes exactos (180 s, 0,70), umbral desde la config, orden y posición propia, volverse privado saca al momento, un cajón nunca es público. Web `/ranking` y `/publico`; e2e `social.spec.ts` |
| Z10 Amigos | Hecha | día 3 | `test_z10_amigos.py` (5): el amigo ve el perfil 1 y sus sesiones, y ni el id ni el nombre de un cajón aparecen en la respuesta (CZ4); pendiente no alcanza; la solicitud no revela emails. Web `/amigos`; e2e |
| Z11 Rival fantasma y reglas | Hecha | día 3 | `test_z11_reglas.py` (13): cada disparador en su borde, zurdo, máximo 3 por prioridad, rotación crc32, umbral desde la config; rival contra arquetipo (simulado no dispara exposición), variante premium con exposición real contra cajones del estilo, cajón y permisos. Web `/rival`; e2e |
| Z12 Denuncias | Hecha | día 3 | `test_z12_denuncias.py` (3): denuncia a perfil visible con notificación al admin, no se denuncia lo que no se ve, `otro` pide detalle. Diálogo en `/publico`; e2e |
| Z13 Panel admin y capacidad | Parcial | día 1 | `services/capacity.py` (uso diario partido por medianoche, rachas, una notificación por episodio). Faltan rutas, panel y tests de 15 y 5 días |
| Z14 Planes | Parcial | día 1 | Premium simulado, cupos (402) y `POST /publicidad/impresion` con test. Falta el componente en resultados |
| Z15 PWA | Parcial | día 1 | `manifest.webmanifest`, íconos 192/512, `sw.js` que no cachea `/api/` ni rangos. Falta Lighthouse y la prueba en Android |
| Z16 a Z18 | Pendiente | | |

## Desvíos (manda el repo)

- **D1 Sin dependencias nuevas.** La instalación de un venv con argon2-cffi, PyJWT y Alembic
  quedó sin aprobar. Todo corre con lo que ya tiene `twinboxing_env` (FastAPI 0.141, SQLAlchemy
  2.0, Pydantic 2.13, PyYAML), sin tocar ese entorno:
  - contraseña con **scrypt** (`hashlib`), como `boxtwin-api`, no argon2id. El hash lleva el
    algoritmo en el prefijo: pasar a argon2id es una rama en `verificar` y rehashear al ingresar;
  - **JWT HS256** estándar firmado con `hmac` de la biblioteca estándar, no PyJWT;
  - **sin Alembic**: `create_all` al arrancar, como A. Para agregar columnas a una base con
    datos hará falta copiar `_columnas_faltantes` de `boxtwin-api/db.py` o instalar Alembic.
  Si Lucas aprueba las tres dependencias, se cambian sin tocar las rutas.
- **D2 SQLite en la estación.** La estación no tiene PostgreSQL (A corre sobre SQLite). La cola
  tiene las dos ramas, como A: SKIP LOCKED en PostgreSQL y UPDATE condicional en SQLite. La base
  va en `~/boxtwinz-datos/boxtwinz.db`, separada de la de A.
- **D3 Sombra y bolsa se siembran solas.** El motor pide una semilla A también en las sesiones
  de una persona (`proponer_individual`; en A el usuario la marca). La spec dice que no hay
  siembra, así que el worker elige el candidato que el motor marca `sugerido` con más tiempo en
  cuadro (`engine_adapter.candidato_individual`). Si otra persona está más tiempo en cuadro que
  el boxeador, se equivoca: por eso estas modalidades llevan "Reconocimiento en evaluación".
- **D4 Sin clasificador.** No hizo falta saltear nada: en A la clasificación es una etapa
  aparte que encadena el worker; el de BoxTwinZ no la encola. El tipo que pueda traer la
  Fight-Card se ignora. La altura (cabeza/cuerpo) la agrega `completar` con la regla de C8 sin
  el clasificador, así que la capa opcional de 6.2 existe sin tocar el motor y se muestra como
  "estimación".
- **D5 Rutas con id como query.** Una exportación estática no genera páginas para ids que no
  existen al compilar: `/sesiones/[id]` es `/sesion?id=`, `/sesiones/[id]/siembra` es
  `/siembra?id=`, `/perfiles/[id]` es `/perfil?id=` y `/publicos/[id]` es `/publico?id=`. Es lo
  mismo que hace la web entrenador.
- **D6 Golpes sin identidad.** El detector corre sobre los keypoints ya asignados a A o B, así
  que en la Fight-Card no existen golpes sin identidad que contar. Se guarda en
  `sesiones.fraccion_sin_identidad` la fracción de cuadros sin pose confiable del peleador:
  `1 − cobertura`, la peor de las dos en sparring. Primero se usó `identidad.sin_asignar` y la
  corrida real mostró que estaba mal: ese campo cuenta los cuadros con cualquier detección sin
  rol, terceros al fondo incluidos, y dio 0,70 en un sparring con cobertura de 0,999 y 0,946.
- **D7 Las sesiones se suben con el perfil 1.** En premium, un cajón recibe sparrings como
  rival, elegido al subir o en la siembra; no se sube una sesión "del cajón". En free, subir con
  un cajón responde 403 `plan_requerido`.
- **D8 Rotación de frases con crc32.** La spec dice `hash(sesion_id) % len(frases)`; el `hash()`
  de Python cambia entre procesos para un str, así que se usa `zlib.crc32`, que es estable.
- **D9 Calidad de pose = `cobertura_A`.** La Fight-Card trae la fracción de cuadros asignados
  al peleador A, que es el boxeador del perfil. Es eso lo que se usa como `calidad_pose`.
- **D10 Worktree.** Se trabaja en el worktree de esta sesión de Claude Code, con la rama
  renombrada a `boxtwinz`, no en `~/Proyectos/TwinBoxing-boxtwinz`.
- **D12 Exposición contra un estilo.** El mapa del rival fantasma sale de los sparrings del
  perfil contra cajones de ese estilo (o contra ese cajón). Contra un arquetipo sin sparrings de
  su estilo se muestra el mapa simulado del arquetipo con rótulo, y ese mapa no dispara
  `exposicion_lado`. Contra un público o un amigo no hay mapa: no hay sparrings enlazados.
- **D13 Mínimo de golpes en el rival fantasma.** Las reglas con "al menos 30 golpes" usan el
  promedio de golpes por sesión de la ventana, no la suma.
- **D11 Correcciones no cambian los números.** Se registran para reentrenar; las métricas
  siguen saliendo del detector, para no mezclar dos fuentes en un mismo número.

## Corrida real de Z4 (3-11)

API en 8003 y worker sobre una base de prueba en el scratchpad, con los modelos de A
(`yolov8l-pose.pt`, `guantes-v2.pt`, `detector_7fuentes.ens.pt`, umbral 0,80, apariencia
encendida). GPU libre al empezar y A sin trabajos. Dos clips de 90 s recortados a 1080p H.264
(la API acepta mp4/mov/m4v): sombra de `video/loma-sombra.webm` (59,9 fps) y sparring de
`videos/sparr.mp4` (4K bajado a 1080p, 59,9 fps), los dos desde el segundo 30.

1. Sombra, sin intervención: `procesar` 297,3 s, semilla automática track 49, `completar`
   2,2 s. 30 golpes detectados (7 izquierda), 20,0 por minuto, calidad de pose 0,993, rótulo
   "Reconocimiento en evaluación". Sin altura: el motor la deja en null sin rival (`sin_rival`).
2. Sparring: `procesar` 297,1 s hasta `esperando_siembra`. Candidatos sugeridos 141 (guantes
   rojos) y 143 (amarillos), comprobados mirando los recortes; sembrados como usuario y rival.
   `completar` 7,4 s. Usuario 19 golpes, rival 22, iniciativa 0,463; exposición 20 por el lado
   izquierdo del usuario y 2 por el derecho; altura estimada en 40 de 41 golpes.
3. Ninguna regla se disparó en el sparring: la mano adelantada del usuario da 0,737 (> 0,70)
   pero son 19 golpes, debajo del mínimo de 30.
4. Costo de GPU: 299,5 s y 304,5 s para 90 s de video, 3,33x y 3,38x tiempo real a 60 fps.
   Es un dato suelto, no la medición del Cap 5: A midió 1,1x a 2,6x sobre otro material.
5. Encontró el error de D6, corregido con un test.

## Riesgos

- **R1 GPU compartida con A.** El worker de BoxTwinZ toma `/tmp/boxtwin-gpu.lock` mientras
  corre el motor, pero el worker de A no conoce ese lock. Si A y BoxTwinZ procesan a la vez, los
  dos procesos de pose comparten los 8 GB de la 2080 y pueden caerse por memoria o tardar el
  doble. Mitigación posible sin tocar A: correr el worker de BoxTwinZ solo cuando la cola de A
  esté vacía (leyendo `/salud` de A). No está implementada.
- **R2 Semilla automática en bolsa y sombra (D3).** Sin medir: es la misma medición de B8.

## Propuestas al motor (no aplicadas)

- Ninguna por ahora.

## Pendiente de Lucas

- Aprobar o no las dependencias de D1.
- Lo de la sección 15 de la spec.
