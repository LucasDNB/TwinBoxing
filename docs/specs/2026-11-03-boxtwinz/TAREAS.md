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
| Z4 Worker y adaptador | Hecha (código) | día 1 | `test_z4_worker.py` (12) con motor falso: comando igual al de A, sombra sin siembra, sparring espera siembra, tiempo de GPU, lock, fallo con reintento y devolución de cupo, adaptador contra una Fight-Card real de la estación. **Falta:** una sesión real de sombra y una de sparring en la estación (ver "Pendiente de Lucas") |
| Z5 Subida, siembra y cupos | Hecha (API) | día 1 | `test_z5_circuito.py` (7) con un video real de 3 s (ffmpeg) y motor falso: subir sparring, candidatos, sembrar, resultado, valorar, corregir; 402 `cupo_agotado`; video por rangos. **Falta:** pantallas `/subir` y `/siembra` y el e2e de navegador |
| Z6 Métricas | Hecha | día 1 | `test_z6_metricas.py` (14): combinaciones encadenadas, segmento corto, resto < 60 s, zurdo, sostén a mano, exposición cruzada, tendencia |
| Z7 Resultado, valoración y correcciones | API hecha | día 1 | Rutas y tests en `test_z5_circuito.py`. Falta la pantalla |
| Z8 Historial, stat general y comparación | API hecha | día 1 | `GET /perfiles/{id}/stats`, `GET /comparar` (422 entre modalidades), cajón premium mirado desde el rival. Falta la pantalla y el e2e |
| Z9 a Z12 | Pendiente | | `rules.py` (consejos) ya escrito y usado en el resultado; faltan sus tests propios (Z11) |
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
  que en la Fight-Card no existen golpes sin identidad que contar. El panel usa
  `identidad.sin_asignar` (fracción de cuadros sin identidad) como proxy, guardado en
  `sesiones.fraccion_sin_identidad`.
- **D7 Las sesiones se suben con el perfil 1.** En premium, un cajón recibe sparrings como
  rival, elegido al subir o en la siembra; no se sube una sesión "del cajón". En free, subir con
  un cajón responde 403 `plan_requerido`.
- **D8 Rotación de frases con crc32.** La spec dice `hash(sesion_id) % len(frases)`; el `hash()`
  de Python cambia entre procesos para un str, así que se usa `zlib.crc32`, que es estable.
- **D9 Calidad de pose = `cobertura_A`.** La Fight-Card trae la fracción de cuadros asignados
  al peleador A, que es el boxeador del perfil. Es eso lo que se usa como `calidad_pose`.
- **D10 Worktree.** Se trabaja en el worktree de esta sesión de Claude Code, con la rama
  renombrada a `boxtwinz`, no en `~/Proyectos/TwinBoxing-boxtwinz`.
- **D11 Correcciones no cambian los números.** Se registran para reentrenar; las métricas
  siguen saliendo del detector, para no mezclar dos fuentes en un mismo número.

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
- Ok para correr Z4 sobre material real en la estación: una sombra y un sparring cortos (de 1
  a 3 minutos cada uno, son minutos de GPU con el worker de A en la misma placa).
- Lo de la sección 15 de la spec.
