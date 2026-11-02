# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). `S` = la corre el SDD en la estación sin `sudo`; `L` = Lucas.

## Código

| Id | Qué | Dónde | Prueba que lo cierra |
|---|---|---|---|
| T1 | `cargar_tercero.py`: valida, crea peleadores con seudónimo, guardia y nota, crea la sesión con el origen en `consentimiento_por`, enlaza o copia el video, encola `procesar`; reanudable; `--ensayo` | `boxtwin-api/scripts/` | `boxtwin-api/tests/test_script_cargar_tercero.py`, 11 pruebas: defaults, cinco rechazos de `validar`, carga completa, la API la lista y la sirve, reanudable sin duplicar, ensayo sin escritura, usuario inexistente |

## Tareas

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S0 | SDD | T1 y esta spec, PR contra `main` | — | suite de `boxtwin-api` en verde | hecha | 2-11: 11 pruebas nuevas; la suite de `boxtwin-api` entera en verde |
| L0 | Lucas | Mergear el PR. Elegir otros seudónimos si `Tercero Z` y `Tercero O` no le sirven | S0 | en `main` | pendiente | |
| S1 | SDD | En la estación: `ffprobe` del video de `anotacion-pacquiao` (fps, duración, ¿rounds netos?), respaldo de la base de A (`boxtwin.db.antes-del-tercero.bak`), `GET /salud` y conteos de sesiones y peleadores de A | L0 | commit `1058d5e`; respaldo existe; decisión del descanso anotada acá | pendiente | |
| L1 | Lucas | OK para correr `procesar` en A: pose y seguimiento del combate entero, unas 2 h de GPU (pose sola, 1 h 45 a 29 fps según E8), sin otra sesión en la cola | S1 | fecha anotada | pendiente | |
| S2 | SDD | `cargar_tercero.py --ensayo` y, si da lo esperado, la carga | L1 | JSON de salida anotado; trabajo `procesar` en cola | pendiente | |
| S3 | SDD | Seguir `procesar` hasta `espera_siembra` | S2 | estado de `GET /jobs/{id}` | pendiente | |
| L2 | Lucas | Siembra en la web: A = el zurdo, B = el ortodoxo; corregir a mano los tramos sin asignar que quiera | S3 | `completar` encolado | pendiente | |
| S4 | SDD | Seguir `completar` y `clasificar` hasta la Fight-Card | L2 | CT2 | pendiente | |
| S5 | SDD | Leer CT1 a CT6; perfil de `Tercero Z` y `GET /rival` contra el boxeador de Lucas; anotar acá qué dispara el Rival | S4 | CT1 a CT6 | pendiente | |
| L3 | Lucas | Capturas o fotos para la presentación y el párrafo del Cap 7 (§6) | S5 | | pendiente | |

## Decisiones

1. 2-11 (D1 a D3, de Lucas): Pacquiao–Margarito, en A, con seudónimos.
2. 2-11: la carga va por script de operador y no por `POST /videos` (§1). Ni marcar un
   consentimiento que nadie dio, ni cambiar A el día antes del congelamiento.
3. 2-11: el congelamiento del 3-11 es de funciones. Cargar y procesar una sesión es uso de A, no
   una función nueva; puede correr después del 3-11.
