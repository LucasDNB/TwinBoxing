# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). `S` = la corre el SDD en la estación sin `sudo`; `L` = Lucas.

## Código

| Id | Qué | Dónde | Prueba que lo cierra |
|---|---|---|---|
| T2 | `cortar_rounds.py`: inicios de round a segundos, validación de cortes (orden, solapamiento, fin del video, nocaut), corte con reencode por round y unión sin reencode, JSON con sha256 del original, reanudable, `--ensayo` | `boxtwin-api/scripts/` | `boxtwin-api/tests/test_script_cortar_rounds.py`, 17 pruebas, una de punta a punta con un video sintético de ffmpeg |
| T1 | `cargar_tercero.py`: valida, crea peleadores con seudónimo, guardia y nota, crea la sesión con el origen en `consentimiento_por`, enlaza o copia el video, encola `procesar`; reanudable; `--ensayo` | `boxtwin-api/scripts/` | `boxtwin-api/tests/test_script_cargar_tercero.py`, 11 pruebas: defaults, cinco rechazos de `validar`, carga completa, la API la lista y la sirve, reanudable sin duplicar, ensayo sin escritura, usuario inexistente |

## Tareas

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S0 | SDD | T1 y esta spec, PR contra `main` | — | suite de `boxtwin-api` en verde | hecha | 2-11: 11 pruebas nuevas; la suite de `boxtwin-api` entera en verde |
| L0 | Lucas | Mergear el PR. Elegir otros seudónimos si `Tercero Z` y `Tercero O` no le sirven | S0 | en `main` | hecha | 2-11 22:24 UTC, PR #45. Seudónimos sin cambios |
| L0b | Lucas | Pasar el video a la estación y marcar el inicio de los doce rounds (mm:ss) | — | doce tiempos anotados acá | hecha | 2-11: `~/Proyectos/twinboxing-finales/pacman3x1.mp4`, ya recortado por Lucas: el round 1 empieza en 0:00 y cada uno 4:00 después (0:00, 4:00, …, 44:00), descansos de 60 s exactos |
| S0b | SDD | `cortar_rounds.py --ensayo` y el corte a `pacman-netos.mp4`; ffprobe del resultado (12 × 180 s) | L0b | `pacman-netos.cortes.json` | no realizada | 2-11: Lucas cortó él mismo los rounds netos en `pac-neto.mp4` (1920×1080, 30 fps), sin `cortar_rounds.py` |
| S1 | SDD | En la estación: `ffprobe` de `pacman-netos.mp4` (fps, duración), respaldo de la base de A (`boxtwin.db.antes-del-tercero.bak`), `GET /salud` y conteos de sesiones y peleadores de A | L0 | commit `1058d5e`; respaldo existe; decisión del descanso anotada acá | hecha | 2-11 ~22:00 (hora local), con `scripts/gemelo-tercero/cargar.sh`: `pac-neto.mp4` dura 2134 s, no 2160 (decisión 6). `/salud`: commit `1058d5e`, clasificador `geometrico`, arrancado 2-11 15:00 UTC. Antes: 19 sesiones, 12 peleadores, 84 trabajos, 0 pendientes. Respaldo `boxtwin.db.antes-del-tercero.bak`. Todo en `~/Proyectos/twinboxing-finales/gemelo/` |
| L1 | Lucas | OK para correr `procesar` en A: pose y seguimiento del combate entero, unas 2 h de GPU (pose sola, 1 h 45 a 29 fps según E8), sin otra sesión en la cola | S1 | fecha anotada | hecha | 2-11 19:31 (hora local), "Ok" de Lucas |
| S2 | SDD | `cargar_tercero.py --video pacman-netos.mp4 --descanso-s 0 --ensayo` y, si da lo esperado, la carga | L1 | JSON de salida anotado; trabajo `procesar` en cola | hecha | 2-11, `cargar_tercero.py` de `main` en `5926972`, con `FORZAR=1`: sesión `3b82b7f828654546b5a2a21855874fab` en cola, cuenta `1556d764…` (usuario `lucas@lucas.com`); `Tercero Z` zurda `4b42ce83…`, `Tercero O` ortodoxa `5acf4b4e…`; video por enlace duro. Ensayo y carga en `gemelo/ensayo.json` y `gemelo/carga.json` |
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
4. 2-11 (de Lucas): el video trae los descansos. No se carga con descanso 60: se cortan los
   rounds netos con `cortar_rounds.py` y se carga con descanso 0 (§2.1). Contar desde el segundo
   0 con descansos fijos corre los rounds y suma los golpes de las repeticiones.
5. 2-11: con el video alineado, 180/60 en A daría las ventanas bien, pero el total del peleador
   (`fightcard._conteo(suyos)`, que alimenta el volumen del perfil) suma todos los golpes, los
   del descanso incluidos; solo `por_round` filtra por ventana. Se mantiene el corte a rounds
   netos (decisión 4), con los inicios 0:00, 4:00, …, 44:00 sin marcar a mano.
6. 2-11 (de Lucas): se carga `pac-neto.mp4` como está, con `FORZAR=1`, aunque dura 2134 s
   contra los 2160 de 12 × 180. Lo más probable es un corte sin reencode (cada round empieza en
   el keyframe más cercano, unos 2,2 s menos por round). El total de golpes no cambia, pero
   las ventanas fijas de 180 s se corren: la del round 12 arranca unos 24 s antes de la campana
   real y queda de 154 s. Lo que va a la tesis o a la presentación por round se declara con ese
   desfase; el volumen total y el perfil no se ven afectados. Si hace falta por round exacto, se
   recorta desde `pacman3x1.mp4` con `cortar_rounds.py` y se carga una sesión nueva.
