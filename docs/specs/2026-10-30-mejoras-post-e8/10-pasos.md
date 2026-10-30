# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). `S` = la corre el SDD en la estación sin `sudo`; `L` = Lucas.
Ninguna tarea escribe en A, en B ni en C. Todo lo que corre en GPU pasa por
`scripts/mejoras/mejoras.sh` y su candado.

## Fase 0 · código, sin GPU (30 y 31-10)

Pruebas primero, en rojo; código; en verde. Rama `mejoras-fase0`, un PR contra `main`. Cada
cambio a código de producto entra con su bandera o parámetro **apagado por defecto**, y una
prueba que verifica que apagado da lo mismo que antes.

| Id | Qué | Dónde | Prueba que lo cierra |
|---|---|---|---|
| T1 | `features_de(..., umbrales=None)`: `None` o un número = comportamiento de hoy; un dict por articulación (nariz, hombro, codo, muñeca, cadera) aplica cada uno | `boxtwin-detector/src/boxtwin_detector/features.py` | con `None`, salida idéntica bit a bit a la de hoy en tres fixtures; con un dict, cada articulación se filtra con el suyo |
| T2 | `calibrar_umbrales.py`: E8b §2 sobre `raiz-v8` y `raiz-y26`, denominador común de `medir_trackeo.py`; escribe `umbrales.json` con cuantiles, fracciones y sha256 | `scripts/mejoras/` | distribuciones sintéticas con cuantil conocido; K-cal en el sintético |
| T3 | Los umbrales llegan al build del detector (`build --umbrales`) y a E4 (`BOXTWIN_UMBRALES` en `datos.py` y `features2d.py`, incluidas las 6 del rival) | `boxtwin-detector/cli.py`, `scripts/clasificador_geo/` | sin la opción, mismo `golpes.npz`; con la opción, el meta registra el sha256 de `umbrales.json` |
| T4 | `rellenar(kp, sc, umbrales, fps, max_s=0.10)` → kp, sc, `kp_rellenado` (E10 §2) | `boxtwin-detector/src/boxtwin_detector/relleno.py` | hueco de 3 a 30 fps se rellena, de 4 no; a 60 fps 6 sí y 7 no; huecos abiertos no; coordenadas medidas intactas; hueco con la detección ausente en un cuadro no se rellena |
| T5 | `build --relleno` y `BOXTWIN_RELLENO=1` en E4; el dataset suma `kp_rellenado` para las métricas por categoría | ídem | sin la opción, tensores idénticos |
| T6 | `segmentar` (E11 §2), `ConfigIdentidadAuto.cortar_mixtos=False`, `Propuesta` por segmento, `AsignarIdentidadAuto` con rangos | `boxtwin-annotator/src/boxtwin/core/identidad_auto.py` | track sintético con el color invertido a mitad: un corte en el cuadro de IoU máximo; separación de semillas < 0,55: ningún corte; menos de 5 muestras por lado: ningún corte; dos cortes como máximo; con `False`, propuestas idénticas a las de hoy sobre una evidencia guardada de prueba |
| T7 | `medir_e11.py`: desarrollo (evidencia de C1, semillas con la regla de C1) y confirmación (anotaciones nuevas); precisión y cobertura por cuadro, C1 por track, P15d | `scripts/mejoras/` | puntaje por cuadro sobre un `annot.json` sintético con un track mixto |
| T8 | `guantes_por_cuadro.py` (E12 §2): extracción por cuadro, asignación de mano, `.npz` por fuente, reanudable, tiempo por cuadro | `scripts/mejoras/` | asignación de mano en casos sintéticos (dos guantes, uno, ninguno, dos a la misma mano) |
| T9 | Guante como keypoints 17 y 18, `PARES_LR` con (17, 18), cinco features, máscara de E12 §3; `build --guantes <dir>`; la TCN toma la dimensión de entrada de los datos | `boxtwin-detector` | sin `--guantes`, 20 features y tensores idénticos; con guante y muñeca baja, cuadro usable con features de muñeca en 0; el espejado lleva el guante izquierdo al derecho |
| T10 | `mejoras.sh`: etapas S3 a S15 con candado (`scripts/gemelo/guardas.py`), `.hecho` por etapa, `MEJORAS_ENSAYO=1` | `scripts/mejoras/` | ensayo imprime los comandos de cada etapa y no corre nada |
| T11 | `lectura.py`: aplica las reglas de E8b, E10, E11 y E12 y escribe `veredictos.json` por experimento | `scripts/mejoras/` | casos borde: +0,030 exacto con 5 de 6 es mejora, +0,029 no; desvío 0,031 es no concluyente |
| T12 | `kreuso.py`: sha256 de los tensores nuevos con la mejora apagada contra los de `~/boxtwin-gemelo/e8/` | `scripts/mejoras/` | dos directorios iguales y uno con un byte cambiado |

Las tareas de código de la fase 2 (E9, E13, E14) se escriben en su propio PR cuando arranque la
fase, con este mismo formato.

## Tareas

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S1 | SDD | T1 a T12, PR `mejoras-fase0` | — | pruebas nuevas en rojo y después en verde; suites completas en verde (las de S1 de E8) | pendiente | |
| L0 | Lucas | Mergear el PR de este spec y el de S1 | S1 | los dos en `main` | pendiente | |
| S2 | SDD | Worktree fijo `~/Proyectos/TwinBoxing-mejoras` desacoplado en la punta de `main` después de L0; huellas de A y de B; sha256 de los originales de A y de los cachés de `raiz-v8` y `raiz-y26` en `~/boxtwin-mejoras/antes.txt` | L0 | hash anotado; los archivos existen; C `inactive` | pendiente | |
| L1 | Lucas | OK para las noches del 31-10 al 3-11 | S2 | fechas anotadas acá | pendiente | |
| S3 | SDD | K-reuso de E8b, E10 y E12 (tensores con las mejoras apagadas contra E8) | L1 | `kreuso.py` sin diferencias | pendiente | |
| S4 | SDD | E8b: `calibrar_umbrales.py`; K-cal; build de Y26-cal; K-mec | S3 | `umbrales.json`; K-cal a ±0,005 | pendiente | |
| S5 | SDD | E8b: `folds` y E4 de Y26-cal (noche del 31-10) | S4 | 35 corridas de cada uno | pendiente | |
| S6 | SDD | E8b: lectura, auditoría, documento, JSON, README, cambios de tesis | S5 | `veredictos.json`; auditoría | pendiente | |
| S7 | SDD | E10: P14d, histograma de huecos por condición, sin etiquetas (de día) | S3 | JSON de huecos | pendiente | |
| S8 | SDD | E10: build y corridas de v8+R (noche del 1-11); S8b Y26-cal+R si S5 terminó; S8c variante 0,20 s si sobra noche | S3, S7 | 35 corridas de cada uno | pendiente | |
| S9 | SDD | E10: lectura, auditoría, documento | S8 | | pendiente | |
| S10 | SDD | E11 desarrollo: `medir_e11.py` sobre la evidencia de C1, con y sin corte; auditoría ítem 2 (sin corte reproduce 85/105) | S2 | JSON de desarrollo | pendiente | |
| L3 | Lucas | Elegir dos sparrings de confirmación para E11 (E11 §4) y anotar su identidad en el anotador con la propuesta automática apagada; hasta el 7-11 | — | los dos `annot.json` con fecha y la nota de cómo se anotó | pendiente | |
| S11 | SDD | E11 confirmación sobre L3: preproceso de pose con los parámetros de A y `analizar` de guantes (GPU, poca, bajo el candado), después `medir_e11.py` | S10, L3 | JSON de confirmación | bloqueada | por L3 |
| S12 | SDD | E11: lectura, auditoría, documento (desarrollo y confirmación separados) | S11, o S10 si L3 no llega el 7-11 | | pendiente | |
| S13 | SDD | E12: `guantes_por_cuadro.py` sobre las siete fuentes (noche del 2-11), con tiempo por cuadro (P16d) | S3 | siete `.npz`; P16c y P16d | pendiente | |
| L2 | Lucas | E12 K-mano: marcar 200 cuadros (la lista la arma S13 con semilla 0) | S13 | ≥ 90% | pendiente | |
| S14 | SDD | E12: build de v8+G y corridas (noche del 3-11); P16e solo si sobra | L2 | 35 corridas | pendiente | |
| S15 | SDD | E12: lectura, auditoría, documento | S14 | | pendiente | |
| S16 | SDD | Cierre: huellas y sha256 después (CM2, CM3), CM1 a CM6, PR de resultados con todos los documentos y cambios de tesis | S6, S9, S12, S15 | iguales a S2 | pendiente | |
| L4 | Lucas | Revisar los `docs/tesis/*-cambios.md` y aplicarlos al `.tex` | S16 | | pendiente | |

## Decisiones

Una línea por decisión, con el motivo, antes de correr la parte afectada.

1. 30-10: "todas esas mejoras" se lee como las ocho, ordenadas por fase según el calendario.
   Lo que no cierra el 9-11 entra al Cap 8 con su pre-registro, sin número; la fase 2 no se
   adelanta (`00-README.md`, Fases).
2. 30-10: E8b, E10 y E12 reusan los cachés y la condición base de E8 (v8 sin ReID, D-H2, K2
   pasado) en lugar de re-cachear: ahorra dos noches y compara contra un número auditado. El
   precio: la base no es exactamente la línea publicada de A (E8 §5.8). K-reuso verifica que el
   código nuevo con la mejora apagada da los mismos tensores.
3. 30-10: la calibración de E8b es por articulación y transductiva sin etiquetas, sobre las siete
   fuentes juntas. Por fold sería más limpia pero obliga a siete builds por condición.
4. 30-10: el tope del relleno de E10 (0,10 s) se fija antes de mirar los huecos; P14d lo
   describe pero no lo mueve.
5. 30-10: E11 separa desarrollo (las seis fuentes de C1, donde se diagnosticó) de confirmación
   (material nuevo), como pidió el documento de C1 §4. Sin material nuevo el 7-11, E11 es
   exploratorio.
6. 30-10: E12 cambia features y máscara juntas y lo declara como una sola variable (E12 §3).
7. 30-10: las licencias de RTMO, OC-SORT, ByteTrack, BoxMOT y SAM 2 se verifican en el humo de
   cada experimento (H9-4, H13-1, H14-1) con fuente y fecha, no desde este spec.
