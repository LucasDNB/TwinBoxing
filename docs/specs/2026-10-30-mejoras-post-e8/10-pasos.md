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
| S1 | SDD | T1 a T12, PR `mejoras-fase0` | — | pruebas nuevas en rojo y después en verde; suites completas en verde (las de S1 de E8) | hecha | 30-10: T1 a T12 en la rama `mejoras-fase0` (`fb406e0`), PR #41 contra `main`, encima de #40. En rojo antes del código: T1, T3, T4, T5, T6 y T9 (fixtures de antes guardadas con el código viejo: `boxtwin-detector/tests/datos/features_hoy.npz`, `boxtwin-annotator/tests/datos/propuesta_mixta_hoy.json`) y T2; las de T7, T8, T10, T11 y T12 se escribieron junto con el script. Suites: boxtwin-annotator 786, boxtwin-api 311, boxtwin-detector 203 (con la decisión 9), boxtwin-guantes 46, scripts/clasificador 32, scripts/clasificador_geo 89 (1 salteada, la de mmpose), scripts/gemelo 69, scripts/mejoras 57, despliegue/tests 20. Nada corrido sobre datos reales |
| L0 | Lucas | Mergear el PR de este spec y el de S1 | S1 | los dos en `main` | hecha | 30-10 21:49 UTC: #40 (`9b88fe1`) y #41 (`547d58b`) mergeados en `main` |
| S2 | SDD | Worktree fijo `~/Proyectos/TwinBoxing-mejoras` desacoplado en la punta de `main` después de L0; huellas de A y de B; sha256 de los originales de A y de los cachés de `raiz-v8` y `raiz-y26` en `~/boxtwin-mejoras/antes.txt` | L0 | hash anotado; los archivos existen; C `inactive` | hecha | 30-10 21:51 UTC: worktree `~/Proyectos/TwinBoxing-mejoras` desacoplado en `547d58b`. `~/boxtwin-mejoras/antes.txt`, 49 sha256: los 21 originales de A (iguales a `~/boxtwin-gemelo/e8/originales-despues.txt`) y 14 de `raiz-v8` y 14 de `raiz-y26` (`pose.npz` y `meta.json`). `huella-A-antes.txt`: HEAD de A `1058d5e`, 17 sesiones, igual a la huella de E8. `huella-B-antes.txt`: HEAD de B `efcec23`, B **apagada** (en E8 estaba encendida; no se toca, el despues se compara apagada con apagada). C `inactive` en las dos unidades (`c-antes.txt`). Incidente: de 18:30 a 21:50 UTC del 30-10 la copia de A estuvo en las ramas de esta campana (un `gh pr checkout 40` en la copia principal, que A usa en modo editable); se volvio a `1058d5e` antes de S2. En ese lapso A no corrio ningun trabajo (el ultimo es del 27-10); la API siguio levantada desde las 18:12, y reiniciarla queda a criterio de Lucas |
| L1 | Lucas | OK para las noches del 31-10 al 3-11 | S2 | fechas anotadas acá | hecha | 30-10 21:55 UTC: "OK, dale" de Lucas, para las noches del 31-10 al 3-11 |
| S3 | SDD | K-reuso de E8b, E10 y E12 (tensores con las mejoras apagadas contra E8) | L1 | `kreuso.py` sin diferencias | hecha | 30-10 21:53 UTC, `547d58b`: `~/boxtwin-mejoras/kreuso/{base,y26}.json`, K-reuso pasa en las dos condiciones, 8 archivos cada una (7 `.det.npz` y `golpes.npz`), 0 diferencias |
| S4 | SDD | E8b: `calibrar_umbrales.py`; K-cal; build de Y26-cal; K-mec | S3 | `umbrales.json`; K-cal a ±0,005 | hecha | 30-10 21:59 UTC: `~/boxtwin-mejoras/e8b/umbrales.json` (sha256 `feaa054c9b3a…`): nariz 0,134, hombro 0,101, codo 0,111, muñeca 0,110, cadera 0,178. K-cal pasa (diferencias < 0,00005). K-mec: 139534 cuadros-carril usables en Y26-cal, contra 147788 (base) y 111472 (Y26). Pasos usables de E4: 0,786 (base 0,842, Y26 0,643) |
| S5 | SDD | E8b: `folds` y E4 de Y26-cal (noche del 31-10) | S4 | 35 corridas de cada uno | pendiente | |
| S6 | SDD | E8b: lectura, auditoría, documento, JSON, README, cambios de tesis | S5 | `veredictos.json`; auditoría | pendiente | |
| S7 | SDD | E10: P14d, histograma de huecos por condición, sin etiquetas (de día) | S3 | JSON de huecos | hecha | 30-10: `~/boxtwin-mejoras/e10/huecos-{v8,y26,y26cal}.json`. Cuadros perdidos de muñeca en huecos de ≤ 0,10 s: v8 0,225, Y26 0,113, Y26-cal 0,192 (codo 0,256, 0,161, 0,250). Debajo de 0,30: el relleno actúa sobre la minoría de la pérdida (02 §2); el tope no cambia |
| S8 | SDD | E10: build y corridas de v8+R (noche del 1-11); S8b Y26-cal+R si S5 terminó; S8c variante 0,20 s si sobra noche | S3, S7 | 35 corridas de cada uno | pendiente | |
| S9 | SDD | E10: lectura, auditoría, documento | S8 | | pendiente | |
| S10 | SDD | E11 desarrollo: `medir_e11.py` sobre la evidencia de C1, con y sin corte; auditoría ítem 2 (sin corte reproduce 85/105) | S2 | JSON de desarrollo | hecha | v1 (30-10 21:53 UTC, `547d58b`, `~/boxtwin-mejoras/e11/desarrollo-v1.json`): auditoría ítem 2, sin corte 85/105; con corte invierte Sparring (precisión 0,887 a 0,059) por la regla de §2.2, punto 1 (decisión 16). v2 (31-10, `9782993`, rama `e11-regla-corte`, PR #42, con la regla corregida; `desarrollo-v2.json`, copiado como `desarrollo.json`): sin corte 85/105 otra vez; precisión por cuadro 0,850 a 0,923 y cobertura 0,788 a 0,872 con corte, ninguna fuente empeora (Sparring 0,887 a 0,960). P15d: 16 tracks partidos, 10 de ellos mixtos según la verdad, de 18 mixtos. Desarrollo: no decide P15a ni P15b |
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
8. 30-10 (S1, T1): la nariz entra en `umbrales.json` y se calibra, pero no entra en la máscara de
   `features_de`, igual que hoy (hoy la nariz no se filtra). Así, con todo en 0,3, la salida es la
   de antes bit a bit. Si se quisiera filtrar la nariz sería otra variable.
9. 30-10 (S1, T4; corrige E10 §2, antes de la primera corrida de S8): el tope del relleno va al
   entero más cercano y no hacia abajo: 3 cuadros a 30 y a 29,97 fps, 6 a 60 y a 59,94. Hacia abajo
   daba 2 a 29,97 (0,10 × 29,97 = 2,997) y 5 a 59,94, cuando los ejemplos de §2 (3 a 30, 6 a 60)
   muestran que el tope se pensó como 3 y 6. Decidido por Lucas el 30-10. En segundos el tope queda
   en 0,1001 s a 29,97 y a 59,94.
10. 30-10 (S1, T4 y T5): "detección presente" es `valid` y no `interpolated` en el export (la pose
    interpolada por la identidad no es una detección); en E4, `valid` de la serie. Un extremo del hueco
    sin detección no cuenta como aceptado.
11. 30-10 (S1, T6): la mediana empata (con un cambio limpio de color, todo k entre m y el cambio da la
    misma diferencia, y el primero corta en la muestra 5). El empate se rompe por la diferencia de
    medias, que tiene el pico en el cambio. Los cortes solo caen entre cuadros distintos.
12. 30-10 (S1, T6): "se repite una sola vez dentro de cada mitad" y "como mucho dos cortes" se leen
    juntos: si las dos mitades piden corte, queda el de mayor diferencia de medianas.
13. 30-10 (S1, T6): sin `cuadros_semilla`, el cuadro del recorte de una semilla partida es la mediana
    de sus cuadros con color, que es el que muestra `candidatos.elegir` fuera del par sugerido.
    `medir_e11.py` le pasa el cuadro real de cada candidato. Los segmentos de un track partido llevan
    id −(3·track + k + 1) en la propuesta; el `Assignment` va al track con el rango del segmento.
14. 30-10 (S1, T3): `datos.py` no cambia: no calcula features ni máscaras. Los umbrales y el relleno de
    E4 entran en `features2d.py` (las 20 del brazo y las 6 del rival).
15. 30-10 (S1): S7 (P14d) no tenía tarea de código; se suma `scripts/mejoras/huecos.py` con su prueba.
    S8c (relleno a 0,20 s) y P16e (E4 con guante) no están en `mejoras.sh`: son informativas, para si
    sobra noche, y piden código que no está en T1 a T12.
16. 31-10 (E11, corrige 03 §2.2 punto 1, después de la primera corrida de S10 y antes de S11): el
    índice de corte sale de la máxima diferencia de MEDIAS, y si se corta lo siguen decidiendo las
    medianas (signo opuesto, las dos ≥ 0,25). La regla escrita (máxima diferencia de medianas) es casi
    plana mientras el color no cambia y el ruido elegía el corte: en Sparring cayó una muestra antes
    del cambio, en el recorte de las semillas, e invirtió la propuesta entera (S10 v1). Se corrigió
    mirando el material de desarrollo, que es para eso (§4); la confirmación no corrió y queda limpia.
    Decidido por Lucas el 31-10. Reemplaza la decisión 11 (el desempate por medias deja de hacer falta).
