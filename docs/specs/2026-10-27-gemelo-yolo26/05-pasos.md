# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). La evidencia es el comando que la verifica o el archivo que produjo.
Ninguna tarea escribe en A ni en B. `S` = la corre el SDD sin `sudo`; `L` = la hace Lucas.

## Fase 0 · sin GPU

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S1 | SDD | T1 a T10 (`01-percepcion.md` §6): pruebas primero, en rojo; código; en verde. Rama `gemelo-yolo26-fase0`, PR contra `main` | — | las pruebas nuevas en rojo y después en verde; suites completas en verde (`boxtwin-api`, `boxtwin-annotator`, `boxtwin-detector`, `boxtwin-guantes`, `scripts/clasificador*/tests`); ensayo de T9 con rutas por entorno | hecha | rama `gemelo-yolo26-fase0` (spec en `gemelo-yolo26-spec`, 916f7d8); pruebas nuevas en rojo y despues en verde, salvo las unidades de T10, que se derivaron de las de B con sed antes de su prueba. Suites: boxtwin-annotator 774, boxtwin-api 311, boxtwin-detector 152, boxtwin-guantes 46, scripts/clasificador 32, scripts/clasificador_geo 84 (1 salteada, la de mmpose), scripts/gemelo 39, despliegue/tests 20 (ensayo de T9 con rutas por entorno y puertos libres, decision 15); `verificar a` y `verificar b` como antes; PR #36 mergeado el 29-10 (`6a0f0db`) |
| L0 | Lucas | Revisar y mergear el PR de este spec y el de S1 | S1 | los dos en `main` | hecha | 29-10: PR #35 (spec, `ce4c5a8`) y PR #36 (fase 0, `6a0f0db`) mergeados en `main` |
| S2 | SDD | Huellas antes: `despliegue/academico/huella-A.sh` y `despliegue/gemelo/huella-B.sh` en `~/boxtwin-gemelo/huella-{A,B}-antes.txt` | L0 | los dos archivos existen; `/salud` de A y de B dan lo de `PRODUCCION.md` | hecha | 29-10, con los scripts de `6a0f0db`: `~/boxtwin-gemelo/huella-A-antes.txt` (HEAD de A `1058d5e`, 17 sesiones) y `huella-B-antes.txt` (HEAD de B `efcec23`, 2 sesiones, B encendida). `produccion.sh verificar a` y `verificar b` ok; A con la version de prueba encendida, como esta registrado el 26-10 |
| L1 | Lucas | OK para la fase 1 y la noche elegida, con las sesiones de prueba de A ya cerradas | S2 | la fecha anotada acá | hecha | 29-10: "ejecuta todo hoy" de Lucas, tomado como OK para la noche del 29-10. Al arrancar: cola de A sin trabajos `en_cola` ni `tomado` (ultimo trabajo del 27-10), GPU sin procesos |

## Fase 1 · E8, de noche y bajo el candado

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S3 | SDD | sha256 de las siete anotaciones, `meta.json` y `pose.npz` originales de la copia principal, en `~/boxtwin-gemelo/e8/originales-antes.txt` | L1 | el archivo existe con 21 líneas | hecha | 29-10 01:55 UTC: `~/boxtwin-gemelo/e8/originales-antes.txt`, 21 lineas |
| S4 | SDD | Bajar `yolo26l-pose.pt` a `~/boxtwin-gemelo/modelos/` y anotar su sha256 | L1 | `sha256sum` anotado acá | hecha | 29-10: release v8.4.0 de Ultralytics, 57995961 bytes, sha256 `ad33da8a29ea5772318c4c980844e47b56792d2b63815ad4e8e09c078c7d1abf` |
| S5 | SDD | Worktree fijo: `produccion.sh worktree-gemelo <punta de main después de L0>`; anotar el hash | L0 | `git -C ~/Proyectos/TwinBoxing-gemelo rev-parse HEAD` igual al anotado; HEAD de A sin cambios | hecha, se repite | 29-10: `684b1dd` (punta de `main` con el PR #37); HEAD de A sin cambios (`1058d5e`). Se repite con el PR del arreglo de H3 y la decision 18 |
| S6 | SDD | Humo H1 a H3 (`01-percepcion.md` §3), con el código de S5 | S4, S5 | los tres pasan, o la rama D-H2 anotada como decisión antes de S7 | en curso | 29-10, con `684b1dd`, en `~/boxtwin-gemelo/humo/humo.json`: H1 pasa (`kpt_shape` [17, 3]; 5 personas con 17 keypoints en GPU y en CPU; 240 pares, mediana propia 5,97 px contra 6,49 px de la mejor permutacion, que cambia las munecas). H2 pasa (600 cuadros; tramo 658 a 1772 de pacquiao con 2 reinicios), pero con ReID por `yolo26n-cls.pt` y no por las features de YOLO26 (decision 18). H3 no midio: el chequeo de GPU conto al propio proceso (arreglado en `otros_en_gpu`). Se repite despues de S5 |
| S7 | SDD | Re-caché de las siete fuentes con `recache.py` dentro de `e8.sh` | S6 | siete cachés en `raiz-y26` con el sha256 del modelo de S4 y el del yaml igual al de A; `recache.json` con tiempo y fps por fuente | pendiente | |
| S8 | SDD | Puente de identidad de las siete fuentes; K1 y K4 | S7 | K1 al 100%; `puente.json` por fuente; cada `annot.json` carga con `load` | pendiente | |
| S9 | SDD | Exports `sequence` y tensores del detector de las dos condiciones, con `--out` en `e8/{base,y26}`; K3 | S8 | catorce exports; los mismos eventos en las dos condiciones; K3 anotado | pendiente | |
| S10 | SDD | `boxtwin-detector folds` de las dos condiciones, cinco semillas, siete folds; K2 del detector | S9 | los dos reportes; K2 dentro de ±0,02 o la explicación anotada | pendiente | |
| S11 | SDD | `datos.py` y E4 de las dos condiciones (`BOXTWIN_RAIZ_FUENTES`, `BOXTWIN_CLASIFICADOR_GEO`); K2 de E4; SH1 | S8 | 1047 `frame_dir` idénticos; 35 corridas por condición; K2 dentro de ±0,02 | pendiente | |
| S12 | SDD | `medir_trackeo.py` (P12c) | S8 | JSON por fuente y total | pendiente | |
| S13 | SDD | `medir_c1.py` sobre `raiz-y26` (P12d) | S8 | resumen de C1 con su JSON por fuente | pendiente | |
| S14 | SDD | Originales después: lo mismo que S3, en `originales-despues.txt` | S10 a S13 | igual a S3 (CG4) | pendiente | |
| S15 | SDD | Lectura contra el pre-registro en `e8/veredictos.json` y borrador del documento | S14 | cada pregunta con su lectura según `02-preregistro-e8.md` §4 | pendiente | |
| S16 | SDD | Auditoría independiente (`02-preregistro-e8.md` §7); documento final, JSON en `docs/experiments/datos/`, entrada del README y cambios de tesis, en un PR | S15 | auditoría sin fallas, o las fallas en el documento (CG6) | pendiente | |

## Fase 2 · la instancia C

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S17 | SDD | `detector_7fuentes_y26.ens.pt` y su `.metricas.json` (`03-instancia-c.md` §5), con el control de la receta | S16 | `meta` con commit y sha256 de los tensores; control anotado | pendiente | |
| S18 | SDD | `clasificador_geo_y26_v1`, en CPU | S16 | `meta.json` con las métricas de P12b | pendiente | |
| S19 | SDD | `clasificador_geo_y26_v1_sin_rival`, en CPU | S16 | `meta.json` con las métricas de E4a sobre Y26 | pendiente | |
| S20 | SDD | Web de C: `produccion.sh web ~/Proyectos/TwinBoxing-gemelo` | S5 | `out/index.html` en el worktree de C; la fecha del de A sin cambios | pendiente | |
| S21 | SDD | `produccion.sh env-gemelo`; unidades copiadas a `~/boxtwin-gemelo/systemd/`; `demo.yaml` sin sesiones; `produccion.sh previo c` | S17 a S20 | `stat -c %a` da `600`; secreto e invitación distintos de A y de B; nada en `:8002` | pendiente | |
| L2 | Lucas | `sudo install -m 600 -o lucasb -g lucasb ~/boxtwin-gemelo/boxtwin-gemelo.env /etc/boxtwin-gemelo.env`; `sudo cp ~/boxtwin-gemelo/systemd/*.service /etc/systemd/system/`; `sudo systemctl daemon-reload`; **sin** `enable` | S21 | `systemctl is-enabled boxtwin-gemelo-api` da `disabled` | pendiente | |
| L3 | Lucas | B apagada; arrancar C: `sudo systemctl start boxtwin-gemelo-api boxtwin-gemelo-worker` | L2 | las dos `active` | pendiente | |
| S22 | SDD | CG2 y CG3: `produccion.sh verificar c`, `ss -ltn`, `tailscale serve status`; sembrar las cuentas de prueba con el entorno de C | L3 | todo lo de `03-instancia-c.md` §8; `:8002` solo en `127.0.0.1` | pendiente | |
| L4 | Lucas | Subir a C un sparring propio y una bolsa propia, con consentimiento, y hacer la siembra | S22 | los videos anotados acá, con quién aparece | pendiente | |
| S23 | SDD | CG5 y CG7 sobre las sesiones de L4; CG8, pico de VRAM mientras procesan | L4 | Fight-Cards con el bloque `pose` y los números de E8; pico anotado | pendiente | |
| S24 | SDD | Huellas después de A y de B (CG1, CG4), CG9, y lectura de CG1 a CG9 | S23 | iguales a S2 | pendiente | |
| L5 | Lucas | Apagar C: `sudo systemctl stop boxtwin-gemelo-worker boxtwin-gemelo-api` | S24 | las dos `inactive` | pendiente | |

## Decisiones

Una línea por decisión, con el motivo, antes de correr la parte afectada.

1. 27-10 (D1): "corre en la estación" se lee como una tercera instancia al lado de A y B, con el
   mismo patrón que `dos-instancias`: worktree congelado, puerto, entorno, unidades y datos
   propios.
2. 27-10 (D2): "alcance local" se lee como sin exposición: `127.0.0.1:8002`, sin Funnel y sin
   `tailscale serve`. Desde la notebook, túnel SSH por la tailnet (`03-instancia-c.md` §7).
3. 27-10 (D3): "sobre las últimas versiones" se lee como la punta de `main` (al 27-10 `2887165`,
   con las versiones, el modo académico y el avatar) más la fase 0. El tipo de golpe es el de A
   (E4 y E4a), reentrenado sobre YOLO26. E7 queda afuera: su 3D tiene licencia de investigación y
   llevarlo a YOLO26 es otra campaña.
4. 27-10 (D4): tamaño L, `yolo26l-pose.pt`, el mismo rol que `yolov8l-pose.pt`.
5. 27-10 (D5): sin preferencia de alcance funcional, se toma la recomendación del 27-10: medir
   antes de servir (E8 primero, el servicio con los modelos de E8 después).
6. 27-10: el umbral del detector queda en 0,80 en las dos condiciones de E8, el de A. Elegir otro
   para Y26 con los mismos datos sería ajustar sobre la evaluación.
7. 27-10: C entra con la versión de prueba (`BOXTWIN_DEMO=1`), como decidió Lucas el 26-10 para
   las versiones. No hay cuentas reales en C.
8. 27-10: a diferencia de B, el `PYTHONPATH` de C incluye `boxtwin-detector/src`, para que la
   inferencia del detector sea la del commit de C.
9. 28-10 (S1, T6): `AssignmentOp` suma `puente`, el `origin.op` que pide §5 de
   `01-percepcion.md`. Es un valor nuevo del enum; A corre un commit desacoplado y no lo lee.
10. 28-10 (S1, T6): el puente pasa solo los roles de peleador. Las asignaciones `ignore` (el
    arbitro) no se transportan: la verdad es por peleador. Los ids nuevos son `as_pe8_*` e
    `in_pe8_*`, y `counters` se copia igual, como pide §5.5. El `iou_at_join` de una
    interpolacion transportada se recalcula sobre las cajas nuevas. "Cuadros con deteccion de
    YOLO26 sin verdad" se cuenta por peleador: cuadros sin verdad de ese peleador donde el cache
    nuevo tiene una deteccion que no se emparejo con nadie.
11. 28-10 (S1, T4): el JSON de E4 de E8 (`docs/experiments/datos/<fecha>-e8-yolo26-E4.json`)
    lleva `{"base": {"E4": ..., "E4a": ...}, "y26": {"E4": ..., "E4a": ...}}`, cada uno con el
    `ensamble_total` que ya escribe `evaluar.py`. `entrenar_final.py --exp E4-y26` lee
    `y26.E4` y se niega a entrenar si el archivo no esta (CG5). El `meta.json` de cada modelo
    suma `comando` (CG9).
12. 28-10 (S1, T7): K3 se calcula en `medir_trackeo.py` (S12) y no en S9: necesita los dos caches
    emparejados contra la verdad, que es lo que ese script ya tiene. El denominador es el mismo
    de P12c.
13. 28-10 (S1, T8): `recache.py` corre el preproceso con `--no-proxy`. El proxy no entra en
    `config_hash` ni en E8, y son unos minutos menos por fuente.
14. 28-10 (S1, T5): `datos.py` nombra el sha256 de cada anotacion relativo a
    `BOXTWIN_RAIZ_FUENTES` y no al repo; con la variable sin definir, la clave es la de hoy.
15. 28-10 (S1, T9): el ensayo de `produccion.sh` (`despliegue/tests/test_produccion_sh.py`)
    elige puertos libres al arrancar en vez de 8100 a 8102: el 28-10 esos tres seguian ocupados
    por los `uvicorn` del ensayo del 26-10 (pids 435864, 435865 y 436003, mas de dos dias
    arriba). No se tocaron. `huella-B.sh` saca `arrancado` de `/salud`, que cambia con cada
    reinicio del proceso y no es de B.
16. 28-10 (S1, T8): `e8.sh` corre `boxtwin-detector folds` sin `--en-distribucion`, sobre los
    `.det.npz` de las siete fuentes. Si ese no es el protocolo del 0,635, K2 lo marca en S10 y se
    corrige antes de leer P12a.
17. 29-10 (T11, pregunta abierta 3): el humo H1 a H3 corre con `scripts/gemelo/humo.py`, como
    etapa S6 de `e8.sh`, bajo el candado. Pruebas de las partes que dan un numero en
    `scripts/gemelo/tests/test_humo.py`, en rojo y despues en verde. Lo que difiere de §3 de
    `01-percepcion.md`, declarado antes de H1:
    - H1 compara contra las 255 permutaciones izquierda-derecha (todo subconjunto de las ocho
      parejas de COCO), con la mediana sobre los 17 keypoints de todos los pares, sin filtrar
      por puntaje. Las personas se emparejan de mayor IoU a menor, cada una una sola vez.
    - H2 no corre `preprocess` sobre copias: corre en el proceso las mismas funciones del
      runner (`_cargar_modelo`, `_inferir`, `_reiniciar_tracker`) sobre los videos originales,
      leidos sin copiar. Es el mismo camino de inferencia y no escribe ningun cache. El tramo de
      `pacquiao_margarito` va de 60 cuadros antes del primer corte (718) a 60 despues del
      segundo (1712), y pasa con dos reinicios.
    - H3 mide solo la inferencia con tracking (sin decodificar), con `torch.cuda.synchronize`
      antes y despues de cada cuadro. El pico es `torch.cuda.max_memory_reserved` del proceso,
      no `nvidia-smi`; CG8 se sigue midiendo con `nvidia-smi` en S23.
    - Sale con 1 si H1 o H2 fallan. Que hacer despues (frenar E8 o entrar en D-H2) lo decide
      Lucas, no el script.
18. 29-10 (S6), decidida por Lucas el 29-10 02:38 UTC ("Si, D-H2"): entrar en la rama D-H2. H2 no fallo, pero mostro
    que `model: auto` no actua igual en las dos condiciones. YOLO26 tiene cabeza de punta a punta
    (`end2end`), y `ultralytics/trackers/track.py` 8.4.41 en ese caso reemplaza `auto` por
    `yolo26n-cls.pt`: una red de ReID aparte, en vez de las features del detector que usa v8.
    Con `botsort.yaml` E8 cambiaria dos cosas, la pose y el ReID. D-H2 (`with_reid: false` en
    las dos condiciones, `botsort-sin-reid.yaml`, la base re-cacheada y pasada por el puente)
    deja una sola. Costo: un re-cache mas de v8 (unas 1 h 45) y el puente de la base; el ReID ya
    se midio sin efecto el 18-08 (31 cambios con ReID, 29 sin). El `botsort.yaml` de A no
    cambia. Es una rama del pre-registro, no una correccion: se puede elegir despues de H1.
19. 29-10 (S6): `boxtwin-detector folds` suma `ensamble_curva` al JSON: recall, precision y F1
    del ensamble por umbral de la grilla (0,10 a 0,90, con 0,80), sumando conteos como el mejor
    umbral. Sin esto el F1 en 0,80 que pide P12a (§4) no se podia leer: `folds` guardaba solo el
    mejor umbral. El mejor umbral sigue igual en `ensamble`, y es el estadistico del 0,635 de K2.

## Preguntas abiertas

1. ¿C en la tailnet, como B pero en otro puerto, para abrirla desde el celular? Hoy no, por D2.
2. Qué sparring y qué bolsa propios entran en L4.
3. ~~El humo H1 a H3 (S6) no tiene script en T1 a T10.~~ Resuelta el 29-10 con T11 (decision
   17). Recomendacion original: sumar T11,
   `scripts/gemelo/humo.py` con pruebas (orden de keypoints contra las permutaciones
   izquierda-derecha, mediana y p95 del tiempo despues del calentamiento), antes de S6. Es una
   correccion del spec y va como decision antes de H1.

## Lo que este spec no hace

Está en `00-README.md`, al final.
