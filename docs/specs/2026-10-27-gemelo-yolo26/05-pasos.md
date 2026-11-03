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
| S5 | SDD | Worktree fijo: `produccion.sh worktree-gemelo <punta de main después de L0>`; anotar el hash | L0 | `git -C ~/Proyectos/TwinBoxing-gemelo rev-parse HEAD` igual al anotado; HEAD de A sin cambios | hecha | 29-10: primero `684b1dd`; repetida en `0c9dd69` (punta de `main` con el PR #38), con `git checkout --detach` sobre el mismo worktree. HEAD de A sin cambios (`1058d5e`) |
| S6 | SDD | Humo H1 a H3 (`01-percepcion.md` §3), con el código de S5 | S4, S5 | los tres pasan, o la rama D-H2 anotada como decisión antes de S7 | hecha | 29-10, con `0c9dd69` y `botsort-sin-reid.yaml` (D-H2), en `~/boxtwin-gemelo/humo/humo.json`: H1 pasa (240 pares, 5,97 px contra 6,49 de la mejor permutacion); H2 pasa (600 cuadros; 2 reinicios en 658 a 1772 de pacquiao); H3 v8 40,6 ms (p95 44,0) y 544 MiB, YOLO26 37,1 ms (p95 40,7) y 350 MiB. El primer humo, con `684b1dd` y ReID, en `humo-684b1dd.json` |
| S7 | SDD | Re-caché de las siete fuentes con `recache.py` dentro de `e8.sh` | S6 | siete cachés en `raiz-y26` con el sha256 del modelo de S4 y el del yaml igual al de A; `recache.json` con tiempo y fps por fuente | hecha | `e8/recache.json` y `recache-v8.json`: 180193 cuadros por condicion, 118 min con YOLO26 y 129 con v8; las siete con el modelo de S4 (`ad33da8a29ea`) y el yaml sin ReID (`27ecd7c5632a`) |
| S8 | SDD | Puente de identidad de las siete fuentes; K1 y K4 | S7 | K1 al 100%; `puente.json` por fuente; cada `annot.json` carga con `load` | hecha | K1 al 100% en las siete fuentes, 0 conflictos; YOLO26 empareja el 95,7% de los cuadros con verdad, mediana del IoU 0,959 a 0,973 (K4 sin debiles); base v8 0,998 a 1,000; los catorce `annot.json` cargan con `load` |
| S9 | SDD | Exports `sequence` y tensores del detector de las dos condiciones, con `--out` en `e8/{base,y26}`; K3 | S8 | catorce exports; los mismos eventos en las dos condiciones; K3 anotado | hecha | catorce exports; los mismos 1046 golpes y 38 amagues por fuente en las dos; cuadros-carril usables 147788 (base) contra 111472 (YOLO26). K3 en S12 (decision 12) |
| S10 | SDD | `boxtwin-detector folds` de las dos condiciones, cinco semillas, siete folds; K2 del detector | S9 | los dos reportes; K2 dentro de ±0,02 o la explicación anotada | hecha | `e8/{base,y26}/folds.json`; K2 del detector 0,637 contra 0,635 |
| S11 | SDD | `datos.py` y E4 de las dos condiciones (`BOXTWIN_RAIZ_FUENTES`, `BOXTWIN_CLASIFICADOR_GEO`); K2 de E4; SH1 | S8 | 1047 `frame_dir` idénticos; 35 corridas por condición; K2 dentro de ±0,02 | hecha | 1047 `frame_dir` identicos (D3 ok en las dos); 35 corridas por variante y condicion; K2 de E4 0,732 contra 0,727; SH1 lectura a (0,572) |
| S12 | SDD | `medir_trackeo.py` (P12c) | S8 | JSON por fuente y total | hecha | `e8/trackeo.json`; denominador 48944 (A) y 49578 (B) |
| S13 | SDD | `medir_c1.py` sobre `raiz-y26` (P12d) | S8 | resumen de C1 con su JSON por fuente | hecha | `e8/y26/c1/resumen.json`: 85/100 = 0,850 (Wilson 0,767 a 0,907) |
| S14 | SDD | Originales después: lo mismo que S3, en `originales-despues.txt` | S10 a S13 | igual a S3 (CG4) | hecha | `e8/originales-despues.txt` igual a S3, 21 lineas (CG4) |
| S15 | SDD | Lectura contra el pre-registro en `e8/veredictos.json` y borrador del documento | S14 | cada pregunta con su lectura según `02-preregistro-e8.md` §4 | hecha | `e8/veredictos.json` con `scripts/gemelo/lectura_e8.py`: P12a no demostrada (0,433 contra 0,623, 0 de 6); P12b no demostrada (0,675 contra 0,732, 1 de 7); P12c mezclada; K3 se declara (0,750 contra 0,896) |
| S16 | SDD | Auditoría independiente (`02-preregistro-e8.md` §7); documento final, JSON en `docs/experiments/datos/`, entrada del README y cambios de tesis, en un PR | S15 | auditoría sin fallas, o las fallas en el documento (CG6) | hecha | auditor independiente: 9 de 9 items pasan, con recalculo propio de P12a y P12b; `docs/experiments/2026-10-29-e8-yolo26.md`, los tres JSON, la entrada del README y `docs/tesis/2026-10-29-e8-cambios.md`, en el PR de resultados |

## Fase 2 · la instancia C

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S17 | SDD | `detector_7fuentes_y26.ens.pt` y su `.metricas.json` (`03-instancia-c.md` §5), con el control de la receta | S16 | `meta` con commit y sha256 de los tensores; control anotado | hecha | `detector_7fuentes_y26.ens.pt` (`8ff9073c33d1`), `.metricas.json` con precision 0,404 y recall 0,480 de P12a en 0,80, `.meta.json` con sha256 de los siete tensores. Control sin umbral: la receta sobre los tensores de A difiere de `detector_7fuentes.ens.pt` hasta 0,124 de P por cuadro en 01-sparring (decision 20) |
| S18 | SDD | `clasificador_geo_y26_v1`, en CPU | S16 | `meta.json` con las métricas de P12b | hecha | `clasificador_geo_y26_v1`, cpu, `0c9dd69`, familia 0,6753 de E8, comando en el meta (decision 21) |
| S19 | SDD | `clasificador_geo_y26_v1_sin_rival`, en CPU | S16 | `meta.json` con las métricas de E4a sobre Y26 | hecha | `clasificador_geo_y26_v1_sin_rival`, cpu, `0c9dd69`, familia 0,6781 de E8 |
| S20 | SDD | Web de C: `produccion.sh web ~/Proyectos/TwinBoxing-gemelo` | S5 | `out/index.html` en el worktree de C; la fecha del de A sin cambios | hecha | 29-10: `out/index.html` en el worktree de C; el de A con el mismo mtime (1793069420) |
| S21 | SDD | `produccion.sh env-gemelo`; unidades copiadas a `~/boxtwin-gemelo/systemd/`; `demo.yaml` sin sesiones; `produccion.sh previo c` | S17 a S20 | `stat -c %a` da `600`; secreto e invitación distintos de A y de B; nada en `:8002` | hecha | `~/boxtwin-gemelo/boxtwin-gemelo.env` en 600, secreto e invitacion distintos de A y de B; unidades en `~/boxtwin-gemelo/systemd/`; `demo.yaml` sin sesiones; `previo c` ok |
| L2 | Lucas | `sudo install -m 600 -o lucasb -g lucasb ~/boxtwin-gemelo/boxtwin-gemelo.env /etc/boxtwin-gemelo.env`; `sudo cp ~/boxtwin-gemelo/systemd/*.service /etc/systemd/system/`; `sudo systemctl daemon-reload`; **sin** `enable` | S21 | `systemctl is-enabled boxtwin-gemelo-api` da `disabled` | hecha | 29-10, con sudo por pedido de Lucas: `/etc/boxtwin-gemelo.env` 600 lucasb; unidades copiadas; `daemon-reload`; `is-enabled` da `disabled` en las dos |
| L3 | Lucas | B apagada; arrancar C: `sudo systemctl start boxtwin-gemelo-api boxtwin-gemelo-worker` | L2 | las dos `active` | hecha | 29-10 07:38 UTC: B apagada (estaba `active`); `previo c` ok; las dos unidades de C `active` |
| S22 | SDD | CG2 y CG3: `produccion.sh verificar c`, `ss -ltn`, `tailscale serve status`; sembrar las cuentas de prueba con el entorno de C | L3 | todo lo de `03-instancia-c.md` §8; `:8002` solo en `127.0.0.1` | hecha | `verificar c` ok (commit `0c9dd69`, `yolo26l-pose.pt` `ad33da8a29ea`, geometrico, modelo individual, secreto fijo, web); `:8002` solo en 127.0.0.1; `tailscale serve` sin `:8002`, `:443` y `:8443` como en S2; cuentas de prueba sembradas (3 cuentas, 0 sesiones) |
| L4 | Lucas | Subir a C un sparring propio y una bolsa propia, con consentimiento, y hacer la siembra | S22 | los videos anotados acá, con quién aparece | pendiente | necesita un sparring y una bolsa propios con consentimiento; los elige Lucas |
| S23 | SDD | CG5 y CG7 sobre las sesiones de L4; CG8, pico de VRAM mientras procesan | L4 | Fight-Cards con el bloque `pose` y los números de E8; pico anotado | bloqueada | por L4: CG5 y CG7 sobre sesiones reales y CG8 con C procesando |
| S24 | SDD | Huellas después de A y de B (CG1, CG4), CG9, y lectura de CG1 a CG9 | S23 | iguales a S2 | hecha, se repite despues de L4 | 29-10: `huella-A-despues.txt` y `huella-B-despues.txt` identicas a las de S2 (CG1); CG9 ok (abajo) |
| L5 | Lucas | Apagar C: `sudo systemctl stop boxtwin-gemelo-worker boxtwin-gemelo-api` | S24 | las dos `inactive` | hecha | 29-10: C `inactive` en las dos unidades; B vuelta a encender como estaba en S2 |

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
20. 29-10 (S17): la receta del detector final es la de `scripts/gemelo/detector_final.py`:
    `entrenar_ensamble` con la `Config` por defecto, cinco semillas y las siete fuentes como
    entrenamiento y como validacion de la parada temprana. El script no esta en `0c9dd69`: corrio
    con el `PYTHONPATH` de C (la biblioteca es la del commit de C) y su meta dice `ff6e159`, el commit
    del script. El control contra el detector de A da hasta 0,124 de diferencia de P por cuadro: se
    declara en el documento de E8.
21. 29-10 (S18 y S19): `entrenar_final.py` del commit de C lee el JSON de E8 de `docs/` de su propio
    checkout. Se copio ahi `2026-10-29-e8-yolo26-E4.json`, identico al del PR de resultados (sha256
    `cfe9773615600a69...`), como archivo sin versionar; el codigo de C no cambio.
22. 29-10 (L3 y L5): B estaba encendida en S2. Se apago para encender C (regla 2) y se volvio a
    encender despues de L5, para que la huella de S24 se compare con B en el mismo estado.

## Criterios de aceptacion, al 29-10

| Id | Estado | Evidencia |
|---|---|---|
| CG1 | cumple | huellas de A y de B de S24 iguales a las de S2, sin ninguna linea distinta |
| CG2 | cumple | `verificar c` ok el 29-10 (S22) |
| CG3 | cumple | `:8002` solo en 127.0.0.1; `tailscale serve` sin entrada hacia `:8002`; `:443` y `:8443` como en S2 |
| CG4 | cumple | sesiones de A 17 y de B 2, sin cambios; S14 igual a S3 |
| CG5 | parcial | el detector de C lleva precision 0,404 y recall 0,480 de E8 y los clasificadores su familia de E8; el bloque `pose` lo prueba T3. Falta verlo en una Fight-Card real (L4) |
| CG6 | cumple | pre-registro `916f7d8` anterior a H1; auditoria 9 de 9 |
| CG7 | pendiente | L4 |
| CG8 | parcial | el candado sin etapas con trabajos de A (auditoria, item 8). El pico de VRAM de C procesando sola falta: necesita L4 |
| CG9 | cumple | cada modelo con su meta; volver a correr `entrenar_final.py` da los mismos sha256 en las diez semillas de los dos clasificadores (el detector entrena en GPU y no se pide) |

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
