# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). La evidencia es el comando que la verifica o el archivo que produjo.
Ninguna tarea escribe en A ni en B. `S` = la corre el SDD sin `sudo`; `L` = la hace Lucas.

## Fase 0 · sin GPU

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S1 | SDD | T1 a T10 (`01-percepcion.md` §6): pruebas primero, en rojo; código; en verde. Rama `gemelo-yolo26-fase0`, PR contra `main` | — | las pruebas nuevas en rojo y después en verde; suites completas en verde (`boxtwin-api`, `boxtwin-annotator`, `boxtwin-detector`, `boxtwin-guantes`, `scripts/clasificador*/tests`); ensayo de T9 con rutas por entorno | pendiente | |
| L0 | Lucas | Revisar y mergear el PR de este spec y el de S1 | S1 | los dos en `main` | pendiente | |
| S2 | SDD | Huellas antes: `despliegue/academico/huella-A.sh` y `despliegue/gemelo/huella-B.sh` en `~/boxtwin-gemelo/huella-{A,B}-antes.txt` | L0 | los dos archivos existen; `/salud` de A y de B dan lo de `PRODUCCION.md` | pendiente | |
| L1 | Lucas | OK para la fase 1 y la noche elegida, con las sesiones de prueba de A ya cerradas | S2 | la fecha anotada acá | pendiente | |

## Fase 1 · E8, de noche y bajo el candado

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| S3 | SDD | sha256 de las siete anotaciones, `meta.json` y `pose.npz` originales de la copia principal, en `~/boxtwin-gemelo/e8/originales-antes.txt` | L1 | el archivo existe con 21 líneas | pendiente | |
| S4 | SDD | Bajar `yolo26l-pose.pt` a `~/boxtwin-gemelo/modelos/` y anotar su sha256 | L1 | `sha256sum` anotado acá | pendiente | |
| S5 | SDD | Worktree fijo: `produccion.sh worktree-gemelo <punta de main después de L0>`; anotar el hash | L0 | `git -C ~/Proyectos/TwinBoxing-gemelo rev-parse HEAD` igual al anotado; HEAD de A sin cambios | pendiente | |
| S6 | SDD | Humo H1 a H3 (`01-percepcion.md` §3), con el código de S5 | S4, S5 | los tres pasan, o la rama D-H2 anotada como decisión antes de S7 | pendiente | |
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

## Preguntas abiertas

1. ¿C en la tailnet, como B pero en otro puerto, para abrirla desde el celular? Hoy no, por D2.
2. Qué sparring y qué bolsa propios entran en L4.

## Lo que este spec no hace

Está en `00-README.md`, al final.
