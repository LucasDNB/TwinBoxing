# M · Mejoras después de E8

30-10-2026 · spec SDD · nada lanzado todavía · registro en `10-pasos.md`

Pedido de Lucas (30-10): "llevemos a cabo todas esas mejoras, dame los specs necesarios". Las
mejoras son las que salieron de revisar E8 el 30-10. E8 no mostró que YOLO26 estime peor la
pose: mostró que el sistema está calibrado a cómo puntúa YOLOv8 (K3: muñecas con puntaje ≥ 0,3
de 0,896 a 0,750; cuadros-carril usables −25%; correlación 0,82 entre pasos perdidos y caída de
familia). Las mejoras atacan los tres cuellos de botella medidos, no el estimador como tal:

| Cuello de botella medido | Dónde está medido | Experimentos |
|---|---|---|
| El umbral de puntaje y el enmascarado tiran pose | E8 §5.1 y §5.7 | E8b, E10, E9 |
| Cobertura de identidad: tracks mixtos que crean coexistencia espuria | C1 §4 (18 de 20 errores) | E11, E13, E14 |
| Recall del detector, 0,53: la muñeca se pierde en el golpe | `boxtwin-detector/README.md`, E8 K3 | E10, E12 |
| El hook: proyección monocular | cinco mediciones convergentes, E5, E7 | E15 |

## Los ocho experimentos

| Id | Qué | Una sola variable | GPU | Fase | Archivo |
|---|---|---|---|---|---|
| E8b | Umbral de puntaje calibrado por estimador (YOLO26) | el umbral de puntaje de keypoint | sí, una noche | 1 | `01-e8b-umbral-calibrado.md` |
| E10 | Relleno temporal de keypoints bajo umbral, en vez de enmascarar | el relleno | sí, una noche | 1 | `02-e10-relleno-temporal.md` |
| E11 | Corte de tracks mixtos por cambio de color del guante, antes de `particionar` | el corte | no (desarrollo); poca (confirmación: pose y guantes de los videos nuevos) | 1 | `03-e11-tracks-mixtos.md` |
| E12 | El guante detectado como features del detector de golpes | 5 features de guante | sí, dos noches | 1b | `04-e12-guantes-en-detector.md` |
| E9 | RTMO-l (rtmlib, Apache-2.0) en lugar de YOLOv8l-pose | el estimador, con la calibración de E8b | sí | 2 | `05-e9-rtmo.md` |
| E13 | Tracker de licencia permisiva (OC-SORT, ByteTrack) en lugar de BoT-SORT | el tracker | sí, poco | 2 | `06-e13-tracker.md` |
| E14 | Identidad por propagación de máscaras de SAM 2 desde la siembra | el mecanismo de identidad | sí | 2 | `07-e14-sam2-identidad.md` |
| E15 | Segunda cámara lateral para el hook: protocolo y piloto | la vista lateral | no | 3 | `08-e15-segunda-camara.md` |

`09-integracion-y-tesis.md` dice cómo un resultado positivo llega a A y qué cambia en cada
capítulo según lo que salga. `10-pasos.md` es el registro de tareas.

## Fases y calendario

El calendario manda sobre el alcance. Congelamiento de funciones de A el 3-11 (contramedida comprometida en el Cap 6),
integración al documento del 10 al 16-11, entrega final el 17-11.

| Fase | Qué | Cuándo | Qué produce para la tesis |
|---|---|---|---|
| 0 | Pruebas y código de T1 a T12 (`10-pasos.md`), PR contra `main`. El commit de este spec es el pre-registro de E8b, E10, E11 y E12 | 30 y 31-10 | nada |
| 1 | E8b (noche del 31-10), E10 (noche del 1-11), E11 desarrollo (de día, sin GPU) | 31-10 al 2-11 | Cap 7 y Cap 8, resultados posteriores a la Entrega 7 |
| 1b | E12 (noches del 2 y 3-11) y E11 confirmación (cuando haya material anotado, L3) | hasta el 9-11 | ídem, si entra; si no, Cap 8 como línea con spec |
| 2 | E9, E13, E14 | después del 17-11 | Cap 8 como líneas futuras con pre-registro escrito |
| 3 | E15, con filmación | cuando Lucas consiga compañero y dos celulares | Cap 8 |

Si una etapa de la fase 1 no entra en su noche, pasa a la siguiente y empuja a las de atrás.
Lo que no termine el 9-11 va al Capítulo 8 como línea con spec y pre-registro, sin número. **No
se adelanta nada de la fase 2 a cambio**: tres experimentos medidos a medias valen menos que uno
cerrado.

## Qué no se toca

- **A** (producción, `127.0.0.1:8000`, Funnel): ni su código desacoplado en `1058d5e`, ni sus
  modelos, ni sus datos. Ningún resultado de este spec pasa a A antes del 17-11
  (`09-integracion-y-tesis.md` §1).
- **B** (académica) y **C** (gemelo YOLO26): sin cambios. C queda apagada como la dejó L5.
- **La sub-hipótesis 1.** Su corte es el 31-10 con lo medido hasta esa fecha. Si E8b o E10 dan
  un número de ocho clases después del corte, se informa como posterior y no cambia el veredicto.
- **Los umbrales pre-registrados** de C1 (0,95), C3, C5 (2×) y C8, el 0,80 del detector y el
  IoU 0,3 del emparejador de eventos. E8b cambia el umbral de puntaje **solo para YOLO26** y
  como condición nueva, no como reemplazo del 0,3 de A.
- **Las anotaciones originales y los cachés de A**: solo lectura, con sha256 antes y después,
  como en E8 (S3 y S14).

## Dónde vive cada cosa

| | Ruta |
|---|---|
| Código | rama `mejoras-fase0`, PR contra `main`. Cada experimento corre desde un worktree fijo `~/Proyectos/TwinBoxing-mejoras`, desacoplado en el commit de `main` posterior a L0 (S2) |
| Datos y corridas | `~/boxtwin-mejoras/<exp>/`, fuera de git |
| Cachés que se reusan | `~/boxtwin-gemelo/raiz-v8/` (v8 sin ReID) y `~/boxtwin-gemelo/raiz-y26/` de E8, solo lectura |
| Evidencia de guantes que se reusa | `~/boxtwin/mediciones/c1/<fuente>.evidencia.json` de C1, solo lectura |
| Resultados | `docs/experiments/<fecha>-<exp>-<slug>.md` y `docs/experiments/datos/` |
| Cambios de tesis | `docs/tesis/<fecha>-<exp>-cambios.md` |
| Orquestador | `scripts/mejoras/mejoras.sh`, con el mismo candado de `scripts/gemelo/guardas.py` |

E8b, E10 y E12 se apoyan en los cachés de E8 y no re-cachean nada: la base de los tres es la
condición base de E8 (v8 sin ReID, D-H2), que ya pasó K2 (0,637 del detector, 0,732 de E4). Así
cada experimento entrena una sola condición nueva y compara contra un número auditado, con un
control (K-reuso) que verifica que el código nuevo reproduce byte a byte los tensores de esa
base cuando la mejora está apagada.

## Prácticas del proyecto que aplica

| Práctica | Dónde |
|---|---|
| Pre-registro antes del primer resultado; cambios solo antes de la primera corrida y anotados como decisión | cada archivo 01 a 04, §"Reglas"; `10-pasos.md` Decisiones |
| Una sola variable por experimento | tabla de arriba |
| Mejora apagada = código viejo, comprobado byte a byte (K-reuso) | 01 §3, 02 §3, 04 §3 |
| Pruebas escritas antes del código; nada que dé un número sin test | T1 a T12 |
| Scripts en `scripts/mejoras/` con POR QUÉ EXISTE, QUÉ HACE y USO; reanudables | T9 |
| Ensamble de cinco semillas; regla de E8 (+0,03 y 5 de 6 folds del detector, 5 de 7 de E4) | 01 a 04 |
| Material de desarrollo separado del de confirmación cuando el diseño salió de mirar ese material | E11 (`03-e11-tracks-mixtos.md` §4) |
| Auditor independiente antes del texto | §"Auditoría" de cada experimento |
| No correr jobs largos sin avisar; candado contra la cola de A | L1, `mejoras.sh` |
| Resultados negativos documentados igual | cada §"Qué decide cada resultado" |

## Criterios de aceptación de la campaña

Se leen al cierre de la fase 1b (9-11), en `10-pasos.md`:

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CM1 | Cada experimento de las fases 1 y 1b corrió bajo su pre-registro, o quedó como no realizado con el motivo | commit de este spec anterior a la primera corrida de cada uno; auditoría sin fallas o con las fallas en el documento |
| CM2 | A, B y C intactas | huellas de A y B iguales antes y después (los scripts de E8, `despliegue/academico/huella-A.sh` y `despliegue/gemelo/huella-B.sh`), salvo la cantidad de sesiones de A; C `inactive` |
| CM3 | Originales intactos | sha256 de las siete anotaciones, `meta.json` y `pose.npz` de A, y de los cachés de `raiz-v8` y `raiz-y26`, iguales antes y después |
| CM4 | Cada mejora apagada reproduce su base | K-reuso de E8b, E10 y E12 pasa: tensores con sha256 igual a los de E8 |
| CM5 | Cada número que entra a la tesis tiene documento en `docs/experiments/` | un documento por experimento, con su JSON |
| CM6 | Nada de esto cambia A antes del 17-11 | HEAD de A `1058d5e` el 17-11 |

## Lo que este spec no hace

- No reabre la sub-hipótesis 1, C1, C3 ni C5 de A.
- No hace fine-tuning de ningún estimador de pose sobre boxeo.
- No reentrena el detector de guantes (`guantes-v2.pt`). E12 lo usa como está.
- No cambia la arquitectura del detector (TCN, campo receptivo 63) ni del clasificador E4,
  salvo la dimensión de entrada en E12.
- No elimina la dependencia de Ultralytics del producto. E9 y E13 miden si las piezas
  permisivas alcanzan; reemplazar la cadena entera (incluido el detector de guantes, entrenado
  con Ultralytics) es otra campaña, ver `05-e9-rtmo.md` §8.
- No mide nada en vivo (modo LIVE).
