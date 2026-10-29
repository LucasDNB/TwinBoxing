# C · Gemelo local de BoxTwin con YOLO26l-pose

27-10-2026 · spec SDD · nada lanzado todavía · registro en `05-pasos.md`

Pedido de Lucas (27-10): un gemelo local de BoxTwin que use YOLO26 en lugar de YOLOv8 y siga
las prácticas del proyecto. Respuestas a las cuatro preguntas: corre en la estación, alcance
local, sobre las últimas versiones, YOLO26-pose tamaño L. Cómo se leyó cada una está en
`05-pasos.md`, decisiones D1 a D5.

La estación pasa a tener tres BoxTwin:

| | A · producción | B · académica | C · gemelo |
|---|---|---|---|
| Qué es | BoxTwin con las versiones | A con el modo académico E7 | A con YOLO26l-pose y el detector y el clasificador reentrenados sobre esas poses |
| Código | copia principal, desacoplada en `1058d5e` (27-10) | worktree `~/Proyectos/TwinBoxing-academico`, en `efcec23` | worktree fijo `~/Proyectos/TwinBoxing-gemelo`, desacoplado en un commit congelado de `main` posterior a la fase 0 (S5) |
| API | `127.0.0.1:8000` | `127.0.0.1:8001` | `127.0.0.1:8002` |
| Afuera | Funnel público, `:443` | solo la tailnet, `:8443` | **nada**: ni Funnel ni `tailscale serve`. Desde la notebook, túnel SSH |
| Datos y base | `~/boxtwin/datos/` | `~/boxtwin-academico/datos/` | `~/boxtwin-gemelo/datos/`, base nueva |
| Pose | `yolov8l-pose.pt` | igual que A | `yolo26l-pose.pt` |
| Detector de golpes | `detector_7fuentes.ens.pt` | igual que A | `detector_7fuentes_y26.ens.pt`, de E8 |
| Tipo de golpe | E4 (`geometrico`), E4a en bolsa y sombra | E7 (`e7`), E4a | E4 y E4a reentrenados sobre YOLO26 (`clasificador_geo_y26_v1`, `_sin_rival`) |

Archivos:

- `01-percepcion.md`: qué cambia en la pose y qué queda fijo, el humo, el re-caché de las siete
  fuentes, el puente de identidad y los cambios de código T1 a T10.
- `02-preregistro-e8.md`: E8, YOLO26l-pose contra YOLOv8l-pose con todo lo demás fijo. Qué se
  mide, controles, reglas de lectura y qué decide cada resultado.
- `03-instancia-c.md`: el despliegue de C. Worktree, entorno, unidades, web, modelos, cuentas
  y acceso local.
- `04-convivencia.md`: qué no comparten A, B y C, la GPU, los datos y la vuelta atrás.
- `05-pasos.md`: el registro de tareas, quién hace cada una, dependencias y evidencia.

## Fases

| Fase | Qué | GPU | Cuándo |
|---|---|---|---|
| 0 | Pruebas y código de T1 a T10 (`01-percepcion.md`), PR contra `main`. El commit de este spec es el pre-registro de E8 | no | 28-10 |
| 1 | Humo H1 a H3, re-caché de las siete fuentes con YOLO26, puente de identidad, controles, E8, auditoría y documento del experimento | sí, de noche y con OK de Lucas | desde la primera noche después de cerrar las sesiones de prueba de esta semana |
| 2 | Modelos finales, despliegue de C, criterios CG, una sesión propia de punta a punta | sí, poco | hasta el 3-11 |

La fase 2 va con cualquier resultado de E8. C es la comparación, no un candidato a producción:
si E8 sale peor, C muestra sus números igual. Llevar YOLO26 a A se decide **después del 17-11**,
con PR propio y C5 medido otra vez. Antes no, por el congelamiento del 3-11.

## Prácticas del proyecto que aplica

| Práctica | Dónde |
|---|---|
| SDD con reparto: el SDD sin `sudo` (S), Lucas con `sudo` o Tailscale (L) | `05-pasos.md` |
| Pre-registro antes del primer resultado; cambios solo antes de H1 y anotados como decisión | `02-preregistro-e8.md` |
| Una sola variable por experimento: el checkpoint de pose | `01-percepcion.md` §1 |
| Commit congelado en un worktree desacoplado; moverlo es repetir los pasos | `03-instancia-c.md` |
| Aislamiento comprobado con huella antes y después | CG1 y CG4 |
| Pruebas escritas antes del código; nada que dé un número sin test | T1 a T10 |
| Scripts versionados en `scripts/`, con POR QUÉ EXISTE, QUÉ HACE y USO; reanudables | T6 a T8 |
| Anotaciones originales y cachés de A solo lectura; todo output a archivo aparte | `01-percepcion.md` §3 y §4 |
| Pesos y datos pesados fuera de git, con sha256 en un `meta.json` | `03-instancia-c.md` §5 |
| Ensamble de cinco semillas; una corrida sola no dice nada | `02-preregistro-e8.md` §4 |
| Auditor independiente antes del texto | `02-preregistro-e8.md` §7 |
| No correr jobs largos sin avisar | L1 |
| Resultados negativos documentados igual | `02-preregistro-e8.md` §5 |

## Criterios de aceptación (escritos antes de lanzar)

Se leen al cierre, en `05-pasos.md`, uno por uno:

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CG1 | A y B quedan idénticas | Huellas de A y de B tomadas en S2 iguales a las de S24, salvo la cantidad de sesiones de A, que puede crecer por su uso |
| CG2 | C es lo que dice ser | `127.0.0.1:8002/salud`: `commit` el congelado en S5, `modelo_pose.archivo` `yolo26l-pose.pt` con el sha256 anotado en S4, `clasificador` `geometrico`, `modelo_individual` `true`, `frontend` `true` |
| CG3 | C es local | `ss -ltn`: `:8002` solo en `127.0.0.1`; `tailscale serve status` sin ninguna entrada hacia `:8002`; el `:443` y el `:8443` como en S2 |
| CG4 | C no escribe en A ni en B | Conteo de sesiones de A y de B sin cambios por usar C; sha256 de las siete anotaciones, `meta.json` y `pose.npz` originales igual antes y después de la fase 1 (S3 y S14) |
| CG5 | Procedencia honesta | Cada Fight-Card de C dice el modelo de pose con su sha256 y muestra la precisión y el recall del detector y la exactitud del clasificador que salen de E8. Ningún número medido con YOLOv8 aparece en C |
| CG6 | E8 corrió bajo su pre-registro | El commit de este spec es anterior al primer resultado (H1); la auditoría (S16) sin fallas, o con las fallas declaradas en el documento |
| CG7 | Punta a punta en C | Un sparring propio y una bolsa propia llegan a Fight-Card con la siembra como única intervención |
| CG8 | La GPU alcanza y no se pisa | Pico de VRAM de C procesando sola ≤ 7,5 GB (`nvidia-smi`, como CA7 de `dos-instancias`); el log del candado de `e8.sh` muestra que ninguna etapa de la fase 1 corrió con trabajos de A en cola o tomados |
| CG9 | Reproducible | Cada modelo de C tiene `meta.json` con commit, sha256 de sus entradas y el comando que lo armó; volver a correr el comando da el mismo sha256 en los modelos entrenados en CPU |

## Lo que este spec no hace

- No toca A ni B, ni sus entornos, unidades, datos, modelos o webs.
- No reabre la sub-hipótesis 1: su corte es el 31-10 y E8 no la rescata, salga lo que salga.
- No vuelve a medir C1, C3 ni C5 de A. Lo que E8 mide sobre identidad y tiempo es de C.
- No publica C, ni en el Funnel ni en la tailnet.
- No reentrena el detector de guantes ni cambia ningún umbral pre-registrado: 0,80 del detector,
  0,3 de puntaje de keypoint, piso de color 0,55, IoU 0,3 del emparejador de eventos.
- No lleva E7 a YOLO26. Sería otra campaña (elevación HMR2 sobre las cajas nuevas y reentreno con
  SMPL, solo investigación). Queda como fase 3 si Lucas la pide.
- No hace fine-tuning de YOLO26 sobre boxeo: se usan los pesos de COCO, igual que con v8.
