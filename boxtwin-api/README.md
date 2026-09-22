# boxtwin-api

La API y el worker del producto. Un entrenador sube el video del sparring, marca cuál de los
dos peleadores es cuál, y recibe la Fight-Card con cada evento enlazado al instante del video.

Reporta **golpes detectados**, nunca golpes lanzados, con la precisión y el recall medidos del
detector a la vista. No reporta conexión, puntuación de round ni veredicto: un sistema
monocular no establece contacto físico.

## Por qué la cola vive en una tabla

Un trabajo dura entre minutos y horas y corre en la única GPU que hay. La API no puede
esperarlo, así que subir devuelve un identificador y el estado se consulta.

La exclusión entre workers se resuelve con `SELECT ... FOR UPDATE SKIP LOCKED`, que es una
línea de SQL, en lugar de Redis y Celery, que son dos piezas para instalar, monitorear y
justificar. Para un nodo y pocos usuarios alcanza, y hay algo que se gana: encolar el trabajo
y crear la sesión pasan en el mismo commit, así que no existe el estado "hay sesión y no hay
trabajo".

`sqlite` no tiene `SKIP LOCKED`. Ahí la exclusión sale de un `UPDATE` condicional, que con un
solo proceso de worker —desarrollo y tests— alcanza. Los dos casos están en `cola.py` y se
distinguen en un solo lugar.

## El corte en dos etapas

No es una decisión de ingeniería. Entre la pose y el detector hay una pregunta que sólo una
persona puede contestar: cuál de los dos cuerpos es cuál. Medido, la identidad automática
resuelve el 82,4% de los tracks sola y el 99,1% con los perfiles sembrados desde dos tracks de
rol conocido. Toda esa diferencia es la siembra, y cuesta dos clicks.

```
POST /videos            → sesión + trabajo "procesar"        (RF1)
   worker: boxtwin procesar                                   → espera_siembra
GET  /jobs/{id}         → dos candidatos con su recorte       (RF3)
POST /jobs/{id}/siembra → trabajo "completar"                 (RF4)
   worker: boxtwin completar                                  → listo, fightcard.json
   worker: clasificar_sesion.py + boxtwin tipos               → tipo estimado  (F9)
GET  /fightcards/{id}
```

## Por qué el worker llama a subprocesos

Por los dos entornos conda, que no se mezclan: el detector corre con torch 2.6 y el
clasificador con torch 2.1 y mmcv pinneado contra CUDA 11.8. Entre ellos va un JSON, que es
como ya funciona el pipeline medido.

El efecto colateral es bueno: una etapa que se cae por memoria de GPU se lleva su proceso y no
el servidor.

Si la imagen con los dos entornos no sale a tiempo, la clasificación corre afuera del
contenedor leyendo el mismo JSON. Es una variable de entorno, no un cambio de código:
`BOXTWIN_CMD_CLASIFICADOR`. Sin ella, el tipo de golpe queda sin estimar y el resto de la
Fight-Card no depende de él.

## Correr

```bash
export BOXTWIN_DATOS=/datos/boxtwin
export BOXTWIN_SECRETO=$(openssl rand -hex 32)
export BOXTWIN_INVITACION=un-codigo
export BOXTWIN_WEB=../boxtwin-web/dist
export BOXTWIN_MODELO_POSE=../yolov8l-pose.pt
export BOXTWIN_MODELO_GUANTES=../boxtwin-guantes/modelos/guantes-v2.pt
export BOXTWIN_MODELO_DETECTOR=../modelos/detector_7fuentes.ens.pt

uvicorn boxtwin_api.app:app --host 127.0.0.1 --port 8000
python -m boxtwin_api.worker
```

Tres variables que no son opcionales en cuanto esto sale a internet, y `GET /salud`
reporta las tres:

- Sin `BOXTWIN_SECRETO` el servidor genera uno al arrancar y todas las sesiones abiertas
  se caen en cada reinicio.
- Sin `BOXTWIN_INVITACION` el **registro queda cerrado**, que es el default a propósito:
  una instancia abierta detrás de un túnel es una GPU ajena gratis para cualquiera que
  tenga la URL. Con un código, hace falta ese código; con `abierto`, cualquiera.
- Con `BOXTWIN_WEB` apuntando al frontend construido, la API lo sirve y queda **un solo
  origen**: alcanza con exponer este puerto y no hace falta CORS.

`/docs`, `/redoc` y `/openapi.json` están **apagados por default**, por el mismo criterio
que el registro: no filtran datos, pero publican la lista de endpoints a cualquiera que
tenga la URL. Para desarrollar, `BOXTWIN_DOCS=1`.

Sin `BOXTWIN_DB` usa sqlite, que alcanza mientras haya un solo worker.

Para ponerlo online: [`despliegue/README.md`](../despliegue/README.md).

## Endpoints

| | |
|---|---|
| `POST /auth/registro`, `POST /auth/login` | cuenta simple, email y clave |
| `POST /videos` | multipart, devuelve `{job_id}` sin esperar |
| `GET /jobs`, `GET /jobs/{id}` | historial y estado, con candidatos de siembra |
| `GET /jobs/{id}/candidatos/{track}` | el recorte |
| `POST /jobs/{id}/siembra` | `{track_a, track_b}` |
| `GET /fightcards/{id}` | el documento |
| `GET /fightcards/{id}/export?formato=csv\|pdf` | |
| `PATCH /fightcards/{id}/golpes/{golpe}` | corrección de tipo |
| `GET /videos/{id}/stream` | con rangos, para saltar a un evento |

La corrección va anidada bajo la sesión y no en `/golpes/{id}` como proponía la spec: el id de
un golpe sale de su carril y su cuadro, y sólo es único adentro de su sesión. Un id global
pediría un contador, y un contador se rompe al reclasificar.

El PDF es la misma página HTML con estilos de impresión. Generar PDF pediría reportlab o un
Chromium adentro del contenedor; para un documento de una carilla, la impresión del navegador
da el mismo resultado y cuesta cero. Está dicho en la página.

## Lo que la API no hace

No toca modelos ni video: eso es del worker. No guarda la Fight-Card en la base, que es un
archivo del directorio de la sesión y lo escribe el worker; la base guarda el estado, para
poder listar sin tocar disco, y las correcciones, que son un dataset.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

58 tests. El bloque que más importa es el de aislamiento entre usuarios (RNF3): no se verifica
leyendo el código sino pidiendo cada recurso con el token del otro.
