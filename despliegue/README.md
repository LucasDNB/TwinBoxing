# Poner el MVP online

Dos cosas distintas se llaman "deploy" en este proyecto y conviene no mezclarlas:

- **Servir el MVP a usuarios reales** (C4, C6). La GPU se queda en `desarrollo-lucas` y la
  app sale por un túnel. Es lo que declara la spec en §6: *HTTPS (túnel hacia la estación de
  desarrollo)*, y es lo que documenta este archivo.
- **El PoC de nube** (§6.3 de CLAUDE.md, Cap 5). El worker en una instancia GPU alquilada,
  presupuesto < 20 USD, para medir throughput y costo por hora de video. Eso no sirve para
  tener usuarios: sirve para un número de la tesis. Para eso están los `Dockerfile`.

## Por qué systemd y no Docker para el primer deploy

La imagen de la API es trivial, pero la del worker lleva los dos entornos conda y mmcv
compilado contra CUDA 11.8, que es donde el riesgo 3 de la spec dice que se van los días. Y
el worker necesita la GPU y los pesos montados igual, así que el contenedor no le ahorra
nada al primer despliegue.

Corriendo directo en `twinboxing_env` con dos units estás online hoy. El Docker queda para
cuando haga falta de verdad, que es el PoC de nube.

Desde el 20-10 la imagen del worker ya no lleva mmcv: un solo entorno con el clasificador
geométrico, que es lo que corre la estación (ver la sección de la imagen, abajo). La razón
de arriba para no usar Docker en la estación sigue en pie: no le ahorra nada.

## Los pasos

```bash
cd ~/Proyectos/TwinBoxing

# 1. el frontend construido, que la API va a servir. Desde el 13-10 es la web entrenador del
#    prototipo (BOXTWIN_WEB apunta a su out/); boxtwin-web quedo como la web vieja del MVP.
#    npm ci y no npm install: instala exactamente lo del package-lock (three va fijado).
cd prototipo/boxtwin-web-entrenador && npm ci && npm run build && npm run presupuesto && cd ../..

# 2. el paquete de la API en el entorno
#    Las dependencias (fastapi, sqlalchemy, uvicorn, python-multipart, email-validator)
#    YA estan instaladas en twinboxing_env desde el 22-09, y no movieron nada: pydantic
#    quedo en 2.13.4 y torch, numpy, opencv y ultralytics no se tocaron.
#    Falta solo el paquete, y recien se puede una vez que esta rama este en el checkout
#    principal: un editable apuntando a un worktree se rompe cuando el worktree se borra.
conda run -n twinboxing_env pip install -e ./boxtwin-api

#    boxtwin-annotator y boxtwin-detector NO hay que reinstalarlos: ya estan editables
#    contra este mismo directorio, asi que el merge les alcanza.

# 3. el entorno de los servicios
mkdir -p ~/boxtwin/datos
sudo install -m 600 -o lucasb -g lucasb despliegue/boxtwin.env.ejemplo /etc/boxtwin.env
sudo -u lucasb editor /etc/boxtwin.env     # BOXTWIN_SECRETO y BOXTWIN_INVITACION

# 4. los servicios
sudo cp despliegue/systemd/boxtwin-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now boxtwin-api boxtwin-worker

#    Y DESPUES DE CADA MERGE, los dos. El paquete es editable, asi que el codigo nuevo
#    esta en disco al instante, pero el proceso sigue con el que importo al arrancar.
#    El frontend NO necesita restart -se sirve de disco- pero si npm ci y npm run build.
#    Mirar antes que la cola este vacia: el worker espera hasta una hora a que termine
#    el trabajo en curso (TimeoutStopSec), pero conviene no reiniciarlo a mitad.
sudo systemctl restart boxtwin-api boxtwin-worker

#    Si el merge cambio una regla que vive en la Fight-Card (combinaciones, altura, rounds),
#    las sesiones viejas siguen con la anterior hasta recalcularlas. No corre pose ni GPU,
#    reescribe fightcard.json y vuelve a aplicar las correcciones del entrenador encima.
#    Antes, un respaldo: es la unica copia de las sesiones.
cp -a ~/boxtwin/datos/sesiones ~/boxtwin/respaldos/sesiones-antes-de-$(date +%F)
boxtwin-annotator enriquecer --todas ~/boxtwin/datos/sesiones

#    Las columnas nuevas de la base las agrega la API sola al arrancar (SQLite). En
#    PostgreSQL, las migraciones de despliegue/migraciones/ en orden de fecha.

# 5. verificar ANTES de exponer nada
curl -s localhost:8000/salud
```

`/salud` tiene que devolver esto:

```json
{"ok": true, "secreto_efimero": false, "registro": "invitacion", "frontend": true,
 "arrancado": "2026-09-22T18:40:00+00:00", "commit": "b28ab63"}
```

**Mirá `arrancado` y `commit` después de cada deploy.** Un servicio que quedó corriendo
código viejo no se distingue de uno al día mirando la app: las rutas que todavía no tiene
contesta **405**, porque las atiende el montaje del frontend, que sólo acepta GET. Eso se
lee como un bug del método y no como lo que es. Costó dos rondas de diagnóstico el 22-09.

```bash
# el commit que corre el servicio contra el que esta en disco
curl -s localhost:8000/salud | python3 -c "import sys,json; print(json.load(sys.stdin)['commit'])"
git -C ~/Proyectos/TwinBoxing rev-parse --short HEAD
```

Si `secreto_efimero` es `true`, faltó `BOXTWIN_SECRETO` y cada reinicio va a desloguear a
todo el mundo. Si `registro` dice `abierto`, cualquiera con la URL se crea una cuenta.

Las units se pueden validar sin arrancarlas:

```bash
systemd-analyze verify despliegue/systemd/boxtwin-*.service
```

Las dos validan desde el 22-09, con `uvicorn` ya instalado. Si falta `/etc/boxtwin.env`,
el servicio no arranca en vez de arrancar sin configuración, que es lo que se quiere: media
configuración es peor que ninguna.

Verificado que `twinboxing_env` sirve la app completa: un solo origen, el video entero por
la API y los 46 eventos de la sesión real. Lo único que todavía corre por `PYTHONPATH` en
vez de por el paquete instalado es `boxtwin_api`, por lo del paso 2.

```bash
# 6. el túnel
tailscale funnel --bg 8000
tailscale funnel status
```

Tailscale ya está instalado y logueado en `desarrollo-lucas`. La primera vez, Funnel hay que
habilitarlo en la consola de administración del tailnet: el comando imprime el link. Da una
URL `https://desarrollo-lucas.<tailnet>.ts.net` con HTTPS, sin abrir puertos en el router y
sin depender de tener IP pública.

Para bajarlo: `tailscale funnel --bg off`.

## Lo que se hizo el 14-10 (para tenerlo de referencia)

Deploy del PR #10 sobre la estacion: checkout de produccion de `claude/ampliacion-13-10` a
`main`, `npm ci` y build de la web entrenador, respaldo de las sesiones, `enriquecer --todas`
(6 sesiones, 0 errores; una sin Fight-Card no se toca), restart y verificacion por `/salud`
(`commit` = el del merge) y `/producto` (banderas nuevas). Antes de reiniciar se miro que la
cola estuviera vacia y que el worker que corria fuera el del servicio: en la estacion corren
tambien workers de las pruebas e2e, con otra `BOXTWIN_DB`, y no hay que confundirlos.

No hay respaldo automatico de `~/boxtwin/datos` ni copia de `/etc/boxtwin.env`: si se
pierde el archivo, `BOXTWIN_SECRETO` y la invitacion se vuelven a cargar a mano.

## Tipo v2 y bolsa y sombra (16-10)

Dos scripts, los dos se corren en la estación como `lucasb` desde la raíz del repo y con la
cola vacía. Hacen copia de `/etc/boxtwin.env` antes de tocarlo.

1. `bash despliegue/encender_v2.sh` enciende el tipo v2. Entrena en CPU lo que falte (el
   modelo final de E4 para sparring y combate, y el de E4a, sin las features del rival, para
   bolsa y sombra), corre las pruebas de equivalencia contra los datos reales, escribe
   `BOXTWIN_CLASIFICADOR=geometrico` y las dos rutas de modelo, reinicia y verifica `/salud`
   (`clasificador: geometrico` y `modelo_individual: true`). `--apagar` vuelve atrás.
2. `bash despliegue/medir_bolsa_sombra.sh` es la medición de B8, que es lo único que falta
   para habilitar bolsa y sombra en la web. `procesar` y `completar` corren la misma cadena
   que el worker sobre un video propio de al menos 3 minutos; `completar` deja el video
   procesado y la planilla de saltos de 2 s. Mirar y llenar la planilla es de Lucas.
   `resultado` da el veredicto con el umbral pre-registrado (95% en cada video) y `encender`
   pone `BOXTWIN_SESIONES_INDIVIDUALES=1` solo si pasaron los dos.

Por qué un modelo aparte para bolsa y sombra: E4 lleva 6 features que ubican la muñeca
respecto del torso del rival. Con una sola persona en cuadro entrarían en cero todo el golpe,
algo que el modelo casi no vio al entrenar. E4a es la misma receta sin esas 6 columnas y ya
estaba medido con una fuente afuera: familia 0,724 contra 0,727 de E4. Las 6 del rival llevan
la altura, no la familia (`docs/experiments/2026-10-16-clasificador-geometrico.md` §5). Ninguno
de los dos se midió sobre bolsa o sombra: el número que muestra la Fight-Card es el de sparring,
y la web ya lo rotula así en una sesión individual (CA-19.3). La altura no aplica: la regla de
C8 necesita al rival y devuelve `sin_rival`.

## La imagen del worker (PoC de nube)

Un solo entorno, `twinboxing_env`, con las versiones de la estación y el clasificador
geométrico. Sin OpenMMLab: PoseConv3D queda afuera del contenedor (decisión del 20-10). No
reemplaza a las units de systemd en la estación; es la imagen del PoC de §6.3.

```bash
# 1. una sola vez: que Docker vea la GPU (requiere sudo). Sin esto `--gpus` falla.
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtimes            # tiene que aparecer nvidia

# 2. /modelos con los nombres que espera la imagen. Enlaces duros: mismo disco, no ocupan
#    lugar, y a diferencia de un symlink se ven desde adentro del contenedor.
cd ~/Proyectos/TwinBoxing
ln -f yolov8l-pose.pt                         ~/boxtwin/modelos/yolov8l-pose.pt
ln -f boxtwin-guantes/modelos/guantes-v2.pt   ~/boxtwin/modelos/guantes-v2.pt
ln -f modelos/detector_7fuentes.ens.pt        ~/boxtwin/modelos/detector_7fuentes.ens.pt
#    clasificador_geo_v1 y clasificador_geo_v1_sin_rival ya estan en ~/boxtwin/modelos

# 3. construir. El contexto es la raiz del repo; .dockerignore deja entrar solo los tres
#    paquetes, no los 27 GB de videos y caches.
docker build -f despliegue/Dockerfile.worker -t boxtwin-worker .

# 4. que la imagen vea la GPU y los modelos, sin tocar la base ni las sesiones
docker run --rm --gpus all -v ~/boxtwin/modelos:/modelos:ro boxtwin-worker \
    /opt/conda/envs/twinboxing_env/bin/python -c \
    "import torch,os; print(torch.cuda.is_available(), sorted(os.listdir('/modelos')))"
```

Con compose, `BOXTWIN_MODELOS=~/boxtwin/modelos`.

## El problema real no es el túnel, es la subida

Un video de 30 minutos de celular son unos 3 GB. Funnel es best-effort y no está pensado
para mover gigabytes, así que sobre el wifi de un gimnasio eso va a ser lento y se va a
cortar. Tres salidas, en el orden en que conviene atacarlas:

| | Costo | Qué resuelve |
|---|---|---|
| Pedir **un round por vez** | cero | 2-3 min ≈ 300 MB, y es como un entrenador revisa igual |
| **Transcodificar en el celular** antes de subir | medio | Es el experimento 4 de §6. Ojo: el preproceso corre a `imgsz=640`, así que el 1080p no le aporta nada a la pose; sí puede aportarle al color del guante, y eso hay que medirlo antes de decidirlo |
| **Subida por trozos**, reanudable | medio | Que no haya que empezar de cero cuando se corta, que sobre 4G va a pasar |

Nada de esto está implementado. Para las sesiones de prueba de C6, la primera fila alcanza.

**Medí una subida real antes de la sesión con Suárez.** Si un round tarda diez minutos en
subir, eso cambia el guion de la prueba y es mejor saberlo antes que adelante del
entrenador.

## Antes de abrir el túnel

- [ ] `BOXTWIN_SECRETO` puesto. `/salud` lo dice.
- [ ] `BOXTWIN_INVITACION` con un código, no en `abierto`.
- [ ] `BOXTWIN_DOCS` sin poner, para que `/docs` no quede público. `/salud` lo dice.
- [ ] Disco: cada sesión deja el video más el cache de pose. No se borra solo.
- [ ] La estación tiene que estar despierta. Revisar que no se suspenda sola.
- [ ] La GPU es la misma con la que trabajás. Un entrenamiento tuyo y una sesión de un
      usuario al mismo tiempo se pelean por 8 GB de VRAM.

## Mirar qué está pasando

```bash
journalctl -u boxtwin-worker -f        # las etapas, con su linea de comando
journalctl -u boxtwin-api -f
systemctl status boxtwin-api boxtwin-worker
```

Cada sesión deja además su propio registro en `~/boxtwin/datos/sesiones/<id>/sesion.json`:
estado, tiempos por etapa y avisos. Es la fuente de verdad de lo que pasó adentro del
procesamiento, y es lo que hay que mirar cuando una sesión queda rara.

## La alternativa sin exponer nada

Si los que prueban son pocos y conocidos, se los puede invitar al tailnet en vez de abrir
Funnel: instalan Tailscale, aceptan la invitación y entran a `http://desarrollo-lucas:8000`
directo, a velocidad de red local y sin nada público.

Es más rápido y más privado, y tiene un costo que para C6 importa: instalar una VPN no es
parte del flujo que se está probando. Si lo que se quiere medir es si un entrenador entiende
la Fight-Card, mejor Funnel; si lo que se quiere es subir un video de 30 minutos sin sufrir,
mejor el tailnet.
