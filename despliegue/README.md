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

## Los pasos

```bash
cd ~/Proyectos/TwinBoxing

# 1. el frontend construido, que la API va a servir
cd boxtwin-web && npm install && npm run build && cd ..

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

# 5. verificar ANTES de exponer nada
curl -s localhost:8000/salud
```

`/salud` tiene que devolver las tres cosas bien:

```json
{"ok": true, "secreto_efimero": false, "registro": "invitacion", "frontend": true}
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
