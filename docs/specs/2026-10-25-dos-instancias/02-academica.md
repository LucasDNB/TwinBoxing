# B · Instancia académica, con E7

## Código

Un worktree **fijo**, fuera de `.claude/worktrees/`, porque esos los borra la limpieza de
sesiones y B quedaría apuntando a la nada:

```
git -C ~/Proyectos/TwinBoxing worktree add ~/Proyectos/TwinBoxing-academico e7-modo-academico
```

Queda en la punta de la rama al momento de lanzar. El hash se anota en `04-pasos.md` y no se
mueve durante la demo. Cambiarlo es repetir los pasos 3 y 7.

`twinboxing_env` resuelve `boxtwin` y `boxtwin_api` hacia la copia principal (editables). B no
reinstala nada: antepone su código con `PYTHONPATH`, que gana sobre las instalaciones editables
(comprobado el 25-10 al correr las pruebas de la rama). Los subprocesos del worker heredan el
entorno (`worker.py`: `env={**os.environ, ...}`), así que el `boxtwin-annotator` que llama
también usa el código de B.

`/salud` lee el commit de `<repo>/.git/HEAD`. En un worktree `.git` es un archivo, así que hoy B
informaría `commit: null`. **Cambio chico pendiente (T1):** que `_commit()` siga el `gitdir:`
del archivo `.git`. Va con su prueba, en la rama de B, y A no lo ve.

## Web

Build propio, con el código de B, en su `out/`:

```
cd ~/Proyectos/TwinBoxing-academico/prototipo/boxtwin-web-entrenador && npm ci && npm run build && npm run presupuesto
```

`npm ci` instala sus propios `node_modules`, así que B no comparte ni toca los de A.

## Entorno · `/etc/boxtwin-academico.env` (permisos 600, dueño lucasb)

```
# B · instancia académica (docs/specs/2026-10-25-dos-instancias). Nunca para usuarios.
BOXTWIN_SECRETO=<nuevo, distinto del de A: python -c "import secrets; print(secrets.token_hex(32))">
BOXTWIN_INVITACION=<nueva, distinta de la de A>
BOXTWIN_DATOS=/home/lucasb/boxtwin-academico/datos
BOXTWIN_DB=sqlite:////home/lucasb/boxtwin-academico/datos/boxtwin.db
BOXTWIN_WEB=/home/lucasb/Proyectos/TwinBoxing-academico/prototipo/boxtwin-web-entrenador/out
PYTHONPATH=/home/lucasb/Proyectos/TwinBoxing-academico/boxtwin-annotator/src:/home/lucasb/Proyectos/TwinBoxing-academico/boxtwin-api/src

# modelos: los mismos archivos que A, solo lectura
BOXTWIN_MODELO_POSE=/home/lucasb/Proyectos/TwinBoxing/yolov8l-pose.pt
BOXTWIN_MODELO_GUANTES=/home/lucasb/Proyectos/TwinBoxing/boxtwin-guantes/modelos/guantes-v2.pt
BOXTWIN_MODELO_DETECTOR=/home/lucasb/Proyectos/TwinBoxing/modelos/detector_7fuentes.ens.pt
BOXTWIN_CMD=/home/lucasb/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator
BOXTWIN_DEVICE=0
BOXTWIN_UMBRAL=0.80

# las mismas decisiones de producto que A
BOXTWIN_C8=rechazado
BOXTWIN_C8_FECHA=2026-10-13
BOXTWIN_C3=rechazado
BOXTWIN_C3_FECHA=2026-10-20
BOXTWIN_SESIONES_INDIVIDUALES=1
BOXTWIN_MODELO_CLASIFICADOR_GEO=/home/lucasb/boxtwin/modelos/clasificador_geo_v1
BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL=/home/lucasb/boxtwin/modelos/clasificador_geo_v1_sin_rival

# lo único distinto: el modo académico E7
BOXTWIN_CLASIFICADOR=e7
BOXTWIN_MODELO_CLASIFICADOR_E7=/home/lucasb/boxtwin/modelos/clasificador_e7_v1
BOXTWIN_CMD_HMR2=/home/lucasb/miniforge3/envs/boxtwin_4dhumans/bin/python /home/lucasb/Proyectos/TwinBoxing-academico/boxtwin-annotator/tools/elevar_hmr2_sesion.py
```

Las rutas de los modelos de A se leen, no se copian: son los mismos archivos que usa A, con su
sha256 verificado al cargar.

## Unidades · `/etc/systemd/system/boxtwin-academico-{api,worker}.service`

Copias de las de A con cuatro diferencias: nombre, `WorkingDirectory`, `EnvironmentFile` y
puerto. **No se habilitan al arrancar** (`enable` no): se encienden para la demo y se apagan
después.

```
[Unit]
Description=BoxTwin académica (E7, modo académico) · API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lucasb
Group=lucasb
WorkingDirectory=/home/lucasb/Proyectos/TwinBoxing-academico
EnvironmentFile=/etc/boxtwin-academico.env
ExecStart=/home/lucasb/miniforge3/envs/twinboxing_env/bin/uvicorn boxtwin_api.app:app \
          --host 127.0.0.1 --port 8001 --proxy-headers
Restart=on-failure
RestartSec=3
NoNewPrivileges=yes
```

```
[Unit]
Description=BoxTwin académica (E7, modo académico) · worker
After=network-online.target boxtwin-academico-api.service

[Service]
Type=simple
User=lucasb
Group=lucasb
WorkingDirectory=/home/lucasb/Proyectos/TwinBoxing-academico
EnvironmentFile=/etc/boxtwin-academico.env
ExecStart=/home/lucasb/miniforge3/envs/twinboxing_env/bin/python -m boxtwin_api.worker
Restart=on-failure
RestartSec=5
TimeoutStopSec=3600
KillSignal=SIGTERM
NoNewPrivileges=yes
```

`Restart=on-failure` y no `always`: si alguien la para a mano, queda parada.

## Afuera

Solo la tailnet, en otro puerto HTTPS y **sin Funnel**:

```
tailscale serve --bg --https=8443 http://127.0.0.1:8001
```

Se abre desde un dispositivo de la tailnet de Lucas en `https://desarrollo-lucas.tail52f6aa.ts.net:8443`.
Para la presentación: la laptop o el celular de Lucas dentro de la tailnet. B no se publica
con Funnel en ningún caso: la enmienda la prohíbe para usuarios y SMPL no tiene licencia
comercial.

## Cómo se comprueba que B es lo que dice

- `curl -s 127.0.0.1:8001/salud`: `clasificador` `e7`, `modelo_e7` `true`, `modelo_individual`
  `true`, `commit` el de la rama (después de T1).
- Una sesión de sparring propia, procesada de punta a punta: `pose3d_hmr2.npz` en su carpeta, y
  `fightcard.json` con `clasificador.metodo` `geometrico-e7` y la `licencia`. En la web, la nota
  "Tipo en modo académico, no apto para uso comercial".
- Una sesión de bolsa: clasifica con E4a, sin `pose3d_hmr2.npz`.
- `ls ~/boxtwin/datos/sesiones | wc -l` igual antes y después: B no escribió en A.
