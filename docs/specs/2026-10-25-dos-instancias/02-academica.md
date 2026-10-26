# B · Instancia académica, con E7

## Código

Un worktree **fijo**, fuera de `.claude/worktrees/`, porque esos los borra la limpieza de
sesiones y B quedaría apuntando a la nada:

```
git -C ~/Proyectos/TwinBoxing worktree add --detach ~/Proyectos/TwinBoxing-academico <hash>
```

`<hash>` es la punta de `cambios-academicos` (PR #29) después de T1. El worktree queda
**desacoplado** en ese commit, no en la rama: un push a `cambios-academicos` durante la demo no
cambia B. El hash se anota en S3 de `04-pasos.md`. Moverlo es repetir S3, S4, L2 y S7.

`twinboxing_env` resuelve `boxtwin` y `boxtwin_api` hacia la copia principal (editables). B no
reinstala nada: antepone su código con `PYTHONPATH`, que gana sobre las instalaciones editables
(comprobado el 25-10 al correr las pruebas de la rama). Los subprocesos del worker heredan el
entorno (`worker.py`: `env={**os.environ, ...}`), así que el `boxtwin-annotator` que llama
también usa el código de B.

`/salud` lee el commit de `<repo>/.git/HEAD`. En un worktree `.git` es un archivo
(`gitdir: <repo principal>/.git/worktrees/<nombre>`), así que hoy B informaría `commit: null`.

**T1.** `_commit()` (`boxtwin-api/src/boxtwin_api/app.py`) resuelve también un worktree. Diff
mínimo: si `.git` es un archivo, seguir su `gitdir:`; las referencias se buscan en el directorio
común que indica `commondir` dentro de ese `gitdir`, primero como archivo suelto y después en
`packed-refs`. Sigue sin subprocesos. Pruebas, con repos falsos armados en `tmp_path`, escritas
antes del código:

1. `.git` carpeta, `HEAD` con `ref:` a una referencia suelta → los 7 primeros del hash (lo de hoy).
2. `.git` carpeta, referencia solo en `packed-refs` → el hash de `packed-refs` (lo de hoy).
3. `.git` archivo con `gitdir:`, `HEAD` del worktree desacoplado (un hash) → ese hash. Es el caso de B.
4. `.git` archivo con `gitdir:` y `HEAD` con `ref:` a una rama; la referencia está en el directorio
   común (`commondir`) → el hash de esa rama.
5. Sin `.git`, o un `gitdir:` que apunta a algo que no existe → `None`, sin excepción.

`_commit()` recibe la raíz como parámetro opcional (por defecto la de hoy), para poder probarla
sin mover el paquete. A no recibe este cambio hasta que se mergee el PR #29 y se actualice la
estación, y eso no es parte de este spec.

## Web

Build propio, con el código de B, en su `out/`:

```
cd ~/Proyectos/TwinBoxing-academico/prototipo/boxtwin-web-entrenador && npm ci && npm run build && npm run presupuesto
```

`npm ci` instala sus propios `node_modules`, así que B no comparte ni toca los de A.

## Entorno · `/etc/boxtwin-academico.env` (permisos 600, dueño lucasb)

Lo arma el SDD en S5: copia `despliegue/academico/boxtwin-academico.env.ejemplo` (versionado, sin
secretos, con los valores de abajo y los dos marcadores) a `~/boxtwin-academico/boxtwin-academico.env`
con permisos 600, y completa el secreto y la invitación con valores nuevos. Lucas lo instala en
`/etc` en L1. El archivo con secretos nunca entra a git.

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

Versionadas en `despliegue/academico/systemd/` (S5) y copiadas a `/etc/systemd/system/` por Lucas
en L1.

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

Son CA2 a CA6 de `00-README.md`, y se registran en S7, S8, S9 y L5 de `04-pasos.md`. En detalle:

- `curl -s 127.0.0.1:8001/salud`: `clasificador` `e7`, `modelo_e7` `true`, `modelo_individual`
  `true`, `commit` el de la rama (después de T1).
- Una sesión de sparring propia, procesada de punta a punta: `pose3d_hmr2.npz` en su carpeta, y
  `fightcard.json` con `clasificador.metodo` `geometrico-e7` y la `licencia`. En la web, la nota
  "Tipo en modo académico, no apto para uso comercial".
- Una sesión de bolsa: clasifica con E4a, sin `pose3d_hmr2.npz`.
- `ls ~/boxtwin/datos/sesiones | wc -l` igual antes y después: B no escribió en A.
