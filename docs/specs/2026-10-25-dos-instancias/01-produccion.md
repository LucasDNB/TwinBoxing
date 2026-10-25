# A · Producción, sin ningún cambio

## Qué es hoy (relevado el 25-10)

- `boxtwin-api.service`: `uvicorn boxtwin_api.app:app --host 127.0.0.1 --port 8000
  --proxy-headers`, `WorkingDirectory=/home/lucasb/Proyectos/TwinBoxing`,
  `EnvironmentFile=/etc/boxtwin.env`.
- `boxtwin-worker.service`: `python -m boxtwin_api.worker`, mismo directorio y mismo entorno.
- Código: la copia principal en `main`, `34e991d` (`/salud` lo informa así). `origin/main` va
  más adelante (`709c22a`) y la estación no se actualizó. Eso también queda como está.
- `twinboxing_env` tiene `boxtwin-annotator`, `boxtwin-api` y `boxtwin-detector` instalados en
  modo editable contra la copia principal.
- `/etc/boxtwin.env`: `BOXTWIN_CLASIFICADOR=geometrico`, sesiones individuales encendidas, C3 y
  C8 rechazados, base en `~/boxtwin/datos/boxtwin.db`, web en
  `prototipo/boxtwin-web-entrenador/out` de la copia principal.
- Afuera: `tailscale serve` con Funnel, `https://desarrollo-lucas.tail52f6aa.ts.net` →
  `http://127.0.0.1:8000`.

## Qué quiere decir "sin ningún cambio"

Nada de esta lista se toca para lanzar B ni mientras B corre:

1. Las dos unidades de systemd y `/etc/boxtwin.env`.
2. La copia principal: ni checkout, ni pull, ni merge, ni build. Mergear #27 y #28 en GitHub no
   la cambia, y no se la actualiza por esto.
3. Las instalaciones editables de `twinboxing_env`. B no instala nada en ese entorno.
4. `~/boxtwin/datos/`: base, sesiones y respaldos.
5. `prototipo/boxtwin-web-entrenador/out` de la copia principal.
6. La configuración de Tailscale para el puerto 443 y su Funnel.
7. Los modelos de `~/boxtwin/modelos/` y de la copia principal. B solo los lee.

## Cómo se comprueba

Huella tomada antes de lanzar B (paso 1 de `04-pasos.md`) y repetida después de lanzarla y de
cada demo. Tienen que dar igual:

- `sha256sum /etc/boxtwin.env /etc/systemd/system/boxtwin-api.service /etc/systemd/system/boxtwin-worker.service`
- `git -C ~/Proyectos/TwinBoxing rev-parse HEAD` → `34e991d…`
- `curl -s 127.0.0.1:8000/salud`: `commit` `34e991d`, `clasificador` `geometrico`
- `tailscale serve status`: el 443 con Funnel apuntando a `127.0.0.1:8000`
- `~/miniforge3/envs/twinboxing_env/bin/python -m pip freeze`, igual al guardado
- `stat -c '%Y' prototipo/boxtwin-web-entrenador/out/index.html` en la copia principal
- La cantidad de sesiones de A, leída sin escribir (`sqlite3` no está instalado):
  `~/miniforge3/envs/twinboxing_env/bin/python -c "import sqlite3; print(sqlite3.connect('file:/home/lucasb/boxtwin/datos/boxtwin.db?mode=ro', uri=True).execute('select count(*) from sesiones').fetchone()[0])"`.
  Puede crecer por el uso normal de A, pero B no escribe nunca ahí: tiene su propia base
  (`03-convivencia.md`)

Lo único que A puede notar es la GPU compartida (`03-convivencia.md`).
