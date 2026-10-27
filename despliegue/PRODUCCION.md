# Rama `produccion`: desplegar todas las versiones de BoxTwin

26-10-2026. Esta rama junta todo lo que hay para desplegar, probado junto, y este archivo dice
en qué orden se despliega. No reemplaza a `README.md` de esta carpeta (los pasos de base, el
túnel, la imagen del worker): lo ordena para la entrega.

## Qué versiones hay y dónde corre cada una

| Instancia | Versiones | Tipo de golpe | Puerto | Afuera | Código | Datos |
|---|---|---|---|---|---|---|
| A · producción | Individual, Entrenador y Gimnasio (`prototipo/specs/versiones-26-10`), más la versión de prueba detrás de `BOXTWIN_DEMO` | E4 (`geometrico`), E4a en bolsa y sombra | 8000 | Funnel público, `:443` | copia principal `~/Proyectos/TwinBoxing` | `~/boxtwin/datos` |
| B · académica | las mismas, con el modo académico E7 (`docs/specs/2026-10-25-e7-modo-academico`) | E7 (`e7`), E4a en bolsa y sombra | 8001 | solo la tailnet, `:8443`, **sin Funnel** | worktree desacoplado `~/Proyectos/TwinBoxing-academico` | `~/boxtwin-academico/datos` |

Una versión no es otra aplicación: es el plan de la cuenta. A sirve las tres versiones a la vez;
lo que cambia por cuenta es qué se ve y cuántas horas tiene (`boxtwin_api/planes.py`). La versión
Entrenador es lo que ya usa la estación, y todas las cuentas existentes quedan como Entrenador al
migrar. Federación y el modo en vivo quedan fuera (`versiones-26-10/00-README.md`).

## Qué trae la rama

Sobre `main` (`709c22a`), en este orden:

1. `versiones-26-10`: B19 a B29 y W27 a W38, con su registro en `prototipo/TAREAS.md`.
2. `cambios-academicos` de GitHub (PR #29, todavía abierto): BM, E7 y el modo académico.
3. `cambios-academicos` local, cuatro commits que no estaban en GitHub: T1 (`/salud` informa el
   commit desde un worktree), los archivos de B en `despliegue/academico/` y el registro de S1 a
   S8 de `docs/specs/2026-10-25-dos-instancias/04-pasos.md` (B ya está arriba, ver abajo).
4. Esto: `despliegue/produccion.sh`, este archivo, `boxtwin.env.ejemplo` al día con lo que corre
   la estación y las variables de las versiones, y `BOXTWIN_CUPO` y `BOXTWIN_DEMO` en el ejemplo
   de B.

Los merges no tuvieron conflictos de código (solo el README raíz) y el modo académico no se
filtra a A: la leyenda de SMPL sale solo con `clasificador.metodo == geometrico-e7`, y
`produccion.sh verificar a` falla si A arranca con `e7`.

### Pruebas sobre la rama (26-10)

| Paquete | Comando | Resultado |
|---|---|---|
| `boxtwin-api` | `python -m pytest` | 308 |
| `boxtwin-annotator` | `PYTHONPATH=src python -m pytest` | 766 |
| `boxtwin-detector` | idem | 152 |
| `boxtwin-guantes` | idem | 46 |
| web entrenador | `npm test`, `lint`, `typecheck`, `build`, `presupuesto`, `sin-demo`, `contraste` | 243; los demás en verde |
| punta a punta | `npx playwright test` | 12 de 12, con WebKit (iPhone 13), incluidos E1 a E3 de las versiones |

### Ensayo del despliegue (26-10, sin tocar la estación)

Las dos instancias levantadas en local desde los ejemplos de entorno de esta rama (A en 8100, B en
8101, datos en un directorio aparte), leyendo cada archivo como lo lee systemd:

- `produccion.sh verificar a`: commit, secreto fijo, registro por invitación, web, `geometrico`,
  versión de prueba apagada. Todo ok.
- `produccion.sh verificar b`: lo mismo con `e7`, `modelo_e7` y `modelo_individual` en `true` con
  los modelos reales de `~/boxtwin/modelos`. Todo ok.
- `produccion.sh demo sembrar` y A con `BOXTWIN_DEMO=1`: los cuatro botones entran cada uno a su
  cuenta (Individual 4 h, Entrenador 15 h, Gimnasio titular y entrenador 50 h), y `verificar a`
  avisa que la versión de prueba está encendida.
- `respaldo`, `env-academico` (600, secreto nuevo de 64 hex, no pisa uno existente), `previo` y
  `verificar` contra un commit viejo (falla con "falta reiniciar"): como se espera.
- Con la web montada, `POST /auth/demo` apagada da **405**, no el 404 que dice RNF-D2. Es la misma
  respuesta que cualquier ruta inexistente con POST (la atiende el montaje de la web), así que no
  delata que la ruta existe, que es lo que el requisito protege. Se deja así y se declara.

## Registro del despliegue de A (26-10, 16:37 a 16:40)

Lucas decidió desplegar antes de las sesiones de prueba del 27-10. Es un **desvío de
`versiones-26-10/06` §1** (y de `dos-instancias/01-produccion.md`, que daba A por intacta): C4, C5
y C6 se miden sobre `c936b0f` y no sobre `34e991d`. Va así a la tesis, con fecha y motivo.

1. Antes: cola vacía, GPU libre, una cuenta real con 12 sesiones. Huella en
   `~/boxtwin/huella-antes-de-produccion.txt`.
2. Respaldos: `~/boxtwin/datos/boxtwin.db.antes-de-produccion-20261026-1637.bak` (12 sesiones,
   abre) y `/etc/boxtwin.env.antes-de-produccion-20261026-1637.bak`. B apagada.
3. Copia principal: `git checkout --detach produccion` → `c936b0f`. Web con `produccion.sh web`
   (el primer build cortó con el aviso de Next y el reintento siguió; presupuesto y sin-demo ok).
4. `/etc/boxtwin.env`: se agregaron `BOXTWIN_CUPO=informar`, `BOXTWIN_DEMO=` y
   `BOXTWIN_DEMO_SUBIR=1` (sigue en 600, lucasb). Restart de `boxtwin-api` y `boxtwin-worker`.
5. Migración al arrancar: columnas nuevas en `usuarios`, `boxeadores`, `sesiones` y
   `correcciones`; la cuenta real quedó Entrenador titular con sus 12 sesiones y 5 peleadores; 12
   duraciones copiadas; índice `uq_boxeador_cuenta_nombre`.
6. `produccion.sh verificar a`: todo ok, versión de prueba apagada. Por el Funnel: `/salud` con
   `c936b0f`, `/ingresar/` 200, `/cuenta` sin token 401.
7. B encendida de nuevo, sin cambios: `verificar b` ok en `efcec23`, sin arranque automático,
   `:8443` solo en la tailnet.
8. Huella después en `~/boxtwin/huella-despues-de-produccion.txt`. Cambian el entorno, el commit,
   `/salud`, la fecha de la web y el hash de `pip freeze`. Este último solo porque las líneas
   editables llevan el commit de la copia; ningún paquete cambió. Sesiones: 12 antes y después.

Sin hacer: CV6 y la versión de prueba (`BOXTWIN_DEMO` sigue vacía), y entrar con la cuenta real
desde el navegador. Eso último lo verifica Lucas, porque la clave no la tiene el SDD.

## Versión de prueba encendida (26-10, 16:50)

Pedido de Lucas. `~/boxtwin/demo.yaml` sin sesiones copiadas (DV-9: ningún video de nadie
expuesto); Atleta 1 y 2 en Entrenador, Atleta 3 a 5 en Gimnasio con su consentimiento de semilla.
`produccion.sh demo sembrar`: 3 cuentas, 4 usuarios, 6 atletas, 3 consentimientos. Respaldo
`/etc/boxtwin.env.antes-de-demo-*.bak`, `BOXTWIN_DEMO=1`, restart solo de `boxtwin-api` (cola vacía).

- Por el Funnel, los cuatro botones disponibles; cada perfil entra a su versión y rol (4, 15, 50 y
  50 h) y ve 0 sesiones.
- **CV6 cumple**: con cada perfil, 53 pedidos contra las 12 sesiones y los 5 peleadores de la cuenta
  real (`/jobs`, `/fightcards`, ticket de video, tramos, perfil): 0 fugas, todos 404.
- Mientras esté encendida, la cuenta real no entra por la web (DV-10).

Apagarla: `BOXTWIN_DEMO=` vacía en `/etc/boxtwin.env` y `sudo systemctl restart boxtwin-api`. Para
dejar las cuentas de prueba como nuevas: `produccion.sh demo reiniciar`.

## Avatar MPFB2 desplegado en A (27-10, 02:51)

PR #30 (avatar MPFB2 y calor sobre la piel, `prototipo/specs/cambios-27-10/`) mergeado a
`produccion` como `1058d5e`. Cambio solo de web: sin migración, sin tocar B ni el worker.

- Copia principal de `c936b0f` a `1058d5e` (`git checkout --detach origin/produccion`),
  `produccion.sh web`: presupuesto en verde, `/peleador/` 340,9 KB, la más pesada
  `/sesion/identificar/` 371,1 KB; `sin-demo` limpio.
- La web nueva sale de disco: `/peleador/` 200 y `/models/boxer_mpfb2.glb` 200, `model/gltf-binary`,
  idéntico byte a byte al del repo.
- `sudo systemctl restart boxtwin-api` (Lucas, 02:51:22). `produccion.sh verificar a`: commit
  `1058d5e`, secreto fijo, invitación, web y `geometrico` ok. Avisa la versión de prueba encendida,
  que sigue así a pedido de Lucas (ventana de ensayo).
- Vuelta atrás: `git checkout --detach c936b0f`, `produccion.sh web` y restart de la API.

## Estado de la estación al 26-10 antes del despliegue (leído, sin tocar)

- A: `boxtwin-api` y `boxtwin-worker` activos, `/salud` `34e991d`, `geometrico`. Sin las versiones.
- B: `boxtwin-academico-api` y `-worker` activos y **no habilitados al arranque**, `/salud`
  `efcec23`, `e7`, `modelo_e7` `true`; `:8443` solo en la tailnet. Frenada en L4 (material de la
  demo) de `dos-instancias/04-pasos.md`.

## Orden de despliegue

### 0. Antes de tocar A

- Que hayan terminado las sesiones de prueba (C4, C5, C6). Regla de `versiones-26-10/06` §1: nada de
  las versiones se despliega con sesiones por delante.
- B apagada mientras se despliega A (`dos-instancias/03-convivencia.md`, regla 1), y la cola de A vacía.

```bash
cd ~/Proyectos/TwinBoxing
bash despliegue/academico/huella-A.sh > ~/boxtwin/huella-antes-de-produccion.txt
sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api        # Lucas
```

### 1. A: las tres versiones

```bash
cd ~/Proyectos/TwinBoxing
bash despliegue/produccion.sh respaldo        # base respaldada y probada (R-12)
git checkout --detach produccion              # la rama ya esta en este repo, sin pull. Detached
                                              # porque git no deja tener la misma rama en dos
                                              # worktrees, y este repo tiene varios
bash despliegue/produccion.sh web             # npm ci, build, presupuesto, sin-demo
sudo -u lucasb editor /etc/boxtwin.env         # Lucas: agregar BOXTWIN_CUPO=informar,
                                              # BOXTWIN_DEMO= (vacia) y BOXTWIN_DEMO_SUBIR=1
                                              # (ver despliegue/boxtwin.env.ejemplo)
sudo systemctl restart boxtwin-api boxtwin-worker                         # Lucas
bash despliegue/produccion.sh verificar a
```

La migración la hace la API al arrancar y es aditiva: tablas `cuentas`, `miembros` y
`consentimientos`, columnas nulas, índice único por cuenta, una cuenta Entrenador por cada usuario
existente y la duración de las sesiones copiada de su `sesion.json`. Los paquetes son editables
contra esta copia, así que no se reinstala nada (PyYAML ya está en `twinboxing_env`).

Después: entrar con una cuenta real (sigue siendo email y clave) y ver sus sesiones; CV1 y CV2 ya
están en las pruebas.

### 2. A: la versión de prueba, solo en las ventanas de ensayo y presentación (DV-10)

```bash
cp despliegue/demo.yaml.ejemplo ~/boxtwin/demo.yaml && editor ~/boxtwin/demo.yaml   # DV-9, Lucas
bash despliegue/produccion.sh demo sembrar
# BOXTWIN_DEMO=1 en /etc/boxtwin.env y restart (Lucas); despues:
bash despliegue/produccion.sh verificar a     # tiene que avisar "version de prueba ENCENDIDA"
# CV6: entrar por cada boton y pedir por id una sesion real conocida: 404 en las cuatro.
# Al cerrar la ventana: BOXTWIN_DEMO vacia, restart, y `demo reiniciar` para el siguiente.
```

### 3. B: la académica

B ya corre (`efcec23`). Tiene el modo académico y no las versiones, y para lo que es (la demo de
E7 en la tailnet) no le hacen falta. Recomendación: **dejarla en `efcec23`** hasta cerrar S9 a S11
de `dos-instancias/04-pasos.md`, y moverla a `produccion` solo si se quiere mostrar E7 con las
versiones. Moverla es repetir S3, S4, L2 y S7:

```bash
sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api        # Lucas
git -C ~/Proyectos/TwinBoxing worktree remove ~/Proyectos/TwinBoxing-academico
bash despliegue/produccion.sh worktree-academico produccion               # S3, anotar el hash
bash despliegue/produccion.sh web ~/Proyectos/TwinBoxing-academico         # S4
sudo systemctl start boxtwin-academico-api boxtwin-academico-worker       # L2, Lucas
bash despliegue/produccion.sh verificar b                                  # S7 (CA2)
```

`/etc/boxtwin-academico.env` no cambia: B arranca con la versión de prueba apagada y el cupo en
informar, que son los valores por defecto. B en una instalación nueva: `produccion.sh
env-academico`, después L1 a L3 de `04-pasos.md`.

### 4. Después de cada demo

```bash
bash despliegue/academico/huella-A.sh > ~/boxtwin/huella-despues.txt     # y comparar (CA1)
sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api        # B apagada, Lucas
```

## Vuelta atrás

- **A:** `git checkout 34e991d` (o `main`), `produccion.sh web`, restart. El código anterior ignora
  las tablas y columnas nuevas. Lo único que se pierde es que las sesiones de un entrenador de
  Gimnasio pasan a verse solo para quien las subió. Si hiciera falta la base de antes, está el
  `.bak` del paso 1.
- **Versión de prueba:** `BOXTWIN_DEMO` vacía y restart. `BOXTWIN_DEMO_SUBIR=0` corta solo la subida.
- **B:** `sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api` y
  `tailscale serve --https=8443 off`. A no se entera.

## Lo que queda de Lucas

- Las tareas con sudo o Tailscale marcadas arriba, y L4, L5 y S10 de `dos-instancias/04-pasos.md`.
- El texto del consentimiento por atleta en `boxtwin-api/src/boxtwin_api/textos/` (DV-5).
- `~/boxtwin/demo.yaml` con material que tenga permiso para mostrarse (DV-9).
- CV3 a CV6 de las versiones, en la estación.
- Next 16.4.0 avisa de una vulnerabilidad conocida al hacer el build. No se actualizó: es un
  cambio de dependencia y pide el OK de Lucas.
- La rama está en GitHub desde el 27-10 (PR #30 la usó de base).
