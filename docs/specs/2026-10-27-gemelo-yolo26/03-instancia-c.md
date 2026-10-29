# C · La instancia del gemelo

## 1. Código

Un worktree fijo, fuera de `.claude/worktrees/` (esos los borra la limpieza de sesiones), y
desacoplado en un commit, no en la rama:

```
bash despliegue/produccion.sh worktree-gemelo <commit>      # T9; S5
```

`<commit>` es la punta de `main` después de mergear la fase 0 (T1 a T10). El mismo commit corre
E8 (fase 1) y el servicio (fase 2): lo que se midió es lo que se sirve. El hash se anota en S5 de
`05-pasos.md`. Moverlo antes de la fase 1 es repetir S5; después, todo desde S6, porque E8 y el
servicio tienen que correr el mismo código.

C no reinstala nada en `twinboxing_env`. Antepone su código con `PYTHONPATH`, que gana sobre las
instalaciones editables de la copia principal (comprobado con B el 25-10). A diferencia de B,
suma también `boxtwin-detector/src`, para que la inferencia del detector sea la del commit de C y
no la de A. Los subprocesos del worker heredan el entorno.

## 2. Entorno · `/etc/boxtwin-gemelo.env` (600, dueño lucasb)

`produccion.sh env-gemelo` (T9) lo arma desde `despliegue/gemelo/boxtwin-gemelo.env.ejemplo`
(T10) en `~/boxtwin-gemelo/boxtwin-gemelo.env`, con secreto e invitación nuevos y distintos de
los de A y de B, comparados sin mostrarlos. Lucas lo instala en L2. Nunca entra a git.

```
# C · gemelo local con YOLO26l-pose (docs/specs/2026-10-27-gemelo-yolo26). Nunca para usuarios.
BOXTWIN_SECRETO=<nuevo, distinto del de A y del de B>
BOXTWIN_INVITACION=<nueva, distinta de la de A y de la de B>
BOXTWIN_DATOS=/home/lucasb/boxtwin-gemelo/datos
BOXTWIN_DB=sqlite:////home/lucasb/boxtwin-gemelo/datos/boxtwin.db
BOXTWIN_WEB=/home/lucasb/Proyectos/TwinBoxing-gemelo/prototipo/boxtwin-web-entrenador/out
PYTHONPATH=/home/lucasb/Proyectos/TwinBoxing-gemelo/boxtwin-annotator/src:/home/lucasb/Proyectos/TwinBoxing-gemelo/boxtwin-api/src:/home/lucasb/Proyectos/TwinBoxing-gemelo/boxtwin-detector/src

# modelos propios de C (fase 2), salvo los guantes, que son los de A en solo lectura
BOXTWIN_MODELO_POSE=/home/lucasb/boxtwin-gemelo/modelos/yolo26l-pose.pt
BOXTWIN_MODELO_GUANTES=/home/lucasb/Proyectos/TwinBoxing/boxtwin-guantes/modelos/guantes-v2.pt
BOXTWIN_MODELO_DETECTOR=/home/lucasb/boxtwin-gemelo/modelos/detector_7fuentes_y26.ens.pt
BOXTWIN_CMD=/home/lucasb/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator
BOXTWIN_DEVICE=0
BOXTWIN_UMBRAL=0.80

# las mismas decisiones de producto que A
BOXTWIN_C8=rechazado
BOXTWIN_C8_FECHA=2026-10-13
BOXTWIN_C3=rechazado
BOXTWIN_C3_FECHA=2026-10-20
BOXTWIN_SESIONES_INDIVIDUALES=1
BOXTWIN_CLASIFICADOR=geometrico
BOXTWIN_MODELO_CLASIFICADOR_GEO=/home/lucasb/boxtwin-gemelo/modelos/clasificador_geo_y26_v1
BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL=/home/lucasb/boxtwin-gemelo/modelos/clasificador_geo_y26_v1_sin_rival

# versiones 26-10: cuentas de prueba con un boton por version (decision del 26-10)
BOXTWIN_CUPO=informar
BOXTWIN_DEMO=1
BOXTWIN_DEMO_SUBIR=1
```

`BOXTWIN_CMD` apunta al ejecutable de `twinboxing_env`, que importa por `PYTHONPATH` el código
del worktree de C. S23 lo comprueba: la Fight-Card de una sesión de C trae el bloque `pose` de T3,
que el código de A no escribe.

## 3. Unidades

`despliegue/gemelo/systemd/boxtwin-gemelo-api.service` y `-worker.service` (T10): copias de las
de B con nombre, descripción, `WorkingDirectory=/home/lucasb/Proyectos/TwinBoxing-gemelo`,
`EnvironmentFile=/etc/boxtwin-gemelo.env` y `--port 8002`. Siempre `--host 127.0.0.1`.
`Restart=on-failure`, sin `enable`: se enciende para usarla y se apaga después.

## 4. Web

Build propio en el worktree, con sus `node_modules`:

```
bash despliegue/produccion.sh web ~/Proyectos/TwinBoxing-gemelo      # S20
```

No hace falta código nuevo de web: el bloque `pose` de T3 viaja en la Fight-Card y el contrato
0.3 lo admite. Mostrarlo en pantalla queda para después, si Lucas lo pide.

## 5. Modelos (fase 2, S17 a S19)

En `~/boxtwin-gemelo/modelos/`, fuera de git, cada uno con su `meta.json` (commit, sha256 de sus
entradas, comando) o, para el detector, con su `.metricas.json` al lado:

| Modelo | Receta | Métricas que muestra la Fight-Card |
|---|---|---|
| `yolo26l-pose.pt` | el checkpoint de S4, sin tocar | — |
| `detector_7fuentes_y26.ens.pt` | la de `detector_7fuentes.ens.pt`: ensamble de cinco semillas (42, 1, 2, 3, 4), 20 features, las siete fuentes de `raiz-y26` | precisión y recall de E8 (P12a), en `detector_7fuentes_y26.ens.metricas.json` (T3) |
| `clasificador_geo_y26_v1` | `entrenar_final.py --exp E4-y26` (T4), en CPU para que sea reproducible bit a bit, como el de A | familia de E4 sobre Y26 (P12b), del JSON de E8 |
| `clasificador_geo_y26_v1_sin_rival` | `entrenar_final.py --exp E4a-y26 --dispositivo cpu` | E4a sobre Y26, del mismo JSON |

El comando exacto con que se armó `detector_7fuentes.ens.pt` no quedó en el repo (relevado el
27-10). S17 lo reconstruye con `boxtwin_detector.ensamble.entrenar_ensamble` sobre las siete
fuentes y lo anota como decisión. Como control sin umbral, la misma receta sobre los tensores de
la base tiene que dar las mismas probabilidades que `detector_7fuentes.ens.pt` sobre una fuente;
si no, se declara la diferencia.

## 6. Cuentas

C usa la versión de prueba (`BOXTWIN_DEMO=1`): un botón por versión, sin email ni invitación
(decisión de Lucas del 26-10). Las cuentas se siembran con el entorno de C:

```
cp despliegue/demo.yaml.ejemplo ~/boxtwin-gemelo/demo.yaml      # sin sesiones copiadas (DV-9)
ENV_A=/etc/boxtwin-gemelo.env REPO=~/Proyectos/TwinBoxing-gemelo \
  bash despliegue/produccion.sh demo sembrar ~/boxtwin-gemelo/demo.yaml
```

`demo` usa la variable `ENV_A` para cualquier entorno: con estos valores lee el de C y escribe en
la base de C. No hay cuentas reales en C.

## 7. Acceso local

C escucha solo en `127.0.0.1:8002`. Desde la estación, `http://127.0.0.1:8002`. Desde la notebook,
por la tailnet y un túnel SSH, sin publicar nada:

```
ssh -N -L 8002:127.0.0.1:8002 lucasb@desarrollo-lucas      # en la notebook
# despues, en el navegador de la notebook: http://127.0.0.1:8002
```

En la notebook el origen es `127.0.0.1:8002`, distinto de los de A y B, así que la sesión del
navegador no se cruza (`dos-instancias`, decisión 2).

## 8. `/salud` esperado (CG2)

`produccion.sh verificar c` (T9) falla si no da todo esto:

| Campo | Valor |
|---|---|
| `commit` | el de S5 |
| `modelo_pose.archivo` | `yolo26l-pose.pt` |
| `modelo_pose.sha256` | los 12 primeros del sha256 anotado en S4 |
| `clasificador` | `geometrico` (nunca `e7`) |
| `modelo_individual` | `true` |
| `secreto_efimero` | `false` |
| `frontend` | `true` |

Y avisa, sin fallar, que la versión de prueba está encendida, que en C es lo esperado.
