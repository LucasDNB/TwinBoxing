# E7 en la app como modo académico · plan

25-10-2026 · la enmienda (`00-enmienda.md`) manda sobre este documento · se implementa después de
aprobarla, en la rama `e7-modo-academico` (sale de `exp/boxmind-prueba-22-10`, PR #27)

## 1. Cómo corre hoy el tipo de golpe

`BOXTWIN_CLASIFICADOR=geometrico` en `/etc/boxtwin.env`. La etapa `clasificar` del worker
(`boxtwin-api/src/boxtwin_api/worker.py`) corre dos comandos en `twinboxing_env`:
`boxtwin clasificar` (`boxtwin/mvp/clasificador_geo.py`, modelo E4 en
`~/boxtwin/modelos/clasificador_geo_v1`) y `boxtwin tipos`. Las sesiones individuales (bolsa y
sombra) usan E4a. El producto no importa nada de `scripts/`: copia lo que necesita y lo prueba
contra el original (CA1, CA2).

## 2. Qué se agrega

| # | Pieza | Dónde | Qué hace |
|---|---|---|---|
| M1 | Modelo final de E7 | `scripts/clasificador_geo/entrenar_final.py --exp E7` → `~/boxtwin/modelos/clasificador_e7_v1` | Las cinco semillas con los 1047 golpes y el 3D de E7, misma receta que E4. `meta.json` con `metodo: geometrico-e7`, familia 0,777 y la leyenda de licencia |
| M2 | Elevación por sesión | herramienta nueva `boxtwin-annotator/tools/elevar_hmr2_sesion.py`, corre en `boxtwin_4dhumans` | De `segmentos.json`, `poses.npz` y el video de la sesión a `pose3d_hmr2.npz`, un registro por segmento: 32 pasos, 17 articulaciones H36M y máscara. Copia `caja`, `cajas_de`, `smpl_a_h36m` y la ventana de `elevar_hmr2.py`; no importa `scripts/` |
| M3 | Features 3D en el producto | `boxtwin/mvp/clasificador_geo.py` | Copia de `features3d.features_de` y `marco_cuerpo`. `clasificar_sesion` gana `--pose3d`: con un modelo de 42 features, concatena las 16 3D a las 26 de hoy. Sin el archivo, error claro y no clasifica con E4 a escondidas |
| M4 | Worker y configuración | `boxtwin-api` `config.py` y `worker.py` | `BOXTWIN_CLASIFICADOR=e7` (además de `geometrico` y `poseconv3d`), `BOXTWIN_CMD_HMR2` y `BOXTWIN_MODELO_CLASIFICADOR_E7`. La etapa `clasificar` corre tres comandos: elevar, clasificar con `--pose3d` y tipos. Bolsa y sombra siguen con E4a, porque E7 necesita al rival |
| M5 | Contrato y Fight-Card | `prototipo/contrato/fightcard-0.3.schema.json` y su `CHANGELOG.md`, `boxtwin/mvp/fightcard.py`, la web si muestra el método | `geometrico-e7` entra al enum de `metodo`. La Fight-Card muestra la leyenda de licencia |
| M6 | Medición de tiempo | `boxtwin-api/scripts/medir_c5.py` sobre dos sesiones de la estación, con E4 y con E7 | Cuánto agrega la elevación al reloj hasta la Fight-Card. Va a `TAREAS.md` y a la Fight-Card si no entra en 2x |

## 3. Pruebas (antes del código, como en E7)

1. CA-E7a: las copias de M2 y M3 dan lo mismo que `scripts/clasificador_geo` (`caja`,
   `cajas_de`, `smpl_a_h36m`, `features_de`) sobre datos sintéticos.
2. CA-E7b: el modelo final de E7 con `--excluir Sparring --semillas 0` da el mismo `pred` que
   E7 s0 de ese fold, igual que CA2 con E4.
3. Config: `e7` se lee bien y cualquier otra cosa sigue siendo el comportamiento de hoy.
4. Worker: con `e7`, la etapa `clasificar` arma los tres comandos en orden. Con bolsa y sombra
   arma los de E4a. Sin `BOXTWIN_CMD_HMR2`, falla con un mensaje claro.
5. `clasificar_sesion` con modelo E7 y sin `pose3d_hmr2.npz` falla, y no cae a E4.
6. La Fight-Card con `geometrico-e7` valida contra el schema y lleva la leyenda.
7. Las pruebas de hoy siguen en verde: API, anotador, `scripts/clasificador_geo` y la web.

## 4. Orden y tiempo

| Paso | Depende de | Estimado |
|---|---|---|
| Aprobar la enmienda y pegar el texto de la licencia de SMPLify | Lucas | — |
| M1 | aprobación | 10 min de GPU |
| Pruebas 1 a 6 en rojo, después M2 a M5 | M1 | un día |
| M6 | M2 a M5 | una hora |
| PR, revisión y build de la web | M6 | — |

No se enciende nada en `/etc/boxtwin.env` sin que Lucas lo pida, y nunca con sesiones de prueba
en curso (enmienda, condición 2).
