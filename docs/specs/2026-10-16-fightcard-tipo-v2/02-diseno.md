# Diseño

Estilo: el del repo. Docstring con POR QUÉ EXISTE, QUÉ HACE y USO, sin tildes; comentarios en
español; diff mínimo en lo existente; nada de reescribir archivos enteros.

## 1. Modelo final (`scripts/clasificador_geo/entrenar_final.py`)

Script nuevo. **No modifica** `entrenar.py`, `modelo.py`, `features2d.py` ni `datos.py`: los
importa. Con `entrenar.entrenar_modelo` y la configuración E4 (columnas 0 a 25):

- Sin `--excluir`: entrena con los 1.047 golpes, semillas 0 a 4, y guarda en
  `~/boxtwin/modelos/clasificador_geo_v1/`: `s0.pt` … `s4.pt` (`state_dict`), y `meta.json` con
  `n_features` 26, nombres de features, `pasos` 32, margen 8, por semilla la media y el desvío
  de estandarización, hiperparámetros, commit, sha256 de cada `.pt` y de `golpes.npz`, y las
  métricas de E4 que muestra la Fight-Card (familia 0,727 con Wilson 0,699–0,753, recall por
  familia), copiadas de `docs/experiments/datos/2026-10-16-clasificador-geo-E4.json`.
- Con `--excluir <fuente>`: lo mismo sin esa fuente, en
  `~/boxtwin/modelos/clasificador_geo_v1/sin_<fuente>/`. Lo usan CA2 y M1.

Las cinco semillas del modelo completo y las de cada fold de M1 (cuatro fuentes) son 25
entrenamientos de segundos.

## 2. Inferencia en producción (`boxtwin-annotator/src/boxtwin/mvp/clasificador_geo.py`)

Módulo nuevo del paquete del anotador (el producto no importa nada de `scripts/`):

- `ClasificadorGolpe`: copia exacta de la clase de `scripts/clasificador_geo/modelo.py`, con
  un comentario que dice de dónde sale y que CA2 verifica que no se desvió.
- `features_golpe(kp, sc, valid, p, brazo, inicio, fin) -> (X (32, 26), M (32,))`: las 20 de
  `boxtwin_detector.features.features_de` (import, no copia) sobre la serie entera del
  peleador, las 6 del rival con la misma fórmula que `scripts/clasificador_geo/features2d.py`
  (copiada, con comentario y verificada por CA1), ventana `[inicio-8, fin+8]` y remuestreo a 32.
  `brazo` en `left`/`right`.
- `ClasificadorGeo.cargar(directorio)`: lee `meta.json`, verifica sha256 de los `.pt`, carga
  las cinco semillas en CPU.
- `ClasificadorGeo.familia(X, M) -> p (n, 3)`: promedio de las cinco, cada una con su
  estandarización.
- `rol(guardia, brazo)`: `boxtwin.core.types.arm_role` con la traducción `ortodoxa → orthodox`,
  `zurda → southpaw` e `izq/der → left/right`.

## 3. Comando `boxtwin-annotator clasificar <sesion>`

Subcomando nuevo en `cli.py`, en `twinboxing_env`. Argumentos: `--modelo` (directorio, por
defecto `~/boxtwin/modelos/clasificador_geo_v1`), `--guardia-a` y `--guardia-b`
(`ortodoxa`, `zurda` o `desconocida`, por defecto `desconocida`).

Lee `segmentos.json` y `poses.npz` de la sesión (los mismos que lee `clasificar_sesion.py`),
calcula features por segmento, clasifica y escribe `tipos.json` versión `0.2`, compatible con
lo que lee `aplicar_tipos`:

```json
{
  "version": "0.2",
  "metodo": "geometrico-v1",
  "checkpoint": "clasificador_geo_v1",
  "exactitud_familia_fuente_no_vista": 0.727,
  "referencia": "docs/experiments/2026-10-16-clasificador-geometrico.md",
  "guardia": {"A": "ortodoxa", "B": "zurda"},
  "guardia_supuesta": {"A": false, "B": true},
  "segmentos_sin_clasificar": 0,
  "tipos": {
    "A-izq-1234": {"tipo": "jab", "confianza": 0.81, "crudo": "straight",
                    "p_familia": [0.81, 0.15, 0.04], "rol": "lead"}
  }
}
```

- El id del golpe es el mismo que arma hoy `clasificar_sesion.py` (`<peleador>-<izq|der>-<inicio>`).
- `confianza` es la probabilidad de la familia elegida. `tipo` es jab o cross para recto según
  el rol, y hook o uppercut si no.
- Un segmento sin ningún paso usable no se clasifica (cuenta en `segmentos_sin_clasificar`),
  igual que hoy los de un cuadro.
- Sesión de un solo peleador (B8): las 6 features del rival van en cero con `rival_ok` 0, como
  en entrenamiento cuando falta el rival.

`aplicar_tipos` (en `mvp/fightcard.py`) se extiende con diff mínimo para copiar al bloque
`clasificador` los campos `metodo`, `referencia` y `guardia_supuesta` si vienen, y a cada golpe
`p_familia` y `rol`. Con un `tipos.json` 0.1 hace exactamente lo de hoy.

## 4. Worker y configuración (`boxtwin-api`)

- `config.py`: `clasificador: Literal["poseconv3d", "geometrico"]` desde
  `BOXTWIN_CLASIFICADOR` (por defecto `poseconv3d`) y `modelo_clasificador_geo: Path` desde
  `BOXTWIN_MODELO_CLASIFICADOR_GEO`.
- `worker.comando_de`, etapa `clasificar`: con `geometrico`, `[[*cfg.cmd_boxtwin, "clasificar",
  dir, "--modelo", ..., "--guardia-a", ga, "--guardia-b", gb], [*cfg.cmd_boxtwin, "tipos",
  dir]]`, con las guardias leídas de los boxeadores de la sesión. Con `poseconv3d`, lo de hoy.
- El encadenamiento después de `completar` hoy depende de `cfg.cmd_clasificador`. Con
  `geometrico` tiene que encadenar aunque `cmd_clasificador` esté vacío: es justamente el caso
  de un worker sin `boxtwin_mmaction`.
- RT8: en `app.py`, el PATCH de boxeador (cuando cambia `guardia`) y la asignación de
  boxeadores a una sesión encolan `clasificar` en las sesiones `listo` afectadas, solo con
  `geometrico`. No se encola si ya hay un `clasificar` en cola para esa sesión.
- `/salud` informa el método de clasificación activo.

## 5. Contrato y web

- `prototipo/contrato/fightcard-0.3.schema.json`: propiedades opcionales nuevas
  (`clasificador.metodo`, `clasificador.referencia`, `clasificador.guardia_supuesta` como
  objeto A/B de booleanos; en `golpe`, `p_familia` y `rol`). La versión del documento sigue en
  0.3: los campos son opcionales y una Fight-Card v1 sigue siendo válida.
- Web (`prototipo/boxtwin-web-entrenador`): el tipo de cliente zod admite los campos nuevos
  como opcionales; donde se muestra la exactitud del clasificador, usar la que trae la
  Fight-Card; el aviso de RT10 en la Fight-Card y en el análisis de sesión cuando
  `guardia_supuesta` de un lado es `true`. Pruebas de unidad e integración con una Fight-Card v2.
  Nada de pantallas nuevas.

## 6. Medición M1 (`scripts/clasificador_geo/medir_v2.py`)

Sigue `01-requisitos.md` §4. Copia las cuatro sesiones a `~/boxtwin/tipo-v2/m1/`, corre en
cada copia `boxtwin-annotator clasificar` con `--modelo .../sin_<fuente>` y las guardias
(dos corridas: anotada y todos ortodoxos), y empareja como `scripts/clasificador_geo/rol_por_brazo.py`
(reusar sus funciones de emparejamiento importándolas). Para la guardia anotada, como en el
producto la guardia es por peleador, se toma la guardia más frecuente de los eventos anotados de
ese peleador en la fuente. Salida: `~/boxtwin/tipo-v2/resultados/m1.json` y su copia en
`docs/experiments/datos/2026-10-16-tipo-v2-m1.json`.

## 7. Pruebas exigidas

- Anotador: `features_golpe` (ventana en bordes, rival ausente, un peleador), carga con sha256
  incorrecto falla, `rol` en los cuatro casos, `clasificar` escribe `tipos.json` 0.2 válido
  sobre una sesión sintética, `aplicar_tipos` con 0.1 idéntico a hoy y con 0.2 copia los campos,
  CA6. Corren con `PYTHONPATH=src` como el resto de la suite del anotador.
- CA1 y CA2: `scripts/clasificador_geo/tests/test_equivalencia_v2.py`, con datos sintéticos
  para la lógica y marcadas para correr contra los datos reales si están (se saltean si no).
  Además, en la estación se corren contra los datos reales y el resultado va a la evidencia.
- API: CA3, CA4, encadenamiento con `cmd_clasificador` vacío y `geometrico`, `/salud`.
- Web: contrato, aviso de guardia supuesta, exactitud desde la Fight-Card.
- `medir_v2.py`: emparejamiento y guardia más frecuente con datos sintéticos.

Comandos:

```
cd boxtwin-annotator && PYTHONPATH=src ~/miniforge3/envs/twinboxing_env/bin/python -m pytest -q
cd boxtwin-api && ~/miniforge3/envs/twinboxing_env/bin/python -m pytest -q
~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/clasificador_geo/tests scripts/clasificador/tests -q
npm --prefix prototipo/boxtwin-web-entrenador test
```

(Si la suite del anotador o de la API se corre de otra forma, la que dicen sus README manda.)
