# 05 · API y datos

Formato del Prompt 11 de la Clase 5: primero el diseño (endpoints con método, ruta, payload y
respuesta; modelo de datos), después los archivos que se tocan, después las pruebas (`06`).
Ninguna tarea B se programa sin que su plan esté aprobado.

La API de hoy cubre el MVP de seis pasos. Lo que sigue es lo que el diseño de nueve pantallas
necesita y no está.

---

## 1. Endpoints que la web usa tal cual

| Método y ruta | Para qué en la web | Pantallas |
|---|---|---|
| `POST /auth/registro`, `POST /auth/login` | Ingreso | 00 |
| `POST /videos` (multipart) | Subida (se le suman campos en B1) | 02, 08 |
| `GET /jobs` | Lista de sesiones (se le suman campos en B1 y B5) | 01, sesiones |
| `GET /jobs/{id}` | Estado, progreso, candidatos | 03, 04 |
| `GET /jobs/{id}/candidatos/{track}?t=` | Recorte de un candidato | 03 |
| `POST /jobs/{id}/siembra` | Confirmar A y B | 03 |
| `GET /fightcards/{id}` | Fight-Card | 03, 04, 05, 09 |
| `GET /fightcards/{id}/export?formato=csv\|html\|pdf` | Exportar | 05, 09 |
| `PATCH /fightcards/{id}/golpes/{golpe}` | Corregir tipo (se amplía en B4) | 04 |
| `GET /videos/{id}/ticket` | Permiso de 30 min para video y recortes | 03, 04 |
| `GET /videos/{id}/stream?t=` | Video original por rangos | 04 |
| `GET\|HEAD /videos/{id}/procesado?t=` | Video con esqueleto | 04 |
| `POST /jobs/{id}/render` | Pedir el video con esqueleto | 04 |
| `POST /boxeadores`, `GET /boxeadores`, `PATCH /boxeadores/{id}` | Peleadores (se amplían en B6) | 01, 02, peleadores |
| `PUT /sesiones/{id}/boxeadores` | Asignar peleadores a A y B | 02 |
| `GET /boxeadores/{id}/perfil` | Perfil (se amplía en B7) | 06, 07 |
| `GET /salud` | Comprobación de despliegue | ninguna |

---

## 2. Cambios de API

### B0 · Servir la web exportada

- `BOXTWIN_WEB` apunta a `boxtwin-web-entrenador/out`. El montaje estático ya existe y va
  último; no cambia.
- La exportación usa `trailingSlash: true`, así que cada página es `carpeta/index.html` y
  `StaticFiles(html=True)` la resuelve. `out/404.html` se sirve para rutas desconocidas.
- Cabeceras para lo estático: `Content-Security-Policy` de `02` §8,
  `Cache-Control: public, max-age=31536000, immutable` para `/_next/static/`, `no-cache` para los
  `.html`.
- Archivos: `app.py` (montaje y cabeceras), `config.py` (sin cambios si `web` ya es configurable).
- Pruebas: una página por prefijo de `02` §4 responde 200 con HTML; `GET /boxeadores` sigue
  devolviendo JSON con el build montado; `GET /sesiones/` devuelve la página y
  `PUT /sesiones/{id}/boxeadores` sigue llegando a la API.

### B1 · Contexto de la sesión y consentimiento

`POST /videos` suma campos de formulario:

| Campo | Tipo | Regla |
|---|---|---|
| `contexto` | `bolsa\|sombra\|sparring\|combate` | por defecto `sparring` |
| `practica` | `tecnica_libre\|combinaciones\|defensa` o vacío | solo con `sombra` |
| `rounds` | entero ≥ 1 o vacío | vacío con "sin rounds" |
| `consentimiento` | `true` | obligatorio; sin él, 422 "falta el consentimiento de las personas filmadas" |

Modelo `Sesion`: columnas nuevas `contexto` (String(16), índice), `practica` (String(24),
nula), `rounds` (Integer, nula), `consentimiento_en` (DateTime), `consentimiento_por` (String(320)).
El consentimiento guarda quién lo declaró y cuándo, que es lo que se puede mostrar si alguien
pregunta (Ley 25.326, R-09).

`sesion_a_dict` suma `contexto`, `practica`, `rounds.cantidad` y
`boxeadores: {"A": {"id", "nombre"} | null, "B": …}` para que la lista no pida los peleadores
uno por uno.

Migración: `db.py` ya avisa que `create_all` no altera tablas existentes. Se agrega
`despliegue/migraciones/2026-10-web-entrenador.sql` con los `ALTER TABLE` para PostgreSQL, y el
chequeo de columnas de `db.py` incorpora las nuevas para fallar con un mensaje claro si la base no
está migrada. Las sesiones existentes quedan con `contexto = 'sparring'` y consentimiento nulo; la
web las muestra igual.

### B2 · Candidatos con color, tramo visible y altura relativa

`GET /jobs/{id}`, campo `candidatos[]`, suma por candidato:

```json
{
  "track": 7,
  "recorte_url": "/jobs/…/candidatos/7",
  "color_guante": {"hex": "#2F74C8", "nombre": "azul"},
  "t_desde": 4.0,
  "t_hasta": 178.2,
  "alto_relativo": 0.97,
  "sugerido": true
}
```

- `color_guante` sale del perfil de color del track que ya calcula la cadena de identidad; nulo
  si el track no tiene cuadros con color (`cuadros_con_color = 0`). El `nombre` es una
  aproximación por tono para la pantalla ("rojos", "azules", "negros", "sin guantes") y no entra
  en ningún cálculo.
- `alto_relativo` es el alto del candidato sobre la mediana de los altos de los candidatos.
- `sugerido` marca a los dos de `pareja_sugerida`.
- Se devuelven hasta 8 candidatos, la pareja sugerida primero y el resto por cuadros de
  coexistencia.
- Archivos: `boxtwin-annotator/src/boxtwin/mvp/candidatos.py` (dataclass `Candidato` y `elegir`),
  `orquesta.py` donde se escribe `sesion.json`. La API no cambia salvo pasar los campos.
- Pruebas: con un `sesion.json` de prueba, los campos llegan; con un track sin color,
  `color_guante` es nulo y la web no rompe.

### B3 · Tramos sin asignar y asignación manual (plan B de C1)

Es la "corrección manual por track en la web como pieza del producto" que el Cap 6 comprometió.

`GET /jobs/{id}/tramos` (sesión en `listo`):

```json
{
  "cobertura": {"A": 0.62, "B": 0.58, "sin_asignar": 0.18},
  "separacion_color": 1.375,
  "tramos": [
    {"track": 26, "t_desde": 192.0, "t_hasta": 220.4, "motivo": "color_insuficiente",
     "recorte_url": "/jobs/…/tramos/26/recorte", "asignado": null}
  ]
}
```

`motivo` sale del diagnóstico de la asignación y se limita a cuatro valores con su texto en la web:
`color_insuficiente` ("guante tapado o poca luz"), `sin_coexistencia` ("cruce o cuerpo a cuerpo"),
`corto` ("salió de cuadro"), `abstencion` ("no se pudo decidir"). Se listan solo los tramos de
más de 1 s, ordenados por duración.

`GET /jobs/{id}/tramos/{track}/recorte?t=`: recorte del cuadro central del tramo, mismo esquema de
ticket que los candidatos.

`PUT /jobs/{id}/tramos`, cuerpo:

```json
{"asignaciones": [{"track": 26, "rol": "A"}, {"track": 29, "rol": "ninguno"}]}
```

- Valida que cada track exista en la sesión. Guarda las asignaciones en `sesion.json`
  (`identidad.asignaciones_manuales`) y en la base (tabla nueva `asignaciones_manuales`: sesión,
  track, rol, usuario, fecha), y encola la etapa `completar` con `manual: true`.
- `completar` con `manual: true` reutiliza la caché de pose y tracking, aplica la asignación por
  color, encima fuerza las manuales, y vuelve a correr detector, guardia y clasificación. No
  repite pose.
- La Fight-Card resultante suma en `identidad`: `tramos_total`, `tramos_asignados_auto`,
  `tramos_asignados_manual`.
- Las asignaciones manuales son también evidencia para el Cap 7: cuántos tramos tuvo que asignar
  una persona es la medida de cuánto le falta a C1 en uso real.
- Archivos: `orquesta.py` (función `completar` y nuevo `listar_tramos`), `app.py` (tres rutas),
  `modelos.py` (tabla), `worker.py` (parámetro `manual`).
- Pruebas: asignar un tramo a A sube `cobertura.A` y suma los golpes de ese tramo; asignar
  "ninguno" no cambia los conteos; el trabajo `completar` manual no invoca la etapa de pose (se
  verifica con el registro de etapas).

### B4 · "No fue un golpe"

- `EntradaCorreccion.tipo` acepta `jab|cross|hook|uppercut|no_golpe`.
- `aplicar_correcciones` marca el golpe con `descartado: true` cuando la corrección es `no_golpe`.
  El golpe sigue en `golpes[]` y en `total` (es lo que el sistema detectó); se agrega
  `peleadores.X.descartados` con el conteo.
- El CSV suma la columna `descartado`.
- Estas correcciones son etiquetas de falso positivo para el detector, que hoy no recibe ninguna
  señal del producto. Se exportan igual que las de tipo.
- Archivos: `esquemas.py`, `boxtwin/mvp/fightcard.py`, `exportar.py`.
- Pruebas: corregir a `no_golpe` y después a `jab` deja el golpe sin `descartado` y con tipo jab;
  el total no cambia en ningún caso.

### B5 · Panel y configuración del producto

`GET /producto` (sin autenticación; no expone nada sensible):

```json
{
  "registro": "invitacion",
  "max_bytes": 6442450944,
  "max_minutos": 30,
  "formatos": [".mp4", ".mov", ".m4v"],
  "factor_procesamiento": {"min": 1.134, "mediana": 1.447, "max": 2.624,
                           "medido_sobre": "campaña de costo del Cap 5, estación de desarrollo"},
  "criterios": {
    "C2": {"estado": "pendiente", "fecha": null},
    "C3": {"estado": "pendiente", "fecha": null}
  },
  "banderas": {"comparacion_rounds": false, "guardia": false,
               "sesiones_individuales": false, "postura_2d": false},
  "dominio": {"pausa_combinacion_s": 0.5, "perfil_preliminar_sesiones": 5}
}
```

- `criterios` se configura por variables de entorno (`BOXTWIN_C2=aprobado|rechazado|pendiente`,
  ídem C3) que Lucas cambia cuando mide, con la fecha. `banderas.comparacion_rounds` es
  `C2 == aprobado`; `banderas.guardia` es `C3 == aprobado`. La web no decide nada sobre esto.
- `factor_procesamiento` reemplaza el "≤ 20 min" del diseño (D2). Cuando se mida C5 sobre las
  sesiones de prueba, se actualizan los tres números.
- `dominio` son decisiones de Lucas; los valores de arriba son iniciales y se confirman antes de
  la prueba con usuarios.

`GET /panel` (autenticado):

```json
{
  "peleadores_activos": 8,
  "peleadores_preliminares": 2,
  "sesiones_mes": {"total": 14, "listo": 12, "en_curso": 1, "espera_siembra": 1, "fallo": 0},
  "rounds_mes": {"total": 41, "sparring": 22, "bolsa": 12, "combate": 4, "sombra": 3},
  "golpes_detectados_mes": 2318,
  "recall_detector": 0.484,
  "pendientes": [{"sesion_id": "…", "estado": "espera_siembra", "titulo": "Sparring de T. Ríos vs N. Ferreyra"}]
}
```

- "Activos": peleadores con al menos una sesión en los últimos 60 días. "Preliminares": activos
  con menos de `perfil_preliminar_sesiones` sesiones.
- `rounds_mes` y `golpes_detectados_mes` se calculan leyendo las Fight-Cards de las sesiones del
  mes; se cachean por sesión con la fecha de modificación del archivo para no leer todo en cada
  pedido.
- `GET /jobs` suma `progreso` (el mismo de `GET /jobs/{id}`) solo para sesiones en curso.
- Archivos: `app.py`, nuevo `panel.py`, `config.py`.
- Pruebas: un usuario no ve sesiones de otro en el panel; una sesión sin Fight-Card no rompe el
  cálculo.

### B6 · Datos del peleador

`EntradaBoxeador` y el modelo `Boxeador` suman `categoria` (texto corto, por ejemplo "Welter"),
`altura_m` (1,20 a 2,20) y `envergadura_m` (1,20 a 2,40), todos opcionales. Migración en el mismo
archivo SQL de B1.

`GET /boxeadores` hoy devuelve id, nombre, guardia, notas y fecha de alta. Suma, calculados con
una consulta agrupada sobre `sesiones` (sin leer Fight-Cards): `sesiones` (cantidad, como A o
como B) y `ultima_sesion` (fecha). Es lo que el panel y la lista de peleadores muestran debajo de
cada nombre.

### B7 · Perfil extendido

`GET /boxeadores/{id}/perfil?contexto=todas|bolsa|sombra|sparring|combate` suma a lo que ya
devuelve:

```json
{
  "sesiones_por_contexto": {"bolsa": 4, "sombra": 2, "sparring": 4, "combate": 1},
  "rounds_total": 29,
  "por_contexto": [
    {"contexto": "sparring", "golpes_por_minuto": 12.8, "jab_fraccion": 0.37,
     "guardia_opuesta_arriba": null, "caida_ultimo_round": null}
  ],
  "rounds_promedio": null,
  "combinaciones": [
    {"secuencia": [{"familia": "jab", "lado": "adelante"}, {"familia": "cross", "lado": "atras"}],
     "n": 64, "estimado": true}
  ]
}
```

- `guardia_opuesta_arriba` es nulo si la bandera de guardia está apagada; `caida_ultimo_round` y
  `rounds_promedio` son nulos si la de rounds está apagada. La API devuelve nulo y no el número:
  así ningún cliente lo puede mostrar por error.
- `caida_ultimo_round`: variación de golpes por minuto del último round contra el anterior, en
  sesiones con tres rounds o más, promedio entre sesiones.
- `combinaciones`: golpes consecutivos del peleador separados por no más de
  `pausa_combinacion_s`; las cinco secuencias de 2 o 3 golpes más frecuentes. `lado` sale de la
  guardia del peleador y del brazo (ortodoxo: izquierdo adelante); nulo si la guardia no está
  cargada. Los golpes descartados (B4) no cuentan; los corregidos cuentan con su tipo corregido.
- Archivos: `perfil.py`, `app.py`.
- Pruebas: un peleador sin guardia cargada recibe `lado: null`; con las banderas apagadas los
  campos condicionados son nulos aunque haya datos.

### B8 · Sesiones de un solo peleador (P1)

- `Siembra.track_b` pasa a opcional cuando la sesión es de `bolsa` o `sombra`; para `sparring` y
  `combate` sigue obligatorio.
- `completar` con una sola semilla asigna por color contra un solo perfil y deja el resto como
  "ninguno". La Fight-Card trae solo `peleadores.A`.
- Antes de encender `banderas.sesiones_individuales` se mide, con criterio fijado ahora: sobre dos
  videos propios de bolsa o sombra, la persona queda asignada en al menos el 95% de los cuadros en
  que está en cuadro, verificado a ojo por Lucas sobre el video procesado. Si no se llega, bolsa y
  sombra quedan como "próximamente" y se declara en el Cap 7.

### B9 · Postura promedio en 2D y distancia de trabajo (P2)

Se especifica para no perderlo, no se compromete:

- Postura: promedio de los keypoints normalizados (centrados en el punto medio de hombros,
  escalados por ancho de hombros, como las features del detector) en cuadros sin golpe, por
  peleador y contexto. Se dibuja de frente en 2D; no se rota ni se presenta como 3D.
- Distancia: separación entre caderas de A y B en anchos de hombro, en bandas larga, media y
  corta. Los límites de las bandas son decisión de dominio de Lucas y se fijan antes de calcular.

---

## 3. Modelo de datos, resumen de cambios

| Tabla | Cambio | Tarea |
|---|---|---|
| `sesiones` | `contexto`, `practica`, `rounds`, `consentimiento_en`, `consentimiento_por` | B1 |
| `boxeadores` | `categoria`, `altura_m`, `envergadura_m` | B6 |
| `asignaciones_manuales` | nueva: `id`, `sesion_id`, `track`, `rol`, `usuario_id`, `creada` | B3 |
| `correcciones` | sin cambio de columnas; `tipo` admite `no_golpe` | B4 |

## 4. Contrato `fightcard.json` 0.2

Compatible hacia atrás: todo lo de 0.1 sigue igual y en el mismo lugar. Se agrega:

| Campo | Tipo | Tarea |
|---|---|---|
| `version` | `"0.2"` | |
| `peleadores.X.golpes[].descartado` | bool | B4 |
| `peleadores.X.descartados` | entero | B4 |
| `identidad.tramos_total`, `tramos_asignados_auto`, `tramos_asignados_manual` | enteros | B3 |
| `identidad.separacion_color` | número | B3 (se copia del diagnóstico con siembra, que ya existe) |

Además, `GET /fightcards/{id}` agrega al responder un bloque `sesion` que no se guarda en el
archivo, porque sale de la base y puede cambiar después del procesamiento:

```json
"sesion": {"contexto": "sparring", "practica": null,
           "boxeadores": {"A": {"id": "…", "nombre": "Tomás Ríos", "guardia": "ortodoxa", "categoria": "Welter"},
                          "B": null}}
```

La web acepta 0.1 y 0.2: a una 0.1 le completa `descartado: false`, `descartados: 0` y los campos
de tramos en nulo, y la pantalla de tramos dice "esta sesión se procesó antes de la revisión de
tramos; reprocesala para revisarlos".
