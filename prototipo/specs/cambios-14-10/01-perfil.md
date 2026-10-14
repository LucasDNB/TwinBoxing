# 01 · Perfil táctico (pantalla 06)

Tareas: B15, B16, B17 (anotador y API) y W20 a W24 (web). Decisiones en `00-README.md`.

---

## 1. B15 · Rounds de 3 minutos (DC-1, DC-5)

### Anotador

`boxtwin/mvp/sesion.py` suma `segmentos(duracion_s, round_s, descanso_s)`, una función pura:

- Con `round_s` declarado devuelve lo mismo que `ventanas_de_round` y `origen = "declarados"`.
- Sin `round_s`, corta en `SEGMENTO_S = 180` sin descanso desde el segundo 0, con
  `origen = "segmentos_180s"`. Si el último segmento dura menos de `MINIMO_ULTIMO_S = 60`, se
  suma al anterior. Un video de menos de 60 s queda en un solo segmento.

`Sesion.nueva` la usa en lugar de `ventanas_de_round`, y la Fight-Card escribe
`video.rounds_origen`. `enriquecer` completa los rounds de una Fight-Card que no los trae
(`video.rounds` vacío): calcula los segmentos y recuenta `por_round` desde los golpes, con la
misma cuenta que `construir`. Reanudable como hasta ahora: entra en la condición de "ya está al
día".

### API

- `config.rounds_estimado` (`BOXTWIN_ROUNDS_ESTIMADO`, por defecto 1). La bandera
  `comparacion_rounds` vale `c2 == "aprobado" or rounds_estimado`, en `/producto` y en el perfil.
- El perfil, para una Fight-Card sin rounds (0.1, 0.2), calcula los segmentos con la misma
  función del anotador (importada, no copiada) y recuenta `por_round` sobre los golpes. Así una
  sesión vieja aporta al volumen por round sin reprocesar.
- `rounds_promedio` suma `origen`: `declarados`, `segmentos_180s` o `mixto`.

### Pruebas (B15)

1. 600 s sin rounds: tres segmentos de 180 s y un último de 60 s (60 no es menos de 60).
2. 590 s sin rounds: el último (50 s) se suma al tercero, que queda de 230 s.
3. 45 s sin rounds: un solo segmento.
4. Con `round_s=120, descanso=30`: las ventanas declaradas, `origen = "declarados"`.
5. `enriquecer` sobre una Fight-Card sin rounds deja `por_round` y la suma por round es igual al
   total.
6. Perfil con una 0.2 sin rounds: `rounds_promedio` no es nulo y dice `segmentos_180s`.
7. `/producto` con C2 pendiente y `BOXTWIN_ROUNDS_ESTIMADO=0`: `comparacion_rounds` falso; con el
   valor por defecto, verdadero, y `criterios.C2` sigue `pendiente`.

## 2. B16 · Distribución por mano y altura, y exposición dirigida (DC-2 a DC-4)

Módulo nuevo `boxtwin-api/src/boxtwin_api/zonas.py`, funciones puras:

- `ZONAS = ("cabeza_mano_izq", "cabeza_mano_der", "cuerpo_mano_izq", "cuerpo_mano_der")`.
- `contar_zonas(golpes)`: sobre los golpes no descartados, cuenta por zona (altura vigente, ya
  corregida si el entrenador la corrigió, por la mano del golpe). Los golpes sin altura van a
  `sin_dato`.
- `corregir(conteo, recall)`: divide cada zona por el recall de su altura y renormaliza las
  fracciones (DC-4). Devuelve las fracciones corregidas; los conteos crudos no se tocan.
- `RECALL_ALTURA = {"cabeza": 0.512, "cuerpo": 0.348}` con su fuente
  (`docs/experiments/2026-10-13-cabeza-cuerpo.md` §7). La diferencia (0,164) supera 0,10, así que
  la corrección se aplica.

`GET /boxeadores/{id}/perfil` suma dos bloques:

```json
"ataque": {
  "zonas": {"cabeza_mano_izq": {"n": 120, "fraccion": 0.41, "fraccion_corregida": 0.36}, …},
  "manos": {"izq": 212, "der": 140},
  "alturas": {"cabeza": 230, "cuerpo": 92, "sin_dato": 30},
  "total": 352
},
"exposicion": {
  "tipo": "dirigidos",
  "zonas": {…igual que ataque…},
  "total": 294, "sin_dato": 22, "sesiones": 6,
  "por_rival": [{"boxeador_id": "…", "nombre": "N. Ferreyra", "total": 140}],
  "sesgo_detector": {"recall_cabeza": 0.512, "recall_cuerpo": 0.348, "corregido": true}
}
```

- `ataque` sale de los golpes del propio peleador; `exposicion`, de los golpes del otro lado de
  cada sesión (solo sparring y combate: en bolsa y sombra no hay rival).
- Los dos son nulos con la bandera `objetivo` apagada (CA-15.3). `manos` del ataque no depende de
  la altura y viaja siempre.
- `por_rival` agrupa por boxeador asignado del otro lado; un rival sin perfil va como
  "sin perfil".
- La bandera `exposicion` de `/producto` vale `c8 == "aprobado" and c9 != "pendiente"` o, por
  decisión de producto, la misma que enciende la altura.

### Pruebas (B16)

1. Cuatro zonas más `sin_dato` suman los golpes no descartados del rival (CA-15.1).
2. Un golpe descartado no cuenta; uno con la altura corregida cuenta en la zona corregida.
3. La corrección: con 50 a la cabeza y 50 al cuerpo, la fracción corregida del cuerpo es
   `(50/0,348) / (50/0,512 + 50/0,348)` ≈ 0,595.
4. En bolsa y sombra no hay exposición.
5. Con la bandera de altura apagada, `ataque.zonas` y `exposicion` vienen nulos y `ataque.manos`
   viene igual.

## 3. B17 · Distancia de trabajo y estilo (DC-7)

- Columnas nuevas en `boxeadores`: `distancia` (`larga`, `media`, `corta` o nulo) y `estilo`
  (`fajador`, `estilista` o nulo). La API las agrega sola al arrancar; la migración de
  PostgreSQL va en `despliegue/migraciones/2026-10-14-perfil.sql`.
- `EntradaBoxeador` las acepta con esos valores; cualquier otro da 422.
- El perfil y `GET /boxeadores` las devuelven.

Pruebas: crear y editar con los dos campos; un valor fuera de la lista da 422; el perfil los
devuelve.

## 4. Web

### W20 · Distribución de golpes

El bloque pasa a ser una tabla de dos por dos: mano (izquierda, derecha) por altura (cabeza,
cuerpo), con el conteo y el porcentaje en cada celda y los totales en los bordes.

- Los totales por mano son medidos (salen del detector). Las celdas y los totales por altura son
  estimación (≈, borde punteado), con la nota de la altura (`textos.estimacionAltura`) y la del
  sesgo: "El detector encuentra cerca de 3 de cada 10 golpes al cuerpo y 5 de cada 10 a la
  cabeza: el cuerpo está subrepresentado."
- Debajo, más chico, la mezcla por tipo que estaba antes, como estimación.
- Con la bandera de altura apagada, solo la fila por mano y la mezcla por tipo.

### W21 · Volumen por round

- Estado "medido" con la bandera `comparacion_rounds`, que ahora se enciende por decisión de
  producto. Rótulo "comparación entre rounds sin validar" mientras C2 no esté aprobado.
- Si el origen es `segmentos_180s`, las filas dicen "S1, S2…" y el bloque aclara: "El video no
  tenía rounds declarados: se dividió en segmentos de 3 minutos." Con `mixto`, "R/S".

### W22 · Mapa de exposición

Lo que pedía `ampliacion-13-10/04` §4, con estas precisiones:

- Título "Hacia dónde le tiran". Subtítulo fijo: "Golpes de sus rivales que el sistema detectó,
  según la altura y la mano con que vinieron. No dice si tocaron."
- Silueta frontal en SVG con cabeza y torso partidos al medio. La mitad que queda enfrente de la
  mano izquierda del rival lleva esa mano; la otra, la derecha. Cada zona con relleno por fracción
  corregida y el porcentaje escrito; nunca solo color.
- Rótulo de la corrección: "Corregido por el detector: encuentra 3 de cada 10 golpes al cuerpo y 5
  de cada 10 a la cabeza." con los conteos crudos en la tabla.
- Tabla de cuatro filas (zona, golpes detectados, fracción corregida) como alternativa accesible.
- Rivales que aportan; si uno pasa del 70% del total, "El mapa describe sobre todo a {nombre}".
- Con menos de 50 golpes dirigidos, "mapa preliminar" y las zonas sin colorear.
- Filtro bolsa o sombra: "En estos contextos no hay rival."

### W23 · Avatar 3D estático (DC-6)

- Componente `components/boxtwin/avatar/Avatar3D.tsx` cargado con `next/dynamic` y `ssr: false`;
  Three.js entra en un chunk aparte que no cuenta en el JS inicial de la ruta.
- Maniquí procedural de primitivas (cápsulas y esferas), en guardia: el brazo adelantado
  semiextendido, el atrasado recogido con la mano a la altura del mentón. Ortodoxo con el pie
  izquierdo adelante; zurdo, espejado.
- Escala: altura total a `altura_m`; brazos repartidos para cerrar `envergadura_m` con un ancho de
  hombros de 0,23 × altura. Sin medidas, proporciones estándar y el aviso "sin medidas cargadas:
  no está a escala".
- Se puede rotar arrastrando. Sin WebGL, un texto con las medidas en lugar del lienzo.
- Rótulo "ilustrativo, no reconstruido del video" y `aria-label` con las medidas.

### W24 · Distancia de trabajo y estilo

- En la hoja de edición del peleador (`HojaPeleador`), dos grupos de opciones: distancia (larga,
  media, corta) y estilo (fajador, estilista), con "sin cargar" para dejar nulo.
- En el perfil, el bloque "Distancia de trabajo" muestra los dos datos con el rótulo "cargado a
  mano", y si faltan, un enlace para cargarlos. También van en la línea de datos del encabezado.

## 5. Criterios de aceptación

- CA-20.1. Con la altura encendida, las cuatro celdas más "sin dato" suman los golpes detectados
  no descartados del perfil filtrado.
- CA-21.1. Una sesión sin rounds declarados de 410 s aporta dos segmentos al volumen por round
  (180 s y 230 s: el resto de 50 s se suma al segundo), y una de 420 s aporta tres (180, 180, 60).
- CA-22.1. El mapa no usa "recibió", "conectó", "le pegaron" ni "alcanzado" (CA-15.2).
- CA-22.2. La tabla accesible tiene las cuatro zonas con los conteos crudos.
- CA-23.1. La ruta del perfil sigue debajo de 380 KB de JS inicial con el avatar.
- CA-24.1. Cargar distancia y estilo en la hoja se ve en el perfil al volver.
