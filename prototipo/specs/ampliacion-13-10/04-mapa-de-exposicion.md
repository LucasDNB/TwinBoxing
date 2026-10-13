# 04 · Mapa de exposición

Pedido (13-10): el mapa de exposición del perfil táctico (pantalla 06).

Tareas: B12 (API), W15 (web). Esfuerzo: 2 días. Depende de B11 (`03`).

---

## 1. Qué mide esta versión, y qué no

El glosario de la tesis (Cap 3) define el mapa de calor de exposición como la frecuencia con la
que cada zona corporal de un peleador "resultó alcanzada" en las sesiones registradas. Alcanzada
es contacto, y un sistema monocular no establece contacto (CLAUDE.md §2). La heurística de
conexión que daría "alcanzada" como estimación es la sub-hipótesis 2, que no está implementada.

Lo que sí se puede construir con lo medido es esto: a qué zonas del peleador fueron dirigidos los
golpes que el sistema detectó de sus rivales. Es la mitad del concepto (la presión que recibe cada
zona) sin la otra mitad (si entró). Esta versión construye eso, con ese nombre en pantalla, y deja
el lugar para "alcanzada" cuando exista la heurística de conexión.

Consecuencia para la tesis: o se ajusta la definición del glosario, o el mapa de esta versión se
llama distinto. La propuesta concreta (ANTES / DESPUÉS) está en `08` §5.

## 2. Las zonas

Dos alturas, que salen de B11: cabeza y cuerpo.

DA-4 · El lado. Tomada: por mano del rival.

- Por mano del rival (tomada): cada altura se parte según la mano con la que vino el golpe
  (izquierda o derecha del rival). Sale del brazo del detector, que es medido. Cuatro zonas:
  cabeza por la izquierda del rival, cabeza por la derecha, cuerpo por la izquierda, cuerpo por la
  derecha. No afirma en qué lado del cuerpo pegó.
- Por lado del cuerpo que recibe: requiere saber si el golpe va al lado izquierdo, al centro o al
  derecho del receptor. Un recto va al centro y un gancho al costado: depende del tipo, que es la
  estimación que no generaliza, o de la posición horizontal de la mano, que con la cámara de
  costado se aplasta. No hay anotación para validarlo.

Con la tomada, el dibujo es la silueta de frente del peleador con la cabeza y el torso
partidos al medio. La mitad de la silueta que queda enfrente de la mano izquierda del rival recibe
los golpes de esa mano. El rótulo lo dice con palabras, para que nadie lo lea como "le pegaron en
el lado izquierdo".

## 3. B12 · Cálculo

Para un peleador X y un filtro de contexto (solo sparring y combate; en bolsa y sombra no hay
rival):

- Para cada sesión en que X es A (o B), se toman los golpes detectados del otro lado, no
  descartados, con `objetivo` no nulo.
- Se cuentan por zona. Se informa el total, la fracción por zona y la tasa por minuto de rival
  visible (cobertura del rival × duración).
- Un rival "sin perfil" cuenta igual: el mapa es de X, no del rival.

`GET /boxeadores/{id}/perfil` suma:

```json
"exposicion": {
  "tipo": "dirigidos",
  "zonas": {
    "cabeza_mano_izq": {"n": 120, "fraccion": 0.41},
    "cabeza_mano_der": {"n": 98,  "fraccion": 0.33},
    "cuerpo_mano_izq": {"n": 31,  "fraccion": 0.11},
    "cuerpo_mano_der": {"n": 45,  "fraccion": 0.15}
  },
  "total": 294, "sin_dato": 22, "sesiones": 6, "por_minuto_rival_visible": 21.3,
  "por_rival": [{"boxeador_id": "…", "nombre": "N. Ferreyra", "total": 140}],
  "sesgo_detector": {"recall_cabeza": null, "recall_cuerpo": null}
}
```

- Nulo entero si la bandera `objetivo` (C8) está apagada.
- `por_rival` existe porque un mapa armado con un solo sparring partner describe a ese rival tanto
  como a X. Si un rival aporta más del 70% del total, la web lo dice.

### Sesgo del detector por zona (C9)

El mapa son fracciones, y una fracción no se ve afectada por el recall bajo si el recall es
parejo entre zonas. Si el detector encuentra menos golpes al cuerpo que a la cabeza (por ejemplo,
porque al cuerpo el brazo se extiende menos y la firma temporal es más débil), el mapa subestima
el cuerpo sin que nadie lo note.

C9, pre-registrado: con la salida de punta a punta sobre las cuatro fuentes de medición de `03`,
recall del detector sobre golpes anotados a la cabeza y sobre golpes anotados al cuerpo.

- Si la diferencia es ≤ 0,10, el mapa se muestra tal cual.
- Si es mayor, el mapa se muestra corregido (cada zona dividida por su recall y renormalizada) y
  lo dice en el rótulo, con los dos recalls a la vista.
- Los dos recalls quedan en `sesgo_detector` y en el Cap 7.

## 4. W15 · Web (perfil, pantalla 06)

- El bloque "Mapa de exposición" sale de "No disponible" y pasa a estimación.
- Silueta frontal en SVG (sin 3D: el 3D es `06`), cuatro zonas con relleno según la fracción,
  usando la escala secuencial del sistema de diseño y nunca solo color: cada zona lleva el
  porcentaje escrito.
- Título "Hacia dónde le tiran" y subtítulo fijo: "Golpes de sus rivales que el sistema detectó,
  según la altura y la mano con que vinieron. No dice si tocaron."
- Debajo, la lista de rivales que aportan al mapa, con el aviso si uno pasa del 70%.
- Si C9 obligó a corregir, el rótulo "corregido por el detector: encuentra {rc} de cada 10 al
  cuerpo y {rh} de cada 10 a la cabeza".
- Alternativa accesible: la misma información como tabla de cuatro filas, que es lo que lee el
  lector de pantalla.
- Con menos de 50 golpes dirigidos, "mapa preliminar" y sin colorear zonas (solo números).

## 5. Lo que queda para cuando exista la heurística de conexión

El campo `tipo: "dirigidos"` está para eso. Cuando la sub-hipótesis 2 se implemente y se mida, el
mismo bloque suma `tipo: "alcanzados_estimados"` con su margen, y la pantalla deja elegir. Las
anotaciones propias ya tienen el resultado estimado por el anotador (`landed`, `blocked`,
`slipped`, `missed`, `unknown`) para medirla: sobre las seis fuentes hay 317 golpes marcados como
conectados.

## 6. Criterios de aceptación

- CA-15.1. La suma de las cuatro zonas más `sin_dato` es igual a los golpes detectados de sus
  rivales en las sesiones filtradas.
- CA-15.2. El mapa nunca usa las palabras "recibió", "conectó", "le pegaron" ni "alcanzado".
- CA-15.3. Con la bandera de C8 apagada, el bloque sigue "No disponible".
- CA-15.4. Filtrar por bolsa o sombra muestra "en estos contextos no hay rival".
