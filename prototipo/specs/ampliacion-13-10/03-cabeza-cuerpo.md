# 03 · Separación entre cabeza y cuerpo en la Fight-Card

Pedido (13-10): que la Fight-Card separe los golpes a la cabeza de los golpes al cuerpo.

Tareas: B11 (etapa `enriquecer`, regla de altura, contrato 0.3, medición C8), W14 (web).
Esfuerzo: 3 a 4 días, de los cuales uno es la medición.

---

## 1. Por qué ahora se puede

`01-producto.md` §2 lo dejó fuera con este motivo: "el clasificador del MVP da familia, no
altura". Es cierto, pero hay otro camino que no pasa por el clasificador, y el relevamiento del
13-10 encontró que ya está todo lo que hace falta para medirlo:

1. Las anotaciones propias tienen la altura de cada golpe. El anotador obliga a elegirla (el
   diálogo arranca sin valor y no deja guardar sin ella), así que no es un valor por defecto.
   Sobre las seis fuentes con identidad: 972 golpes, 810 a la cabeza y 162 al cuerpo.

| Fuente | Golpes | Cabeza | Cuerpo | Cuerpo % |
|---|---|---|---|---|
| 01-sparring | 99 | 72 | 27 | 27,3 |
| 02-sparring | 67 | 64 | 3 | 4,5 |
| 03-sparring | 131 | 114 | 17 | 13,0 |
| 04-sparring | 124 | 101 | 23 | 18,5 |
| sparring-3 | 400 | 326 | 74 | 18,5 |
| pacquiao_margarito | 151 | 133 | 18 | 11,9 |

2. Cada sesión procesada guarda `poses.npz` con los keypoints de A y B por cuadro (forma
   `2 × T × 17 × 2`, en píxeles, con máscara de validez). La altura del golpe se puede calcular
   con la mano del que pega y el cuerpo del que recibe en el mismo cuadro, sin volver a correr
   pose.

La geometría normalizada es lo que ya demostró que transfiere entre fuentes (el detector
generaliza por eso; el clasificador, que dependía de píxeles, no). Esta regla usa lo mismo.

## 2. La regla (B11)

Para cada golpe detectado de un peleador P contra el rival R:

1. Cuadro pico: el cuadro dentro de `[t_inicio, t_fin]` en que la muñeca del brazo del golpe está
   más lejos de su propio hombro, en anchos de hombro de P. Si el detector de guantes dio caja en
   ese cuadro, se usa el centro del guante en lugar de la muñeca.
2. Referencia del rival en ese cuadro: línea de hombros de R (punto medio de los dos hombros) y
   largo del torso de R (de punto medio de hombros a punto medio de caderas).
3. Altura relativa: `h = (y_mano − y_hombros_R) / torso_R`. Negativa es por encima de los hombros
   del rival (y de la imagen crece hacia abajo).
4. Objetivo: `cabeza` si `h < τ`, `cuerpo` si `h ≥ τ`. Si R no es válido en el cuadro pico o en
   los ±2 cuadros vecinos, el objetivo queda `null` con motivo `sin_rival`.
5. Confianza: `|h − τ|` llevado a 0..1 con una logística de pendiente fija; se informa para
   ordenar y para marcar en ámbar los dudosos, no como probabilidad calibrada.

τ se elige en la etapa de calibración (§4) dentro de una grilla fijada ahora:
`τ ∈ {−0,10; 0,00; 0,10; 0,20; 0,30; 0,40}`.

Lo que la regla no ve, dicho de antemano: un golpe de un peleador más alto a la cabeza de uno
agachado, y viceversa, lo resuelve bien porque usa el cuerpo del rival en ese cuadro. Lo que no
resuelve es la profundidad: con la cámara de costado, un golpe que pasa por delante del cuerpo del
rival a la altura del pecho sin llegar se cuenta como "cuerpo". Es "a dónde fue dirigido", no "a
dónde llegó".

DA-5 · Bolsa y sombra. No hay rival. Tomada: no se calcula.

- No se calcula (tomada): el bloque dice "en bolsa y sombra no hay contra quién medir la
  altura". No hay anotación de bolsa para validar nada.
- Respecto de los propios hombros: se puede calcular, pero no se puede medir contra nada sin
  anotar sesiones de bolsa.

## 3. Etapa `enriquecer` del worker

Comando nuevo del anotador, `boxtwin-annotator enriquecer <directorio de sesión>`:

- Lee `poses.npz`, la Fight-Card y, si existen, las cajas de guantes de la caché de identidad.
- Escribe en la Fight-Card los campos de este spec, de `01` (combinaciones) y de `04` (mapa).
- No corre pose, ni detector, ni clasificador. Corre en el entorno `twinboxing_env` sin GPU.
- Lo llama `completar` al final, y también se puede correr sobre sesiones ya hechas:
  `boxtwin-annotator enriquecer --todas ~/boxtwin/datos/sesiones`. Reanudable: saltea las que ya
  tienen `version: "0.3"` y la misma versión de regla.
- Registra su tiempo en `sesion.json` (cuenta para C5).

## 4. Calibración y medición (C8), pre-registradas

Se escriben ahora y no se tocan después de ver resultados.

Partición por fuente, para no calibrar y medir sobre lo mismo:

- Calibración: 01-sparring y 02-sparring (166 golpes, 30 al cuerpo). Ahí se elige τ.
- Medición: 03-sparring, 04-sparring, sparring-3 y pacquiao_margarito (806 golpes, 132 al
  cuerpo). Ahí se reporta, una sola vez, con el τ elegido.

Dos niveles, porque mezclan errores distintos:

1. Regla sola: sobre los golpes anotados, con la identidad anotada y los keypoints de la caché de
   cada proyecto de anotación. Aísla el error de la regla.
2. Punta a punta: sobre los golpes que el sistema detectó y emparejó con uno anotado (IoU 0,3).
   Muestra lo que ve el entrenador.

Métrica: exactitud balanceada (promedio del recall de cabeza y del recall de cuerpo) y recall de
cuerpo. No se usa la exactitud simple, porque con 83% de golpes a la cabeza la regla "todo es
cabeza" ya da 0,83 y no sabe nada; su exactitud balanceada es 0,50. Esa es la línea de base a
superar y se reporta al lado.

Criterio C8, nivel 1 sobre la partición de medición:

- exactitud balanceada ≥ 0,75 y recall de cuerpo ≥ 0,60, con el intervalo de Wilson al 95% del
  recall de cuerpo reportado (con 132 golpes al cuerpo el intervalo ronda ±0,08).
- Si pasa: la bandera `objetivo` se enciende y el bloque pasa a "estimación, medida".
- Si no pasa: el bloque queda "No disponible" con el número medido en el Cap 7, igual que C3.
- Se registra además por fuente. Si una fuente queda por debajo de 0,60 de exactitud balanceada,
  se dice cuál y por qué se cree que falla (ángulo de cámara, árbitro), sin sacarla de la cuenta.

El nivel 2 se reporta sin umbral.

Resultado a `docs/experiments/AAAA-MM-DD-cabeza-cuerpo.md`, con la matriz de confusión por fuente.
Script versionado en `boxtwin-annotator/scripts/medir_objetivo.py` con su test.

## 5. Contrato 0.3

Por golpe:

```json
{"id": "g_0142", "objetivo": "cuerpo", "objetivo_confianza": 0.71, "objetivo_motivo": null}
```

`objetivo_motivo` vale `sin_rival`, `individual` (bolsa y sombra) o `null`.

Por peleador, en `total` y en cada `por_round`: `cabeza`, `cuerpo` y `objetivo_sin_dato`
(enteros; `cabeza + cuerpo + objetivo_sin_dato = total`).

En la raíz: `metodos.objetivo: {"regla": "altura_relativa_v1", "tau": 0.2, "medido": {"exactitud_balanceada": …, "recall_cuerpo": …} | null}`.

La corrección del entrenador se amplía: `PATCH /fightcards/{id}/golpes/{golpe}` acepta
`objetivo: "cabeza" | "cuerpo"` además de `tipo`. Se guarda en la tabla `correcciones` con una
columna nueva `campo` (`tipo` u `objetivo`). Como las de tipo, son etiquetas para recalibrar τ.

## 6. W14 · Web

- Fight-Card (05 y 09): fila nueva "A la cabeza / al cuerpo", A contra B, como barra partida en
  dos tonos, con el conteo y el porcentaje, en estilo de estimación. Debajo, "N sin dato" cuando
  hay golpes sin rival en cuadro. Con la bandera apagada, sigue el bloque "No disponible" de hoy
  con el texto `nodisp.altura` actualizado: "La altura del golpe se muestra cuando la regla pase
  la validación contra lo marcado a mano."
- Análisis de sesión (04): cada golpe de la lista lleva "cabeza" o "cuerpo"; la hoja de corrección
  suma el par cabeza/cuerpo debajo del tipo.
- Perfil (06): la distribución de golpes suma la división cabeza/cuerpo por familia, que el diseño
  ya dibujaba.
- Rótulo con el método: "Altura estimada por la posición de la mano respecto del rival.
  Exactitud medida: {x} sobre {n} golpes de fuentes no usadas para ajustarla."

## 7. Criterios de aceptación

- CA-14.1. `cabeza + cuerpo + sin dato` es igual a golpes detectados, por sesión y por round.
- CA-14.2. Con la bandera apagada no aparece "cabeza" ni "cuerpo" en ninguna pantalla.
- CA-14.3. Corregir la altura de un golpe cambia la fila al volver, sin reprocesar.
- CA-14.4. `enriquecer --todas` sobre las sesiones existentes no corre pose (se verifica con el
  registro de etapas) y deja todas en 0.3.
