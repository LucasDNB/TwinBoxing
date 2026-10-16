# E6 · Rol (mano adelantada o atrasada) por el brazo del detector y la guardia

16-10-2026 · spec: `docs/specs/2026-10-16-clasificador-geometrico/` §5 del diseño, pre-registrado
en `703fd11d8a3f9a74d871777357491ba8993cd9ba` · script `scripts/clasificador_geo/rol_por_brazo.py`
· datos en `datos/2026-10-16-rol-por-brazo-E6.json` · decisiones en
`scripts/clasificador_geo/TAREAS.md`

## 1. Qué se mide (pre-registrado)

P7: si el rol sacado del brazo del detector más la guardia acierta más que el clasificador.
En producción jab y cross salen del argmax del clasificador sobre seis clases; fuera de
distribución ese clasificador acierta el eje lead/rear en 0,656 (E1, 15-10), por debajo de
decir siempre adelantada. El detector, en cambio, ya trabaja por brazo.

- Regla: rol = `arm_role(guardia, brazo del detector)`: adelantado si guardia ortodoxa y brazo
  izquierdo, o zurda y brazo derecho. La guardia es la del evento anotado emparejado.
- Variante sin umbral: la misma regla con todos ortodoxos.
- Verdad: el rol anotado.
- Lectura: la regla mejora al clasificador si su exactitud de rol lo supera en 0,05 o más.
  Advertencia escrita de antemano: el detector de producción se entrenó con estas fuentes, así
  que el acuerdo de brazo es optimista.

Datos: las cuatro sesiones de C8 nivel 2 (`~/boxtwin/mediciones/c8-nivel2/`). Orientación A/B
con `medir_sesion`, como `pares_nivel2`. Emparejamiento por peleador y sin separar por brazo,
con `emparejar` a IoU 0,3, de todos los golpes de la Fight-Card contra los anotados completos:
así un golpe del detector en el brazo equivocado se empareja igual y se ve el error, cosa que el
emparejamiento por carril de `pares_nivel2` no puede.

## 2. Lo que cambió respecto del diseño

Las cuatro Fight-Cards de C8 nivel 2 se completaron sin clasificador: los 1285 golpes tienen
`tipo` en None y el checkpoint del clasificador está vacío. El comparador del diseño ("`tipo`
de la Fight-Card") no existe en esas sesiones. Se usó E1, que es la configuración de producción
medida con la fuente afuera (semilla 0, `pred.pkl` del 15-10): el argmax de sus seis clases
sobre el evento anotado emparejado, jab = adelantado, cross = atrasado, y un hook o uppercut
sobre un recto anotado cuenta como error. Es la lectura más estricta: el checkpoint de
producción vio las cuatro fuentes y daría un número en distribución. Queda como pregunta
abierta si se quiere también ese número (decisión 11 de `TAREAS.md`).

## 3. Resultados

390 pares detector-anotación sobre las cuatro fuentes; 259 son rectos anotados.

### 3.1 Total

| | Exactitud (aciertos / n) | Wilson 95% |
|---|---|---|
| Acuerdo de brazo, todos los pares | 0,979 (382 / 390) | 0,960–0,990 |
| Rol, regla | 0,985 (255 / 259) | 0,961–0,994 |
| Rol, regla con todos ortodoxos | 0,811 (210 / 259) | 0,759–0,854 |
| Rol, clasificador (E1) | 0,726 (188 / 259) | 0,669–0,777 |
| Rol, siempre adelantada | 0,699 (181 / 259) | 0,640–0,752 |

### 3.2 Por fuente

| Fuente (golpes FC, pares, rectos) | A/B invertidos | Acuerdo de brazo | Regla | Todos ortodoxos | Clasificador (E1) | Siempre adelantada |
|---|---|---|---|---|---|---|
| 03-sparring (83, 62, 43) | no | 0,952 | 0,977 | 0,977 | 0,628 | 0,674 |
| 04-sparring (75, 58, 36) | no | 0,966 | 0,972 | 0,417 | 0,694 | 0,722 |
| sparring-3-rounds (244, 199, 120) | no | 0,985 | 0,983 | 0,983 | 0,717 | 0,667 |
| pacquiao_margarito (883, 71, 60) | sí | 1,000 | 1,000 | 0,583 | 0,833 | 0,767 |

## 4. Lectura contra el pre-registro

P7: mejora, contra E1 como sustituto del clasificador de producción (§2), no contra el
checkpoint de producción. La regla acierta el rol en 0,985 contra 0,726, 0,259 por encima del
mínimo de 0,05, y los intervalos de Wilson no se tocan. Pero el 0,985 es una cota superior, por
construcción: la regla usa la guardia anotada, que es la misma con la que se deriva el rol
verdadero, así que la regla acierta exactamente cuando el detector acierta el brazo (la
auditoría lo verificó en las 259 filas). Lo que mide es el acuerdo de brazo del detector sobre
los rectos. Hay que decir al lado:

1. El acuerdo de brazo (0,979) es optimista: el detector de producción se entrenó con estas
   cuatro fuentes. Sobre una fuente nueva sería más bajo, y la regla hereda cada error de brazo.
   En estos datos, los cuatro errores de la regla son exactamente los cuatro rectos con el brazo
   cambiado (`filas` del JSON).
2. La guardia sale de la anotación. Con todos ortodoxos la regla baja a 0,811, y en las fuentes
   con un zurdo (04-sparring, pacquiao_margarito) cae debajo del clasificador. El producto
   necesita la guardia por peleador; cuánto pierde la regla con una guardia estimada no se
   midió.

E6 mide el rol solo sobre los rectos que el detector encontró (259), no sobre todos los anotados.

## 5. Auditoría

Auditor independiente con las instrucciones de `auditor-medicion`, 16-10-2026, ítem 9: PASA.
Desde `filas`, regla 255/259, todos ortodoxos 210/259, clasificador 188/259, diferencia
+0,2587, mejora. El código empareja por peleador sin separar por brazo a IoU 0,3 y solo contra
completos; los 1285 golpes de las cuatro Fight-Cards tienen `tipo` None; el orden de los
`pred.pkl` de E1 sigue al fold en las cuatro fuentes y el argmax coincide con las 259 filas.
Verificó también que regla = verdad si y solo si el brazo es correcto, en las 259 filas (ver
§4). Huellas intactas.

## 6. Consecuencias

- Para jab contra cross, el brazo del detector más la guardia es mejor fuente que el
  clasificador, con las dos salvedades del §4. El pre-registro §4.3 dice que, si E6 pasa, el
  cambio en la Fight-Card es otro spec: no se tocó nada en producción.
- Ese spec necesita una guardia por peleador que no venga de la anotación (la siembra del
  entrenador, por ejemplo), y medir el acuerdo de brazo sobre una fuente que el detector no vio.
