# E-06 — Valores para la hoja Supuestos

`BoxTwin_Cap5_Flujo_de_fondos.xlsx` no está en el repositorio, así que no pude ver la
hoja ni escribir en ella. Estos son los tres valores para cargarlos como celdas de
entrada, en [supuestos_planilla.csv](supuestos_planilla.csv) y acá abajo en claro.

| Celda | Valor | Unidad | Estado |
|---|---|---|---|
| S-19 | `1,447` | horas de cómputo por hora de video | medido |
| S-17 | dejar como está | W | **no medido**, sigue siendo supuesto |
| S-04 | `0,8755` | USD por hora de video | derivado de S-19 |

Rangos, por si la hoja tiene celdas de mínimo y máximo:

| Celda | Mínimo | Máximo |
|---|---|---|
| S-19 | 1,134 | 2,624 |
| S-04 | 0,6861 | 1,5876 |

Tres cosas antes de pegar.

El CSV usa punto decimal porque es formato de máquina. Si la planilla está en configuración
regional española hay que pegar con coma, o el 1.447 entra como mil cuatrocientos cuarenta
y siete.

S-17 va vacío a propósito. No se midió y no corresponde poner un cero ni dejar el 250
como si estuviera respaldado. Si la hoja necesita un número para calcular, hay que
marcarlo como supuesto en la celda.

El máximo de S-04, 1,5876, sale de arrastrar el máximo de S-19 al mismo precio de lista.
Si en cambio se quiere el techo por instancia más cara, con `g5.xlarge` a 1,006 USD/h da
1,675 USD por hora de video.
