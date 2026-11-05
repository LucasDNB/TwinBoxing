# Cierre · ¿bajar la resolución antes de procesar arregla el 4K?

5-11-2026 · pedido de Lucas · se commitea antes de medir · motivo: `docs/experiments/2026-11-04-c4-c5-sesiones.md`

## 1. Qué se mide

`sparr` (4K, 59,94 fps) y `sparr01` (4K, 30 fps) dieron 8,88 y 4,24 veces su duración en A. Pregunta:
si el video se baja antes de procesar, ¿entra en 2,0 sin perder pose?

Offline, con el comando `procesar` del entorno de A y los mismos modelos, escribiendo solo en
`~/boxtwin-mejoras/cierre/4k/`. No se suben sesiones a A. Colas vacías y candado de GPU.

| Condición | Video | Transcodificado |
|---|---|---|
| `4k` | sparr01 | ninguno: control |
| `1080` | los dos | libx264, `crf 20`, `veryfast`, mismo fps |
| `720` | los dos | libx264, `crf 23`, `veryfast`, mismo fps |
| `1080-30` | sparr | como `1080`, y a 30 fps |

## 2. Reglas

**Factor.** `(transcodificar + procesar + completar y clasificar de A) / duración`. `completar` y
`clasificar` no se corren: se toman de la sesión de A en 4K, así que el número es conservador si esas
etapas también bajan. Al lado se informa sin el transcodificado, que es el caso de hacerlo en el celular.

**Entra en tiempo** si el factor es 2,0 o menos, con el transcodificado incluido.

**Conserva la pose** si contra el caché 4K de la sesión de A se empareja al menos el 98% de las cajas y
el error mediano de muñeca es 2% del alto de la caja o menos, en todo el video. Los umbrales son los de
720p. En `1080-30` se compara cada cuadro con su cuadro de origen.

**Sirve** si entra en tiempo y conserva la pose, en los dos videos para `1080` y `720`.

**Control K-4k.** `4k` offline da `preproceso + evidencia` a menos de 10% de lo que dio A (742 s). Si no,
los tiempos offline no se comparan con A y se declara.

**K-A.** Si A o BoxTwinZ toman un trabajo durante una condición, esa condición se repite.

## 3. Queda fuera

Identidad, detector y Fight-Card sobre el video bajado: no se corre la siembra. Un cambio en A: si
sirve, se propone, no se despliega.
