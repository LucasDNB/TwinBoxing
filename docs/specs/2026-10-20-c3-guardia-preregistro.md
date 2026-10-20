# C3 · Pre-registro de la definición operativa de guardia

20-10-2026 · decisión de dominio de Lucas, tomada antes de marcar o correr C3 · código:
`boxtwin-annotator/src/boxtwin/mvp/guardia.py` (`ConfigGuardia`) · criterio original en
`docs/specs/MVP.md` §7

## 1. Qué se fija

| Parámetro | Antes | Desde el 20-10 | Quién |
|---|---|---|---|
| Zona de guardia (`radio_guardia`) | muñeca a menos de 0,6 anchos de hombro de la nariz | **1,0 ancho de hombro** (puño al costado del mentón) | Lucas, criterio de dominio |
| Lectura de "volvió a la guardia" | sin decidir: por mediana de `sparring-3` o por tiempo límite | **tiempo límite fijo**: vuelve si la muñeca entra a la zona dentro de **1000 ms** del fin del golpe | Lucas |
| Tiempo límite (`limite_busqueda_ms`) | 1500 ms | **1000 ms** | Lucas |
| Próximo golpe del mismo brazo antes de volver | cuenta como no vuelve | sin cambios | — |
| Retorno lento (`retorno_lento_ms`) | None, a calibrar por mediana | None: la lectura por mediana no se usa | Lucas |
| Mano opuesta caída (`fraccion_mano_caida`) | afuera de la zona en ≥ 50% del golpe | sin cambios | — |

## 2. C3, sin cambios en el resto

- Muestra: 100 golpes de `sparring-3`, marcados a mano por Lucas con dos respuestas por golpe:
  volvió a la guardia dentro de un segundo (sí/no) y mano opuesta caída durante el golpe
  (sí/no).
- Lucas marca **sin ver la salida del indicador**, con el mismo criterio que fija este
  documento: un segundo y el puño al costado del mentón.
- Golpes no medibles (sin pose de la mano) se informan aparte y no entran al acuerdo.
- Umbral: acuerdo ≥ 80% en cada uno de los dos indicadores. Si no pasa, F5 sale del MVP y se
  declara como línea futura (`BOXTWIN_C3=rechazado`).
- Al lado, sin umbral: kappa de cada indicador y las matrices de confusión.

## 3. Lo que hay que declarar

El 1,0 no se eligió a ciegas. El 22-09, sobre `amateur_estatico`, se midió la distancia
muñeca-nariz (p10 entre 0,75 y 1,09; mediana 1,39 a 1,75) y se vio que con 0,6 el indicador
marcaba la mano opuesta caída en 40 de 41 golpes
(`docs/experiments/2026-09-22-mvp-sobre-video-real.md`). La decisión se apoya en el criterio
de boxeo, pero se tomó conociendo esa distribución. Lo que la protege es que C3 se mide sobre
`sparring-3`, que no entró en la decisión, y que este documento se commitea antes de marcar
la muestra o correr el indicador sobre ella.
