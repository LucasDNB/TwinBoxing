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

## 4. Agregado del 20-10, antes de sortear la muestra

Lo que el §2 no fijaba, decidido por Lucas y commiteado antes de que exista la muestra, de
marcar un golpe o de correr el indicador sobre `sparring-3`. Script:
`boxtwin-annotator/scripts/medir_c3.py`.

**Golpes.** Los golpes anotados completos (`completeness = full`) de `sparring-3-rounds`, que
son 381 de 400: sin fintas ni abortados. Tiempos (inicio y fin) e identidad son los de la
anotación. C3 mide la validez del indicador, no la del detector (recall 0,48) ni la de la
identidad (C1, 0,810): con golpes detectados, los errores de esas dos piezas se contarían
como errores de guardia. Se declara que el fin del golpe es el anotado y no el del detector.

**Muestreo.** 100 golpes por muestreo aleatorio simple sin reposición, semilla `20102026`,
sobre los 381 ordenados por id de evento. Se reporta la distribución por peleador y brazo.

**Marcado a ciegas.** Cada golpe se muestra como un clip desde 0,5 s antes del inicio
anotado hasta 1,5 s después del fin anotado, con una caja sobre el peleador, el brazo que
pega escrito y un contador desde el fin del golpe que marca el límite de 1,0 s. No se dibuja
esqueleto, zona de guardia ni nada que salga del indicador. La herramienta de marcado no
carga el indicador. Dos respuestas por golpe: volvió a la guardia dentro de 1,0 s
(sí / no / no se ve) y mano opuesta caída durante el golpe (sí / no / no se ve).

**Indicador.** `guardia.medir` con `ConfigGuardia()` por defecto (radio 1,0, límite 1000 ms,
fracción 0,5), sobre la serie de keypoints por peleador de `series_por_peleador` con la
identidad anotada, el mismo camino que el producto. Para el carril del próximo golpe se usan
todos los golpes anotados de la fuente, no solo los de la muestra. Volvió = `retorno_ms` con
número; no volvió = `retorno_ms` None con la mano vista; no medible = `medible` falso. Mano
opuesta caída = `mano_opuesta_caida`; None es no medible.

**Acuerdo.** Por indicador, sobre los golpes que tienen respuesta sí/no de los dos lados.
Los "no se ve" de Lucas y los no medibles del indicador se informan aparte y no entran. El
criterio es la proporción puntual ≥ 0,80 en cada indicador; Wilson 95%, kappa de Cohen y
matriz de confusión van al lado, sin umbral. El script no calcula el acuerdo hasta que estén
marcados los 100.
