# C1 · Pre-registro de la medición de identidad con siembra humana

20-10-2026 · criterio original en `docs/specs/MVP.md` §7 · script
`boxtwin-annotator/scripts/medir_c1.py` (pruebas en `tests/test_script_medir_c1.py`) · se
commitea antes de correr la medición

## 1. Qué se mide

C1, tal como está en la spec: tracks bien asignados con siembra humana sobre las seis fuentes
de gimnasio, umbral **≥ 95%**. Si no pasa, el plan B ya está construido: corrección manual por
track en la web (B3/W9).

Fuentes: `Sparring`, `01-sparring`, `02-sparring`, `03-sparring`, `04-sparring` y
`sparring-3-rounds`. Pacquiao-Margarito es transmisión y no entra.

## 2. Cómo se mide

Por fuente, sobre el cache de pose del proyecto de anotación (los `track_id` tienen que ser
los mismos que los de la anotación), con `ConfigIdentidadAuto()` por defecto y el detector
`guantes-v2.pt`:

1. Evidencia de guante con `analizar`, la misma función que usa `procesar`.
2. Propuesta automática y candidatos con `proponer` y `elegir`, como en el producto.
3. **Siembra simulada, regla fija:** la persona ve el par sugerido y los candidatos en el
   orden de la pantalla. Si el par sugerido son dos peleadores distintos según la anotación,
   lo confirma diciendo cuál es A. Si no (un árbitro, la misma persona dos veces, o no hay
   par), elige de la lista el primer candidato que es A y el primero que es B. Si no hay uno
   de cada lado, la fuente queda **sin siembra posible** y C1 no pasa.
4. Propuesta con esas semillas, como `completar` en el producto.
5. Puntaje contra la anotación manual **sin probar la correspondencia cruzada**: con
   semillas, A es A, y un track invertido es un error.

La anotación asigna roles por rango de cuadros y un track puede cambiar de persona. La verdad
de un track es su rol mayoritario por cuadros entre A y B; los tracks sin cuadros de
peleador en el cache quedan afuera.

## 3. Números

- **Primaria (con umbral):** aciertos sobre tracks evaluables (peleadores de la anotación que
  la propuesta decidió como A o B), agregados sobre las seis fuentes, con Wilson 95%. Es la
  misma base que el 82,4% sin siembra y el 99,1% del color, así que se pueden poner al lado.
- Al lado, sin umbral: cobertura de tracks (aciertos sobre todos los peleadores de la
  anotación; lo no decidido cuenta como error), cobertura pesada por cuadros de peleador (un
  track mixto suma solo los cuadros de su rol), tracks mixtos, el puntaje sin siembra de la
  misma corrida y el caso de siembra de cada fuente.
- Una sola corrida. Si una fuente falla por algo del entorno (falta el cache, el video), se
  arregla el entorno y se corre; no se cambia nada de esta regla después de ver un número.

## 4. Lo que hay que declarar

- La siembra es simulada con la anotación: mide si la identidad alcanza **dado** que la
  persona contesta bien, no si la persona contesta bien. Que el usuario reconozca a los dos
  peleadores en dos recortes se ve en la prueba con usuarios (C6), no acá.
- Se mide sobre los caches de anotación, no sobre sesiones nuevas del producto: el detector de
  guantes vio negativos de este mismo material (`2026-09-20-negativos-del-propio-material`),
  así que el número es en distribución para el detector de guantes.
- La verdad mayoritaria es una decisión de este pre-registro: la versión del 82,4%
  (`puntuar_contra`) tomaba la última asignación de A o B de cada track.
