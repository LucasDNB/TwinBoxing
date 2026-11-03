# Identidad estable: seguimiento y asignación de rol (spec SDD)

2-11-2026. Pedido de Lucas: "afinar al 100% la corrección de tracking y la asignación de
identidad para tener un producto estable". El commit de este archivo es el pre-registro de §6.

## 1. Qué quiere decir "100%" acá

Con una sola cámara y clinches, la identidad automática no llega al 100%: C1 dio 0,810 por
track (`docs/experiments/2026-10-20-c1-identidad-con-siembra.md`) y el mejor arreglo medido,
el corte de E11, deja 0,923 de precisión por cuadro en desarrollo. Prometer el 100% automático
sería escribir un número que no se puede sostener.

Lo que sí se puede exigir, y es lo que hace estable a un producto, es esto:

> **Ningún cuadro con el rol equivocado llega a la Fight-Card sin haber pasado por los ojos de
> la persona.** Lo que la máquina no sabe, lo pregunta; lo que sospecha, lo muestra.

El 100% es de la Fight-Card después de la revisión, y el costo se mide en decisiones de la
persona. La parte automática sirve para que esas decisiones sean pocas.

## 2. El problema, leído del código (main `5926972`)

1. **Los errores son invisibles.** `listar_tramos` (`boxtwin-annotator/src/boxtwin/mvp/tramos.py`)
   devuelve solo tracks SIN ASIGNAR de más de 1 s, y `PUT /jobs/{id}/tramos` rechaza cualquier
   track que no esté en esa lista. Los 20 tracks mal asignados de C1 (19%) no aparecen nunca en
   la web y no se pueden corregir. El módulo de identidad dice en su cabecera que "un rol
   equivocado se dibuja sobre un cuerpo y se ve perfecto": la revisión no lo compensaba.
2. **La unidad de corrección es el track entero.** 18 de los 20 errores de C1 salen de tracks
   mixtos (el tracker pasa de una persona a la otra). A mano tampoco se puede partir un track.
3. **El corte de E11 está escrito y apagado.** `ConfigIdentidadAuto.cortar_mixtos` existe, con
   tests, pero `completar` no lo usa. Desarrollo (S10 v2): precisión por cuadro 0,850 a 0,923,
   cobertura 0,788 a 0,872, ninguna fuente empeora. Sin confirmación (L3 suspendido el 2-11).

El tracker (BoT-SORT, `track_buffer` 120) no se toca en esta spec: cambiarlo es E13, con
pre-registro propio y fase 2. Esta spec trabaja sobre lo que el tracker entrega.

## 3. Diseño, por capas

### Capa 1. Automática: partir los mixtos

`completar` acepta `--cortar-mixtos` y la API lo pasa con `BOXTWIN_CORTE_MIXTOS=1` (apagada por
defecto). Usa `segmentar` de E11 tal como está, con el cuadro del recorte que vio la persona como
`cuadros_semilla` (sale de `sesion.json`, `candidatos`).

### Capa 2. La cola de revisión ve también lo asignado

Módulo nuevo `boxtwin/mvp/revision.py`. La unidad pasa a ser el **tramo**: `(track, desde, hasta)`
en cuadros, con clave `"<track>"` si cubre el track entero y `"<track>-<desde>"` si no. Sobre los
rangos finales (automáticos más correcciones), la cola junta:

| Motivo | Qué es | Prioridad |
|---|---|---|
| `conflicto` | dos tramos con el mismo rol en el mismo cuadro, en al menos `min_coexistencia` cuadros medidos. Una misma persona no está dos veces: uno de los dos está mal | 1 |
| `color_contradice` | tramo asignado con al menos `min_guantes_voto` guantes y mayoría de esos guantes más cerca del perfil del OTRO rol. Solo si los perfiles separan (`separacion_minima`) | 2 |
| `posible_mezcla` | track sin partir donde el mejor corte de E11 tiene medianas de signo opuesto pero no llega al margen (la mitad de `margen_orientacion` sí) | 3 |
| los cuatro de hoy | sin asignar de más de 1 s (`abstencion`, `corto`, `color_insuficiente`, `sin_coexistencia`) | 4 |

Adentro de cada prioridad, del más largo al más corto. Ninguno de los umbrales es nuevo: todos
son parámetros que el módulo ya tenía.

Con `degradar_conflictos` (bandera `BOXTWIN_IDENTIDAD_ESTRICTA=1`), los dos tramos de un
conflicto que no hayan sido decididos por una persona pierden el rol y pasan a sin asignar:
la asimetría del módulo (un sin asignar se ve, un rol equivocado no) aplicada al resultado.
Si se enciende o no, lo decide §6.

### Capa 3. La persona corrige por rango y puede partir

- `PUT /jobs/{id}/tramos` acepta, además de `{track, rol}`, `{track, rol, desde, hasta}` en
  segundos, para cualquier track de la sesión (no solo los de la lista). Se guarda en
  `sesion.json` → `identidad.correcciones`. Lo viejo (`asignaciones_manuales`) se sigue leyendo
  como corrección del track entero.
- Una corrección manda sobre la propuesta en su rango y en nada más: no se usa como semilla para
  el resto del video (la misma regla de `aplicar_manuales`).
- `GET /jobs/{id}/tramos` devuelve cada ítem con `clave`, `tipo` (`sin_asignar` o `sospechoso`),
  `rol_actual` y el motivo; con `?todos=1` suma los tramos asignados sin sospecha, para revisar
  a mano lo que se quiera.
- La web: en cada tramo, al ver el video, un botón "Acá cambia de persona" parte el tramo en el
  instante del reproductor; cada mitad lleva su propio A / B / Ignorar.
- La Fight-Card suma `identidad.revision_pendiente` (ítems sospechosos sin decisión) para que la
  pantalla no dé por buena una identidad que tiene preguntas abiertas.

## 4. Lo que esta spec no hace

- No cambia el tracker ni el estimador de pose (E13, E9, E14 siguen en fase 2).
- No propaga una corrección al resto del video (rearmar la partición con la corrección como
  restricción es mejor, pero es otro diseño).
- No mide C1 otra vez ni cambia su resultado: C1 sigue en 0,810 y no pasa.
- No toca A antes del 17-11 (congelamiento del 3-11): todo va detrás de banderas apagadas, en
  una rama contra `main`.

## 5. Tareas

| Id | Qué | Dónde | Prueba |
|---|---|---|---|
| IE1 | `revision.py`: tramos desde la propuesta, aplicar correcciones por rango, sospechosos, degradar conflictos, cola ordenada | `boxtwin-annotator/src/boxtwin/mvp/` | tests sintéticos: corrección parcial parte el rango; conflicto detectado y degradado; color que contradice; mezcla débil; orden de la cola; sin correcciones el resultado es idéntico al de hoy |
| IE2 | `completar` con `--cortar-mixtos`, `--estricta`, correcciones por rango; tramos con clave y recorte por clave | `mvp/orquesta.py`, `cli.py`, `mvp/candidatos.py` | `sesion.json` viejo (solo `asignaciones_manuales`) da lo mismo que hoy |
| IE3 | API: entrada por rango, cualquier track, `?todos=1`, recorte por clave, banderas al worker | `boxtwin-api` | tests de la API; sesiones viejas siguen andando |
| IE4 | Web: tipo y rol actual en cada ítem, textos de los motivos nuevos, "Acá cambia de persona" | `prototipo/boxtwin-web-entrenador` | vitest, lint y build |
| IE5 | `scripts/identidad/medir_identidad_estable.py`, offline sobre la evidencia de C1 | `scripts/identidad/` | tests con verdad sintética |
| IE6 | Lucas corre IE5 en la estación (sin GPU, segundos) y se lee con §6 | estación | JSON en `~/boxtwin/mediciones/identidad-estable/` |

## 6. Medición y regla (pre-registro)

**Material.** Las seis fuentes de gimnasio de C1 con su evidencia guardada
(`~/boxtwin/mediciones/c1/<fuente>.evidencia.json`) y la misma siembra simulada de C1 y de E11.
Es material de **desarrollo**: ahí se diagnosticó el problema y se diseñó E11. Lo que salga se
informa con esa etiqueta y no va a la tesis como resultado confirmado.

**Condiciones.** `base` (el producto de hoy), `corte` (capa 1), `corte+estricta` (capa 1 con
`degradar_conflictos`). Las tres con la cola de la capa 2.

**Medidas por cuadro** (los cuadros donde el track tiene detección, igual que E11):

- precisión y cobertura automáticas;
- **error silencioso**: cuadro con rol A o B propuesto, distinto del de la verdad, que no cae
  adentro de ningún ítem de la cola;
- error visible: el mismo error, adentro de un ítem de la cola;
- tamaño de la cola: ítems, segundos de video a mirar e ítems por minuto de video;
- **usuario simulado**: contesta cada ítem con la verdad. Un ítem cuya verdad tiene A y B (al
  menos 5 cuadros cada uno) cuenta dos decisiones (partir y asignar); el resto, una. Después se
  recalculan precisión y cobertura.

**Regla.**

- E1 (estabilidad): errores silenciosos ≤ 0,5% de los cuadros de peleador sumando las seis
  fuentes, y ninguna fuente por encima del 2%.
- E2 (mecánica de la cola): con el usuario simulado, precisión por cuadro ≥ 0,99.
- E3 (costo): decisiones por minuto de video, sin umbral. Se informa por fuente.

**Qué se enciende.** Entre las condiciones que pasan E1 y E2, la de menos decisiones por minuto.
Si ninguna pasa E1, la de menos errores silenciosos, y el producto se declara **no estable** en
identidad: la Fight-Card muestra la advertencia y esta spec no se da por cumplida. Empate a 0,1
decisiones por minuto: la que tenga menos banderas encendidas.

**Lo que no decide.** No es confirmación (la confirmación necesita material nuevo anotado a mano
con la propuesta apagada, L3 de E11, hoy suspendido). La verdad es la anotación de C1; el usuario
simulado no se equivoca y una persona sí (eso es C6).

## 7. Salidas

`docs/experiments/<fecha>-identidad-estable.md` con las tres condiciones, el JSON en
`docs/experiments/datos/`, entrada del README y, si cambia lo que se dice del producto, una nota
en `docs/tesis/` para el Cap 7 (la revisión de tramos ve los asignados sospechosos) y el Cap 6
(R-02: la contramedida pasa de "corrección manual de lo sin asignar" a "revisión de todo lo
dudoso").
