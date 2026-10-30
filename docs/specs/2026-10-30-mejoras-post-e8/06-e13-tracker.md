# E13 · Tracker alternativo (pre-registro, fase 2)

30-10-2026. Fase 2: no corre antes del 17-11. Correcciones hasta antes de H13-1.

## 1. La pregunta

Los tracks mixtos los produce el tracker: 23 con v8 y 29 con YOLO26 sobre el denominador común
de E8 (P12c). E11 los corta después; E13 prueba si otro tracker los produce menos. Además, el
tracker es una de las tres piezas de Ultralytics del producto (`05-e9-rtmo.md` §8).

## 2. Condiciones

Las mismas detecciones de pose de `raiz-v8` (v8 sin ReID), guardadas por cuadro, re-trackeadas
offline con cada tracker. Una sola variable: el algoritmo de asociación.

| | Tracker | Licencia a verificar en H13-1 | Uso posible |
|---|---|---|---|
| Base | BoT-SORT sin ReID de Ultralytics (`botsort-sin-reid.yaml`) | AGPL-3.0 o Enterprise | la de E8 |
| T1 | OC-SORT, implementación oficial | según su repositorio | producto, si es permisiva |
| T2 | ByteTrack, implementación oficial | según su repositorio | producto, si es permisiva |
| T3 | BoostTrack o Deep OC-SORT con ReID, vía BoxMOT | según su repositorio | solo investigación si es copyleft |

Parámetros: los de cada implementación por defecto, más dos ajustes fijados desde ahora porque
la base también los tiene: buffer de pérdida equivalente a 4 s y reinicio en cortes de plano con
el mismo detector de cortes (umbral 0,3). No se ajusta nada contra la verdad.

## 3. Humo

| Id | Qué |
|---|---|
| H13-1 | Licencias de las tres implementaciones, con fuente y fecha, en `~/boxtwin-mejoras/e13/licencias.md` |
| H13-2 | Re-trackear la base con BoT-SORT desde las detecciones guardadas reproduce los `track_id` del caché de `raiz-v8` en el 100% de los cuadros (si no, el offline no es el mismo tracker y E13 compara contra el re-trackeo, declarado) |

## 4. Preguntas y reglas

| Id | Pregunta | Regla |
|---|---|---|
| P18a | Tracks mixtos y cuadros dentro de tracks mixtos, por tracker | **Mejora** si baja los tracks mixtos al menos un 30% respecto de la base sin subir los cambios de track por minuto más de un 20% |
| P18b | Cambios de track por minuto y tracks distintos por peleador | sin umbral |
| P18c | C1 con la definición del pre-registro de C1, seis fuentes, sin y con el corte de E11 | sin umbral; no es la medición de C1 |
| P18d | Tiempo del tracker por cuadro | sin umbral |

La verdad es la de `raiz-v8` pasada por el puente de E8 (K1 al 100%). El 30% de P18a es grande a
propósito: con 23 tracks mixtos en la base, menos que eso está dentro de lo que cambia de una
fuente a otra.

## 5. Qué decide cada resultado

- **T1 o T2 mejoran (licencia permisiva)**: candidato para A con PR propio y C1 y C5 medidos otra
  vez; resuelve una de las tres piezas de §8 de E9.
- **Solo T3 mejora**: se informa como investigación; el ReID ayuda pero no se puede usar en el
  producto con esa licencia.
- **Ninguno mejora**: Cap 8: el problema no es el algoritmo de asociación sino la oclusión del
  clinch, y lo que queda es E14 o la corrección manual.
