# E14 · Identidad por propagación de máscaras de SAM 2 desde la siembra (pre-registro, fase 2)

30-10-2026. Fase 2: no corre antes del 17-11. Correcciones hasta antes de H14-1.

## 1. La idea

El producto ya le pide a una persona que señale a A y a B (la siembra). Hoy esa señal se usa
para orientar componentes de coexistencia. E14 la usa de otra manera: **las dos cajas sembradas
son el prompt de un modelo de segmentación de video que propaga una máscara por peleador** a lo
largo de todo el video, y cada detección de pose se asigna a A o a B según con qué máscara se
superpone. Deja de depender de los `track_id` del tracker, que es de donde salen los tracks
mixtos.

El costo es el tiempo: la literatura de deportes lo presenta como una opción para análisis
offline donde la continuidad de la trayectoria importa más que la velocidad
(Eichner et al., ISD 2026), y SAM 2 tiene un problema conocido de expansión de máscaras en
escenas densas (SAMIDARE, arXiv 2604.22162). El clinch es exactamente ese caso.

## 2. Diseño (fijo desde ahora)

- Modelo: SAM 2.1 `hiera-small` (el más grande que se espera que entre en 8 GB con descarga de
  estado a CPU); si no entra, `hiera-tiny`, anotado como decisión. Licencia: la tarjeta de los
  pesos en Hugging Face declara Apache-2.0; H14-1 la verifica en el `LICENSE` del repositorio y
  la del backbone, con fecha.
- Entorno nuevo `boxtwin_sam2`; se comunica con `twinboxing_env` por archivo.
- Prompt: en el cuadro del recorte que vio la persona, las dos cajas de las semillas.
- Propagación por bloques de 300 cuadros con `offload_video_to_cpu` y `offload_state_to_cpu`;
  cada bloque arranca con las máscaras del último cuadro del anterior como prompt. Sin
  re-prompts automáticos en v1: si una máscara se pierde, se pierde, y eso se mide.
- Asignación: una detección de pose va a A si al menos el 60% de sus keypoints con puntaje
  ≥ 0,3 cae dentro de la máscara de A y menos del 20% dentro de la de B (y simétrico). Si no,
  sin asignar. Un cuadro con dos detecciones en la misma máscara: la de mayor fracción.
- Salida: un `annot.json` con asignaciones por rango, como el de la identidad automática.

## 3. Humo

| Id | Qué |
|---|---|
| H14-1 | Licencias (código, pesos, backbone) con fuente y fecha |
| H14-2 | Un minuto de sparring-3-rounds: pico de VRAM ≤ 7,5 GB y ms por cuadro (mediana, p95) |
| H14-3 | Las dos máscaras del cuadro de prompt contienen al menos el 90% de los keypoints de su semilla |

## 4. Preguntas y reglas

Material: el mismo de E11 (desarrollo: seis fuentes; confirmación: los sparrings nuevos
anotados de L3). La comparación es contra la identidad del producto **con** el corte de E11 si
E11 mejoró, y sin él si no.

| Id | Pregunta | Regla |
|---|---|---|
| P19a | Precisión por cuadro | la de E11 §5: mejora con +0,03 sin perder más de 0,02 de cobertura |
| P19b | Cobertura por cuadro | ídem |
| P19c | Tiempo de la etapa, en múltiplos de la duración del video, y su suma a C5 | sin umbral; si la cadena pasa de 2×, se declara que no entra en el producto como está |
| P19d | Dónde se pierde: cuadros con máscara vacía, con máscaras superpuestas (IoU > 0,5) y en qué segundo de cada clinch | descriptiva |

## 5. Qué decide cada resultado

- **Mejora y entra en el tiempo**: candidato para A con PR propio; reemplazaría la partición por
  coexistencia, que es la pieza que hoy falla.
- **Mejora y no entra en el tiempo**: Cap 8 como modo de "análisis diferido" (procesar de noche),
  con el costo medido; el Cap 5 tendría que sumar ese cómputo.
- **No mejora**: Cap 8; refuerza la corrección manual por track.
