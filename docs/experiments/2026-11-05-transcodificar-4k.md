# Bajar la resolución antes de procesar video 4K

5-11-2026 · pre-registro: `docs/specs/2026-11-04-cierre/01-transcodificar-4k.md` (`5e35fc3`, antes de
medir) · datos: `docs/experiments/datos/2026-11-05-transcodificar-4k.json` · código: `scripts/r11/medir_4k.py`

## Veredicto

**Ninguna condición sirve por la regla, en ningún video.** Transcodificar en el servidor no deja el 4K
dentro de 2,0 conservando la pose.

Ayuda mucho, pero no alcanza. `sparr01` baja de 4,23 a 2,21 con 1080p, y `sparr` de 8,88 a 3,02 con
1080p a 30 fps. El caso que se acerca es transcodificar **en el celular**: `sparr01` a 1080p da 1,75 y
conserva la pose. Si el video queda a 60 fps no hay condición que entre.

## Resultados

| Condición | Transcodificar | Procesar | Factor con transcodificado | Sin transcodificado | Cajas | Error de muñeca | ¿Sirve? |
|---|---|---|---|---|---|---|---|
| sparr01 4K (control) | 0 s | 759 s | 4,23 | 4,23 | 100% | 0% | no, tiempo |
| sparr01 1080p | 88 s | 292 s | 2,21 | **1,75** | 98,1% | 0,51% | no, tiempo |
| sparr01 720p | 61 s | 235 s | **1,76** | 1,44 | 95,6% | 0,90% | no, pose |
| sparr 1080p | 169 s | 590 s | 4,71 | 3,77 | 97,5% | 0,54% | no, los dos |
| sparr 720p | 124 s | 481 s | 3,85 | 3,16 | 93,6% | 0,92% | no, los dos |
| sparr 1080p a 30 fps | 142 s | 313 s | 3,02 | 2,23 | 96,8% | 0,78% | no, los dos |

Control K-4k: **pasa**. `preproceso + evidencia` offline da 740,5 s contra 742,4 s en A, así que estos
tiempos se comparan con los de A. K-A: ningún trabajo entró en A ni en BoxTwinZ durante la medición.

## Qué aprendí

1. **El costo era de la resolución.** Con el video a 1080p, el preproceso baja de 123 ms a unos 46 ms
   por cuadro, lo mismo que una sesión a 1080p. El modelo no cambió, así que el 2,5x no está en la
   inferencia. Bajar a YOLOv8m, el plan B de C5, no ataca esto. No lo medí.
2. **60 fps sigue siendo caro.** A la misma resolución son el doble de cuadros, y bajar a 30 fps da
   2,23 sin contar el transcodificado.
3. **El transcodificado en el servidor pesa.** Cuesta de 61 a 169 s con libx264 en CPU, entre 0,3 y
   0,9 puntos de factor. No probé `h264_nvenc`, ni decodificar directo a menor resolución.
4. **El error de muñeca no es el problema.** Queda debajo de 1% en todo. Lo que falla es el
   emparejamiento de cajas, que baja con la resolución: a 720p se pierde lo más chico o lejano.

## Limitaciones

1. **Dos videos, del mismo material.** Son una sola situación de filmación.
2. **La vara de la pose es el caché 4K, no una verdad.** No sé si las cajas que se pierden son de los
   peleadores o de gente del fondo. Si son de fondo, los fallos de pose no importan para el producto.
   La regla no las separa, y no lo miré. El umbral de 98% venía de `720p`, que comparaba otra cosa:
   allí detecciones sobre tramos con dos personas, acá la salida del tracker en todo el video.
3. **`completar` y `clasificar` se tomaron de la sesión 4K.** Son 35 y 88 s. Si bajan con el video, el
   factor real es menor. `sparr01` a 1080p con transcodificado no entraría igual: aun con esas etapas en
   cero da 2,02. `sparr` a 1080p y 30 fps sin transcodificado sí podría entrar: da 2,23 con los 88 s,
   y 1,74 sin ellos.
4. **Sin identidad, detector ni Fight-Card** sobre el video bajado. No sé si cambian los golpes.
5. **Transcodificar en el celular no se probó.** El factor sin transcodificado supone que llega hecho y
   gratis, con un codificador que no es el del teléfono.
6. **`sparr01` tiene 28,89 fps medidos** y el transcodificado puede haber normalizado cuadros.

## Qué decide esto

- C5 sigue sin cumplirse para 4K de celular.
- La vía más prometedora es pedir o forzar 1080p a 30 fps en la subida. Es una propuesta: nada se
  desplegó, y haría falta medir identidad y golpes sobre ese video.
