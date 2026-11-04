# Cambios a la tesis por el cierre (4-11)

Fuentes: los documentos de `docs/experiments/` del 1 al 4-11. Las fuentes LaTeX no están en el
repositorio: estos textos se pegan a mano. Como en los cambios anteriores, va el DESPUÉS con dónde
entra; el ANTES lo pone Lucas. Sigue vigente lo que ya está en los otros `*-cambios.md`.

Lo marcado con corchetes lo tiene que completar o confirmar Lucas. No está medido por mí.

## 1. Cap 4, perfiles de despliegue

DESPUÉS: "Los perfiles no los declaré, los medí. En la estación, con una RTX 2080 SUPER y dos
peleadores en cuadro, YOLOv8l-pose a 640 exportado a TensorRT FP16 tarda 10,3 ms de mediana entre
decodificar y estimar pose, con p95 de 11,3 ms y 93 cuadros por segundo sostenidos. Ese es el perfil
A. El perfil B colapsa en A: en esta máquina bajar a 15 cuadros por segundo no compra calidad, porque
lo mejor que hay ya entra en 40 ms. El perfil C, solo ONNX, es el mismo modelo a 26,0 ms y 38 cuadros
por segundo. El mAP de pose sobre las personas de COCO val2017 da 0,672 en los dos."

"Esto vale para la estación. La notebook quedó sin medir, así que no extrapolo estos perfiles a otra
máquina. Tampoco dice que el modo en vivo entre en 40 ms: mide la pose, sin tracker, identidad ni
dibujo."

QUÉ CAMBIÓ: los perfiles estaban pendientes de medición. Fuente: `2026-11-04-r11-perfiles.md`.

## 2. Cap 4 o Cap 5, transcodificado en el dispositivo

DESPUÉS: "Comparé la pose sobre video transcodificado a 720p contra la de 1080p, en los mismos cuadros.
Se empareja más del 99% de las cajas y el error mediano de muñeca es de 0,8% y 1,1% del alto de la
caja. Con la regla que fijé antes de medir, 720p es apto. Es acuerdo con 1080p y no precisión contra
una verdad anotada, y vale para un codificador y una calidad."

QUÉ CAMBIÓ: la estrategia de transcodificado tenía la pérdida sin medir. Fuente: `2026-11-04-pose-720p.md`.

## 3. Cap 7, clasificador a 15 fps

DESPUÉS: "Simulé pose a 15 cuadros por segundo y reentrené el clasificador. La exactitud de familia
baja de 0,732 a 0,718, dentro de la caída de 0,03 que había fijado. El uppercut es la excepción: su
recall baja de 0,541 a 0,412, con 85 golpes. No medí el detector en esas condiciones, así que esto
no alcanza para decir que el perfil B funciona de punta a punta."

QUÉ CAMBIÓ: resultado nuevo. Fuente: `2026-11-04-clasificador-15fps.md`.

## 4. Cap 7, "Mejoras posteriores a E8": E11

DESPUÉS: "El corte de tracks mixtos se diseñó sobre las seis fuentes del diagnóstico de C1. Sobre ese
material sube la precisión por cuadro de 0,850 a 0,923 y la cobertura de 0,788 a 0,872, sin empeorar
ninguna fuente. El número es exploratorio porque diseñé y medí sobre lo mismo. La confirmación pedía
dos sparrings nuevos anotados a mano, que no llegué a tener, así que la hipótesis queda sin decidir y
el corte sigue apagado."

QUÉ CAMBIÓ: E11 no tenía texto. Fuente: `2026-11-04-e11-tracks-mixtos.md`.

## 5. Cap 7, "Mejoras posteriores a E8": E12

DESPUÉS: "Sumé el guante detectado como entrada del detector de golpes. Con la evaluación corregida
para que use la máscara de pose del producto, la diferencia es de +0,001 de F1: no demostrada. El
primer número, 0,429, era un defecto de la evaluación y no del guante. Además cuesta 12,6 ms por
cuadro, y con eso C5 se iría de 2,0."

QUÉ CAMBIÓ: E12 no tenía texto. Fuentes: `2026-11-01-e12-guantes-en-detector.md` y `2026-11-01-r1-evaluacion-producto.md`.

## 6. Cap 7, "Mejoras posteriores a E8": E13

DESPUÉS: "Probé si otro tracker produce menos tracks mixtos, re-trackeando las mismas detecciones.
Antes verifiqué que el camino offline reproduce el tracker original en los 180193 cuadros. Ninguno
cumplió la regla, que pedía bajar los mixtos un 30% sin subir los cambios de track por minuto más de un
20%. OC-SORT y ByteTrack empeoraron las dos cosas. BoostTrack con reidentificación bajó los mixtos de
23 a 18 y subió los cambios un 49,8%."

"BoostTrack igual deja un dato que la regla no mira: corta a la mitad los cuadros dentro de tracks
mixtos y la precisión de identidad por cuadro pasa de 0,882 a 0,977, sobre las fuentes de desarrollo.
Cambia mezcla por fragmentación. Lo dejo como línea futura, sin número de aceptación."

QUÉ CAMBIÓ: E13 estaba previsto para después del 17-11. Fuente: `2026-11-04-e13-tracker.md`.

## 7. Cap 7, identidad estable

DESPUÉS: "Con la revisión por tramos y un usuario simulado, la identidad queda estable en desarrollo
con una precisión de 0,985, a un costo de unas 23 decisiones por minuto de video. Es material de
desarrollo, y el usuario simulado contesta la verdad, cosa que una persona no siempre hace. Falta
medirlo con personas."

QUÉ CAMBIÓ: identidad estable no tenía texto. Fuentes: `2026-11-03-identidad-estable.md` y `2026-11-03-ie7-apariencia.md`.

## 8. Cap 7, criterios C4, C5 y C6

DESPUÉS, si Lucas confirma las sesiones: "Sobre [las N sesiones de prueba], el tiempo hasta la
Fight-Card dio una mediana de [1,892] veces la duración del video, contra un máximo de 2,0. Las dos
sesiones largas, de 28 y 36 minutos, quedaron en 2,04 y 2,06: un video largo tarda más de dos veces su
duración. [Las siete] llegaron a la Fight-Card sin errores de procesamiento."

C6: [Lucas: lo que hayan dicho Suárez o el entrenador. Si no hubo sesión, escribir que C6 no se hizo.]

QUÉ CAMBIÓ: C4 y C5 estaban sin medir. Fuente: `2026-11-04-c4-c5-sesiones.md`. Son provisionales.

## 9. Cap 8, líneas futuras

DESPUÉS: "Quedan con pre-registro escrito y sin número: la confirmación de E11, E9, E14 y E15. Sin
medir quedan la notebook, el despliegue en nube y el detector a 15 cuadros por segundo. Con E13, el
pre-registro decía que si ningún tracker mejoraba, el problema era la oclusión del clinch y no la
asociación. Los datos de BoostTrack no me dejan afirmarlo así: una asociación con apariencia reduce
la mezcla, a costa de fragmentar. Medirlo bien pide material nuevo y resolver la licencia."

QUÉ CAMBIÓ: separa lo pre-registrado de lo que no se llegó a medir, y matiza §5 de E13.

## 10. Declaración de uso de IA, una línea

DESPUÉS: "La campaña posterior a E8 y las mediciones del cierre las ejecutó Claude Code en modo agente,
con pre-registros y reglas fijados antes de medir y las decisiones anotadas por mí."
