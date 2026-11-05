# Cierre del 4-11 · pre-registro de lo que falta medir

4-11-2026 · pedido de Lucas: cerrar todo hoy · se commitea antes de medir 720p y 15 fps · `CLAUDE.md` §6.2 y §6.4

## 1. Pose a 720p (§6.4)

**Pregunta.** Cuánto se pierde si el dispositivo transcodifica a 720p antes de subir.

**Medición.** Los mismos cuadros de R-11 (`~/boxtwin-mejoras/r11/cuadros/`, sparring-3 y pacquiao).
Cada video se transcodifica con `ffmpeg`, libx264, `-crf 23`, `-preset veryfast`, a 720 de alto, y
pasa por `medir_celda.py` con `yolov8l-pose.pt` en PyTorch a 640, el camino de A. Cajas y keypoints
vuelven a la escala 1080p y `regla.concordancia` los compara con el caché de `raiz-v8`, sin los 200
cuadros de warm-up.

**Regla.** 720p **es apto** si, en los dos videos, se empareja al menos el 98% de las cajas y el error
mediano de muñeca queda en 2% del alto de la caja o menos. Si no, **no es apto**.

**Queda fuera.** La verdad anotada: es acuerdo con 1080p, no precisión absoluta. Un codificador de
celular real: `crf 23` es una elección, otro bitrate daría otro número.

## 2. Clasificador a 15 fps (§6.2)

**Pregunta.** Cuánto cae E4, el clasificador del producto, si la pose se muestrea a 15 fps.

**Medición.** Condición `v8-15`: las series de E8 base, dejando un cuadro de cada `round(fps / 15)`
(2 a 30 fps, 4 a 59,94). Los cuadros del medio se interpolan linealmente entre los conservados, con
el menor puntaje de los dos; si falta un vecino válido, quedan con puntaje 0. E4 completo, 35
corridas, igual que E10. Se apaga con `BOXTWIN_MUESTREO_FPS` sin definir.

**Regla.** El perfil B **es viable para el clasificador** si la exactitud de familia del ensamble cae
0,03 o menos contra la base de E8. Si cae más, **no es viable**.

**Queda fuera.** El detector de golpes: esto es solo E4. Identidad y tracker a 15 fps tampoco se miden.
Se simula sobre la pose a 30 fps; no es una corrida real a 15 fps.

## 3. C4 y C5 sobre las sesiones de A (declaración, no medición nueva)

C5 ya se corrió el 4-11 sobre las siete sesiones creadas desde el 27-10 **antes** de escribir este
archivo. La lista de sesiones de prueba no estaba fijada en ningún documento. Por eso se informa como
**provisional**, con el resultado de todos los recortes posibles y sin elegir uno. Lo confirma Lucas.

## 4. Lo que no se mide hoy y se declara

- Notebook (§6.1): no hay acceso desde esta máquina.
- PoC en nube (§6.3): necesita alquilar una instancia, con gasto y credenciales de Lucas.
- C6: no se realiza. Decisión de Lucas del 5-11: BoxTwin quedó fuera del alcance y para BoxTwinZ no es necesario.
- C2 y C7: quedan sin medir y se declaran.
- E11 confirmación: L3 no llegó (`sparr` tiene dos tracks asignados y `sparr01` nada). Se aplica la
  regla de `03-e11-tracks-mixtos.md` §4: E11 se informa solo con desarrollo, rotulado como exploratorio.
