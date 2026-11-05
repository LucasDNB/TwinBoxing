# R-11 · Benchmark de perfiles de despliegue (pre-registro)

3-11-2026 · pedido de Lucas para la noche del 3 al 4-11 · se commitea antes de la primera medición
de tiempo · §6.1 de `CLAUDE.md`

No había protocolo en el repo: `CLAUDE.md` §6.1 y `docs/tareas/2026-10-20-proxima-sesion.md` §3 dan
el pedido, no las reglas. Este archivo las fija. Cambios solo antes de la primera corrida, anotados
al final como decisión. Después de esa corrida, lo que cambie se declara como desvío en el documento
de resultados.

## 1. Qué se mide y qué no

Cuánto tarda la estimación de pose por cuadro, con dos peleadores en cuadro, en la estación (RTX 2080
SUPER, 8 GB), para decidir qué configuraciones entran en cada presupuesto.

Queda fuera, y se declara en el resultado:

- **La notebook.** `CLAUDE.md` §6.1 pedía desktop y notebook. Esta corrida es solo la estación: los
  perfiles quedan asignados para esa máquina y no se extrapolan.
- **El resto de la cadena en vivo.** Se mide decodificación más pose. Tracker, identidad, detector
  de golpes y clasificador no están en el número. Que una configuración entre en 40 ms dice que la
  pose entra, no que el modo en vivo entra: la sub-hipótesis 3 no se puede dar por confirmada solo
  con esto.
- **La degradación del clasificador a 15 fps** (§6.2). El perfil B se asigna por tiempo; si es
  viable para el clasificador lo decide esa otra medición.

## 2. Condiciones

- Modelos: YOLOv8-pose n, s, m y l, pesos oficiales. Los cuatro `.pt` se registran con su sha256.
- Resoluciones: 416, 480 y 640 px, entrada cuadrada fija y lote 1.
- Formatos, exportados en esta máquina con Ultralytics 8.4.41:
  - **TensorRT FP16** (`format=engine, half=True`), TensorRT 10.9.
  - **ONNX FP32** con ONNX Runtime 1.22, proveedor CUDA. Se corre FP32 porque es la forma portable:
    no depende de compilar un motor por placa.
- Parámetros de inferencia: los de A (`conf=0.25`, `iou=0.7`).
- Referencia, sin perfil: l a 640 en PyTorch FP32, el camino de A, una celda por video. Sirve para
  ubicar los números, no compite.
- Entorno: venv `~/boxtwin-mejoras/r11/venv`, que ve el torch y el Ultralytics de `twinboxing_env` y
  suma TensorRT y ONNX Runtime solo adentro de ese venv. `twinboxing_env` no se modifica.

La entrada cuadrada es más cara que el rectángulo que usa A en PyTorch (640×384 en 16:9). Se elige
cuadrada porque un motor fijo tiene que aceptar cualquier relación de aspecto. Va como limitación.

## 3. Barrido en dos etapas

**Etapa 1.** n, s, m y l a 640, en los dos formatos: 8 celdas, cada una sobre los dos videos.

**Etapa 2.** Por formato, y solo a partir de lo que dio la etapa 1:

- cada tamaño que no pasó 40 ms a 640 se mide a 480 y a 416;
- si en ese formato los cuatro pasaron 40 ms, se mide solo l a 480 y a 416, para tener la curva.

No se bajan de resolución los tamaños que pasan a 640: la hipótesis (declarada) es que a igual
modelo, menos resolución no da más calidad, y entonces no cambiarían ninguna asignación. Las celdas
que no entran en la etapa 2 quedan **no medidas**, no descartadas.

Corte: si la etapa 2 no terminó a las 4:00 del 4-11, no arranca ninguna celda nueva y se documenta lo
que haya. Las asignaciones se hacen con lo medido y se marcan como parciales.

## 4. Protocolo de una celda

**Material.** `sparring-3-rounds` y `pacquiao_margarito`. Los cuadros se eligen una sola vez, antes de
medir, con el caché de `raiz-v8` y sin mirar tiempos: tramos contiguos de al menos 150 cuadros donde
el caché tiene al menos dos detecciones con `det_conf ≥ 0,5` y sin corte de plano adentro, tomados en
orden desde el principio del video hasta juntar 2200 cuadros. La lista se guarda con su sha256 y es la
misma para todas las celdas. Los primeros 200 son warm-up y se descartan; quedan 2000 medidos.

**Medición por cuadro**, en un solo hilo y en serie, sin solapar:

- `t_dec`: decodificar el cuadro con OpenCV;
- `t_pose`: del cuadro BGR en memoria a los keypoints en CPU, con preproceso, inferencia y NMS;
- `t_total = t_dec + t_pose`.

**Los presupuestos se leen sobre `t_total`.** Es la lectura conservadora: una implementación que
solape decodificación e inferencia puede dar mejor. `t_pose` se informa al lado. Los saltos entre
tramos no se cronometran.

**Estadísticos**, por celda y video, sobre los 2000 cuadros de cada ronda y sobre las dos rondas
juntas: mediana, p95, % de cuadros sobre 40 ms y % sobre 66 ms, para `t_total` y para `t_pose`.
También FPS sostenidos (cuadros sobre tiempo de pared del tramo medido) y la fracción de cuadros
donde la celda misma detectó al menos dos personas.

**Rondas.** Dos: la primera en orden de celdas, la segunda en orden inverso, para que la deriva
térmica no caiga siempre sobre las mismas. Si la mediana de una celda difiere más de 10% entre
rondas, se marca en el resultado; no se repite.

**Máquina.** Reloj de la GPU sin fijar (no hay permiso para hacerlo): temperatura, reloj y potencia se
registran durante cada celda. A y BoxTwinZ quedan levantados y sin trabajos.

**Contaminación.** Se toma `/tmp/boxtwin-gpu.lock` para toda la corrida. A no conoce ese candado, así
que antes y después de cada celda se lee su cola y `nvidia-smi`. Si durante una celda A tuvo trabajos
o hubo otro proceso de cómputo en la GPU, la celda se descarta, se espera a que la cola se vacíe y se
repite. Las descartadas se cuentan en el resultado.

## 5. Calidad

`CLAUDE.md` §6.1 pide mAP de keypoints. El boxeo no tiene verdad de keypoints, así que hay dos cosas
y ninguna es calidad sobre boxeo:

- **mAP pose (COCO val2017)** de cada celda medida, con el mismo motor o archivo ONNX. Fija el orden
  de calidad para la regla de §6. Corre después de los tiempos. Si el 4:00 llega antes, o COCO no se
  puede bajar, no se mide y se declara.
- **Concordancia con la base**, sobre los mismos cuadros del benchmark: error de muñeca contra el
  caché de `raiz-v8` (v8l FP32), normalizado por el alto de la caja, y fracción de muñecas con puntaje
  ≥ 0,3. Sin umbral: es una descripción.

## 6. Regla de asignación de perfiles

**Pasa un presupuesto** una celda cuyo p95 de `t_total`, con las dos rondas juntas, queda en o por
debajo del presupuesto **en los dos videos**. Equivale a pedir como mucho 5% de cuadros sobre
presupuesto. Dos videos, sin promediar: gana el peor.

**Orden de calidad.** mAP pose de §5, de mayor a menor. Diferencias menores a 0,5 puntos cuentan como
empate y gana la de menor mediana de `t_total`. Si el mAP no se midió: tamaño (l > m > s > n), después
resolución, y la asignación se marca como provisional.

| Perfil | Presupuesto | Formatos elegibles | Qué es |
|---|---|---|---|
| **A** | 40 ms (25 fps) | TensorRT FP16 y ONNX | la de mayor calidad que pasa: el modo en vivo |
| **B** | 66 ms (15 fps) | TensorRT FP16 y ONNX | la de mayor calidad que pasa |
| **C** | 66 ms (15 fps) | solo ONNX | la de mayor calidad que pasa: el perfil portable, para máquinas donde no se compila un motor |

Consecuencias fijadas desde ahora:

- Si ninguna celda pasa, ese perfil queda **vacío**. No se relaja el presupuesto.
- Si la configuración de B es la misma que la de A, **B colapsa en A** y se informa así: en esta máquina
  15 fps no compran calidad. No se fuerza una distinta.
- Lo mismo C respecto de A o B.

**Lectura para la sub-hipótesis 3 y el modo en vivo.** Si el perfil A queda vacío, la pose sola no
entra en 25 fps en la estación y la sub-hipótesis 3 queda refutada para esta máquina con estos
modelos. Si no queda vacío, la condición necesaria se cumple y sigue abierta por lo que §1 deja afuera.

## 7. Controles

| Id | Qué | Si falla |
|---|---|---|
| K-r11a | Cada exportado corre y detecta: en los 200 cuadros de warm-up de sparring-3, al menos dos personas en el 90% de los cuadros | la celda queda fuera y se declara |
| K-r11b | Los cuadros medidos son los de la lista: el sha256 de la lista es igual en todas las celdas | se para |
| K-r11c | A y BoxTwinZ intactos: commit de `/salud`, servicios activos y colas sin trabajos propios, antes y después | se declara |

## 8. Dónde vive cada cosa

- Código y pruebas: `scripts/r11/`.
- Datos: `~/boxtwin-mejoras/r11/`, fuera de git.
- Resultado: `docs/experiments/2026-11-04-r11-perfiles.md`, `docs/experiments/datos/` y entrada en el
  `README`.
- Tesis: nada se escribe; los cambios se proponen en el resumen de la noche.
