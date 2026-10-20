# Registro de tareas (SDD) · clasificador geométrico y 3D, noche del 15 al 16-10

Specs: `docs/specs/2026-10-16-clasificador-geometrico/` (00 a 03). Estados: `pendiente`,
`en curso`, `hecha` (pruebas en verde, evidencia escrita), `bloqueada` (qué la bloquea),
`no realizado` (decisión delegada que la saca, con el Id de la decisión). La evidencia es el
comando que la verifica o el archivo que deja.

Orden: G0 primero. Después G1 a G6 en orden. G7 y G8 son la parte 3D y pueden quedar
`no realizado` sin afectar al resto. G9 a G12 cierran.

| Id | Tarea | Spec | Depende de | Estado | Evidencia / notas |
|---|---|---|---|---|---|
| G0 | Arranque: rama `exp/clasificador-geometrico`, sesión en la raíz, pruebas de `scripts/clasificador` en verde, `huellas.py verificar` (las huellas del 14-10, en `~/boxtwin/clasificador-2026-10/huellas.json`), commit del pre-registro solo, hash en `<trabajo>/preregistro.txt`, `nvidia-smi` con versión de driver en `NOCHE.md` | 00, 01 | — | hecha | rama `exp/clasificador-geometrico`; pre-registro commit 703fd11 (solo); 29 pruebas de scripts/clasificador ok; huellas intactas; driver 595.99.02 en NOCHE.md |
| G1 | `datos.py` + pruebas: golpes por fuente, ventanas, etiquetas, `rol` por `Event.arm_role`; chequeo contra los folds de E1 | 02 §1, §7.1 | G0 | hecha | `datos.py`: 1047 golpes, conteos por fuente = pre-registro, mismo conjunto de frame_dir que folds E1; series por fuente en `<trabajo>/datos/series/`; 5 pruebas ok |
| G2 | `features2d.py` + pruebas: 20 del brazo (import de `features_de`), 6 del rival, remuestreo a 32; escribe `golpes.npz` | 02 §2, §7.2 | G1 | hecha | `features2d.py`: X (1047, 32, 26), 84,2% de pasos usables, 13 golpes sin ningun paso usable (ver decisiones); 5 pruebas ok; huellas intactas |
| G3 | `modelo.py`, `entrenar.py` + pruebas: `ClasificadorGolpe`, una fuente afuera, cinco semillas, reanudable, CPU o GPU | 02 §3, §7.3, §7.4 | G2 | hecha | `modelo.py`, `entrenar.py`; 8 pruebas nuevas (forma, máscara, sobreajuste, misma semilla = mismo pred en CPU, sin fuga de la fuente de prueba, reanudable, CropResize) |
| G4 | `regla_c8.py` + prueba: la regla de C8 sobre los 775 completos de las cuatro fuentes de C8 | 02 §6 | G1 | hecha | `regla_c8.py`: 775 golpes, 128 al cuerpo (= pre-registro), ventana anotada sin margen, sin dato = error; 2 pruebas |
| G5 | `evaluar.py` + pruebas: ensambles, métricas, ocho clases, veredictos del pre-registro §3, fusión F | 02 §6, §7.7 | G3, G4 | hecha | `evaluar.py`: 11 pruebas (perfectas = 1, 0,029 / 0,031 con 4 y 5 de 7, desvío > 0,03, 3 de 4, ocho clases, fusión, orden de pred.npz) |
| G6 | Correr E4 y E4a (7 folds × 5 semillas cada una) y evaluar; F si están los `pred.pkl` de E3 | 01 §2, §3 | G3, G5 | hecha | 70 corridas (log `logs/G6-entrenar.log`, cuda, ~3 s c/u); `evaluar.py --exp E4 E4a`: E4 familia 0,727 vs E1 0,656, 5 de 7 folds, desvío 0,015 → P5 familia **mejora**; altura C8 0,719 vs regla 0,729, 1 de 4 → **no demostrada**; C8b no; ocho clases 0,623 (lectura a). F 0,741. E3 recalculado desde pred.pkl = 0,667 / 0,605 (igual al 15-10). Huellas intactas |
| G7 | `elevar3d.py`: MotionBERT sobre las ventanas del atacante; control de calidad de largos de hueso; `pose3d.npz` | 02 §4.1 | G1 | hecha | `elevar3d.py` (boxtwin_mmaction, cuda, 117 s, lote 8): 1047 golpes, 99,7% de pasos con 3D; pesos sha256 d80af323… en `pesos/pesos.sha256`; QC en `datos/pose3d.qc.json` (CV antebrazo mediana 0,08 a 0,14 por fuente, brazo 0,05 a 0,08); 4 pruebas de funciones puras |
| G8 | `features3d.py` + pruebas (invariancia a rotación y espejo); correr E5 y evaluar | 02 §4.2, §7.5 | G6, G7 | hecha | `features3d.py` (16 features, ver decisión 9), 4 pruebas (invariancia a rotación y escala, espejo); E5 35 corridas (42 features); `evaluar.py`: familia 0,738 vs E4 0,727, +0,011, 5 de 7 → P6 **no demostrada**; recall hook E5 0,590 vs E4 0,620. Huellas intactas |
| G9 | `rol_por_brazo.py` + pruebas; correr E6 | 02 §5, §7.6 | G0 | hecha | `rol_por_brazo.py`, 8 pruebas (regla en los 4 casos, brazo cambiado, inversión A/B, umbral 0,05); E6: 390 pares, acuerdo de brazo 0,979; 259 rectos: regla 0,985, todos ortodoxos 0,811, clasificador (E1, ver decisión 11) 0,726, siempre adelantada 0,699 → P7 **mejora** (+0,259). Huellas intactas |
| G10 | `auditor-medicion` sobre E4 a E6 (adaptar sus ítems: pre-registro antes que resultados, huellas, sin fuga por fold, recálculo independiente de la familia de E4 y de E6) | 01 | G6, G9 (G8 si se hizo) | hecha | Auditor independiente con las instrucciones de `auditor-medicion` (el agente del repo no estaba cargado en la sesión): 11 de 11 PASA; observaciones incorporadas a los documentos (desvío 44→42, mejora de E4 no robusta por semilla: 3 de 5, P7 contra E1 y regla = acuerdo de brazo por construcción, cadencia de MotionBERT, ocho clases de E3 en F) |
| G11 | Documentos: `docs/experiments/2026-10-16-clasificador-geometrico.md` (E4, E4a, F, E5) y `docs/experiments/2026-10-16-rol-por-brazo.md` (E6), con el formato de `2026-10-15-clasificador-tipo-altura.md`; datos a `docs/experiments/datos/`; entradas cortas en el README raíz las deja propuestas en el documento (el README pide confirmación) | 01 §5 | G10 | hecha | `docs/experiments/2026-10-16-clasificador-geometrico.md`, `2026-10-16-rol-por-brazo.md`, datos en `docs/experiments/datos/2026-10-16-*.json`; entrada del README propuesta en el §8 del primero, README sin tocar |
| G12 | Notas de tesis `docs/tesis/2026-10-16-clasificador-geometrico-cambios.md` (Cap 7, Cap 8 con BoxMind, R-01, Declaración de uso de IA) y resumen final en `NOCHE.md` | 01 §5 | G11 | hecha | `docs/tesis/2026-10-16-clasificador-geometrico-cambios.md`; resumen al final de `NOCHE.md` |

## Decisiones tomadas durante la noche

(Claude Code agrega acá cada elección de D10, con motivo.)

1. G2. La máscara por cuadro es la `usable` de `features_de` tal cual, sin cruzarla con `valid`
   del atacante: el diseño §2.1 la nombra así y un cuadro sin pose ya tiene score 0. 13 golpes
   quedan sin ningún paso usable; no se sacan (el pre-registro fija 1.047 golpes), y en el
   pooling enmascarado su vector queda en cero, así que el modelo los clasifica solo con el sesgo.
2. G2. Inclinación del rival como ángulo de `c_r → s_r` contra la vertical hacia arriba de la
   imagen: `cos = -v_y/|v|`, `sin = v_x/|v|`, en píxeles sin espejar, como dice §2.2.
3. G3. CropResize se aplica sobre la grilla de 32 pasos ya remuestreada (recorte de 80 a 100% e
   interpolación lineal a 32). Como la grilla es uniforme sobre la ventana, es el mismo recorte
   que sobre la ventana original, sin volver a leer las series en cada época.
4. G3. Para que la máscara importe de verdad, la entrada se multiplica por M antes de la primera
   convolución: si no, las convoluciones dilatadas llevan a los pasos usables lo que hay en los no
   usables. Un lote de una sola muestra se saltea (BatchNorm no entrena con una); con los tamaños
   de fold reales no pasa.
5. G5. "Gana en un fold" se lee como estrictamente mayor; un empate no cuenta. Para P6, el
   freno del punto 6 se aplica si el desvío de familia de E4 o el de E5 pasa de 0,03 (la lectura
   más estricta, D10). El freno de dispersión no se aplica a la altura, porque el punto 6 habla
   solo de la familia.
6. G5. P5 altura compara la balanceada de E4 sobre las cuatro fuentes de C8 (ensamble, 775
   golpes) con la de la regla sobre esos mismos golpes, y por fuente fold contra fuente.
7. G7. `inference_pose_lifter_model` de MMPose 1.3.2 arma `lifting_target` de un cuadro y falla
   con secuencias de 243 (el demo eleva cuadro por cuadro, con seq_len 1 del config, y así
   MotionBERT pierde el contexto temporal). Se usa el mismo pipeline del config y el mismo
   `model.test_step`, armando el data_info con las formas de 243 cuadros. Una pasada por golpe,
   contexto centrado en la ventana; la salida se interpola a los 32 pasos en coordenadas de
   cuadro. `keypoints_visible` en 1 y `norm_pose_2d` apagado, como en esa función: la
   normalización la hace el codec de MotionBERT con `camera_param` = (ancho, alto).
8. G7. Lote de 64 secuencias no entra en los 8 GB (OOM); se bajó a 8. No cambia el resultado.
9. G8. El diseño §4.2 dice "18 en total" pero la lista nombra 16 (9 coordenadas, ángulo del
   codo, elevación, azimut cos y sin, velocidad horizontal y vertical de la muñeca, velocidad
   del azimut). Se implementan las 16 nombradas y E5 entra con 26 + 16 = 42, no 44. Inventar dos
   features para llegar al número sería decidir algo de fondo después de escribir el
   pre-registro. Va también a preguntas abiertas.
10. G8. E5 usa la misma máscara que E4 (la 2D), para que lo único que cambie entre las dos sean
    las features; donde no hay 3D (0,3% de los pasos) las 16 van en cero. Las velocidades 3D
    van por segundo: derivada sobre los 32 pasos con dt = duración de la ventana / 31.
11. G9. Las cuatro Fight-Cards de C8 nivel 2 se completaron sin clasificador: `tipo` en None en
    los 1.285 golpes y `clasificador.checkpoint` vacío. No es D4 (las sesiones están). Lectura
    más estricta: el clasificador de producción medido fuera de distribución es E1 (config de
    producción con la fuente afuera, s0, `pred.pkl` del 15-10), que es el 0,656 de rol que motiva
    E6 en el README. Su argmax de seis clases se toma sobre el evento anotado emparejado; un hook
    o uppercut sobre un recto anotado cuenta como error, como pide el diseño §5. La alternativa
    (correr el checkpoint de producción sobre las sesiones) daría un número en distribución,
    optimista, y va a preguntas abiertas.

## Preguntas abiertas para Lucas

(Lo que no se pudo resolver sin decidir algo de fondo.)

1. Diseño §4.2: ¿cuáles eran las dos features 3D que faltan para llegar a 18? Se corrió E5 con
   las 16 que nombra la lista (decisión 9). Si había dos más pensadas, E5 se vuelve a correr con
   ellas como otra configuración, no como reemplazo.
2. E6: ¿se quiere además el rol del checkpoint de producción corriendo `clasificar_sesion.py`
   sobre copias de las sesiones de C8 nivel 2? Ese checkpoint vio las cuatro fuentes, así que el
   número saldría en distribución; P7 se decidió contra E1 (decisión 11).
3. E6 usa la guardia anotada del evento. En el producto la guardia no viene de la anotación;
   cuánto pierde la regla con una guardia estimada no se midió (la variante todos ortodoxos da
   0,811, una cota pesimista).
4. Cap 8: verificar el identificador arXiv de BoxMind (2601.11492) antes de citarlo; viene del
   spec y no se verificó en esta sesión.

### Respuestas de Lucas (20-10-2026)

1. E5 con 18 features: no se corre. Queda como línea futura.
2. Rol con el checkpoint de producción: no se corre. P7 queda decidido contra E1.
3. Guardia estimada en E6: sigue abierta como limitación declarada.

Sub-hipótesis 1: Lucas sigue intentando hasta el corte del 31-10, con las mediciones primero
(C1 y benchmark R-11 esta semana) y el clasificador después en la GPU.
