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
| G4 | `regla_c8.py` + prueba: la regla de C8 sobre los 775 completos de las cuatro fuentes de C8 | 02 §6 | G1 | pendiente | |
| G5 | `evaluar.py` + pruebas: ensambles, métricas, ocho clases, veredictos del pre-registro §3, fusión F | 02 §6, §7.7 | G3, G4 | pendiente | |
| G6 | Correr E4 y E4a (7 folds × 5 semillas cada una) y evaluar; F si están los `pred.pkl` de E3 | 01 §2, §3 | G3, G5 | pendiente | |
| G7 | `elevar3d.py`: MotionBERT sobre las ventanas del atacante; control de calidad de largos de hueso; `pose3d.npz` | 02 §4.1 | G1 | pendiente | |
| G8 | `features3d.py` + pruebas (invariancia a rotación y espejo); correr E5 y evaluar | 02 §4.2, §7.5 | G6, G7 | pendiente | |
| G9 | `rol_por_brazo.py` + pruebas; correr E6 | 02 §5, §7.6 | G0 | pendiente | |
| G10 | `auditor-medicion` sobre E4 a E6 (adaptar sus ítems: pre-registro antes que resultados, huellas, sin fuga por fold, recálculo independiente de la familia de E4 y de E6) | 01 | G6, G9 (G8 si se hizo) | pendiente | |
| G11 | Documentos: `docs/experiments/2026-10-16-clasificador-geometrico.md` (E4, E4a, F, E5) y `docs/experiments/2026-10-16-rol-por-brazo.md` (E6), con el formato de `2026-10-15-clasificador-tipo-altura.md`; datos a `docs/experiments/datos/`; entradas cortas en el README raíz las deja propuestas en el documento (el README pide confirmación) | 01 §5 | G10 | pendiente | |
| G12 | Notas de tesis `docs/tesis/2026-10-16-clasificador-geometrico-cambios.md` (Cap 7, Cap 8 con BoxMind, R-01, Declaración de uso de IA) y resumen final en `NOCHE.md` | 01 §5 | G11 | pendiente | |

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

## Preguntas abiertas para Lucas

(Lo que no se pudo resolver sin decidir algo de fondo.)
