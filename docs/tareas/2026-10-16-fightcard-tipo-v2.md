# Registro de tareas (SDD) · Fight-Card tipo v2

Specs: `docs/specs/2026-10-16-fightcard-tipo-v2/` (00 a 03). Estados: `pendiente`, `en curso`,
`hecha` (pruebas en verde y evidencia escrita), `bloqueada` (qué la bloquea), `no realizado`
(con la decisión delegada). Evidencia: el comando que la verifica y su resultado, o el archivo.

| Id | Tarea | Spec | Depende de | Estado | Evidencia / notas |
|---|---|---|---|---|---|
| V0 | Arranque: rama `feat/fightcard-tipo-v2` desde `main` con el PR de `exp/clasificador-geometrico` ya mergeado (si no lo está, desde esa rama, y se dice); sesión en la raíz; suites existentes en verde (línea de base, con conteo de pruebas por suite en `NOCHE.md`); huellas; hash del commit que trajo el spec (es el que fija M1: `git log -1 --format=%H -- docs/specs/2026-10-16-fightcard-tipo-v2/01-requisitos.md`) en `~/boxtwin/tipo-v2/spec.txt` | 00, 01 | — | hecha | Rama desde `main` (4ff0712, PR #14 mergeado). Base: anotador 696, detector 152, API 208, scripts 75 (+1 skip), web 197, todo en verde (`~/boxtwin/tipo-v2/base/`). `huellas.py verificar`: intactas. spec.txt = 4ff0712 |
| V1 | `entrenar_final.py`: modelo completo (s0 a s4) y `sin_<fuente>` para Sparring y las cuatro de C8; `meta.json` con sha256 | 02 §1 | V0 | pendiente | |
| V2 | `mvp/clasificador_geo.py` + pruebas: clase, `features_golpe`, carga con sha256, ensamble, `rol` | 02 §2, §7 | V0 | pendiente | |
| V3 | CA1 y CA2 contra datos reales (`test_equivalencia_v2.py`) | 01 §3, 02 §7 | V1, V2 | pendiente | |
| V4 | `boxtwin-annotator clasificar` + `tipos.json` 0.2 + extensión de `aplicar_tipos` + pruebas (incluida CA6) | 02 §3 | V2 | pendiente | |
| V5 | Contrato: campos opcionales en `fightcard-0.3.schema.json`; pruebas de contrato del anotador | 02 §5 | V4 | pendiente | |
| V6 | API: bandera, `comando_de`, encadenamiento sin `cmd_clasificador`, guardias desde boxeadores, reclasificación (RT8), `/salud`; pruebas (CA3, CA4) | 02 §4 | V4 | pendiente | |
| V7 | Web: zod, exactitud desde la Fight-Card, aviso de guardia supuesta; pruebas | 02 §5 | V5 | pendiente | |
| V8 | CA7: copia de una sesión real por `clasificar` v2 y `tipos`, tiempo (RNF1), validación contra el contrato | 01 §3 | V4, V5 | pendiente | |
| V9 | `medir_v2.py` + pruebas; correr M1 | 01 §4, 02 §6 | V1, V4 | pendiente | |
| V10 | CA8: todas las suites en verde con la bandera apagada; conteo contra V0 | 01 §3 | V4 a V7 | pendiente | |
| V11 | `auditor-medicion` sobre CA1, CA2 y M1 | 01 | V3, V9 | pendiente | |
| V12 | Documentos: `docs/experiments/2026-10-16-tipo-v2.md` (CA1, CA2, CA7, M1 y auditoría), notas de tesis `docs/tesis/2026-10-16-tipo-v2-cambios.md` (Cap 4 §3.3 y prototipo, Cap 6 R-01 y R-10, Cap 7), cómo encender la bandera, resumen en `NOCHE.md` | 00 | V10, V11 | pendiente | |

## Decisiones tomadas durante la noche

## Preguntas abiertas para Lucas
