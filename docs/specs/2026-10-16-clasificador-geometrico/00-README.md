# Clasificador geométrico y 3D (E4 a E6) · noche del 15 al 16-10-2026

Pedido de Lucas (15-10): mejorar el clasificador de tipo y altura a partir de lo que hacen
trabajos parecidos, con Claude Code trabajando de corrido la noche del 15 al 16 por SDD.

## Por qué estas tres cosas

La campaña E1 a E3 (`docs/experiments/2026-10-15-clasificador-tipo-altura.md`) dejó:

1. La familia, fuera de distribución, en 0,656, contra 0,632 de decir siempre recto.
2. El rol (mano adelantada o atrasada) en 0,656, por debajo de decir siempre adelantada
   (0,668). En producción jab y cross salen de ese número.
3. La altura aprendida (0,541 a 0,612 de balanceada) por debajo de la regla geométrica de C8
   (0,726).

La hipótesis que sale del repo y de la literatura:

- El detector generaliza porque trabaja con 20 features centradas, escaladas y espejadas
  (`boxtwin-detector/src/boxtwin_detector/features.py`). PoseConv3D trabaja con mapas de
  calor en píxeles y no generaliza. → **E4**: el mismo modelo chico sobre esas features, más
  features del rival para la altura.
- El hook es un problema de proyección monocular. BoxMind (arXiv 2601.11492) clasifica
  técnica y altura en boxeo de élite con pose 3D estimada desde una cámara. Skarimva
  (arXiv 2602.23231) mide que 3D baja el error a la mitad. → **E5**: elevar a 3D con
  MotionBERT, que ya trae MMPose 1.3.2 en `boxtwin_mmaction`, y sumar features 3D en un
  marco del cuerpo.
- El rol no necesita un modelo: el detector ya trabaja por brazo. → **E6**: rol = brazo del
  detector + guardia, contra el clasificador.

Referencia para calibrar expectativas: BoxMind, con 10.900 eventos, anotación de
entrenadores profesionales, 3D y partición por round, da F1 0,692 de técnica y 0,747 de
altura con pose sola. Llegar cerca de eso sería un buen resultado; 0,80 en ocho clases no está
al alcance esta semana, y el pre-registro lo dice.

## Archivos

| Archivo | Qué es |
|---|---|
| `01-preregistro.md` | Preguntas, configuraciones, métricas y lectura. Se commitea antes de medir y no se toca después. |
| `02-diseno.md` | Datos, features, modelo, entrenamiento, 3D, E6, evaluación, layout y pruebas exigidas. |
| `03-decisiones-delegadas.md` | Qué hace Claude Code sin preguntar cuando algo sale mal de noche. |
| `scripts/clasificador_geo/TAREAS.md` | Registro SDD: tareas G0 a G12 con estado y evidencia. Es lo que Claude Code actualiza. |
| `.claude/skills/noche-geometrico/SKILL.md` | El bucle de trabajo desatendido (`/noche-geometrico`). |

## Cómo se lanza (Lucas, antes de dormir)

1. Leer `01-preregistro.md` (cinco minutos). Si algo no te cierra, cambialo ahora: después
   de lanzar queda fijo.
2. En la estación, desde la raíz del repo, con el parche aplicado:
   `claude` → `/noche-geometrico`.
3. Modo automático activado. No hace falta aprobar nada más: la campaña no usa `tmux` ni
   pasos con `ask`.

A la mañana: `scripts/clasificador_geo/TAREAS.md` dice qué se hizo y qué quedó;
`~/boxtwin/clasificador-geo-2026-10/NOCHE.md` es la bitácora; los documentos de resultados
están en `docs/experiments/`.

## Regla de oro

Nada se mide antes del commit del pre-registro, y ninguna lectura se cambia después de ver un
número. Si aparece un bug en la evaluación después de ver resultados, se arregla, se corre todo
de nuevo y se documentan las dos corridas.
