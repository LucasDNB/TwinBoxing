# E7 · 3D de 4D-Humans en lugar de MotionBERT

24-10-2026

- `01-preregistro.md`: pregunta, condiciones de uso, configuración fija y reglas de lectura. Se
  commitea antes de correr nada y después no se edita.
- `02-diseno.md`: cómo se arma, archivos, pruebas y pasos.

De dónde sale: en BM, el mismo código de BoxMind con los mismos golpes de BoxingWeb pierde 0,139
de F1 de técnica al pasar de la pose de BoxingWeb (4D-Humans, identidad verificada) a la de
BoxTwin (YOLOv8l-pose más MotionBERT): BM-0 0,617, BM-0b 0,478
(`docs/experiments/2026-10-23-boxmind-prueba.md`). Ese número mezcla el estimador 3D con la
identidad y con el 2D. E5 ya había mostrado que el 3D de MotionBERT no mejora el hook. E7 cambia
solo el estimador 3D, sobre los 1047 golpes de BoxTwin y con todo lo demás de E5 fijo, sin usar
nada de BoxingWeb.

Pasos manuales de Lucas antes de correr, con lo que no se puede automatizar:

| Id | Paso |
|---|---|
| Q1 | Aprobar este pre-registro |
| Q2 | Registrarse en `smpl.is.tue.mpg.de`, aceptar la licencia y bajar `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` a `~/boxtwin/e7-4dhumans/smpl/` |
| Q3 | Autorizar el entorno conda nuevo `boxtwin_4dhumans` (diseño §1). No toca `boxtwin_mmaction` ni `twinboxing_env` |
| Q4 | GPU libre la noche de la corrida |

Fecha de corte: 31-10-2026. Si no termina, se informa como no realizado.
