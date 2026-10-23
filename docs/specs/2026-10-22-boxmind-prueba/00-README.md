# BM · Prueba académica con el modelo de BoxMind

22-10-2026

- `01-preregistro.md`: qué se mide, condiciones de uso del material de terceros, diseño,
  reglas de lectura y el gatillo de la propuesta. Se commitea antes de correr nada.
- `02-dataset-propio.md`: la propuesta de dataset propio para un BoxTwin comercial, que se
  presenta como respaldada o no según el gatillo.
- Scripts y pasos de corrida: `scripts/boxmind_prueba/` y su `TAREAS.md`.

Lo que BoxMind publicó (verificado en `e06add2`): la arquitectura del clasificador de
atributos, un `train.py` atado a sus datos y el cargador de BoxingWeb. No hay pesos, ni
detector, ni seguidor, ni licencia. Por eso la prueba entrena su arquitectura con poses
propias y cubre lo mismo que E4 (tipo y altura de golpes ya delimitados), no detección ni
identidad.

Para la tesis: si el documento ya quedó cerrado el 20-10, BM entra, si termina antes del
31-10, como anexo de resultados y en la presentación del 17-11, con la aclaración de fecha;
conviene confirmar con el director que se acepta un resultado posterior al cierre.
