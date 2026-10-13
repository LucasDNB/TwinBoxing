# Contrato de la Fight-Card

`fightcard-0.2.schema.json` es el JSON Schema que validan los dos lados (`06` §3):

- el worker: `boxtwin-annotator/tests/test_mvp_fightcard_contrato.py` valida lo que produce
  `boxtwin.mvp.fightcard.construir`, con y sin correcciones;
- la web: `boxtwin-web-entrenador/tests/unit/contrato.test.ts` valida los fixtures de una sesión
  real (0.1 y 0.2) y comprueba que zod acepte lo mismo que el schema.

Si uno de los dos cambia el documento sin tocar el schema, su prueba falla. Así la web y el
worker no se separan en silencio.

El spec ubica este archivo en `docs/specs/web-entrenador/`; vive acá porque todo el trabajo del
prototipo está dentro de `prototipo/`.
