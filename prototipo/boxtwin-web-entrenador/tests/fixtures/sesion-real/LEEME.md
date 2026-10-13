# Fixtures de una sesión real

Salen de `~/boxtwin/datos/sesiones/4ad5553ab817425e8e80bd23c5c931dd` (sparring de 174,9 s,
procesado el 22-09) con `scripts/armar_fixtures.py`.

- `fightcard-0.1.json`: el documento tal cual lo escribió el worker. Sin tipos: esa sesión se
  procesó sin clasificador.
- `fightcard-0.2.json`: el mismo, llevado a 0.2. **Los tipos son de prueba** (rotan jab, cross,
  hook, uppercut) y el checkpoint lo dice: `prueba-fixture-no-es-un-modelo`. Tiene un golpe
  marcado "no fue un golpe" y los campos de tramos con valores de ejemplo.
- `job-<estado>.json`: `GET /jobs/{id}` en cada estado, con los candidatos reales.

Nada de esto alimenta un número de la tesis: son formas de respuesta para probar la web.
