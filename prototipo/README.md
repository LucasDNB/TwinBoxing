# prototipo · web entrenador de BoxTwin

Desarrollo guiado por specs (SDD) de la web entrenador, a partir de `specs-prototipo.zip`
(12-10-2026).

```
prototipo/
├── specs-prototipo.zip        el original, sin tocar
├── specs/                     el zip descomprimido: la fuente de verdad de cada tarea
├── TAREAS.md                  registro SDD: tarea, spec que lee, estado y evidencia
└── boxtwin-web-entrenador/    la web nueva (Next.js, exportacion estatica)
```

Los cambios de API (tareas B) van en `boxtwin-api/` y `boxtwin-annotator/`, en su lugar, porque
la web nueva corre sobre la misma API que la vieja y las dos tienen que convivir hasta la
compuerta de reemplazo (`specs/07-plan.md` §2).

## Desvíos de ubicación respecto de los specs

Los specs ubican el paquete en `boxtwin-web-entrenador/` en la raíz y el JSON Schema en
`docs/specs/web-entrenador/`. Acá viven dentro de `prototipo/` porque así se pidió la carpeta.
Si la compuerta pasa, mover `prototipo/boxtwin-web-entrenador` a la raíz es un `git mv` y un
cambio de `BOXTWIN_WEB`.

## Cómo se trabaja

Una tarea por vez, en el orden de `specs/07-plan.md`. Cada tarea lee solo los apartados que
cita, y se cierra con sus pruebas en verde. Las tareas B siguen el Prompt 11: el plan
(endpoint, payload, tabla, prueba) ya está escrito en `specs/05-api-y-datos.md`; el código
se escribe contra ese plan y la prueba lo verifica.
