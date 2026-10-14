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

## Cómo correrlo

```bash
cd prototipo/boxtwin-web-entrenador
npm install
npm run dev          # http://localhost:3000, reescribe la API a localhost:8000
npm test             # unitarias e integración (vitest + msw)
npm run build        # genera out/; la API lo sirve con BOXTWIN_WEB=<ruta>/out
E2E_SIN_WEBKIT=1 npm run e2e   # punta a punta contra la API real, sin GPU
```

Las pruebas de la API y del anotador: `python -m pytest` desde `boxtwin-api/`, y
`PYTHONPATH=src python -m pytest tests/` desde `boxtwin-annotator/`.

## Dependencias y `npm audit`

`npm audit` reporta 5 vulnerabilidades "high" (revisado el 14-10). Son una sola cadena,
`eslint-config-next` → `@next/eslint-plugin-next` → `fast-glob` → `micromatch` → `braces`, y
solo de desarrollo: `npm audit --omit=dev` da 0, así que no llegan al navegador ni a la API.
`braces` no tiene versión corregida, y lo que propone `npm audit fix --force` es bajar
`eslint-config-next` a la 14, que no es compatible con Next 16. No se aplica; se revisa cuando
salga una versión de `eslint-config-next` que lo resuelva.

## Estado

El detalle por tarea, con su evidencia, está en `TAREAS.md`. Lo que queda abierto y necesita
a Lucas está al final de ese archivo y en el resumen de la sesión del 13-10.
