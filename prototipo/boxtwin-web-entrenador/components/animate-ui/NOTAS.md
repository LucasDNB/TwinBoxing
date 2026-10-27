# Cambios a mano sobre lo que instaló el CLI

Lo que vive en `components/animate-ui/` y `components/ui/` lo copió el CLI de shadcn y no se
edita salvo por tokens (`02` §3). Cada cambio queda acá.

| Fecha | Archivo | Cambio | Por qué |
|---|---|---|---|
| 13-10 | `components/ui/*.tsx` | `import { cn } from "cn"` → `"@/lib/utils"` | El CLI 4 resolvió el alias de utils como el paquete npm `cn`, que no es el helper de shadcn; se desinstaló el paquete |
| 13-10 | `components/radix/sidebar.tsx` | `openDelay` → `delayDuration`; `side` y `align` pasan de `Tooltip` a `TooltipContent` | El registro trae el sidebar escrito contra la API de tooltip de la familia Base UI; con la de Radix no compila (`tsc`) |
| 13-10 | `primitives/effects/highlight.tsx` | se quita `aria-selected` de los atributos de cada ítem | El highlight lo ponía en todo ítem, también en los botones con rol `radio` de los toggle-group, donde ARIA no lo permite: axe lo marca como violación seria en cada pantalla con un toggle-group (encontrado en el e2e). El estado ya lo dicen los roles de Radix (`aria-checked`, `aria-selected` de la pestaña) |
| 13-10 | `components/ui/sonner.tsx` | `theme="dark"` fijo, sin `next-themes` | Tema oscuro único (`03` §1); `next-themes` no hace falta |

Versión de `motion` fijada en 12.43.0 (`01` §7, riesgo 3). La prueba
`tests/unit/humo-animate-ui.test.tsx` monta cada componente y falla si cambia la versión.
