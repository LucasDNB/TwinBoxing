# Cambios a mano sobre lo que instaló el CLI

Lo que vive en `components/animate-ui/` y `components/ui/` lo copió el CLI de shadcn y no se
edita salvo por tokens (`02` §3). Cada cambio queda acá.

| Fecha | Archivo | Cambio | Por qué |
|---|---|---|---|
| 13-10 | `components/ui/*.tsx` | `import { cn } from "cn"` → `"@/lib/utils"` | El CLI 4 resolvió el alias de utils como el paquete npm `cn`, que no es el helper de shadcn; se desinstaló el paquete |
| 13-10 | `components/ui/sonner.tsx` | `theme="dark"` fijo, sin `next-themes` | Tema oscuro único (`03` §1); `next-themes` no hace falta |

Versión de `motion` fijada en 12.43.0 (`01` §7, riesgo 3). La prueba
`tests/unit/humo-animate-ui.test.tsx` monta cada componente y falla si cambia la versión.
