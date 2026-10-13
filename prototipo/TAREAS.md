# Registro de tareas (SDD)

Estado: `hecha` (pruebas en verde), `parcial` (lo que falta está dicho), `pendiente`,
`bloqueada` (qué la bloquea). La evidencia es el comando que la verifica.

| Id | Tarea | Spec | Estado | Evidencia / notas |
|---|---|---|---|---|
| W0 | Paquete Next 16, TS estricto, Tailwind 4, export condicional, ESLint, Vitest, Playwright | 02 §2, §3, §9 | hecha | `npm run build` genera `out/`; `npm test` corre vacío |
| W1 | shadcn, registro `@animate-ui`, componentes de 03 §4, `motion@12.43.0`, tokens, fuentes, humo | 03 §2, §4 | hecha | `npm test` (humo: 5 pruebas; contraste: 3); `npm run contraste`. Hallazgo: letra clara sobre relleno A da 4,11:1 y sobre B 3,92:1; la marca A/B se dibuja con contorno y letra en el color de trazo (≥ 4,83:1) y nunca sobre `--color-elevado` |
