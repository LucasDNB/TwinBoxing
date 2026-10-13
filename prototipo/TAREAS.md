# Registro de tareas (SDD)

Estado: `hecha` (pruebas en verde), `parcial` (lo que falta está dicho), `pendiente`,
`bloqueada` (qué la bloquea). La evidencia es el comando que la verifica.

| Id | Tarea | Spec | Estado | Evidencia / notas |
|---|---|---|---|---|
| W0 | Paquete Next 16, TS estricto, Tailwind 4, export condicional, ESLint, Vitest, Playwright | 02 §2, §3, §9 | hecha | `npm run build` genera `out/`; `npm test` corre vacío |
| W1 | shadcn, registro `@animate-ui`, componentes de 03 §4, `motion@12.43.0`, tokens, fuentes, humo | 03 §2, §4 | hecha | `npm test` (humo: 5 pruebas; contraste: 3); `npm run contraste`. Hallazgo: letra clara sobre relleno A da 4,11:1 y sobre B 3,92:1; la marca A/B se dibuja con contorno y letra en el color de trazo (≥ 4,83:1) y nunca sobre `--color-elevado` |
| W2 | Layout (barra inferior y lateral), guarda, ingreso, cliente con token y XHR, zod, `formato` portado | 02 §5 a §7, 04 §00 | hecha | `npm test`: `ingreso.test.tsx` cubre CA-00.1 (funcional) y CA-00.2; `formato.test.ts` con los casos de la web vieja sin cambios. El "< 2 s en 4G" de CA-00.1 se mide con Lighthouse (06 §5), pendiente. El lint excluye el código copiado por el CLI (no pasa las reglas del React Compiler); el sidebar trae un bug de tipos del registro, corregido y anotado |
| B0 | Servir `out/` desde la API con cabeceras | 05 B0 | hecha | `pytest tests/test_web_entrenador_b0_b5.py`. Desvío: la CSP suma `script-src 'self' 'unsafe-inline'`, porque la exportación de Next mete el payload RSC en scripts en línea y sin eso no hidrata |
| B5a | `GET /producto` con criterios y banderas por entorno | 05 B5 | hecha | mismo archivo de pruebas; `BOXTWIN_C2`, `BOXTWIN_C3` (`pendiente`/`aprobado`/`rechazado`) y `_FECHA`; un valor mal escrito cuenta como pendiente |
| B1 | Contexto, rounds, consentimiento; migración SQL | 05 B1 | hecha (falta aplicar en la estación) | `pytest tests/test_web_entrenador_b1.py` (9 pruebas). Migración en `despliegue/migraciones/2026-10-web-entrenador.sql` (idempotente, también trae B3 y B6). `db.py` ya agrega solas las columnas nulas, así que la API arranca sobre una base sin migrar; la prueba lo verifica con una base anterior a B1. Aplicarla en la estación queda para Lucas. Choque resuelto: la web vieja no mandaba consentimiento y la API ahora lo exige; se le sumó la casilla (diff de 14 líneas en `boxtwin-web`) para que las dos webs convivan hasta la compuerta |
