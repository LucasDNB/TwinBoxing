# Decisiones delegadas

Cada vez que se aplica una, va una línea a `~/boxtwin/tipo-v2/NOCHE.md`. Lo que no está acá y
pide decidir algo de fondo va a "Preguntas abiertas" del registro de tareas, y se sigue con otra
tarea.

| Id | Si pasa esto | Se hace esto |
|---|---|---|
| T1 | Una tarea no pasa sus pruebas después de tres intentos | `bloqueada` con el error; se sigue con lo independiente. |
| T2 | Una suite que estaba en verde se pone en rojo por un cambio de esta noche | Se arregla el cambio, nunca la aserción vieja. Si no se puede, se revierte el cambio con un commit nuevo (no `git reset`) y la tarea queda `bloqueada`. |
| T3 | CA1 o CA2 fallan con datos reales | Se busca la diferencia y se corrige el código de producción para que reproduzca a E4. No se toca nada de `scripts/clasificador_geo/` que ya produjo resultados. Si la diferencia está en E4 (un bug en lo que se midió), se frena, se documenta y queda como pregunta abierta: no se re-mide E4 esta noche. |
| T4 | `huellas.py verificar` falla | Se frena todo. |
| T5 | Faltan las sesiones de C8 nivel 2 | M1 queda `no realizado`. CA7 se hace con la sesión de `amateur_estatico` si está, o con una sesión sintética, y se dice. |
| T6 | La web no compila o `npm test` no corre en la estación | La parte web queda `bloqueada` con el error. El resto sigue: la bandera apagada no depende de la web. |
| T7 | Algo del diseño choca con el código existente | Gana el código existente y su contrato; se adapta el diseño con el cambio mínimo y se anota la decisión con el motivo. |
| T8 | Tentación de mejorar el modelo, las features o los hiperparámetros | No. Va a "Próximos pasos". |
| T9 | Algo pide encender la bandera en la estación, reiniciar servicios, tocar la base de producción, `git push`, borrar o instalar | No se hace. Se deja escrito qué haría falta. |
| T10 | Se termina todo antes de la mañana | Revisión final con `auditor-medicion` sobre M1, documento, notas de tesis y resumen. No se empieza nada nuevo. |
