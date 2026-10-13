# Cambios del contrato de la Fight-Card

Cada versión es compatible hacia atrás: una Fight-Card vieja se sigue leyendo, y un cliente que
no conoce los campos nuevos los ignora. Lo que se rompe se dice acá.

## 0.3 · ampliación del 13-10 (`prototipo/specs/ampliacion-13-10/`)

- `peleadores.X.combinaciones` (B10, `01` §3): las ternas izq-der-izq con menos de 0,5 s de
  inicio a inicio, sin solapar. Lleva su regla adentro (`patron`, `separacion`, `max_s`) para
  que el documento diga con qué se contó. Se recalcula al aplicar tipos y correcciones.
- **Rompe** en `GET /boxeadores/{id}/perfil`: `combinaciones` deja de ser la lista de las cinco
  secuencias más frecuentes por tipo y pasa a ser un objeto (`patron`, `total`, `por_minuto`,
  `por_contexto`, `tipos_frecuentes`, `sesiones_con_combinaciones`). La única cliente es la web
  entrenador, que se cambió en el mismo PR.
- **Rompe** en `GET /producto`: `dominio.pausa_combinacion_s` se reemplaza por
  `dominio.combinacion` (`patron`, `separacion`, `max_s`). La variable
  `BOXTWIN_PAUSA_COMBINACION` deja de existir: la regla es una decisión de dominio versionada en
  `boxtwin.mvp.combinaciones.REGLA`, no una perilla por entorno.

## 0.2 · web entrenador (B3, B4)

- `descartado` por golpe y `descartados` por peleador ("no fue un golpe").
- Campos de tramos en `identidad`.

## 0.1 · MVP
