# Dos instancias en paralelo · producción y académica

25-10-2026 · spec, nada lanzado todavía · registro en `04-pasos.md`

En la estación conviven dos BoxTwin:

| | A · producción | B · académica |
|---|---|---|
| Qué es | BoxTwin tal como venía funcionando | BoxTwin con el modo académico E7 (PR #27 y #28) |
| Código | copia principal, `main` en `34e991d` | worktree fijo `~/Proyectos/TwinBoxing-academico`, rama `e7-modo-academico` |
| API | `127.0.0.1:8000` | `127.0.0.1:8001` |
| Afuera | Funnel público, `https://desarrollo-lucas.tail52f6aa.ts.net` (sin cambios) | solo la tailnet, `https://desarrollo-lucas.tail52f6aa.ts.net:8443`, **sin Funnel** |
| Servicios | `boxtwin-api`, `boxtwin-worker` | `boxtwin-academico-api`, `boxtwin-academico-worker` |
| Entorno | `/etc/boxtwin.env` | `/etc/boxtwin-academico.env` |
| Datos y base | `~/boxtwin/datos/` | `~/boxtwin-academico/datos/`, base nueva y vacía |
| Tipo de golpe | E4 (`geometrico`) | E7 (`e7`), con E4a en bolsa y sombra |

Archivos:

- `01-produccion.md`: qué quiere decir "sin ningún cambio" y cómo se comprueba.
- `02-academica.md`: la instancia B, con el entorno y las unidades escritas.
- `03-convivencia.md`: GPU, red, datos, sesiones de prueba y vuelta atrás.
- `04-pasos.md`: el orden, con la verificación de cada paso.

## Antes de lanzar B (bloqueante)

La enmienda del modo académico (`docs/specs/2026-10-25-e7-modo-academico/00-enmienda.md`) no deja
encender E7 sin el texto de la licencia de SMPLify (condición 3), y B lo enciende. Hay dos
salidas, y las dos son de Lucas:

1. Pegar el texto de la licencia en `02-registro.md` del modo académico, y que permita una
   demostración académica sin cobro ni distribución. Es la recomendada.
2. Enmendar por escrito la condición 3 para B: solo en la tailnet, sin usuarios y solo con
   material propio.

Sin una de las dos, B se puede armar entera pero no se arranca.
