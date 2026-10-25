# Dos instancias en paralelo · producción y académica

25-10-2026 · spec SDD, revisado el 25-10 (cookies, commit congelado, reparto de tareas) · nada
lanzado todavía · registro en `04-pasos.md`

En la estación conviven dos BoxTwin:

| | A · producción | B · académica |
|---|---|---|
| Qué es | BoxTwin tal como venía funcionando | BoxTwin con el modo académico E7 (PR #29, rama `cambios-academicos`) |
| Código | copia principal, `main` en `34e991d` | worktree fijo `~/Proyectos/TwinBoxing-academico`, **desacoplado en un commit congelado** de `cambios-academicos` (se anota en S3) |
| API | `127.0.0.1:8000` | `127.0.0.1:8001` |
| Afuera | Funnel público, `https://desarrollo-lucas.tail52f6aa.ts.net` (sin cambios) | solo la tailnet, `https://desarrollo-lucas.tail52f6aa.ts.net:8443`, **sin Funnel** |
| Servicios | `boxtwin-api`, `boxtwin-worker` | `boxtwin-academico-api`, `boxtwin-academico-worker` |
| Entorno | `/etc/boxtwin.env` | `/etc/boxtwin-academico.env` |
| Datos y base | `~/boxtwin/datos/` | `~/boxtwin-academico/datos/`, base nueva |
| Sesión del navegador | token en `localStorage` del origen `:443` | token en `localStorage` del origen `:8443` |
| Tipo de golpe | E4 (`geometrico`) | E7 (`e7`), con E4a en bolsa y sombra |

Archivos:

- `01-produccion.md`: qué quiere decir "sin ningún cambio" y cómo se comprueba.
- `02-academica.md`: la instancia B, con el entorno, las unidades y T1.
- `03-convivencia.md`: sesión del navegador, GPU, red, datos, sesiones de prueba y vuelta atrás.
- `04-pasos.md`: el registro de tareas, quién hace cada una, dependencias y evidencia.

## Quién hace qué

- **SDD (Claude Code, sin `sudo`):** las tareas `S`. Huella de A, T1 con su prueba, el worktree,
  el build de la web, los archivos de entorno y de las unidades dentro de `~/boxtwin-academico/`
  y `despliegue/academico/`, y todas las verificaciones.
- **Lucas (con `sudo`, o en Tailscale):** las tareas `L`. Instalar el entorno y las unidades en
  `/etc`, arrancar y parar B, publicar el 8443, crear la cuenta y elegir el material de la demo.

El SDD se detiene en cada `L` pendiente y sigue con lo que no dependa de ella.

## Criterios de aceptación (escritos antes de lanzar)

Se leen al cierre, en `04-pasos.md`, uno por uno:

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CA1 | A queda idéntica | La huella de `01-produccion.md` tomada en S1 es igual a la de S11, salvo la cantidad de sesiones de A, que puede crecer por el uso de A |
| CA2 | B es lo que dice ser | `127.0.0.1:8001/salud`: `clasificador` `e7`, `modelo_e7` `true`, `modelo_individual` `true`, `commit` el congelado en S3 |
| CA3 | B no es pública | `tailscale serve status`: el 8443 sin Funnel; el 443 con Funnel apuntando solo a `127.0.0.1:8000` |
| CA4 | B no escribe en A | La cantidad de carpetas en `~/boxtwin/datos/sesiones` no cambia por usar B. En la prueba S10, ninguna sesión de B aparece en la base de A |
| CA5 | E7 funciona de punta a punta en B | Sparring propio: `pose3d_hmr2.npz`, Fight-Card con `geometrico-e7` y `licencia`, y la leyenda en el pie del video y de la web. Bolsa propia: E4a sin `pose3d_hmr2.npz` |
| CA6 | Las sesiones del navegador no se pisan | En el mismo navegador, con sesión abierta en A y en B, cerrar o abrir sesión en una no cambia la otra |
| CA7 | La GPU alcanza | Pico de VRAM con A y B procesando a la vez ≤ 7,5 GB (`03-convivencia.md`). Si no, T2 antes de la primera demo |
