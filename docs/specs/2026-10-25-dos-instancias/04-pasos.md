# Pasos y registro

En orden. Cada paso tiene su verificación, y ninguno escribe en A. Los que piden `sudo` los
corre Lucas o se le piden.

| # | Paso | Verificación | Estado |
|---|---|---|---|
| 0 | Licencia de SMPLify pegada, o condición 3 enmendada para B (`00-README.md`) | texto en `docs/specs/2026-10-25-e7-modo-academico/02-registro.md` | hecha, 25-10 |
| 1 | Huella de A (`01-produccion.md`, "Cómo se comprueba"), guardada en `~/boxtwin-academico/huella-A-antes.txt` | el archivo existe y `/salud` da `34e991d` y `geometrico` | pendiente |
| 2 | T1: `_commit()` sigue el `gitdir:` de un worktree, con prueba, en `e7-modo-academico` | prueba nueva en verde; suite de la API en verde | pendiente |
| 3 | Worktree fijo `~/Proyectos/TwinBoxing-academico` en `e7-modo-academico`; anotar el hash | `git -C ~/Proyectos/TwinBoxing-academico rev-parse HEAD` | pendiente |
| 4 | Build de la web de B (`npm ci`, `build`, `presupuesto`) | `out/index.html` en el worktree de B; `out/` de A sin cambios (fecha) | pendiente |
| 5 | `~/boxtwin-academico/datos/` y `/etc/boxtwin-academico.env` (600), con secreto e invitación nuevos | `sudo -u lucasb test -r`; los valores no son los de A | pendiente |
| 6 | Unidades de B en `/etc/systemd/system/`, `daemon-reload`, **sin** `enable` | `systemctl is-enabled boxtwin-academico-api` da `disabled` | pendiente |
| 7 | Arrancar B: `systemctl start boxtwin-academico-api boxtwin-academico-worker` | `curl 127.0.0.1:8001/salud`: `e7`, `modelo_e7: true`, commit del paso 3 | pendiente |
| 8 | `tailscale serve --bg --https=8443 http://127.0.0.1:8001`, sin Funnel | `tailscale serve status`: 8443 sin Funnel y 443 igual que antes | pendiente |
| 9 | R1: pico de VRAM con A y B procesando a la vez | número anotado acá; si pasa 7,5 GB, T2 (`03-convivencia.md`) | pendiente |
| 10 | Prueba de B: una sesión de sparring y una de bolsa propias, de punta a punta | lo de "Cómo se comprueba" en `02-academica.md` | pendiente |
| 11 | Huella de A repetida | igual a la del paso 1 | pendiente |

Después de cada demo: paso 11 de nuevo y B apagada (`03-convivencia.md`, regla 2).

## Lo que este spec no hace

- No actualiza A ni le mergea nada, ni siquiera los PR #27 y #28.
- No mide C5 con E7: eso ya está en M6. La regla 1 de `03-convivencia.md` apaga B durante C5.
- No publica B fuera de la tailnet.
