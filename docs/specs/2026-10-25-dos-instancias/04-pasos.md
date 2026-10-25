# Registro de tareas (SDD)

Estado: `hecha` (con evidencia), `en curso`, `pendiente`, `bloqueada` (qué la bloquea), `no
realizada` (con el motivo). La evidencia es el comando que la verifica o el archivo que produjo.
Ninguna tarea escribe en A. `S` = la corre el SDD sin `sudo`; `L` = la hace Lucas.

| Id | Quién | Tarea | Depende de | Verificación | Estado | Evidencia |
|---|---|---|---|---|---|---|
| L0 | Lucas | Licencia de SMPLify pegada | — | texto en `docs/specs/2026-10-25-e7-modo-academico/02-registro.md` | hecha | 25-10 |
| S1 | SDD | Huella de A (`01-produccion.md`, "Cómo se comprueba") en `~/boxtwin-academico/huella-A-antes.txt` | — | el archivo existe; `/salud` da `34e991d` y `geometrico` | hecha | 25-10, `despliegue/academico/huella-A.sh`: `/salud` `34e991d` y `geometrico`; 443 con Funnel a `127.0.0.1:8000`; 12 sesiones de A |
| S2 | SDD | T1 (`02-academica.md`): `_commit()` resuelve un worktree. Primero las pruebas, después el código; commit en `cambios-academicos` | — | las cinco pruebas de T1 en verde; la suite de la API en verde | hecha | 25-10, `tests/test_commit_worktree.py` 5 en rojo y después en verde; API 243 passed; en un worktree real da su commit |
| S3 | SDD | Worktree fijo y desacoplado: `git worktree add --detach ~/Proyectos/TwinBoxing-academico <punta de cambios-academicos después de S2>`; anotar el hash | S2 | `git -C ~/Proyectos/TwinBoxing-academico rev-parse HEAD` igual al anotado | hecha | 25-10, congelado en `efcec23f20c0b5f820deffc937e13e65e00dc9f5`; HEAD de A sigue en `34e991d` |
| S4 | SDD | Build de la web de B en su worktree (`npm ci`, `npm run build`, `npm run presupuesto`) | S3 | `out/index.html` en el worktree de B; la fecha de `out/index.html` de A sin cambios | hecha | 25-10, `out/index.html` de B 16:45; presupuesto ok en las 11 rutas; leyenda académica en el bundle; A sigue en `1792122654`. Logs en `~/boxtwin-academico/build-web-B*.log` (decisión 4) |
| S5 | SDD | Archivos de B: `~/boxtwin-academico/datos/`; `~/boxtwin-academico/boxtwin-academico.env` (600), armado desde `despliegue/academico/boxtwin-academico.env.ejemplo` con secreto e invitación nuevos; las dos unidades versionadas en `despliegue/academico/systemd/` | S3 | `stat -c %a` da `600`; el secreto y la invitación no son los de A (se comparan sin mostrarlos); las unidades difieren de las de A solo en nombre, directorio, entorno, puerto y `Restart` | hecha | 25-10, `600`; secreto e invitación distintos de A (comparados sin mostrarlos); el entorno difiere de A solo en datos, base, web, clasificador, secreto e invitación, más `PYTHONPATH` y las dos variables de E7; `diff` de unidades: solo nombre, descripción, directorio, entorno, puerto y `Restart`; `Config` de B carga `e7` con su código (decisión 5) |
| S6 | SDD | Chequeo previo al arranque | S4, S5 | `ss -ltn` sin nada en `:8001`; `tailscale serve status` sin `:8443`; `curl 127.0.0.1:8000/salud` igual que en S1 | hecha | 25-10, las tres; la huella completa (`~/boxtwin-academico/huella-A-S6.txt`) es igual a la de S1 |
| L1 | Lucas | `sudo install -m 600 -o lucasb -g lucasb ~/boxtwin-academico/boxtwin-academico.env /etc/boxtwin-academico.env`; `sudo cp ~/boxtwin-academico/systemd/*.service /etc/systemd/system/` (copia idéntica a las versionadas); `sudo systemctl daemon-reload`; **sin** `enable` | S6 | `systemctl is-enabled boxtwin-academico-api` da `disabled` | hecha | 25-10, las dos unidades `disabled`; `/etc/boxtwin-academico.env` 600 lucasb, igual al armado en S5 |
| L2 | Lucas | Arrancar B: `sudo systemctl start boxtwin-academico-api boxtwin-academico-worker` | L1 | — | hecha | 25-10 16:48, las dos `active` |
| S7 | SDD | CA2 | L2 | `curl 127.0.0.1:8001/salud` | hecha | 25-10, `clasificador` `e7`, `modelo_e7` `true`, `modelo_individual` `true`, `commit` `efcec23` (el de S3); la web responde 200; la base nueva quedó en `~/boxtwin-academico/datos/`; la huella de A sigue igual a S1 |
| L3 | Lucas | `tailscale serve --bg --https=8443 http://127.0.0.1:8001`, sin Funnel | S7 | — | pendiente | |
| S8 | SDD | CA3 | L3 | `tailscale serve status` | pendiente | |
| L4 | Lucas | Material de la demo: elegir los videos propios (al menos un sparring con los dos peleadores y una bolsa), crear la cuenta en B con la invitación nueva y subirlos desde `https://…:8443` | S8 | la lista de videos anotada acá, con quién aparece y su consentimiento | pendiente | |
| S9 | SDD | CA5 y CA4 sobre lo que subió Lucas | L4 | lo de CA5 en las carpetas de sesión de B; conteo de sesiones de A igual que en S1 | pendiente | |
| L5 | Lucas | CA6: en el mismo navegador, sesión abierta en A y en B; cerrar sesión en B y recargar A | L4 | A sigue con su sesión; B pide ingresar | pendiente | |
| S10 | SDD | CA7 (R1): A y B procesando a la vez, pico de VRAM con `nvidia-smi --query-gpu=memory.used --format=csv -lms 500` | L4, y que Lucas habilite usar A para la prueba con material propio | número anotado acá; si pasa 7,5 GB, T2 queda pendiente y B no se usa en una demo con A procesando | pendiente | |
| S11 | SDD | CA1: la huella de A de nuevo, y lectura de CA1 a CA7 | S9, L5, S10 | igual a S1 | pendiente | |

Después de cada demo: S11 de nuevo y B apagada (`03-convivencia.md`, regla 2), con
`sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api` (Lucas).

## Decisiones

Una línea por decisión tomada al implementar o al correr, con el motivo, antes de correr la parte
afectada.

1. 25-10: B corre un commit congelado (worktree desacoplado), no la rama: un push a
   `cambios-academicos` durante la demo no la cambia. Moverla es repetir S3, S4, L2 y S7.
2. 25-10: la sesión del navegador no choca entre A y B. La web guarda el token en `localStorage`
   (`lib/sesion-usuario.ts`) y lo manda como `Authorization: Bearer`; el video usa un ticket en
   la URL; la API no pone cookies. `localStorage` es por origen y el puerto es parte del origen.
   La única cookie es la del menú lateral (`components/animate-ui/.../sidebar.tsx`): se comparte
   entre los dos puertos y solo guarda si el menú está abierto. CA6 lo verifica en el navegador.
3. 25-10: el umbral de 7,5 GB de CA7 es elegido, no medido. La 2080 informa 7786 MiB utilizables
   (`torch.cuda.get_device_properties`, E7-1) y se dejan unos 300 MiB de margen para la
   fragmentación del asignador. Si la medición de S10 queda cerca del umbral, se informa igual.

4. 25-10: el primer `npm run build` de B generó las 11 páginas y salió con error: Next.js corta
   una vez para recomendar una actualización de seguridad y pide repetir el comando. Se repitió y
   pasó. No se actualizó Next: B lleva el mismo código congelado y A tampoco se toca. Recomendación
   de seguridad pendiente, para después de la presentación y con PR propio.
5. 25-10: las unidades conservan el `[Install]` de las de A (no se habilitan igual, L1 no hace
   `enable`), y se copian también a `~/boxtwin-academico/systemd/` porque el worktree de B está
   congelado antes de que existieran. `boxtwin-academico.env` se lee como lo lee systemd (la línea
   entera es el valor): no se carga con `source` de bash, porque `BOXTWIN_CMD_HMR2` lleva un espacio.

## Preguntas abiertas

(vacío)

## Lo que este spec no hace

- No actualiza A ni le mergea nada, ni siquiera el PR #29.
- No mide C5 con E7: eso ya está en M6. La regla 1 de `03-convivencia.md` apaga B durante C5.
- No publica B fuera de la tailnet.
