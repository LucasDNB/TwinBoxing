# Convivencia de A y B

## Qué no comparten

| | A | B | Por qué |
|---|---|---|---|
| Base y cola | `~/boxtwin/datos/boxtwin.db` | `~/boxtwin-academico/datos/boxtwin.db` | La cola de trabajos es una tabla de la base: con bases separadas, ningún worker toma trabajos del otro |
| Sesiones y videos | `~/boxtwin/datos/sesiones` | `~/boxtwin-academico/datos/sesiones` | B no ve ni toca sesiones de usuarios de A |
| Cuentas | las de A | nuevas, otra invitación y otro secreto | Una sesión iniciada en A no vale en B |
| Puerto, URL | 8000, 443 con Funnel | 8001, 8443 solo en la tailnet | B no es pública |
| Código, web | copia principal | worktree fijo y su `out/` | Mergear o cambiar B no cambia A |

**Sesión del navegador:** no se comparte, aunque A y B usen el mismo nombre de host. La web
guarda el token en `localStorage` y lo manda como `Authorization: Bearer`; el video va con un
ticket en la URL; la API no pone cookies. `localStorage` es por origen y el puerto es parte del
origen, así que `:443` y `:8443` tienen cada una su sesión. La única cookie es la del menú
lateral, que solo guarda si el menú está abierto y se comparte entre los dos puertos sin efecto
sobre datos ni sesiones (decisión 2 de `04-pasos.md`; CA6 lo verifica).

**Qué comparten:** `twinboxing_env` (B solo lo lee, por `PYTHONPATH`), los archivos de modelos
(solo lectura) y la GPU.

## Datos en B

B arranca vacía. Solo entra material propio: videos de Lucas o de quienes hayan dado su
consentimiento para la demo (enmienda, condición 2). No se copian sesiones ni la base de A: son
de usuarios de A y su consentimiento no incluye un modelo con licencia de investigación. Para
la presentación se carga lo que se vaya a mostrar.

## GPU (la única cosa que A puede notar)

Una sola 2080 de 8 GB para los dos workers. Lo medido: HMR2 2,8 GB pico con lote 8 (E7-1). Lo
que no está medido: el pico del worker de A en `procesar` (YOLOv8l-pose, BoT-SORT y guantes) con
otro proceso al lado. Reglas, de la más fuerte a la más débil:

1. **Con sesiones de prueba o medición de C5 en curso, B está apagada**
   (`systemctl stop boxtwin-academico-worker boxtwin-academico-api`). C5 mide el reloj hasta la
   Fight-Card de A, y un segundo worker en la GPU lo ensuciaría.
2. B se enciende para la demo y para prepararla, y se apaga después. Sus unidades no arrancan
   con la máquina.
3. Antes de la primera demo, la medición R1 (S10 de `04-pasos.md`, CA7): las dos instancias
   procesando a la vez, pico de VRAM con `nvidia-smi --query-gpu=memory.used --format=csv -lms 500`.
   El umbral de 7,5 GB es elegido, no medido: 7786 MiB utilizables menos unos 300 MiB de margen
   (decisión 3 de `04-pasos.md`). Si lo pasa, la regla 2 pasa a ser "B procesa solo con la cola
   de A vacía", y eso es un cambio de código en B con su prueba (T2), no una costumbre.

Con E7, la etapa `clasificar` de B tarda de 1,1x a 1,4x la duración del video (M6). Mientras
corre, A sigue andando pero puede ir más lenta.

## Si algo sale mal

- **Apagar B:** `sudo systemctl stop boxtwin-academico-worker boxtwin-academico-api` y
  `tailscale serve --https=8443 off`. A no se entera.
- **Sacar B del todo:** lo anterior, más borrar las dos unidades y `/etc/boxtwin-academico.env`,
  `sudo systemctl daemon-reload`, `git worktree remove ~/Proyectos/TwinBoxing-academico` y, si se
  quiere, `~/boxtwin-academico/`.
- **Si A se ve afectada:** apagar B primero y comparar con la huella de `01-produccion.md`. Ningún
  paso de este spec escribe en A, así que una diferencia ahí no viene de B.
