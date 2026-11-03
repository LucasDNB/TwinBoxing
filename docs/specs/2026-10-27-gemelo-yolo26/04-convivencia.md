# Convivencia de A, B y C

## Qué no comparten

| | A | B | C | Por qué |
|---|---|---|---|---|
| Base y cola | `~/boxtwin/datos/boxtwin.db` | `~/boxtwin-academico/datos/boxtwin.db` | `~/boxtwin-gemelo/datos/boxtwin.db` | La cola es una tabla de la base: ningún worker toma trabajos de otro |
| Sesiones y videos | `~/boxtwin/datos/sesiones` | `~/boxtwin-academico/datos/sesiones` | `~/boxtwin-gemelo/datos/sesiones` | C no ve sesiones de usuarios de A |
| Cuentas | las de A | las de B | de prueba, sembradas en su base | Una sesión de un origen no vale en otro |
| Puerto, acceso | 8000, Funnel | 8001, tailnet | 8002, solo `127.0.0.1` | C no se publica |
| Código, web | copia principal | worktree de B | worktree de C y su `out/` | Cambiar C no cambia A ni B |
| Modelos | `~/boxtwin/modelos`, copia principal | los de A | `~/boxtwin-gemelo/modelos`, más los guantes de A en solo lectura | Los de C salen de E8 |
| Trabajo de E8 | — | — | `~/boxtwin-gemelo/raiz-y26`, `~/boxtwin-gemelo/e8` | Las anotaciones y los cachés de A solo se leen |

Comparten `twinboxing_env` (C solo lo lee, por `PYTHONPATH`), `guantes-v2.pt` (solo lectura) y la
GPU.

## Datos en C

C arranca vacía. Entra solo material propio: videos de Lucas o de quienes dieron consentimiento
para mostrarlos. No se copian sesiones ni la base de A. Para comparar una misma sesión en A y en C
se sube el mismo video propio a las dos.

En la fase 1, los videos de las siete fuentes se leen desde su lugar por enlace simbólico. Nada de
`raiz-y26`, `e8` ni `modelos` entra a git. No entra nada de BoxingWeb ni de SMPL.

## GPU

Una sola 2080 de 8 GB para tres instancias. Reglas, de la más fuerte a la más débil:

1. **Con sesiones de prueba o medición de C5 en curso, C está apagada y la fase 1 no corre.** C5
   mide el reloj de A.
2. **B y C nunca encendidas a la vez.** Cada una con A está acotada (CA7 de B, CG8 de C); las
   tres juntas no se miden y no hace falta.
3. **Los trabajos largos de la fase 1 corren de noche, con OK de Lucas (L1), y bajo el candado
   de `e8.sh` (T8).** Antes de cada etapa, el candado espera mientras la base de A tenga trabajos
   `en_cola` o `tomado` (consulta de solo lectura, como la huella) o `nvidia-smi
   --query-compute-apps=pid --format=csv,noheader` muestre otro proceso. Cada espera queda en
   `~/boxtwin-gemelo/e8/candado.log` (CG8). Una etapa empezada no se interrumpe: si A recibe un
   trabajo en el medio, ese trabajo de A va más lento, y queda en el log para poder decirlo.
4. El servicio C se enciende para usarlo y se apaga después. Sus unidades no arrancan con la
   máquina.

CG8 se mide en S23 con C procesando sola: pico de VRAM con `nvidia-smi
--query-gpu=memory.used --format=csv -lms 500`, contra el mismo umbral elegido de 7,5 GB de B
(7786 MiB utilizables menos unos 300 de margen; `dos-instancias`, decisión 3).

## Si algo sale mal

- **Apagar C:** `sudo systemctl stop boxtwin-gemelo-worker boxtwin-gemelo-api`. A y B no se
  enteran.
- **Cortar la fase 1:** `Ctrl-C` o `kill` del proceso de `e8.sh`. Todo es reanudable y nada de lo
  que escribe está en A.
- **Sacar C del todo:** lo anterior, más borrar las dos unidades y `/etc/boxtwin-gemelo.env`,
  `sudo systemctl daemon-reload`, `git -C ~/Proyectos/TwinBoxing worktree remove
  ~/Proyectos/TwinBoxing-gemelo` y, si se quiere, `~/boxtwin-gemelo/`.
- **Si A o B se ven afectadas:** apagar C primero y comparar con las huellas de S1. Ningún paso de
  este spec escribe en A ni en B, así que una diferencia no viene de C.
