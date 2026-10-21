# Próxima sesión · desde el 20-10

Estado al cierre del 20-10 y qué sigue, en orden. Las sesiones de prueba (C4, C5, C6) son la
semana del 27-10 y la presentación el 17-11.

## Lo que se cerró el 20-10

| Criterio | Resultado | Documento |
|---|---|---|
| C1, identidad con siembra | 0,810 contra 0,95: no pasa. Plan B (corrección por track en la web) | `docs/experiments/2026-10-20-c1-identidad-con-siembra.md` |
| C3, guardia | 0,580 y 0,360 contra 0,80: no pasa. F5 sale del MVP | `docs/experiments/2026-10-20-c3-guardia.md` |
| C5, tiempo | Pre-registrado; ensayo 2,741x; arreglo de lectura de video en `analizar` (guantes 4-5x más rápido, evidencia idéntica). Se mide en las sesiones de prueba | `docs/experiments/2026-10-20-c5-ensayo-tiempo.md` |
| Imagen del worker | Construida por primera vez (`boxtwin-worker:cc78cc9`, 15,2 GB), sin mmaction, versiones de la estación. Sin probar con GPU | `despliegue/README.md`, sección de la imagen |

La copia principal (`~/Proyectos/TwinBoxing`) quedó en `main` con el arreglo de C5 (`34e991d`).

## 1. Pendientes rápidos de Lucas (unos 10 minutos)

- [x] `BOXTWIN_C3=rechazado` y `BOXTWIN_C3_FECHA=2026-10-20` en `/etc/boxtwin.env`;
      `sudo systemctl restart boxtwin-api`; mirar `/producto`. Hecho el 21-10: `/producto`
      da C3 `rechazado` (2026-10-20) y la bandera `guardia` en `false`.
- [x] Borrar del remoto las ramas mergeadas: `c1-medicion-resultado-20-10`,
      `c3-medicion-20-10`, `c5-medicion-20-10`. Hecho el 20-10 junto con el resto de las
      ramas contenidas en `main` (PR de CLAUDE.md, rescate de `d843f26`).
- [ ] Opcional: `docker builder prune` libera unos 15 GB del primer build.

## 2. Preparar las sesiones de prueba del 27-10 (lo primero)

Es lo único con fecha fija, y de ahí salen C4, C5 y C6, que no se repiten.

- [ ] **Guion de C6**, corto y escrito antes de la sesión (spec §7). Lo redacta Claude, lo
      ajusta Lucas con criterio de entrenador. Tiene que reflejar que F5 (guardia) salió.
- [ ] **Subida real por el túnel desde un celular**, cronometrada. Si un round tarda diez
      minutos, cambia el guion (`despliegue/README.md`, "El problema real no es el túnel").
- [ ] **Confirmar el arreglo de C5 con una sesión nueva**: subir un video, sembrar y correr
      `boxtwin-api/scripts/medir_c5.py` sobre esa sesión. No cuenta para C5. La proyección es
      1,74x a 1,86x: si da cerca de 2x, conviene saberlo antes del 27-10. La sesión
      `464e9b87` (en espera de siembra) no sirve: su evidencia se calculó con el código viejo.
- [ ] Checklist "Antes de abrir el túnel" de `despliegue/README.md`.

Para C5 el 27-10: `medir_c5.py --db sqlite:////home/lucasb/boxtwin/datos/boxtwin.db
--datos ~/boxtwin/datos --sesiones <ids de las sesiones de prueba>`. No usar la GPU por fuera
del worker durante esas sesiones: el script solo detecta solapamiento entre trabajos del
worker.

## 3. Benchmark de perfiles de despliegue (§6.1 de CLAUDE.md)

Bloquea el Capítulo 4. También informa el plan B de C5 (YOLOv8m). Horas de GPU: de noche y
nunca durante las sesiones de prueba. Pedir el ok antes de lanzarlo.

## 4. Imagen del worker y PoC de nube (§6.3)

- [ ] Lucas: instalar el NVIDIA Container Toolkit (requiere sudo; comandos en
      `despliegue/README.md`). Hoy `docker info` solo tiene `runc`.
- [ ] Lucas: los tres enlaces duros en `~/boxtwin/modelos` (mismo README).
- [ ] Prueba de humo: `torch.cuda.is_available()` en el contenedor y `/modelos` visible.
- [ ] Prueba de equivalencia: procesar una sesión corta adentro del contenedor (por ejemplo
      `fb031164`, 24 s) y comparar con el resultado de la estación.
- [ ] Después de las sesiones de prueba: el PoC en una instancia alquilada (< 20 USD).

## Cosas que conviene recordar

- Esta sesión no tiene permiso de `git push`: Lucas pushea y Claude abre el PR.
- Los tests del anotador y de la API importan la copia principal (instalación editable). Para
  probar cambios de un worktree: `PYTHONPATH=src python -m pytest`.
- La estación corre por systemd desde la copia principal. Después de un `git pull`, los
  subprocesos del worker toman el código nuevo en el próximo trabajo; la API y el proceso
  del worker necesitan restart para el suyo.
- La frase "la diferencia entre 82,4% y 99,1% es la siembra" está corregida (20-10): el 99,1%
  era color con perfiles de oráculo. No volver a citarla así.
