# Identidad estable: registro de tareas y pasos en la estación

| Id | Estado | Evidencia |
|---|---|---|
| IE1 `revision.py` | hecha (2-11) | `boxtwin-annotator/tests/test_revision_identidad.py`, 14 pruebas |
| IE2 `completar` | hecha (2-11) | `boxtwin-annotator/tests/test_mvp_identidad_estable.py`, 5 pruebas de punta a punta (corte parcial a mano, corrección de un asignado, conflicto y estricta, sin banderas igual que antes, corte de E11); `test_mvp_tramos.py` actualizado al contrato nuevo |
| IE3 API | hecha (2-11) | `boxtwin-api/tests/test_identidad_estable.py`, 7 pruebas; `test_web_entrenador_b3.py` actualizado (las sesiones viejas se leen igual) |
| IE4 web | hecha (2-11) | `tests/integracion/tramos.test.tsx`, 4 pruebas nuevas; 261 en verde, lint y tipos limpios. El build no se pudo probar en la sesión de Claude (Google Fonts bloqueado por la red); hay que construir en la estación |
| IE5 script | hecha (2-11) | `scripts/identidad/tests`, 8 pruebas |
| IE6 medición | hecha (3-11): **no estable**, se elige `base`, banderas apagadas | `docs/experiments/2026-11-03-identidad-estable.md`; JSON en `docs/experiments/datos/2026-11-03-identidad-estable.json`. Auditoría: `base` y `corte` iguales a E11 S10 v2 en todos los decimales. Ninguna condición pasa E1 (1,54%; 03-sparring 12,8%) ni E2 (mejor 0,984) |
| IE7 apariencia | hecha (3-11): **estable** en desarrollo con el simulado v2; se elige `base`, `BOXTWIN_IDENTIDAD_APARIENCIA=0`. El 3-11, después de leerla, Lucas lo enciende por defecto (no lo pide la regla) | pre-registro `02-ie7-apariencia.md` (`92e0105`); código `ff49041`; `docs/experiments/2026-11-03-ie7-apariencia.md`. Auditoría contra IE6 bien. Silenciosos 0,052%, precisión después de revisar 0,9993 en las dos condiciones |

Suites al 2-11 en la sesión de Claude (Python 3.13, torch CPU): anotador 759 pasan y 1 falla
de entorno (`test_render`, sin códec avc1 en ese opencv); API 346 pasan; `scripts/mejoras` 63.

## IE6: correr la medición (sin GPU, segundos)

En la estación, con la rama mergeada o en un worktree, sobre `twinboxing_env`:

```bash
cd ~/Proyectos/TwinBoxing          # o el worktree con la rama
~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/identidad/tests -q
~/miniforge3/envs/twinboxing_env/bin/python scripts/identidad/medir_identidad_estable.py \
    --raiz ~/Proyectos/TwinBoxing --evidencia-dir ~/boxtwin/mediciones/c1 \
    --out ~/boxtwin/mediciones/identidad-estable/desarrollo.json
```

Imprime, por fuente y condición, errores silenciosos, tamaño de la cola, decisiones y la
precisión después de revisar; al final, la condición elegida por la regla de §6 y las banderas
que le corresponden. Antes de leerlo, la auditoría mínima: la condición `base` tiene que dar la
misma precisión y cobertura por cuadro que `sin_corte` de E11 S10 v2 (0,850 y 0,788), y `corte`
las de `con_corte` (0,923 y 0,872). Si no, algo cambió en el camino y no se lee nada.

## Encender en producción (después de IE6 y después del 17-11)

A está congelada desde el 3-11 y corre la rama `produccion`. Esto entra con un PR contra
`produccion`, con la base respaldada (`despliegue/README.md`), y las banderas que diga IE6:

```bash
# /etc/boxtwin.env
BOXTWIN_CORTE_MIXTOS=1            # si IE6 elige "corte" o "corte+estricta"
BOXTWIN_IDENTIDAD_ESTRICTA=1      # solo si IE6 elige "corte+estricta"
sudo systemctl restart boxtwin-api
```

La cola de revisión nueva (asignados sospechosos) no tiene bandera: entra con el código. Las
sesiones ya procesadas siguen mostrando su lista vieja hasta que se reprocesen o se les aplique
una asignación (que vuelve a correr `completar` y arma la lista nueva).
