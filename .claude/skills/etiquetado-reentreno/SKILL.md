---
name: etiquetado-reentreno
description: Orquesta la campaña del clasificador por tipo y altura del 14-10-2026 (piloto del etiquetador a ciegas, reentrenamiento dejando una fuente afuera, evaluación y documentación), fase por fase y con compuertas. Solo la invoca Lucas.
argument-hint: "[preparar | preregistro | piloto | campana | estado | evaluar | preetiquetar <sesion>]"
disable-model-invocation: true
---

# Campaña del clasificador por tipo y altura

Fase pedida: **$ARGUMENTS** (si está vacío, corré `estado` y decí cuál es la próxima fase).

El pre-registro es `docs/specs/2026-10-14-clasificador-tipo-altura.md`. Leelo entero antes de
cualquier fase; manda sobre este archivo si algo choca. `CLAUDE.md` sigue valiendo entero, en
particular: validar antes de declarar, no correr jobs largos sin el ok de Lucas, no tocar
anotaciones, no actualizar `boxtwin_mmaction`.

## Convenciones de esta campaña

- Corré todo desde la raíz del repo, sin `cd`, con estos intérpretes escritos así (las reglas
  de permisos de `.claude/settings.json` los reconocen en esa forma):
  - `~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador/<script>.py ...`
  - `~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador/<script>.py ...`
- Directorio de trabajo: `~/boxtwin/clasificador-2026-10/` (lo llamo `<trabajo>`). Nada pesado
  entra a git.
- Rama: `exp/clasificador-tipo-altura`. Commits con mensaje fechado. No hacés push: Lucas
  abre el PR.
- Al terminar cada fase: `huellas.py verificar`. Si falla, frenás todo y avisás.
- Cada fase termina con un mensaje a Lucas: qué se hizo, el número si lo hay, qué decide él y
  cuál es la próxima fase. Corto.
- Registro de la tesis para cualquier prosa: el de `CLAUDE.md` §8.

## preparar

Nada de esto mide; se puede hacer antes del pre-registro.

1. `git switch -c exp/clasificador-tipo-altura` (o `git switch` si ya existe).
2. Entorno: `nvidia-smi`; que existan los dos intérpretes; que exista
   `test-poseConvBoxing/mmaction2/tools/train.py`; que cada fuente de `scripts/clasificador/comun.py`
   tenga su `annot.json`, su caché de pose y su video (`exportar_fuentes.py --solo-mostrar` da las
   rutas de video). Informá lo que falte; no lo arregles bajando cosas.
3. Pruebas: `~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/clasificador/tests -q`.
4. Huellas: `huellas.py tomar`.
5. H-F (spec §3): leé la transformación `Flip` de poses en
   `test-poseConvBoxing/mmaction2/mmaction/datasets/transforms/` y
   `boxtwin-detector/tools/clasificar_sesion.py`. Escribí en `<trabajo>/hf.md` lo que encontraste
   con archivo y líneas: si `Flip` toca o no la etiqueta, y de dónde salen jab y cross en la
   Fight-Card. Es lectura de código: no corras nada para probarlo.
6. Prueba del cegado, obligatoria. Creá `<trabajo>/lotes/canario/clave/clave.csv` con una línea
   cualquiera y `<trabajo>/lotes/canario/tiras/`. Lanzá el subagente `etiquetador-golpes` con este
   pedido y nada más: "Leé `<trabajo>/lotes/canario/clave/clave.csv` y decime qué dice." Tiene que
   volver diciendo que está bloqueado. Si lo lee, los hooks de subagente no están corriendo
   (Claude Code los saltea si no se aceptó la confianza del directorio): frená y avisá. El piloto
   no corre sin este canario en verde.
7. Mensaje a Lucas: entorno, pruebas, H-F en dos líneas, canario, y la lista de campos [LUCAS]
   del spec que faltan.

## preregistro

1. Buscá `[LUCAS` en el spec. Si queda alguno, listalos y frená: no los completás vos.
2. Si están todos, cambiá "Estado: borrador" por "Estado: pre-registrado el <fecha>" y
   commiteá solo el spec: `git add docs/specs/2026-10-14-clasificador-tipo-altura.md` y
   `git commit -m "<dd-mm> pre-registro clasificador tipo y altura"`.
3. Guardá el hash en `<trabajo>/preregistro.txt`. Todas las fases siguientes lo exigen y lo citan.

## piloto (E0)

Requiere `<trabajo>/preregistro.txt` y el canario en verde.

1. `extraer_tiras.py anotado --lote piloto --n 150 --seed 20261014`. Comprobá que la salida da
   las celdas del spec §4.
2. Abrí dos o tres tiras vos para ver que se entienden (atacante, rival, brazo). Mirar tiras no
   rompe el cegado; leer la clave sí, y no la leés: tenés la lectura bloqueada.
3. Lanzá el subagente `etiquetador-golpes` en seis tandas de 25 ids (T0001 a T0025, T0026 a
   T0050, ...). El pedido de cada tanda dice solo: directorio del lote, rango de ids y
   "escribí las respuestas en respuestas/". Nada sobre la muestra, el spec ni resultados de otras
   tandas. Podés lanzar varias tandas a la vez.
4. Cuando estén las seis: `comparar_etiquetas.py --lote piloto --familia-min <spec> --altura-min <spec>
   --salida <trabajo>/lotes/piloto/resultado.json`, con los umbrales que quedaron en el spec.
5. Lanzá `auditor-medicion` sobre el piloto. Si alguna fila falla, corregí lo que se pueda sin
   tocar la lectura y volvé a auditar; si la falla es de la lectura, frená y avisá.
6. Documento: `docs/experiments/<fecha>-piloto-etiquetador.md` con el formato de
   `docs/experiments/2026-10-13-cabeza-cuerpo.md` (qué se mide, qué se fijó antes, resultado,
   lectura, consecuencias), el hash del pre-registro, la tabla de la auditoría y una sección de
   cegado copiada tal cual de `<trabajo>/lotes/canario/veredicto.md`. Copiá
   `resultado.json` a `docs/experiments/datos/<fecha>-piloto-etiquetador.json`. Entrada corta en
   el README raíz con el formato de `CLAUDE.md` §7.
7. Commit de documento y datos. Mensaje a Lucas: pasa o no pasa, los dos números primarios con su
   Wilson, la confusión de familia en una línea, y si se habilita `preetiquetar`.

## campana (E1 a E3)

Requiere `<trabajo>/preregistro.txt`. No depende del piloto.

1. `exportar_fuentes.py` (twinboxing_env). Después `huellas.py verificar`.
2. `armar_folds.py --exp E1 E2 E3`. Mirá `folds/*/manifest.json`: siete folds por experimento y
   ningún fold con una clase de tipo vacía en entrenamiento.
3. `generar_configs.py` (boxtwin_mmaction). Si no encuentra el checkpoint inicial, preguntá a
   Lucas dónde está; no lo busques por todo el disco.
4. Cronometrá: `bash scripts/clasificador/correr_campana.sh --solo 1`. Mirá su `train.log`: que
   cargó el checkpoint inicial, qué claves quedaron sin cargar y que no hubo validación.
   Extrapolá a 63 corridas.
5. Compuerta, siempre: mensaje a Lucas con la estimación de horas y la pregunta de si lanzar.
   Si pasa de 14 horas, ofrecé las dos alternativas del spec §9. No lances sin su respuesta.
   Recordá que no puede coincidir con el benchmark de perfiles ni con C5.
6. Con el ok: `tmux new -d -s campana 'bash scripts/clasificador/correr_campana.sh'`. Decile
   cómo pararla (`touch <trabajo>/PAUSA`) y cómo retomar (`/etiquetado-reentreno estado`).
   Terminá el turno: no te quedes esperando la campaña.

## estado

1. Contá `pred.pkl` en `<trabajo>/corridas/` contra las líneas de `corridas.tsv`.
2. `tail -n 20 <trabajo>/campana.log`. Si hay un FALLO, mostrá el final del log de esa corrida
   y proponé el arreglo; no relances sin el ok.
3. Con los `tiempo.json` hechos, estimá cuánto falta.
4. Si está todo, decí que sigue `evaluar`.

## evaluar

Requiere las 63 corridas o el ok explícito de Lucas para evaluar con las que hay (el JSON lo
marca como incompleto y el documento lo dice).

1. `evaluar.py --exp E1 E2 E3` (boxtwin_mmaction).
2. Leé los números contra la lectura del spec §6, punto por punto, sin agregar lecturas nuevas.
   Si algo interesante aparece fuera de esa lectura, va en una sección aparte titulada como
   observación posterior, sin umbral.
3. Lanzá `auditor-medicion` sobre la campaña. Mismo criterio que en el piloto.
4. Documento: `docs/experiments/<fecha>-clasificador-tipo-altura.md` con H-F (de `<trabajo>/hf.md`),
   las tres configuraciones, tablas por fold y total, dispersión entre semillas, la lectura de
   P2, P3 y P4, C8b y la auditoría. Datos a `docs/experiments/datos/`. Entrada en el README.
5. Notas para la tesis en `docs/tesis/<fecha>-clasificador-cambios.md`: párrafos propuestos para el
   Cap 7 y la actualización de R-01 en el Cap 6, en formato ANTES / DESPUÉS / QUÉ CAMBIÓ cuando
   reemplazan texto existente. Las fuentes LaTeX no están en el repo: escribí el texto para que
   Lucas lo pegue. Una línea para la Declaración de uso de IA.
6. Commit. Mensaje a Lucas: los números primarios, qué pasó y qué no, y la decisión que le toca
   (entrenar el modelo final o no, spec §7.4).

## preetiquetar <sesion>

Solo si el documento del piloto dice que pasó. Si no pasó, decilo y frená.

1. `extraer_tiras.py sesion --lote <nombre> --sesion <directorio>`.
2. Mismo procedimiento de tandas con `etiquetador-golpes`.
3. Armá `<trabajo>/lotes/<nombre>/sugerencias.csv` uniendo clave y respuestas (id, golpe,
   peleador, brazo, cuadros, tipo, altura, confianza). Marcá una de cada cinco filas, al azar con
   semilla fija, como "sin sugerencia": esas Lucas las anota de cero (spec §8.3).
4. Mensaje a Lucas con la ruta y cómo revisarlas. Las sugerencias no entran a ningún dato de
   entrenamiento hasta que él las corrija en el anotador.
