"""
BoxTwin - Recorte de clips por evento.

POR QUE EXISTE
  Sirve para mirar el dataset con los ojos, que es como se detectan los errores que ninguna
  validacion encuentra: un golpe etiquetado hook que es un uppercut, una ventana que empieza
  tarde, un clip que no contiene ningun golpe. En la verificacion de BoxingVI ese fue el
  unico metodo que funciono.

  Se corta por NUMERO DE CUADRO y no por timestamp. Cortar por tiempo con un fps de
  30000/1001 desplaza los bordes, y sobre ventanas de veinte cuadros un desplazamiento de
  dos ya cambia lo que se ve. El filtro `select` de ffmpeg elige por indice de cuadro
  exacto, lo que obliga a reencodear: no hay corte frame-exacto sin reencodear, porque los
  cortes sin reencodear caen en keyframes.

  Consecuencia honesta: este export NO es reproducible byte a byte. La salida de libx264
  depende de su version y sus flags. La garantia de determinismo esta en el manifest y en
  los rangos de cuadros, no en los bytes del mp4.

  El clip marca al peleador anotado con su caja. Sin eso el clip es ambiguo y no se nota:
  en boxeo los golpes se solapan, asi que la ventana de un evento contiene seguido el golpe
  del otro peleador. Paso al revisar la primera muestra de Sparring.mp4: el clip de ev_0005,
  un straight-left de fighter_B, contiene tambien el right hook de fighter_A en los cuadros
  237-248, y leido sin saber a quien mirar la etiqueta parece equivocada. Un clip que se
  puede leer mal es un clip que va a ser leido mal, por el que lo anoto dos meses despues y
  por cualquiera que reciba el dataset.

QUE HACE
  Recorta un clip por evento en carpetas por clase, marcando al peleador anotado, y escribe
  un manifest.csv con todos los campos del evento y la ruta del clip.

USO
  export clips --classes 12   (label-space side por defecto)
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from boxtwin.core.export.base import (
    ExportContext,
    ExportResult,
    base_metadata,
    escribir_json,
    registrar,
)
from boxtwin.core.constants import ROLE_COLOR_A, ROLE_COLOR_B
from boxtwin.core.export.labels import LabelSpace, class_name
from boxtwin.core.export.windows import ventana_de_evento
from boxtwin.core.types import FighterId

__all__ = ["exportar", "CAMPOS"]

CAMPOS = [
    "clip", "clase", "event_id", "video", "fighter", "guard", "arm_role",
    "start_frame", "peak_frame", "end_frame", "n_frames",
    "side", "punch_type", "target", "completeness", "landed", "quality",
    "frames_sin_caja", "notes",
]


def _marca(cajas: list[tuple[float, float, float, float] | None], color: str) -> str:
    """
    Filtros drawbox que siguen al peleador anotado, uno por cuadro del clip.

    Se emite un drawbox por cuadro con `enable=eq(n,k)` en vez de una caja fija: el peleador
    se mueve, y una caja estatica en el promedio marca el lugar equivocado justo en el
    momento del golpe, que es cuando mas se desplaza. `n` cuenta desde 0 porque estos filtros
    van despues de select, que reinicia la numeracion del stream de salida.

    Los cuadros sin deteccion no dibujan nada. Eso es honesto: no se sabe donde esta.
    """
    partes = []
    for k, caja in enumerate(cajas):
        if caja is None:
            continue
        x1, y1, x2, y2 = (round(v) for v in caja)
        partes.append(
            f"drawbox=x={x1}:y={y1}:w={max(1, x2 - x1)}:h={max(1, y2 - y1)}"
            f":color={color}:t=3:enable='eq(n\\,{k})'"
        )
    return "".join("," + x for x in partes)


def _cortar(
    video: Path,
    destino: Path,
    desde: int,
    hasta: int,
    crf: int,
    preset: str,
    marca: str = "",
) -> str | None:
    """
    Corta [desde, hasta] por indice de cuadro. Devuelve el error si fallo.

    El seek de entrada se hace unos cuadros antes y el filtro select recorta exacto: sin el
    seek previo, ffmpeg decodifica desde el principio del archivo en cada clip y el corte de
    un video largo pasa de minutos a horas.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(video),
        "-vf", f"select='between(n\\,{desde}\\,{hasta})',setpts=N/FRAME_RATE/TB{marca}",
        "-fps_mode", "vfr",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an",
        str(destino),
    ]
    salida = subprocess.run(cmd, capture_output=True, text=True)
    if salida.returncode != 0:
        return salida.stderr.strip()[:300]
    return None


@registrar("clips")
def exportar(ctx: ExportContext) -> ExportResult:
    space = LabelSpace(ctx.opcion("label_space", LabelSpace.SIDE.value))
    classes = int(ctx.opcion("classes", 12))
    pad = int(ctx.opcion("pad", 0))
    crf = int(ctx.opcion("crf", 20))
    preset = ctx.opcion("preset", "veryfast")
    marcar = bool(ctx.opcion("mark_fighter", True))
    # Los mismos colores que el overlay de la aplicacion: quien anoto reconoce el rojo y el
    # azul sin tener que aprender una convencion nueva para revisar.
    color = {
        FighterId.A: "0x%02X%02X%02X" % ROLE_COLOR_A,
        FighterId.B: "0x%02X%02X%02X" % ROLE_COLOR_B,
    }

    base = ctx.video_path.stem
    raiz = ctx.out_dir / "clips"
    raiz.mkdir(parents=True, exist_ok=True)

    filas: list[dict[str, object]] = []
    avisos: list[str] = []
    fallidos = 0
    descartados = 0

    for ev in ctx.doc.events:
        clase = class_name(ev, space, classes)
        if clase is None:
            descartados += 1
            continue
        v = ventana_de_evento(ev, ctx.doc, pad=pad)
        carpeta = raiz / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        # El nombre lleva los cuadros: mirando el archivo se sabe de donde salio.
        destino = carpeta / f"{base}_{ev.id}_{v.start_frame}_{v.end_frame}.mp4"

        marca = ""
        sin_caja = 0
        if marcar:
            cajas = []
            for f in range(v.start_frame, v.end_frame + 1):
                pose = ctx.resolver.by_fighter(f)[ev.fighter]
                cajas.append(None if pose is None else tuple(float(x) for x in pose.bbox))
            sin_caja = sum(1 for c in cajas if c is None)
            marca = _marca(cajas, color[ev.fighter])

        error = _cortar(
            ctx.video_path, destino, v.start_frame, v.end_frame, crf, preset, marca
        )
        if error:
            fallidos += 1
            avisos.append(f"{ev.id}: {error}")
            continue

        filas.append(
            {
                "clip": str(destino.relative_to(ctx.out_dir)),
                "clase": clase,
                "event_id": ev.id,
                "video": ctx.video_path.name,
                "fighter": ev.fighter.value,
                "guard": ev.guard.value,
                "arm_role": ev.arm_role.value,
                "start_frame": v.start_frame,
                "peak_frame": "" if ev.peak_frame is None else ev.peak_frame,
                "end_frame": v.end_frame,
                "n_frames": v.n_frames,
                "side": ev.side.value,
                "punch_type": ev.punch_type.value,
                "target": ev.target.value,
                "completeness": ev.completeness.value,
                "landed": ev.landed.value,
                "quality": ev.quality.value,
                "frames_sin_caja": sin_caja,
                "notes": ev.notes,
            }
        )

    manifest = ctx.out_dir / f"{base}.manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)

    if descartados:
        avisos.append(f"{descartados} eventos quedaron fuera del espacio de {classes} clases")

    meta = base_metadata(ctx, "clips")
    meta["label_space"] = space.value
    meta["mark_fighter"] = marcar
    meta["counts"] = {"clips": len(filas), "fallidos": fallidos, "descartados": descartados}
    meta["determinism"] = (
        "El corte es frame-exacto pero reencodea, asi que los bytes del mp4 dependen de la "
        "version de libx264. El determinismo esta garantizado sobre el manifest y los "
        "rangos de cuadros, no sobre los archivos de video."
    )
    meta_path = ctx.out_dir / f"{base}.clips.meta.json"
    escribir_json(meta_path, meta)

    return ExportResult(
        formato="clips",
        archivos=[manifest, meta_path],
        resumen=meta["counts"],
        avisos=avisos,
    )
