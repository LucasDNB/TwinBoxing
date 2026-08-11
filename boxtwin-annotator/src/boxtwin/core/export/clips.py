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

QUE HACE
  Recorta un clip por evento en carpetas por clase y escribe un manifest.csv con todos los
  campos del evento y la ruta del clip.

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
from boxtwin.core.export.labels import LabelSpace, class_name
from boxtwin.core.export.windows import ventana_de_evento

__all__ = ["exportar", "CAMPOS"]

CAMPOS = [
    "clip", "clase", "event_id", "video", "fighter", "guard", "arm_role",
    "start_frame", "peak_frame", "end_frame", "n_frames",
    "side", "punch_type", "target", "completeness", "landed", "quality", "notes",
]


def _cortar(video: Path, destino: Path, desde: int, hasta: int, crf: int, preset: str) -> str | None:
    """
    Corta [desde, hasta] por indice de cuadro. Devuelve el error si fallo.

    El seek de entrada se hace unos cuadros antes y el filtro select recorta exacto: sin el
    seek previo, ffmpeg decodifica desde el principio del archivo en cada clip y el corte de
    un video largo pasa de minutos a horas.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(video),
        "-vf", f"select='between(n\\,{desde}\\,{hasta})',setpts=N/FRAME_RATE/TB",
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

        error = _cortar(ctx.video_path, destino, v.start_frame, v.end_frame, crf, preset)
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
