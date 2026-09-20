"""
BoxTwin - Layout de un proyecto de anotacion.

POR QUE EXISTE
  Las rutas las necesitan la GUI, el preproceso, el empaquetado y los exports. Si viven en
  el modulo de la interfaz, el pipeline de export termina importando la GUI para saber donde
  esta un archivo, y esa dependencia rompe la regla de que todo el export pueda correr en el
  entorno de entrenamiento, sin Qt ni pantalla.

  Por el mismo motivo vive aca `cargar_o_crear`. El documento de anotacion lo creaba la
  GUI al abrir un proyecto, asi que un video recien preprocesado no tenia archivo hasta que
  alguien lo abria a mano, y los comandos pensados para correr sin intervencion humana
  -la asignacion automatica de identidad, sobre todo- se negaban a arrancar sobre
  exactamente el caso que justifica que existan.

QUE HACE
  Resuelve las rutas del proyecto a partir de la ruta del video, y carga o crea su documento.

USO
  from boxtwin.core.project import project_paths, cargar_o_crear
  paths = project_paths(Path("proyecto/videos/spar.mp4"))
  doc, migrados = cargar_o_crear(paths, cache.meta)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ProjectPaths", "project_paths", "cargar_o_crear"]


@dataclass(frozen=True)
class ProjectPaths:
    project: Path
    video: Path
    npz: Path
    meta: Path
    proxy: Path
    annot: Path
    config: Path
    exports: Path


def project_paths(video: Path) -> ProjectPaths:
    """
    Deduce el proyecto del video.

    Si el video esta en <proyecto>/videos/, el proyecto es el padre; si no, el directorio
    del video. Es la convencion del layout y evita tener que pasar dos rutas a cada comando.

    NO se resuelven los symlinks. Se usa abspath, que normaliza los ".." de forma lexica y
    deja el enlace intacto. La diferencia no es cosmetica: con resolve(), un proyecto de
    prueba armado con el video enlazado al original hace que todo apunte al proyecto
    original, y las escrituras caen sobre la anotacion de verdad. Paso dos veces el
    19-08-2026 armando pruebas aisladas; la segunda pisó un reanno.json de 35 intentos.
    """
    video = Path(os.path.abspath(video))
    project = video.parent.parent if video.parent.name == "videos" else video.parent
    base = video.stem
    return ProjectPaths(
        project=project,
        video=video,
        npz=project / "cache" / f"{base}.pose.npz",
        meta=project / "cache" / f"{base}.meta.json",
        proxy=project / "cache" / f"{base}.proxy.mp4",
        annot=project / "annotations" / f"{base}.annot.json",
        config=project / "config.yaml",
        exports=project / "exports",
    )


def cargar_o_crear(paths: "ProjectPaths", meta: dict) -> tuple:
    """
    El documento del proyecto: el que hay, o uno vacio y valido si todavia no existe.

    Todo lo que necesita sale del meta del preproceso y del cache, asi que crear no inventa
    nada: copia lo que ya se midio del video. Devuelve (documento, lista de migraciones).
    """
    from datetime import datetime

    from boxtwin.core.annotations import load as load_doc
    from boxtwin.core.annotations import save as save_doc
    from boxtwin.core.schema import PoseRef, VideoInfo, new_document
    from boxtwin.core.types import FpsSource, Guard, KeypointFormat
    from boxtwin.core.video import sha256_file
    from boxtwin.version import __version__

    if paths.annot.is_file():
        return load_doc(paths.annot)

    v = meta["video"]
    doc = new_document(
        app_version=__version__,
        now=datetime.now().astimezone(),
        video=VideoInfo(
            path=str(paths.video),
            sha256=v["sha256"],
            size_bytes=int(v["size_bytes"]),
            mtime=datetime.fromisoformat(v["mtime"]),
            fps=float(v["fps"]),
            fps_declared=v.get("fps_declared"),
            fps_source=FpsSource(v["fps_source"]),
            width=int(v["width"]),
            height=int(v["height"]),
            total_frames=int(v["total_frames"]),
            total_frames_declared=v.get("total_frames_declared"),
            duration_s=float(v["duration_s"]),
            codec=str(v["codec"]),
        ),
        pose=PoseRef(
            npz_path=str(paths.npz),
            meta_path=str(paths.meta),
            meta_sha256=sha256_file(paths.meta) if paths.meta.is_file() else "0" * 64,
            keypoint_format=KeypointFormat(meta.get("keypoint_format", "coco17")),
            keypoint_sources={"0-16": "yolov8l-pose"},
        ),
        # Punto de partida, no un dato: se elige ortodoxa porque es lo mas frecuente, no
        # porque se sepa. La real la fija el anotador desde el panel de identidad.
        guard_a=Guard.ORTHODOX,
        guard_b=Guard.ORTHODOX,
    )
    save_doc(doc, paths.annot)
    return doc, []
