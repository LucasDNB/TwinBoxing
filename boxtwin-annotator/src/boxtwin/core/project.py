"""
BoxTwin - Layout de un proyecto de anotacion.

POR QUE EXISTE
  Las rutas las necesitan la GUI, el preproceso, el empaquetado y los exports. Si viven en
  el modulo de la interfaz, el pipeline de export termina importando la GUI para saber donde
  esta un archivo, y esa dependencia rompe la regla de que todo el export pueda correr en el
  entorno de entrenamiento, sin Qt ni pantalla.

QUE HACE
  Resuelve las rutas del proyecto a partir de la ruta del video.

USO
  from boxtwin.core.project import project_paths
  paths = project_paths(Path("proyecto/videos/spar.mp4"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ProjectPaths", "project_paths"]


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
