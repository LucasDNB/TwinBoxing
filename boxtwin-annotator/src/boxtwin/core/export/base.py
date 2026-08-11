"""
BoxTwin - Contexto y contrato comun de los exports.

POR QUE EXISTE
  Un dataset exportado sin saber de que anotacion salio no se puede reproducir ni auditar.
  Meses despues, con el annot.json ya modificado, la unica forma de saber si un modelo se
  entreno sobre los datos que uno cree es comparar hashes. Por eso todo export lleva en su
  metadata el sha256 del documento que lo genero.

  El hash se calcula sobre la serializacion canonica, no sobre el archivo en disco: asi dos
  exports del mismo contenido dan el mismo hash aunque el archivo se haya reescrito, y dos
  contenidos distintos nunca coinciden aunque el archivo se vea igual.

QUE HACE
  Define el contexto que recibe cada exportador, el resultado que devuelve, el hash de la
  anotacion y el registro de formatos disponibles.

USO
  from boxtwin.core.export import exportadores
  exportadores()["stats"](ctx)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from boxtwin.core.annotations import dumps
from boxtwin.core.identity import IdentityResolver
from boxtwin.core.posecache import PoseCache
from boxtwin.core.schema import AnnotationDoc
from boxtwin.version import __version__

__all__ = [
    "ExportContext",
    "ExportResult",
    "annot_hash",
    "base_metadata",
    "exportadores",
    "registrar",
]


def annot_hash(doc: AnnotationDoc) -> str:
    """sha256 de la serializacion canonica del documento."""
    return hashlib.sha256(dumps(doc).encode("utf-8")).hexdigest()


@dataclass
class ExportContext:
    """Todo lo que un exportador necesita. Los exportadores no abren archivos por su cuenta."""

    doc: AnnotationDoc
    cache: PoseCache
    resolver: IdentityResolver
    video_path: Path
    out_dir: Path
    opciones: dict[str, Any] = field(default_factory=dict)

    def opcion(self, clave: str, default: Any = None) -> Any:
        return self.opciones.get(clave, default)


@dataclass
class ExportResult:
    formato: str
    archivos: list[Path]
    resumen: dict[str, Any]
    avisos: list[str] = field(default_factory=list)


def base_metadata(ctx: ExportContext, formato: str) -> dict[str, Any]:
    """
    Metadata comun. Va en todos los exports y es lo que permite rastrear un dataset.

    created_at queda en UTC con offset explicito: un dataset que se compara entre maquinas
    no puede depender de la zona horaria de la que lo genero.
    """
    return {
        "kind": "boxtwin.export",
        "format": formato,
        "app_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {
            "name": ctx.video_path.name,
            "sha256": ctx.doc.video.sha256,
            "fps": ctx.doc.video.fps,
            "width": ctx.doc.video.width,
            "height": ctx.doc.video.height,
            "total_frames": ctx.doc.video.total_frames,
        },
        "annot_sha256": annot_hash(ctx.doc),
        "pose_meta_sha256": ctx.doc.pose.meta_sha256,
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in ctx.opciones.items()},
    }


def escribir_json(path: Path, data: dict[str, Any]) -> None:
    """Escritura estable: claves en orden de insercion, indentacion fija y newline final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8"
    )


_REGISTRO: dict[str, Callable[[ExportContext], ExportResult]] = {}


def registrar(nombre: str) -> Callable:
    def _deco(fn: Callable[[ExportContext], ExportResult]):
        _REGISTRO[nombre] = fn
        return fn

    return _deco


def exportadores() -> dict[str, Callable[[ExportContext], ExportResult]]:
    # Los modulos se importan aca para que registrarse no dependa del orden de import.
    from boxtwin.core.export import clips, mmaction, sequence, stats  # noqa: F401

    return dict(_REGISTRO)
