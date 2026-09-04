"""
BoxTwin - Sesion de anotacion: lo que une video, cache y documento.

POR QUE EXISTE
  Abrir un video para anotar implica encontrar cuatro archivos, comprobar que se
  correspondan entre si y armar el documento si no existe. Si eso queda desparramado en la
  ventana principal, el dia que haya que abrir dos videos o correr algo sin interfaz hay
  que desarmarlo todo.
  La comprobacion de correspondencia no es formalidad. El cache guarda el sha256 del video
  con el que se genero: si alguien reemplaza el archivo de video dejando el mismo nombre,
  los keypoints siguen cargando y caen sobre otro material. Es un error que no se ve, se
  anota igual y arruina el dataset.

QUE HACE
  Resuelve las rutas del proyecto, carga el cache de pose, carga o crea el annot.json,
  abre la fuente de cuadros preferiendo el proxy y arma el resolver de identidad.

USO
  sesion = Session.open(Path("proyecto/videos/spar.mp4"))
  sesion.resolver.resolve_frame(120)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from boxtwin.core.annotations import load as load_doc
from boxtwin.core.annotations import save as save_doc
from boxtwin.core.identity import IdentityResolver
from boxtwin.core.posecache import PoseCache
from boxtwin.core.project import ProjectPaths, project_paths
from boxtwin.core.metrics import new_session_metrics
from boxtwin.core.schema import (
    AnnotationDoc,
    AnnotatorInfo,
    PoseRef,
    Totals,
    VideoInfo,
    new_document,
)
from boxtwin.core.undo import UndoStack
from boxtwin.core.types import FpsSource, Guard, KeypointFormat
from boxtwin.core.video import sha256_file
from boxtwin.gui.player.decoder import FrameSource
from boxtwin.version import __version__

__all__ = ["Session", "SessionError", "ProjectPaths", "project_paths"]


class SessionError(RuntimeError):
    pass


@dataclass
class Session:
    paths: ProjectPaths
    cache: PoseCache
    doc: AnnotationDoc
    source: FrameSource
    resolver: IdentityResolver
    fps: float
    total_frames: int
    video_size: tuple[int, int]
    seams: list[int]
    using_proxy: bool
    migrated: list[int]
    has_video: bool = True
    annotator: str = "desconocido"
    undo: UndoStack | None = None
    _hires: FrameSource | None = None

    @classmethod
    def open(cls, video: Path, *, buffer_mb: int = 512) -> Session:
        """
        Abre una sesion. El video original es OPCIONAL si esta el proxy.

        Anotar no necesita GPU, asi que el preproceso y la anotacion pueden vivir en
        maquinas distintas. Sobre fuentes 4K el original pesa 490 MB cada 10 minutos y el
        proxy 97, asi que exigir el original obligaria a mover cinco veces mas datos para
        nada. Lo unico que se pierde sin el es el zoom en resolucion original.
        """
        paths = project_paths(video)
        if not paths.npz.is_file():
            raise SessionError(
                f"falta el cache de pose ({paths.npz.name}).\n"
                f"Correr primero: boxtwin-annotator preprocess {paths.video}"
            )
        if not paths.video.is_file() and not paths.proxy.is_file():
            raise SessionError(
                f"no hay imagen para {paths.video.name}: falta el video original y "
                f"tambien el proxy ({paths.proxy.name}).\n"
                "Hace falta al menos uno de los dos."
            )

        cache = PoseCache.open(paths.npz)
        meta = cache.meta
        video_meta = meta["video"]

        total_frames = int(video_meta["total_frames"])
        fps = float(video_meta["fps"])
        ancho, alto = int(video_meta["width"]), int(video_meta["height"])

        doc, migrated = cls._load_or_create(paths, meta)
        cls._check_correspondence(doc, meta, paths)

        fuente_path, usando_proxy, escala = cls._pick_source(paths, ancho)
        source = FrameSource(
            fuente_path, total_frames=total_frames, buffer_mb=buffer_mb, scale=escala
        )

        return cls(
            paths=paths,
            cache=cache,
            doc=doc,
            source=source,
            resolver=IdentityResolver(doc, cache),
            fps=fps,
            total_frames=total_frames,
            video_size=(ancho, alto),
            seams=[int(s["frame"]) for s in meta.get("resume_seams", [])],
            using_proxy=usando_proxy,
            migrated=migrated,
            has_video=paths.video.is_file(),
            annotator=cls._leer_anotador(paths),
            undo=UndoStack(doc),
        )

    # -- armado ------------------------------------------------------------

    @staticmethod
    def _pick_source(paths: ProjectPaths, ancho_original: int) -> tuple[Path, bool, float]:
        """
        Prefiere el proxy. Medido, decodificar 4K a un hilo da 41 fps contra 919 del proxy,
        asi que sobre el original la reproduccion hacia atras es directamente inviable.
        """
        if paths.proxy.is_file():
            import cv2

            cap = cv2.VideoCapture(str(paths.proxy))
            ancho_proxy = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if ancho_proxy > 0:
                return paths.proxy, True, ancho_proxy / ancho_original
        return paths.video, False, 1.0

    @property
    def hires_available(self) -> bool:
        return self.has_video

    @staticmethod
    def _leer_anotador(paths: ProjectPaths) -> str:
        """
        Quien anota. Sale de config.yaml y si no del usuario del sistema.

        Es obligatorio en las metricas de proceso: sin saber quien anoto que, el analisis
        de acuerdo entre anotadores no se puede hacer, y ese analisis es parte del aporte
        metodologico, no un extra.
        """
        import os

        if paths.config.is_file():
            import yaml

            data = yaml.safe_load(paths.config.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict) and data.get("annotator"):
                return str(data["annotator"])
        return os.environ.get("USER") or os.environ.get("USERNAME") or "desconocido"

    @staticmethod
    def _load_or_create(paths: ProjectPaths, meta: dict) -> tuple[AnnotationDoc, list[int]]:
        if paths.annot.is_file():
            return load_doc(paths.annot)

        video_meta = meta["video"]
        ahora = datetime.now().astimezone()
        doc = new_document(
            app_version=__version__,
            now=ahora,
            video=VideoInfo(
                path=str(paths.video),
                sha256=video_meta["sha256"],
                size_bytes=int(video_meta["size_bytes"]),
                mtime=datetime.fromisoformat(video_meta["mtime"]),
                fps=float(video_meta["fps"]),
                fps_declared=video_meta.get("fps_declared"),
                fps_source=FpsSource(video_meta["fps_source"]),
                width=int(video_meta["width"]),
                height=int(video_meta["height"]),
                total_frames=int(video_meta["total_frames"]),
                total_frames_declared=video_meta.get("total_frames_declared"),
                duration_s=float(video_meta["duration_s"]),
                codec=str(video_meta["codec"]),
            ),
            pose=PoseRef(
                npz_path=str(paths.npz),
                meta_path=str(paths.meta),
                meta_sha256=sha256_file(paths.meta) if paths.meta.is_file() else "0" * 64,
                keypoint_format=KeypointFormat(meta.get("keypoint_format", "coco17")),
                keypoint_sources={"0-16": "yolov8l-pose"},
            ),
            # Punto de partida, no un dato: se elige ortodoxa porque es lo mas frecuente,
            # no porque se sepa. La real la fija el anotador desde el panel de identidad
            # cuando ve pegar unos golpes, y al cambiarla se le ofrece reescribir los
            # eventos ya anotados. Dejar esto sin interfaz costo tres correcciones a mano
            # sobre 175 eventos.
            guard_a=Guard.ORTHODOX,
            guard_b=Guard.ORTHODOX,
        )
        save_doc(doc, paths.annot)
        return doc, []

    @staticmethod
    def _check_correspondence(doc: AnnotationDoc, meta: dict, paths: ProjectPaths) -> None:
        if doc.video.sha256 != meta["video"]["sha256"]:
            raise SessionError(
                f"{paths.annot.name} fue hecho sobre otro video.\n"
                f"  anotacion: {doc.video.sha256[:16]}\n"
                f"  cache    : {meta['video']['sha256'][:16]}\n"
                "Los keypoints caerian sobre material distinto del que se anoto."
            )

    # -- ciclo de vida -----------------------------------------------------

    def hires(self) -> FrameSource | None:
        """
        Fuente en resolucion original, abierta recien cuando se la pide.

        Se usa solo para el cuadro en pausa con zoom alto. Ampliar el proxy 4x deja la
        imagen tan borrosa que juzgar si un guante llego a la cara se vuelve adivinanza, y
        ese juicio es justamente lo que se esta anotando. Decodificar un cuadro del
        original cuesta unos 25 ms sobre 4K, que en pausa no molesta y en reproduccion
        seria imposible: por eso solo en pausa.
        """
        if not self.has_video:
            return None  # se anota solo con el proxy: no hay original que abrir
        if not self.using_proxy:
            return self.source
        if self._hires is None:
            try:
                self._hires = FrameSource(
                    self.paths.video, total_frames=self.total_frames, buffer_mb=64, scale=1.0
                )
            except Exception:  # noqa: BLE001
                self._hires = False  # type: ignore[assignment]
        return self._hires or None

    def begin_session(self, app_version: str) -> str:
        """
        Abre una sesion de trabajo en las metricas de proceso.

        Cada corrida es una sesion propia aunque sea sobre el mismo video: fusionarlas
        perderia la informacion de cuantas veces se volvio sobre el material, que es
        justamente lo que distingue una anotacion de una tanda.
        """
        from boxtwin.core.annotations import new_id

        ahora = datetime.now().astimezone()
        sid = new_id(self.doc, "session")
        self.doc.process.sessions = [
            *self.doc.process.sessions,
            new_session_metrics(
                session_id=sid, annotator=self.annotator, ahora=ahora, app_version=app_version
            ),
        ]
        if not any(a.id == self.annotator for a in self.doc.process.annotators):
            self.doc.process.annotators = [
                *self.doc.process.annotators,
                AnnotatorInfo(id=self.annotator, name=self.annotator),
            ]
        return sid

    def end_session(self, active_ms: int) -> None:
        """Cierra la sesion y recalcula los totales sobre todas las sesiones del archivo."""
        if not self.doc.process.sessions:
            return
        actual = self.doc.process.sessions[-1]
        actual.ended_at = datetime.now().astimezone()
        actual.active_ms = active_ms

        total_ms = sum(s.active_ms for s in self.doc.process.sessions)
        tiempos = sorted(m.active_ms for m in self.doc.process.event_metrics.values())
        mediana = tiempos[len(tiempos) // 2] if tiempos else None
        self.doc.process.totals = Totals(
            active_ms=total_ms, events=len(self.doc.events), median_ms_per_event=mediana
        )

    def save(self) -> None:
        from boxtwin.core.annotations import touch

        touch(self.doc, datetime.now().astimezone())
        save_doc(self.doc, self.paths.annot)

    def close(self) -> None:
        self.source.close()
        if self._hires:
            self._hires.close()
