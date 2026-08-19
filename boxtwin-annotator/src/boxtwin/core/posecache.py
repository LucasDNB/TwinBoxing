"""
BoxTwin - Formato del cache de pose y su lector.

POR QUE EXISTE
  El anotador necesita las detecciones de un frame cualquiera en tiempo constante, porque
  retroceder cuadro a cuadro tiene que ser instantaneo. Guardar una lista de dicts por
  frame obligaria a parsear al azar y a cargar todo en memoria; guardar un array por frame
  desperdicia espacio porque la cantidad de detecciones varia.
  La solucion es un indice CSR: todas las detecciones concatenadas en arrays contiguos mas
  un vector de offsets. Las detecciones del frame f son la rebanada
  [frame_index[f], frame_index[f+1]), que es una vista sin copia.

  frame_status existe por un motivo que no se ve de entrada: con CSR, un frame procesado
  sin nadie en cuadro y un frame que nunca se proceso producen los dos una rebanada vacia.
  Sin ese array, la reanudacion no puede saber donde quedo.

QUE HACE
  Define el layout del npz, lo escribe y lo lee. El archivo es inmutable: la anotacion
  nunca lo modifica, todo lo que decide el anotador vive en el annot.json.

USO
  from boxtwin.core.posecache import PoseCache
  cache = PoseCache.open(Path("cache/spar.pose.npz"))
  dets = cache.detections(1234)
  dets.track_id, dets.bbox, dets.keypoints, dets.kp_score
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CACHE_FORMAT_VERSION",
    "N_KEYPOINTS",
    "FrameStatus",
    "PoseDetections",
    "PoseCache",
    "PoseArrays",
    "write_pose_cache",
]

CACHE_FORMAT_VERSION = 1
N_KEYPOINTS = 17


class FrameStatus(IntEnum):
    """
    Estado de procesamiento de cada frame.

    Distinguir NO_PROCESADO de OK sin detecciones es lo que hace posible reanudar: los dos
    casos producen una rebanada vacia en el indice CSR.
    """

    NO_PROCESADO = 0
    OK = 1
    ERROR_DECODE = 2


@dataclass(frozen=True)
class PoseDetections:
    """
    Detecciones de un frame. Los arrays son vistas del cache, no copias.

    No se escriben nunca: el cache es inmutable. Si hace falta modificarlos, copiar antes.
    """

    frame: int
    status: FrameStatus
    track_id: np.ndarray  # int32 (n,)
    bbox: np.ndarray  # float32 (n, 4) xyxy en pixeles del video original
    det_conf: np.ndarray  # float32 (n,)
    keypoints: np.ndarray  # float32 (n, 17, 2)
    kp_score: np.ndarray  # float32 (n, 17)

    def __len__(self) -> int:
        return int(self.track_id.shape[0])

    @property
    def processed(self) -> bool:
        return self.status is not FrameStatus.NO_PROCESADO

    def index_of_track(self, track_id: int) -> int | None:
        """Posicion de un track dentro de este frame, o None si no esta."""
        hits = np.flatnonzero(self.track_id == track_id)
        return int(hits[0]) if hits.size else None


@dataclass
class PoseArrays:
    """Los arrays del cache ya concatenados, tal como van al npz."""

    frame_index: np.ndarray  # int64 (n_frames + 1,)
    frame_status: np.ndarray  # uint8 (n_frames,)
    track_id: np.ndarray  # int32 (N,)
    bbox: np.ndarray  # float32 (N, 4)
    det_conf: np.ndarray  # float32 (N,)
    keypoints: np.ndarray  # float32 (N, 17, 2)
    kp_score: np.ndarray  # float32 (N, 17)

    @property
    def n_frames(self) -> int:
        return int(self.frame_status.shape[0])

    @property
    def n_detections(self) -> int:
        return int(self.track_id.shape[0])

    def validate(self) -> None:
        """
        Chequea las invariantes del layout. Barato y evita depurar un npz corrupto.

        La autoridad sobre cuantas detecciones hay es frame_index, no track_id. Si el
        largo esperado se sacara de track_id, la forma de track_id no quedaria validada
        contra nada y un array corto pasaria sin que nadie lo note.
        """
        n_frames = self.n_frames

        if self.frame_index.shape != (n_frames + 1,):
            raise ValueError(
                f"frame_index tiene forma {self.frame_index.shape}, se esperaba ({n_frames + 1},)"
            )
        if self.frame_index[0] != 0:
            raise ValueError(f"frame_index[0] tiene que ser 0, es {self.frame_index[0]}")
        if np.any(np.diff(self.frame_index) < 0):
            raise ValueError("frame_index no es monotono creciente")

        n_det = int(self.frame_index[-1])
        esperado = {
            "track_id": (n_det,),
            "det_conf": (n_det,),
            "bbox": (n_det, 4),
            "keypoints": (n_det, N_KEYPOINTS, 2),
            "kp_score": (n_det, N_KEYPOINTS),
        }
        for nombre, forma in esperado.items():
            real = getattr(self, nombre).shape
            if real != forma:
                raise ValueError(
                    f"{nombre} tiene forma {real} y frame_index declara {n_det} "
                    f"detecciones, se esperaba {forma}"
                )


def write_pose_cache(path: Path, arrays: PoseArrays, meta: dict[str, Any]) -> None:
    """
    Escribe el npz de forma atomica.

    El meta va embebido ademas de vivir en su propio json, para que el npz sea
    autocontenido: si alguien mueve el archivo suelto, todavia se puede saber con que
    modelo y que configuracion se genero.
    """
    import json

    arrays.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(
            tmp,
            format_version=np.int32(CACHE_FORMAT_VERSION),
            meta_json=np.array(json.dumps(meta, ensure_ascii=False, indent=2)),
            frame_index=arrays.frame_index.astype(np.int64, copy=False),
            frame_status=arrays.frame_status.astype(np.uint8, copy=False),
            track_id=arrays.track_id.astype(np.int32, copy=False),
            bbox=arrays.bbox.astype(np.float32, copy=False),
            det_conf=arrays.det_conf.astype(np.float32, copy=False),
            keypoints=arrays.keypoints.astype(np.float32, copy=False),
            kp_score=arrays.kp_score.astype(np.float32, copy=False),
        )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class PoseCache:
    """Lector del cache. Mantiene el npz abierto y devuelve vistas sin copiar."""

    def __init__(self, arrays: PoseArrays, meta: dict[str, Any], path: Path | None = None) -> None:
        arrays.validate()
        self._a = arrays
        self.meta = meta
        self.path = path

    @classmethod
    def open(cls, path: Path) -> PoseCache:
        import json

        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != CACHE_FORMAT_VERSION:
                raise ValueError(
                    f"{path} usa el formato de cache {version} y este binario entiende "
                    f"{CACHE_FORMAT_VERSION}; hay que reprocesar el video"
                )
            meta = json.loads(str(z["meta_json"]))
            arrays = PoseArrays(
                frame_index=z["frame_index"],
                frame_status=z["frame_status"],
                track_id=z["track_id"],
                bbox=z["bbox"],
                det_conf=z["det_conf"],
                keypoints=z["keypoints"],
                kp_score=z["kp_score"],
            )
        return cls(arrays, meta, path)

    # -- acceso ------------------------------------------------------------

    def __len__(self) -> int:
        return self._a.n_frames

    @property
    def n_detections(self) -> int:
        return self._a.n_detections

    def status(self, frame: int) -> FrameStatus:
        return FrameStatus(int(self._a.frame_status[frame]))

    def detections(self, frame: int) -> PoseDetections:
        """Detecciones del frame. O(1): es una rebanada del indice CSR."""
        if not 0 <= frame < self._a.n_frames:
            raise IndexError(f"frame {frame} fuera de [0, {self._a.n_frames})")
        lo = int(self._a.frame_index[frame])
        hi = int(self._a.frame_index[frame + 1])
        return PoseDetections(
            frame=frame,
            status=self.status(frame),
            track_id=self._a.track_id[lo:hi],
            bbox=self._a.bbox[lo:hi],
            det_conf=self._a.det_conf[lo:hi],
            keypoints=self._a.keypoints[lo:hi],
            kp_score=self._a.kp_score[lo:hi],
        )

    def track_ids(self) -> np.ndarray:
        """Todos los track_id distintos del video, ordenados."""
        return np.unique(self._a.track_id)

    def frames_of_track(self, track_id: int) -> np.ndarray:
        """Frames en que aparece un track. Util para detectar los huecos a interpolar."""
        hits = np.flatnonzero(self._a.track_id == track_id)
        if hits.size == 0:
            return np.empty(0, dtype=np.int64)
        # searchsorted sobre los offsets convierte indice de deteccion en numero de frame.
        return np.searchsorted(self._a.frame_index, hits, side="right").astype(np.int64) - 1

    def unprocessed_frames(self) -> np.ndarray:
        return np.flatnonzero(self._a.frame_status == FrameStatus.NO_PROCESADO).astype(np.int64)
