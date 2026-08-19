"""
BoxTwin - Escritura por shards y estado de reanudacion del preproceso.

POR QUE EXISTE
  Medido en esta maquina, yolov8l-pose con BoT-SORT corre a 29 fps sostenidos, asi que un
  video de 24.000 cuadros son 14 minutos y uno de una hora son mas de dos horas. Perder
  eso por un corte de luz o un OOM no es aceptable, y NPZ no se puede ir escribiendo de a
  poco: se escribe entero o no se escribe.
  De ahi los shards. Cada N frames se cierra un archivo chico y se actualiza el estado, y
  al terminar se concatena todo en el npz final.

  El problema real de reanudar no es el archivo, es el tracker. BoT-SORT tiene estado
  interno: si el proceso muere en el cuadro 10.000 y arranca una instancia nueva, los
  track_id vuelven a empezar en 1 y colisionan con los del tramo anterior. Una asignacion
  de identidad hecha sobre el track 1 del primer tramo se aplicaria en silencio al track 1
  del segundo, que es otra persona. Por eso los ids que emite el tracker se desplazan por
  un offset persistido, y la costura queda registrada en el meta para que el anotador sepa
  que ahi hay un corte de identidad garantizado.

QUE HACE
  Acumula detecciones, las vuelca a shards, mantiene el estado de reanudacion, aplica el
  offset de track_id y concatena todo en el npz final.

USO
  writer = ShardWriter(work_dir, video_sha256=..., n_frames_expected=..., config_hash=...)
  inicio = writer.resume_from()      # primer frame a procesar
  writer.add_frame(idx, status, dets)
  writer.finalize(destino_npz, meta)
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, write_pose_cache

__all__ = ["FrameDetections", "ResumeState", "ShardWriter", "ResumeMismatch", "SHARD_FRAMES"]

# 2000 cuadros son ~70 s de trabajo a 29 fps. Es lo maximo que se pierde ante un corte.
SHARD_FRAMES = 2000


class ResumeMismatch(RuntimeError):
    """El estado guardado no corresponde a este video o a esta configuracion."""


@dataclass
class FrameDetections:
    """Detecciones crudas de un frame, tal como salen del tracker."""

    track_id: np.ndarray  # (n,) int
    bbox: np.ndarray  # (n, 4) float xyxy
    det_conf: np.ndarray  # (n,) float
    keypoints: np.ndarray  # (n, 17, 2) float
    kp_score: np.ndarray  # (n, 17) float

    @classmethod
    def empty(cls) -> FrameDetections:
        return cls(
            track_id=np.empty(0, np.int32),
            bbox=np.empty((0, 4), np.float32),
            det_conf=np.empty(0, np.float32),
            keypoints=np.empty((0, N_KEYPOINTS, 2), np.float32),
            kp_score=np.empty((0, N_KEYPOINTS), np.float32),
        )

    def __len__(self) -> int:
        return int(self.track_id.shape[0])


@dataclass
class ResumeState:
    """
    Estado persistido entre corridas.

    `id_offset` es lo que se le suma a cada track_id que emite el tracker actual.
    `seams` lista los cuadros donde el tracker se reinicio: ahi hay un corte de identidad
    garantizado y ningun track cruza de un lado al otro.
    """

    video_sha256: str
    config_hash: str
    n_frames_expected: int
    next_frame: int = 0
    max_track_id: int = 0
    id_offset: int = 0
    shards: list[str] = field(default_factory=list)
    seams: list[dict[str, int]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    @classmethod
    def from_json(cls, texto: str) -> ResumeState:
        return cls(**json.loads(texto))


class ShardWriter:
    """Acumula frames, vuelca shards y reconstruye el npz final."""

    def __init__(
        self,
        work_dir: Path,
        *,
        video_sha256: str,
        config_hash: str,
        n_frames_expected: int,
        shard_frames: int = SHARD_FRAMES,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.shard_frames = shard_frames
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "state.json"

        self.state = self._cargar_o_crear(video_sha256, config_hash, n_frames_expected)

        # Buffer del shard en curso.
        self._buf_desde: int = self.state.next_frame
        self._buf_status: list[int] = []
        self._buf_dets: list[FrameDetections] = []

    # -- estado ------------------------------------------------------------

    def _cargar_o_crear(self, sha: str, config_hash: str, n_frames: int) -> ResumeState:
        if not self.state_path.exists():
            return ResumeState(
                video_sha256=sha, config_hash=config_hash, n_frames_expected=n_frames
            )

        previo = ResumeState.from_json(self.state_path.read_text(encoding="utf-8"))
        if previo.video_sha256 != sha:
            raise ResumeMismatch(
                f"{self.work_dir} tiene trabajo a medias de otro video "
                f"(sha {previo.video_sha256[:12]} contra {sha[:12]}). "
                "Borrar el directorio o pasar --restart."
            )
        if previo.config_hash != config_hash:
            raise ResumeMismatch(
                "la configuracion de inferencia cambio desde la corrida anterior. "
                "Reanudar mezclaria dos configuraciones en un mismo cache: no se puede. "
                "Borrar el directorio o pasar --restart."
            )
        return previo

    def _guardar_estado(self) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(self.state.to_json(), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def resume_from(self) -> int:
        """
        Primer frame a procesar.

        Si hay trabajo previo, ademas prepara el offset de track_id para la instancia nueva
        del tracker y registra la costura. Se llama una sola vez, antes del bucle.
        """
        if self.state.next_frame > 0:
            self.state.id_offset = self.state.max_track_id
            self.state.seams.append(
                {"frame": self.state.next_frame, "id_offset": self.state.id_offset}
            )
            self._guardar_estado()
        return self.state.next_frame

    def cut_seam(self, frame: int) -> None:
        """
        Declara que la identidad se corta en `frame`. Se llama al reiniciar el tracker.

        Misma mecanica que `resume_from`, otra causa: alla el tracker se reinicia porque el
        proceso se corto, aca porque cambio el plano y arrastrar la identidad a traves de un
        corte de camara le pone a un peleador el cuerpo del otro. En los dos casos los ids
        nuevos arrancan de cero y hay que desplazarlos para que no pisen a los anteriores.
        """
        self.state.id_offset = self.state.max_track_id
        self.state.seams.append({"frame": frame, "id_offset": self.state.id_offset})
        self._guardar_estado()

    @property
    def id_offset(self) -> int:
        return self.state.id_offset

    # -- acumulacion -------------------------------------------------------

    def add_frame(self, frame: int, status: FrameStatus, dets: FrameDetections) -> None:
        """
        Agrega un frame. Los track_id se desplazan por el offset vigente aca y no antes,
        para que el llamador no tenga que acordarse.
        """
        esperado = self._buf_desde + len(self._buf_status)
        if frame != esperado:
            raise ValueError(f"se esperaba el frame {esperado} y llego {frame}")

        if len(dets) and self.state.id_offset:
            dets = FrameDetections(
                track_id=dets.track_id + self.state.id_offset,
                bbox=dets.bbox,
                det_conf=dets.det_conf,
                keypoints=dets.keypoints,
                kp_score=dets.kp_score,
            )
        if len(dets):
            self.state.max_track_id = max(self.state.max_track_id, int(dets.track_id.max()))

        self._buf_status.append(int(status))
        self._buf_dets.append(dets)

        if len(self._buf_status) >= self.shard_frames:
            self.flush()

    def flush(self) -> None:
        """Cierra el shard en curso y avanza el estado. Idempotente si no hay nada acumulado."""
        if not self._buf_status:
            return

        n = len(self._buf_status)
        indice = self._shard_path(self._buf_desde)
        arrays = _concatenar(self._buf_dets)

        tmp = indice.with_name(indice.name + ".tmp.npz")
        np.savez(
            tmp,
            frame_start=np.int64(self._buf_desde),
            frame_status=np.asarray(self._buf_status, np.uint8),
            frame_index=arrays["frame_index"],
            track_id=arrays["track_id"],
            bbox=arrays["bbox"],
            det_conf=arrays["det_conf"],
            keypoints=arrays["keypoints"],
            kp_score=arrays["kp_score"],
        )
        tmp.replace(indice)

        # El estado se actualiza DESPUES del shard: si el proceso muere entre las dos
        # cosas, el shard huerfano se reescribe en la proxima corrida y no se pierde nada.
        self.state.next_frame = self._buf_desde + n
        if indice.name not in self.state.shards:
            self.state.shards.append(indice.name)
        self._guardar_estado()

        self._buf_desde = self.state.next_frame
        self._buf_status = []
        self._buf_dets = []

    def _shard_path(self, frame_start: int) -> Path:
        return self.work_dir / f"shard_{frame_start:09d}.npz"

    # -- finalizacion ------------------------------------------------------

    def finalize(self, destino: Path, meta: dict[str, Any], *, total_frames: int) -> None:
        """
        Concatena los shards en el npz final y borra el directorio de trabajo.

        `total_frames` es el conteo real medido decodificando, que puede ser menor que el
        estimado del contenedor. Los frames que nunca se procesaron quedan como
        NO_PROCESADO, que es informacion y no un hueco silencioso.
        """
        self.flush()

        status = np.zeros(total_frames, np.uint8)
        offsets = [np.zeros(1, np.int64)]
        piezas: list[dict[str, np.ndarray]] = []
        acumulado = 0

        proximo_esperado = 0
        for nombre in sorted(self.state.shards):
            with np.load(self.work_dir / nombre, allow_pickle=False) as z:
                inicio = int(z["frame_start"])
                st = z["frame_status"]
                # Un hueco entre shards desalinearia todo el indice CSR en silencio y el
                # cache saldria con los keypoints corridos. Mejor fallar aca.
                if inicio != proximo_esperado:
                    raise ValueError(
                        f"shards no contiguos en {self.work_dir}: {nombre} arranca en "
                        f"{inicio} y se esperaba {proximo_esperado}"
                    )
                proximo_esperado = inicio + int(st.shape[0])
                fin = min(inicio + st.shape[0], total_frames)
                if inicio >= total_frames:
                    continue
                recorte = fin - inicio
                status[inicio:fin] = st[:recorte]

                idx = z["frame_index"][: recorte + 1]
                offsets.append(idx[1:] + acumulado)
                n_det = int(idx[-1])
                acumulado += n_det
                piezas.append(
                    {
                        "track_id": z["track_id"][:n_det],
                        "bbox": z["bbox"][:n_det],
                        "det_conf": z["det_conf"][:n_det],
                        "keypoints": z["keypoints"][:n_det],
                        "kp_score": z["kp_score"][:n_det],
                    }
                )

        frame_index = np.concatenate(offsets) if len(offsets) > 1 else np.zeros(1, np.int64)
        # Los frames sin shard (nunca procesados) heredan el ultimo offset: rebanada vacia.
        if frame_index.shape[0] < total_frames + 1:
            faltan = total_frames + 1 - frame_index.shape[0]
            frame_index = np.concatenate([frame_index, np.full(faltan, frame_index[-1], np.int64)])

        vacio = _concatenar([])
        arrays = PoseArrays(
            frame_index=frame_index,
            frame_status=status,
            track_id=_apilar(piezas, "track_id", vacio["track_id"]),
            bbox=_apilar(piezas, "bbox", vacio["bbox"]),
            det_conf=_apilar(piezas, "det_conf", vacio["det_conf"]),
            keypoints=_apilar(piezas, "keypoints", vacio["keypoints"]),
            kp_score=_apilar(piezas, "kp_score", vacio["kp_score"]),
        )

        meta = dict(meta)
        meta["resume_seams"] = self.state.seams
        meta["max_track_id"] = self.state.max_track_id

        write_pose_cache(destino, arrays, meta)
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def iter_shards(self) -> Iterator[Path]:
        for nombre in sorted(self.state.shards):
            yield self.work_dir / nombre


def _concatenar(dets: list[FrameDetections]) -> dict[str, np.ndarray]:
    """Arma el bloque CSR de una lista de frames."""
    frame_index = np.zeros(len(dets) + 1, np.int64)
    if dets:
        frame_index[1:] = np.cumsum([len(d) for d in dets], dtype=np.int64)

    def junta(nombre: str, forma: tuple[int, ...], dtype) -> np.ndarray:
        partes = [getattr(d, nombre) for d in dets if len(d)]
        if not partes:
            return np.empty((0, *forma), dtype)
        return np.concatenate(partes).astype(dtype, copy=False)

    return {
        "frame_index": frame_index,
        "track_id": junta("track_id", (), np.int32),
        "bbox": junta("bbox", (4,), np.float32),
        "det_conf": junta("det_conf", (), np.float32),
        "keypoints": junta("keypoints", (N_KEYPOINTS, 2), np.float32),
        "kp_score": junta("kp_score", (N_KEYPOINTS,), np.float32),
    }


def _apilar(piezas: list[dict[str, np.ndarray]], clave: str, vacio: np.ndarray) -> np.ndarray:
    partes = [p[clave] for p in piezas if p[clave].shape[0]]
    return np.concatenate(partes) if partes else vacio
