"""
BoxTwin - Pase de pose y tracking sobre todo el video.

POR QUE EXISTE
  No se infiere pose durante la reproduccion. Medido en esta maquina, yolov8l-pose con
  BoT-SORT sostiene 29 fps, o sea que la inferencia en vivo ni siquiera alcanza el
  framerate nativo, y menos con la UI encima. Peor: retroceder obligaria a reinferir, y
  como el tracker tiene estado, el resultado dependeria del camino recorrido. Un cache
  precalculado hace que ir y venir por el video sea gratis y deterministico.

  El cache es inmutable. Todo lo que decide el anotador, incluida la identidad, vive en el
  annot.json. Asi se puede reanotar sin reprocesar y reprocesar sin reanotar.

QUE HACE
  Decodifica el video cuadro a cuadro, corre deteccion de pose con tracking, persiste por
  shards para poder reanudar, cuenta los frames de verdad en vez de creerle al contenedor,
  genera el proxy en paralelo y escribe el npz final mas su meta.json.

USO
  from boxtwin.preprocess.runner import PreprocessConfig, preprocess
  resultado = preprocess(Path("videos/spar.mp4"), Path("proyecto"), PreprocessConfig())
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus
from boxtwin.core.video import (
    ProbeError,
    VideoProbe,
    probe,
    reconcile_fps,
    require_ffmpeg,
    sha256_bytes,
    sha256_file,
)
from boxtwin.preprocess.proxy import ProxyJob
from boxtwin.preprocess.writer import SHARD_FRAMES, FrameDetections, ShardWriter
from boxtwin.version import __version__

__all__ = [
    "PreprocessConfig", "PreprocessResult", "preprocess", "default_tracker_path",
    "AnotacionEnRiesgoError",
]

META_FORMAT_VERSION = 1


def default_tracker_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "botsort.yaml"


@dataclass
class PreprocessConfig:
    """
    Parametros de inferencia. Todo esto entra en el hash de configuracion.

    half queda en False por defecto a proposito: FP16 daria algo de velocidad en Turing
    pero cambia los numeros, y un cache que no se puede reproducir bit a bit no sirve para
    comparar corridas. 29 fps alcanzan.
    """

    model: str = "yolov8l-pose.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "0"
    half: bool = False
    tracker: Path = field(default_factory=default_tracker_path)
    # Detectar cortes de plano y reiniciar el tracker en cada uno. Sobre camara fija no
    # encuentra ninguno y no cuesta nada; sobre transmision es lo que evita que la
    # identidad se arrastre entre camaras.
    detect_cuts: bool = True
    cut_threshold: float = 0.3
    shard_frames: int = SHARD_FRAMES
    make_proxy: bool = True
    proxy_width: int = 960
    proxy_gop: int = 12
    proxy_crf: int = 20
    proxy_preset: str = "veryfast"

    def hash_payload(self, tracker_text: str, model_sha: str) -> dict[str, Any]:
        """Lo que define si dos corridas son la misma configuracion."""
        return {
            "model": self.model,
            "model_sha256": model_sha,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "half": self.half,
            "tracker_config": tracker_text,
            # Cambiarlas cambia donde se reinicia el tracker y por lo tanto los track_id:
            # dos corridas con distinto valor no son la misma configuracion.
            "detect_cuts": self.detect_cuts,
            "cut_threshold": self.cut_threshold,
        }

    def config_hash(self, tracker_text: str, model_sha: str) -> str:
        payload = json.dumps(self.hash_payload(tracker_text, model_sha), sort_keys=True)
        return sha256_bytes(payload.encode("utf-8"))


@dataclass
class PreprocessResult:
    npz_path: Path
    meta_path: Path
    proxy_path: Path | None
    total_frames: int
    n_detections: int
    seams: list[dict[str, int]]
    runtime_s: float
    suspected_vfr: bool
    short_decode: bool
    frames_expected: int


class AnotacionEnRiesgoError(RuntimeError):
    """Rehacer el cache dejaria la anotacion apuntando a personas equivocadas."""


def _anotacion_en_riesgo(project_dir: Path, base: str) -> tuple[int, int] | None:
    """
    Cuantos eventos y asignaciones cuelgan del cache que se va a rehacer.

    Los track_id son del tracker, no del video: rehacer el cache los renumera, y las
    asignaciones de identidad pasan a apuntar a otra persona. No se ve en el overlay, porque
    el esqueleto sigue estando sobre un cuerpo; se ve en el export, cuando ya es tarde.
    """
    annot = project_dir / "annotations" / f"{base}.annot.json"
    if not annot.is_file():
        return None
    try:
        import json

        d = json.loads(annot.read_text(encoding="utf-8"))
        n_ev = len(d.get("events", []))
        n_as = len(d.get("identity", {}).get("assignments", []))
    except Exception:  # noqa: BLE001
        # Un archivo ilegible tambien es trabajo: no se pisa por no poder contarlo.
        return (-1, -1)
    return (n_ev, n_as) if (n_ev or n_as) else None


def preprocess(
    video: Path,
    project_dir: Path,
    cfg: PreprocessConfig | None = None,
    *,
    restart: bool = False,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreprocessResult:
    """
    Corre el preproceso completo sobre un video, reanudando si hay trabajo previo.

    `on_progress(frame_actual, total_estimado)` se llama cada tanto; el total es estimado
    porque el real solo se sabe al terminar de decodificar.

    Se niega a rehacer un cache del que cuelga una anotacion, salvo `force`. Reanudar no
    cuenta: ahi los track_id ya emitidos no cambian.
    """
    import cv2  # se importa aca para que el modulo se pueda inspeccionar sin opencv

    cfg = cfg or PreprocessConfig()
    require_ffmpeg()

    video = Path(video).resolve()
    project_dir = Path(project_dir).resolve()
    cache_dir = project_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = probe(video)
    sha_video = sha256_file(video)

    modelo_path = _resolver_modelo(cfg.model)
    sha_modelo = sha256_file(modelo_path)
    tracker_text = Path(cfg.tracker).read_text(encoding="utf-8")
    hash_cfg = cfg.config_hash(tracker_text, sha_modelo)

    base = video.stem
    npz_path = cache_dir / f"{base}.pose.npz"
    meta_path = cache_dir / f"{base}.meta.json"
    proxy_path = cache_dir / f"{base}.proxy.mp4"
    work_dir = cache_dir / f"{base}.pose.partial"

    # El guard va antes de borrar nada. Reanudar es inofensivo: los track_id ya escritos no
    # se tocan. Lo que destruye la anotacion es empezar de cero, sea por --restart o porque
    # cambio la configuracion y el cache anterior dejo de valer.
    rehace = restart or (npz_path.is_file() and not work_dir.is_dir())
    if rehace and not force:
        riesgo = _anotacion_en_riesgo(project_dir, base)
        if riesgo is not None:
            n_ev, n_as = riesgo
            detalle = (
                "no se pudo leer, pero existe"
                if n_ev < 0
                else f"{n_ev} eventos y {n_as} asignaciones de identidad"
            )
            raise AnotacionEnRiesgoError(
                f"{base}.annot.json tiene {detalle}.\n"
                "  Rehacer el cache renumera los track_id y esas asignaciones pasan a\n"
                "  apuntar a otra persona. No se ve en el overlay: el esqueleto sigue\n"
                "  estando sobre un cuerpo, y el error aparece recien en el export.\n"
                "  Si de verdad querés rehacerlo, --force, y despues hay que reasignar\n"
                "  identidad desde cero."
            )

    if restart:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)

    # Estimacion para la barra de progreso. El numero real sale del conteo al decodificar.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ProbeError(f"opencv no pudo abrir {video}")
    estimado = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (info.nb_frames_declared or 0)

    writer = ShardWriter(
        work_dir,
        video_sha256=sha_video,
        config_hash=hash_cfg,
        n_frames_expected=estimado,
        shard_frames=cfg.shard_frames,
    )
    inicio = writer.resume_from()

    cortes: list[int] = []
    if cfg.detect_cuts:
        from boxtwin.preprocess.cuts import detectar_cortes

        cortes = detectar_cortes(video, fps=info.fps, umbral=cfg.cut_threshold)
    pendientes = [c for c in cortes if c > inicio]

    proxy_job = None
    if cfg.make_proxy:
        proxy_job = ProxyJob(
            src=video, dst=proxy_path, width=cfg.proxy_width, gop=cfg.proxy_gop,
            crf=cfg.proxy_crf, preset=cfg.proxy_preset,
        ).start()

    t0 = time.perf_counter()
    n_untracked = 0
    n_errores = 0
    total_frames = inicio

    try:
        if inicio > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos != inicio:
                raise ProbeError(
                    f"el seek al frame {inicio} cayo en {pos}; este contenedor no permite "
                    "reanudar de forma exacta, hay que reprocesar con --restart"
                )

        modelo = _cargar_modelo(modelo_path, cfg)

        frame_idx = inicio
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if pendientes and frame_idx >= pendientes[0]:
                # El corte se declara ANTES de inferir el primer cuadro del plano nuevo: si
                # se infiriera primero, ese cuadro ya se habria asociado con los tracks del
                # plano anterior, que es exactamente lo que se quiere evitar.
                _reiniciar_tracker(modelo)
                writer.cut_seam(frame_idx)
                while pendientes and pendientes[0] <= frame_idx:
                    pendientes.pop(0)

            dets, sin_track = _inferir(modelo, frame, cfg)
            n_untracked += sin_track
            writer.add_frame(frame_idx, FrameStatus.OK, dets)

            frame_idx += 1
            total_frames = frame_idx
            if on_progress is not None and frame_idx % 25 == 0:
                on_progress(frame_idx, max(estimado, frame_idx))

        writer.flush()
    except BaseException:
        writer.flush()
        if proxy_job is not None:
            proxy_job.cancel()
        raise
    finally:
        cap.release()

    runtime = time.perf_counter() - t0
    veredicto = reconcile_fps(info, total_frames)

    # cap.read() devuelve False tanto al terminar el archivo como ante un error de
    # decodificacion a mitad, y opencv no distingue los dos casos. Lo unico observable es
    # que hayan salido bastantes menos cuadros de los estimados. No se puede afirmar cual
    # de los dos paso, pero callarse tampoco sirve: se reporta y decide el que anota.
    short_decode = bool(estimado) and total_frames < int(estimado * 0.99)

    proxy_ok = False
    if proxy_job is not None:
        proxy_ok = proxy_job.wait()

    meta = _armar_meta(
        video=video, info=info, sha_video=sha_video, veredicto=veredicto, cfg=cfg,
        modelo_path=modelo_path, sha_modelo=sha_modelo, tracker_text=tracker_text,
        hash_cfg=hash_cfg, runtime=runtime, n_untracked=n_untracked, n_errores=n_errores,
        proxy_path=proxy_path if proxy_ok else None, estimado=estimado,
        short_decode=short_decode, total_frames=total_frames,
    )
    meta["scene_cuts"] = {
        "detectados": cfg.detect_cuts,
        "umbral": cfg.cut_threshold,
        "n": len(cortes),
        "frames": cortes,
    }

    writer.finalize(npz_path, meta, total_frames=total_frames)

    # El meta.json se escribe con lo que quedo en el npz, incluidas las costuras.
    from boxtwin.core.posecache import PoseCache

    cache = PoseCache.open(npz_path)
    meta_final = dict(cache.meta)
    meta_final["counts"]["n_detections"] = cache.n_detections
    _escribir_json(meta_path, meta_final)

    return PreprocessResult(
        npz_path=npz_path,
        meta_path=meta_path,
        proxy_path=proxy_path if proxy_ok else None,
        total_frames=total_frames,
        n_detections=cache.n_detections,
        seams=meta_final.get("resume_seams", []),
        runtime_s=runtime,
        suspected_vfr=veredicto.suspected_vfr,
        short_decode=short_decode,
        frames_expected=estimado,
    )


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------


def _resolver_modelo(nombre: str) -> Path:
    """
    Ubica los pesos.

    Se hashea el archivo, asi que hace falta la ruta real y no alcanza con el nombre: dos
    checkpoints distintos con el mismo nombre producirian caches incomparables.
    """
    p = Path(nombre)
    if p.is_file():
        return p.resolve()
    for base in (Path.cwd(), Path.cwd().parent):
        cand = base / nombre
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"no se encontro el checkpoint {nombre!r}. Pasar la ruta completa con --model."
    )


def _cargar_modelo(path: Path, cfg: PreprocessConfig):
    from ultralytics import YOLO

    modelo = YOLO(str(path))
    return modelo


def _reiniciar_tracker(modelo) -> None:
    """
    Vacia el estado del tracker sin recrear el modelo.

    Recrear el modelo costaria cargar los pesos de nuevo en cada corte, y en una transmision
    profesional son decenas por round. `reset()` existe en BYTETracker y en BOTSORT; si una
    version futura lo saca, se avisa en vez de seguir en silencio con la identidad arrastrada.
    """
    trackers = getattr(getattr(modelo, "predictor", None), "trackers", None) or []
    if not trackers:
        return
    for t in trackers:
        reset = getattr(t, "reset", None)
        if reset is None:
            raise RuntimeError(
                "el tracker de esta version de ultralytics no tiene reset(); sin eso la "
                "identidad se arrastra a traves de los cortes de plano. Correr con "
                "--no-detect-cuts si se acepta ese riesgo."
            )
        reset()


def _inferir(modelo, frame, cfg: PreprocessConfig) -> tuple[FrameDetections, int]:
    """
    Corre deteccion y tracking sobre un cuadro.

    Solo se persisten las detecciones que el tracker le puso id. Una deteccion sin track no
    se puede asignar a un peleador ni encadenar con el cuadro siguiente, asi que en el
    cache seria ruido; se cuentan aparte para que el descarte sea medible y no invisible.
    """
    res = modelo.track(
        frame,
        persist=True,
        tracker=str(cfg.tracker),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        half=cfg.half,
        verbose=False,
    )[0]

    cajas = res.boxes
    if cajas is None or len(cajas) == 0:
        return FrameDetections.empty(), 0

    if cajas.id is None:
        return FrameDetections.empty(), int(len(cajas))

    ids = cajas.id.cpu().numpy().astype(np.int32)
    xyxy = cajas.xyxy.cpu().numpy().astype(np.float32)
    conf = cajas.conf.cpu().numpy().astype(np.float32)

    kp = res.keypoints
    n = ids.shape[0]
    if kp is None:
        xy = np.zeros((n, N_KEYPOINTS, 2), np.float32)
        sc = np.zeros((n, N_KEYPOINTS), np.float32)
    else:
        xy = kp.xy.cpu().numpy().astype(np.float32)
        sc = (
            kp.conf.cpu().numpy().astype(np.float32)
            if kp.conf is not None
            else np.ones((n, N_KEYPOINTS), np.float32)
        )

    return FrameDetections(track_id=ids, bbox=xyxy, det_conf=conf, keypoints=xy, kp_score=sc), 0


def _armar_meta(
    *, video: Path, info: VideoProbe, sha_video: str, veredicto, cfg: PreprocessConfig,
    modelo_path: Path, sha_modelo: str, tracker_text: str, hash_cfg: str, runtime: float,
    n_untracked: int, n_errores: int, proxy_path: Path | None, estimado: int,
    short_decode: bool, total_frames: int,
) -> dict[str, Any]:
    import torch
    import ultralytics
    import yaml

    st = video.stat()
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}

    return {
        "format_version": META_FORMAT_VERSION,
        "kind": "boxtwin.pose_meta",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "app_version": __version__,
        "keypoint_format": "coco17",
        "video": {
            "path": str(video),
            "sha256": sha_video,
            "size_bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds"),
            "width": info.width,
            "height": info.height,
            "codec": info.codec,
            "fps": veredicto.fps,
            "fps_rational": str(info.fps_rational),
            "fps_declared": info.fps_declared,
            "fps_source": veredicto.source.value,
            "fps_from_count": veredicto.fps_from_count,
            "fps_relative_error": veredicto.relative_error,
            "suspected_vfr": veredicto.suspected_vfr,
            "duration_s": info.duration_s,
            "total_frames": veredicto.total_frames,
            "total_frames_declared": info.nb_frames_declared,
        },
        "inference": {
            **cfg_dict,
            "model_path": str(modelo_path),
            "model_sha256": sha_modelo,
            "tracker_config": yaml.safe_load(tracker_text),
            "tracker_config_sha256": sha256_bytes(tracker_text.encode("utf-8")),
            "config_hash": hash_cfg,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "counts": {
            "n_detections": 0,  # lo completa el llamador leyendo el npz final
            "n_frames": veredicto.total_frames,
            "n_frames_decode_error": n_errores,
            "untracked_dropped": n_untracked,
        },
        "decode": {
            "frames_expected": estimado,
            "frames_decoded": total_frames,
            "short_decode": short_decode,
        },
        "proxy": (
            {
                "path": str(proxy_path),
                "width": cfg.proxy_width,
                "gop": cfg.proxy_gop,
                "crf": cfg.proxy_crf,
                "preset": cfg.proxy_preset,
            }
            if proxy_path
            else None
        ),
        "runtime_s": round(runtime, 2),
    }


def _escribir_json(path: Path, data: dict[str, Any]) -> None:
    import os

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
