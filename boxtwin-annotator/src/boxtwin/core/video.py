"""
BoxTwin - Sondeo de metadatos reales del video.

POR QUE EXISTE
  Los contenedores mienten y hay que tratarlos como una hipotesis, no como un dato. En los
  nueve videos del proyecto ninguno declara nb_frames y aparecen cuatro fps nativos
  distintos donde la fuente declaraba uno solo. Un error de fps de 30 contra 29,97 son
  cuatro cuadros de deriva cada dos minutos, suficiente para que las fronteras anotadas
  dejen de caer sobre el golpe.
  El numero de frames tampoco se puede creer. Se verifica decodificando, que es gratis
  porque el preproceso decodifica todo el video igual.

QUE HACE
  Corre ffprobe, expone el fps como fraccion exacta, cuenta frames decodificando y
  reconcilia las dos fuentes. Si el fps racional del contenedor concuerda con el conteo
  real sobre la duracion, se usa el racional porque es exacto; si no concuerda, se usa el
  medido y se marca el video como sospechoso de VFR.
  Tambien calcula el sha256 del archivo, que es la identidad del video en el resto del
  sistema.

USO
  from boxtwin.core.video import probe, sha256_file, reconcile_fps
  info = probe(Path("videos/spar.mp4"))
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from boxtwin.core.types import FpsSource

__all__ = [
    "VideoProbe",
    "FpsVerdict",
    "ProbeError",
    "probe",
    "sha256_file",
    "reconcile_fps",
    "count_frames_by_decoding",
    "require_ffmpeg",
]

# Tolerancia relativa entre el fps racional del contenedor y el medido sobre el conteo
# real. 0,5% deja pasar el error de redondeo de la duracion sin dejar pasar un VFR.
FPS_TOLERANCE = 0.005


class ProbeError(RuntimeError):
    """ffprobe no esta disponible o el archivo no tiene stream de video legible."""


@dataclass(frozen=True)
class VideoProbe:
    """Lo que dice el contenedor. Nada de esto se da por cierto hasta reconciliarlo."""

    path: Path
    width: int
    height: int
    codec: str
    fps_rational: Fraction
    fps_declared: float
    duration_s: float
    nb_frames_declared: int | None

    @property
    def fps_container(self) -> float:
        return float(self.fps_rational)


@dataclass(frozen=True)
class FpsVerdict:
    """Resultado de reconciliar el fps declarado contra el conteo real de frames."""

    fps: float
    source: FpsSource
    total_frames: int
    fps_from_count: float
    relative_error: float
    suspected_vfr: bool


def require_ffmpeg() -> None:
    """Falla temprano y claro si faltan las herramientas, en vez de a mitad del preproceso."""
    faltantes = [b for b in ("ffprobe", "ffmpeg") if shutil.which(b) is None]
    if faltantes:
        raise ProbeError(
            f"faltan binarios en el PATH: {', '.join(faltantes)}. "
            "Instalar ffmpeg antes de preprocesar."
        )


def probe(path: Path) -> VideoProbe:
    """Corre ffprobe y devuelve lo que declara el contenedor, sin interpretarlo."""
    require_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"no existe el archivo: {path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    salida = subprocess.run(cmd, capture_output=True, text=True)
    if salida.returncode != 0:
        raise ProbeError(f"ffprobe fallo sobre {path}: {salida.stderr.strip()}")

    data = json.loads(salida.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ProbeError(f"{path} no tiene stream de video")
    st = streams[0]

    fps_rational = _parse_rational(st.get("r_frame_rate"))
    if fps_rational is None or fps_rational <= 0:
        raise ProbeError(f"{path} no declara un r_frame_rate utilizable: {st.get('r_frame_rate')!r}")

    avg = _parse_rational(st.get("avg_frame_rate"))
    duracion = _first_float(st.get("duration"), (data.get("format") or {}).get("duration"))
    if duracion is None or duracion <= 0:
        raise ProbeError(f"{path} no declara duracion utilizable")

    nb = st.get("nb_frames")
    nb_frames = int(nb) if nb not in (None, "N/A") else None

    return VideoProbe(
        path=path,
        width=int(st["width"]),
        height=int(st["height"]),
        codec=str(st.get("codec_name", "unknown")),
        fps_rational=fps_rational,
        fps_declared=float(avg) if avg else float(fps_rational),
        duration_s=duracion,
        nb_frames_declared=nb_frames,
    )


def reconcile_fps(info: VideoProbe, total_frames: int) -> FpsVerdict:
    """
    Cruza el fps racional del contenedor contra el conteo real de frames.

    Si concuerdan, gana el racional: 30000/1001 es exacto y 29,97 es una aproximacion que
    acumula deriva sobre un video largo. Si no concuerdan, el contenedor esta mintiendo o
    el video es de framerate variable, y ahi se usa el medido y se marca la sospecha.
    """
    if total_frames <= 0:
        raise ValueError("total_frames tiene que ser positivo")

    fps_medido = total_frames / info.duration_s
    error = abs(fps_medido - info.fps_container) / info.fps_container
    concuerda = error <= FPS_TOLERANCE

    return FpsVerdict(
        fps=info.fps_container if concuerda else fps_medido,
        source=FpsSource.CONTAINER_VERIFIED if concuerda else FpsSource.MEASURED,
        total_frames=total_frames,
        fps_from_count=fps_medido,
        relative_error=error,
        suspected_vfr=not concuerda,
    )


def count_frames_by_decoding(path: Path) -> int:
    """
    Cuenta frames decodificando de verdad.

    Solo hace falta cuando se quiere el numero sin correr el preproceso completo. El
    preproceso lleva su propio conteo mientras infiere, asi que ahi este pase no se usa.
    """
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ]
    salida = subprocess.run(cmd, capture_output=True, text=True)
    if salida.returncode != 0:
        raise ProbeError(f"ffprobe -count_frames fallo sobre {path}: {salida.stderr.strip()}")
    texto = salida.stdout.strip()
    if not texto.isdigit():
        raise ProbeError(f"conteo de frames ilegible para {path}: {texto!r}")
    return int(texto)


def sha256_file(path: Path, *, chunk_size: int = 1 << 22) -> str:
    """
    Hash del archivo completo.

    Se paga una sola vez en el preproceso, que ya lee el video entero. Un hash parcial
    seria mas rapido pero no distinguiria dos recortes del mismo material, que es
    exactamente el caso que hay que detectar.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for bloque in iter(lambda: fh.read(chunk_size), b""):
            h.update(bloque)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_rational(value: str | None) -> Fraction | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _first_float(*valores: object) -> float | None:
    for v in valores:
        if v in (None, "N/A"):
            continue
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None
