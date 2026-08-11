"""
BoxTwin - Generacion del proxy de baja resolucion.

POR QUE EXISTE
  Medido sobre los videos del proyecto, decodificar 4K a un hilo da 41 fps y un cuadro
  3840x2160 en BGR ocupa 24,9 MB. Con eso, un buffer circular de 40 cuadros ya es 1 GB de
  RAM y retroceder cuadro a cuadro es inviable. El mismo material a 960 px decodifica a
  919 fps y ocupa 1,55 MB por cuadro. El proxy no es una optimizacion, es lo que hace
  posible el reproductor.
  El GOP corto es el otro parametro que importa: con GOP 12 el peor caso de un salto hacia
  atras decodifica 11 cuadros en vez de 249. Cuesta 42% mas de tamano y lo vale.

  El costo de generarlo es cero en la practica: se lanza en paralelo a la inferencia, que
  es GPU-bound. Medido, el proxy de un video 1080p de 24.000 cuadros son 74 s contra 830 s
  del pase de pose, asi que se esconde entero debajo.

QUE HACE
  Lanza ffmpeg en segundo plano y avisa cuando termino. Escribe a un temporal y renombra
  al final, para que un proxy a medias nunca parezca completo.

USO
  job = ProxyJob(src, dst, width=960, gop=12).start()
  ...  # correr la inferencia mientras tanto
  job.wait()
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from boxtwin.core.video import require_ffmpeg

__all__ = ["ProxyJob", "ProxyError"]


class ProxyError(RuntimeError):
    pass


@dataclass
class ProxyJob:
    src: Path
    dst: Path
    width: int = 960
    gop: int = 12
    crf: int = 20
    preset: str = "veryfast"

    _proc: subprocess.Popen | None = None
    _tmp: Path | None = None

    def already_done(self) -> bool:
        return self.dst.is_file() and self.dst.stat().st_size > 0

    def start(self) -> ProxyJob:
        """Lanza ffmpeg y vuelve enseguida. Si el proxy ya existe, no hace nada."""
        if self.already_done():
            return self
        require_ffmpeg()
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.dst.with_name(self.dst.name + ".tmp.mp4")

        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(self.src),
            # min(w, iw) para no ampliar: si el video ya es mas chico que el ancho del
            # proxy, escalarlo hacia arriba gasta espacio y tiempo sin agregar un pixel de
            # informacion. -2 mantiene la altura par, que h264 exige.
            "-vf", f"scale=w='min({self.width},iw)':h=-2",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-g", str(self.gop),
            "-pix_fmt", "yuv420p",
            "-an",
            str(self._tmp),
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait(self) -> bool:
        """Espera a que termine. Devuelve True si el proxy quedo escrito."""
        if self._proc is None:
            return self.already_done()

        _, err = self._proc.communicate()
        if self._proc.returncode != 0:
            if self._tmp:
                self._tmp.unlink(missing_ok=True)
            raise ProxyError(
                f"ffmpeg fallo generando el proxy de {self.src.name}: "
                f"{err.decode('utf-8', 'replace').strip()}"
            )
        if self._tmp:
            self._tmp.replace(self.dst)
        return True

    def cancel(self) -> None:
        """Mata el proceso y limpia el temporal. Se llama si el preproceso aborta."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._tmp:
            self._tmp.unlink(missing_ok=True)
