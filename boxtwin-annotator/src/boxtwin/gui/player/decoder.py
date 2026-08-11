"""
BoxTwin - Decodificador indexado por numero de cuadro.

POR QUE EXISTE
  La navegacion es por numero de cuadro y no por timestamp. Con timestamps, un fps de
  30000/1001 acumula deriva y el cuadro que devuelve un salto deja de ser el que se pidio,
  que en una herramienta cuyo producto son fronteras temporales es inaceptable.
  Verificado sobre los videos del proyecto: el seek por CAP_PROP_POS_FRAMES cae en el
  cuadro exacto pedido. Igual se comprueba al abrir, porque no todos los contenedores se
  portan asi y descubrirlo a mitad de la anotacion seria tarde.

  Se decodifica del proxy y no del original. Medido: 4K a un hilo da 41 fps y el proxy a
  960 px da 919. A 1,0x hace falta un cuadro cada 33 ms y decodificar uno del proxy cuesta
  ~1 ms, por eso no hace falta hilo aparte: decodificar sincronico dentro del bucle de la
  interfaz alcanza de sobra y evita toda una clase de bugs de sincronizacion.

QUE HACE
  Sirve cuadros por numero, usando el buffer circular y decidiendo cuando alcanza con
  seguir leyendo y cuando hay que saltar. En un salto hacia atras rellena tambien los
  cuadros anteriores, que son los que se van a pedir enseguida.

USO
  src = FrameSource(Path("cache/spar.proxy.mp4"), total_frames=53412)
  img = src.frame(1234)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boxtwin.gui.player.ringbuffer import FrameRing

__all__ = ["FrameSource", "DecodeError"]

# Cuantos cuadros previos se rellenan cuando el salto es hacia atras. Con GOP 12 el seek
# ya tuvo que decodificar parte de esto, asi que guardarlo sale casi gratis.
PREFILL_BACK = 48


class DecodeError(RuntimeError):
    pass


class FrameSource:
    """Acceso aleatorio por numero de cuadro, con cache."""

    def __init__(
        self,
        path: Path,
        *,
        total_frames: int,
        buffer_mb: int = 512,
        scale: float = 1.0,
    ) -> None:
        import cv2

        self.path = Path(path)
        self.total_frames = total_frames
        # Factor entre este video y el original. El proxy a 960 sobre un 1920 da 0,5, y es
        # lo que convierte coordenadas de keypoint a pixeles de esta imagen.
        self.scale = scale

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise DecodeError(f"no se pudo abrir {self.path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0  # proximo cuadro que devolveria un read() secuencial

        self.ring = FrameRing.for_budget(mb=buffer_mb, frame_nbytes=self.width * self.height * 3)
        self.seeks = 0
        self.decoded = 0

    # -- ciclo de vida -----------------------------------------------------

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- acceso ------------------------------------------------------------

    def frame(self, n: int) -> np.ndarray | None:
        """
        Cuadro n, o None si esta fuera del video.

        Tres caminos, de mas barato a mas caro: ya esta en memoria, viene justo despues del
        ultimo leido, o hay que saltar.
        """
        if not 0 <= n < self.total_frames:
            return None

        self.ring.set_cursor(n)
        cacheado = self.ring.get(n)
        if cacheado is not None:
            return cacheado

        # Leer hacia adelante sale mas barato que saltar mientras la distancia sea corta.
        if self._next <= n <= self._next + PREFILL_BACK:
            return self._leer_hasta(n)

        desde = max(0, n - PREFILL_BACK) if n < self._next else n
        self._seek(desde)
        return self._leer_hasta(n)

    def prefetch(self, desde: int, cantidad: int) -> int:
        """
        Adelanta la decodificacion de un tramo. Se llama cuando la interfaz esta ociosa.

        Devuelve cuantos cuadros quedaron en memoria de los pedidos.
        """
        puestos = 0
        for f in range(desde, min(desde + cantidad, self.total_frames)):
            if f in self.ring:
                puestos += 1
                continue
            if self.frame(f) is not None:
                puestos += 1
        return puestos

    # -- internos ----------------------------------------------------------

    def _seek(self, n: int) -> None:
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, n)
        pos = int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES))
        if pos != n:
            raise DecodeError(
                f"el salto al cuadro {n} cayo en {pos}. Este contenedor no permite "
                "navegacion exacta por numero de cuadro y no se puede anotar sobre el."
            )
        self._next = n
        self.seeks += 1

    def _leer_hasta(self, n: int) -> np.ndarray | None:
        """Lee secuencialmente hasta n, guardando todo lo que pasa por el camino."""
        img = None
        while self._next <= n:
            ok, cuadro = self._cap.read()
            if not ok:
                return None
            self.ring.put(self._next, cuadro)
            self.decoded += 1
            if self._next == n:
                img = cuadro
            self._next += 1
        return img

    def verify_exact_seek(self, muestras: int = 5) -> bool:
        """
        Comprueba que el salto por numero de cuadro caiga donde dice.

        Se corre al abrir el video. Un contenedor que no lo cumple no sirve para anotar y
        hay que saberlo antes de empezar, no despues de marcar cien eventos.
        """
        if self.total_frames < 2:
            return True
        objetivos = [
            int(self.total_frames * i / (muestras + 1)) for i in range(1, muestras + 1)
        ]
        try:
            for n in objetivos:
                self._seek(n)
        except DecodeError:
            return False
        finally:
            self._seek(0)
        return True
