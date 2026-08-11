"""
BoxTwin - Entrega de cuadros al navegador.

POR QUE EXISTE
  El navegador no puede navegar un video cuadro a cuadro. El elemento <video> de HTML5
  busca por tiempo, no por numero de cuadro, y con un fps de 30000/1001 el cuadro que
  devuelve un salto no es el que se pidio. Como el producto de esta herramienta son
  fronteras temporales medidas en cuadros, esa imprecision no es un detalle: es el dato.
  Por eso el servidor manda cuadros sueltos, indexados por numero, y el navegador nunca
  decodifica video.

  Medido sobre el proxy a 960 px de este proyecto: decodificar cuesta ~1 ms y comprimir a
  JPEG con calidad 85 cuesta 1,2 ms mas, con 48 KB por cuadro. A 30 cuadros por segundo son
  11 Mbit/s, que en una red local sobra; a 0,25x, que es la velocidad de anotacion, son
  2,8 Mbit/s. WebP se descarto: 29 ms por cuadro, veinticinco veces mas caro.

QUE HACE
  Decodifica del proxy, comprime a JPEG y cachea el resultado comprimido, que es lo caro y
  lo que se vuelve a pedir al ir y venir sobre una frontera.

USO
  render = FrameRenderer(session)
  cuerpo = render.jpeg(1234)
"""

from __future__ import annotations

from boxtwin.server.ringbuffer import FrameRing
from boxtwin.server.session import Session

__all__ = ["FrameRenderer", "DEFAULT_QUALITY"]

# 85 es el punto donde el artefacto de compresion deja de verse sobre el guante, que es lo
# que hay que juzgar. Bajar a 60 ahorra 40% de bytes y empieza a ensuciar los bordes.
DEFAULT_QUALITY = 85

# Cuantos cuadros comprimidos se guardan. Un JPEG son ~48 KB, asi que 600 cuadros son
# 29 MB: barato comparado con volver a decodificar y comprimir.
JPEG_CACHE_FRAMES = 600


class FrameRenderer:
    """Convierte numeros de cuadro en JPEG, con cache de lo ya comprimido."""

    def __init__(
        self, session: Session, *, quality: int = DEFAULT_QUALITY, stamp: bool = False
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.session = session
        self.quality = quality
        # Modo diagnostico: escribe el numero de cuadro sobre la propia imagen. Es la unica
        # forma de saber, mirando una captura de pantalla, de que cuadro es lo que se esta
        # viendo, con independencia de lo que crea el cliente. Sin esto, un desfasaje entre
        # imagen y esqueleto solo se puede discutir de memoria.
        self.stamp = stamp
        # El cache guarda bytes ya comprimidos y reusa la politica de desalojo por
        # distancia al cursor, que es la que aguanta el ir y venir sobre una frontera.
        self._cache = FrameRing(JPEG_CACHE_FRAMES)
        self.encoded = 0
        self.hits = 0

    def jpeg(self, frame: int, *, quality: int | None = None) -> bytes | None:
        """JPEG del cuadro, o None si esta fuera del video."""
        if not 0 <= frame < self.session.total_frames:
            return None

        q = quality or self.quality
        if q == self.quality:
            cacheado = self._cache.get(frame)
            if cacheado is not None:
                self.hits += 1
                return bytes(cacheado)

        img = self.session.source.frame(frame)
        if img is None:
            return None
        if self.stamp:
            img = self._estampar(img, frame)

        ok, buf = self._cv2.imencode(".jpg", img, [self._cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        self.encoded += 1

        datos = buf.tobytes()
        if q == self.quality:
            self._cache.set_cursor(frame)
            self._cache.put(frame, buf)
        return datos

    def hires_jpeg(self, frame: int, *, quality: int = 92) -> bytes | None:
        """
        Cuadro en resolucion original, para el zoom con la reproduccion detenida.

        No se cachea: se pide de a uno, cuando el anotador se detiene a mirar de cerca.
        Si no esta el video original, devuelve None y el cliente se queda con el proxy.
        """
        fuente = self.session.hires()
        if fuente is None:
            return None
        img = fuente.frame(frame)
        if img is None:
            return None
        ok, buf = self._cv2.imencode(".jpg", img, [self._cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def _estampar(self, img, frame: int):
        """Escribe el numero de cuadro sobre la imagen, para el modo diagnostico."""
        cv2 = self._cv2
        copia = img.copy()
        texto = f"IMG {frame}"
        cv2.rectangle(copia, (0, 0), (200, 44), (0, 0, 0), -1)
        cv2.putText(copia, texto, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        return copia

    def stats(self) -> dict[str, int]:
        return {
            "encoded": self.encoded,
            "cache_hits": self.hits,
            "cached_frames": len(self._cache),
            "decoder_seeks": self.session.source.seeks,
            "decoded_frames": self.session.source.decoded,
        }
