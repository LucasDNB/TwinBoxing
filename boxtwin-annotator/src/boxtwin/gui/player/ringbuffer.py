"""
BoxTwin - Buffer circular de cuadros decodificados.

POR QUE EXISTE
  Retroceder cuadro a cuadro tiene que ser instantaneo, y decodificar hacia atras no
  existe: hay que volver al keyframe anterior y avanzar. Sin cache, cada paso hacia atras
  cuesta un seek mas la decodificacion de medio GOP, y anotar fronteras temporales, que es
  ir y venir sobre los mismos veinte cuadros, se vuelve insoportable.

  El presupuesto es de memoria y no de cantidad de cuadros. Un cuadro 4K en BGR ocupa
  24,9 MB y uno del proxy a 960 px ocupa 1,55 MB: fijar "300 cuadros" daria 7,5 GB en un
  caso y 465 MB en el otro. Por eso la capacidad se calcula del tamano real del cuadro.

  La politica de descarte es por distancia al cursor y no por antiguedad. Con LRU, ir y
  venir sobre una frontera desalojaria justo los cuadros que se estan mirando; por
  distancia, la ventana sigue al cursor sola y no importa en que direccion se mueva.

QUE HACE
  Guarda cuadros indexados por numero, desaloja el mas lejano al cursor cuando se llena y
  reporta que tramo tiene en memoria.

USO
  ring = FrameRing.for_budget(mb=512, frame_nbytes=w * h * 3)
  ring.put(120, img); ring.get(119)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

__all__ = ["FrameRing"]

# Con menos que esto ni siquiera se cubre un paso de 5 cuadros para los dos lados.
MIN_CAPACITY = 16


class FrameRing:
    """Cache de cuadros por numero, con desalojo por distancia al cursor."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("la capacidad tiene que ser al menos 1")
        self.capacity = capacity
        self._data: dict[int, np.ndarray] = {}
        self._cursor = 0

    @classmethod
    def for_budget(cls, *, mb: int, frame_nbytes: int) -> FrameRing:
        """Capacidad derivada de un presupuesto de memoria y del tamano real del cuadro."""
        if frame_nbytes <= 0:
            raise ValueError("frame_nbytes tiene que ser positivo")
        cabidas = (mb * 1024 * 1024) // frame_nbytes
        return cls(max(MIN_CAPACITY, int(cabidas)))

    # -- estado ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, frame: int) -> bool:
        return frame in self._data

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._data))

    @property
    def cursor(self) -> int:
        return self._cursor

    def set_cursor(self, frame: int) -> None:
        """Mueve el centro de la ventana. No desaloja: eso pasa recien al insertar."""
        self._cursor = frame

    def span(self) -> tuple[int, int] | None:
        """Primer y ultimo cuadro en memoria, sin garantia de que el tramo sea continuo."""
        if not self._data:
            return None
        return min(self._data), max(self._data)

    def contiguous_span(self) -> tuple[int, int] | None:
        """
        Tramo continuo alrededor del cursor.

        Es lo que se puede reproducir sin volver a decodificar. Si el cursor no esta en
        memoria, no hay tramo.
        """
        if self._cursor not in self._data:
            return None
        lo = hi = self._cursor
        while lo - 1 in self._data:
            lo -= 1
        while hi + 1 in self._data:
            hi += 1
        return lo, hi

    # -- acceso ------------------------------------------------------------

    def get(self, frame: int) -> np.ndarray | None:
        return self._data.get(frame)

    def put(self, frame: int, img: np.ndarray) -> None:
        self._data[frame] = img
        while len(self._data) > self.capacity:
            # El cuadro recien insertado no se considera para desalojo: se acaba de pedir y
            # tirarlo dejaria el cache inutil. Pero si queda como unico candidato, hay que
            # dejarlo igual, si no el buffer creceria por encima de su presupuesto.
            candidatos = [f for f in self._data if f != frame]
            if not candidatos:
                break
            # Desempate por numero de cuadro menor: con dos equidistantes se conserva el
            # posterior, que es el que se va a pedir si se sigue avanzando. Ademas hace que
            # el desalojo sea deterministico y no dependa del orden de insercion.
            lejano = max(candidatos, key=lambda f: (abs(f - self._cursor), -f))
            del self._data[lejano]

    def clear(self) -> None:
        self._data.clear()

    def discard_outside(self, lo: int, hi: int) -> None:
        """Tira todo fuera de un rango. Se usa al cambiar de video o de fuente."""
        for f in [f for f in self._data if f < lo or f > hi]:
            del self._data[f]
