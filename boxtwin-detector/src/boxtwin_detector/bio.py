"""
BoxTwin - Segmentos y etiquetas BIO.

POR QUE EXISTE
  El export `sequence` escribe una etiqueta por cuadro en un espacio BIO de 13 clases: O mas
  B e I por cada una de las 6 familias. El detector no usa ese espacio, por dos razones que
  se miden:

  1. B se escribe en UN SOLO cuadro por evento. En el espacio de 13, B-uppercut-lead tiene
     tantos cuadros en todo el video como uppercuts lead haya: entre 5 y 12. No se aprende
     una clase de ocho ejemplos de un cuadro cada uno.
  2. La familia del golpe es el eje dificil y esta medido tres veces que no es separable en
     pose monocular. Mezclarla con la deteccion hace que un fracaso no se pueda atribuir.

  Se colapsa entonces a O / B / I. B se conserva, aunque cueste un cuadro por golpe, porque
  sin ella dos golpes pegados del mismo brazo se leen como uno solo largo. Medido sobre
  sparring-3: pasa 1 vez en 400 eventos, asi que es barato y cubre el caso raro.

QUE HACE
  Convierte etiquetas BIO por cuadro en segmentos y al reves. Trabajar con segmentos es lo
  que permite remuestrear a otro fps sin perder la B, que un submuestreo ingenuo tira.

USO
  segs = segmentos_de(labels[p, carril])
  nuevas = etiquetas_de(segs, T)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Segmento", "FUERA", "B", "I", "segmentos_de", "etiquetas_de", "es_b"]

FUERA, B, I = 0, 1, 2


@dataclass(frozen=True)
class Segmento:
    """Un golpe en un carril, con cuadros inclusivos. clase es el indice en el espacio original."""

    inicio: int
    fin: int
    clase: int

    def __post_init__(self) -> None:
        if self.fin < self.inicio:
            raise ValueError(f"segmento invertido: {self.inicio} > {self.fin}")

    @property
    def largo(self) -> int:
        return self.fin - self.inicio + 1


def es_b(etiqueta: int) -> bool:
    """En el espacio del export, B son los impares: _bio(idx) = (1 + 2*idx, 2 + 2*idx)."""
    return etiqueta != FUERA and etiqueta % 2 == 1


def segmentos_de(carril: np.ndarray) -> list[Segmento]:
    """
    Segmentos de un carril BIO del export, en su espacio de 13 clases.

    Un segmento arranca en cada B y se extiende mientras haya I. Una I sin B previa no la
    escribe el export, pero si aparece se abre segmento igual y no se pierde el golpe.
    """
    segs: list[Segmento] = []
    T = len(carril)
    f = 0
    while f < T:
        e = int(carril[f])
        if e == FUERA:
            f += 1
            continue
        clase = (e - 1) // 2
        inicio = f
        f += 1
        while f < T and carril[f] != FUERA and not es_b(int(carril[f])):
            f += 1
        segs.append(Segmento(inicio, f - 1, clase))
    return segs


def etiquetas_de(segmentos: list[Segmento], T: int) -> np.ndarray:
    """
    Etiquetas O/B/I por cuadro. Los segmentos se asumen ordenados y sin solaparse.

    Un segmento que quede fuera de [0, T) se recorta; si queda vacio, se descarta. Pasa al
    remuestrear el ultimo golpe de un video.
    """
    out = np.zeros(T, np.int8)
    for s in segmentos:
        a, b = max(s.inicio, 0), min(s.fin, T - 1)
        if a > b:
            continue
        out[a] = B
        if b > a:
            out[a + 1 : b + 1] = I
    return out
