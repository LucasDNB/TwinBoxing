"""
BoxTwin - De probabilidades por cuadro a golpes.

POR QUE EXISTE
  La salida del modelo son tres numeros por cuadro. Eso todavia no es una lista de golpes, y
  el paso que falta no es cosmetico: es donde se ELIGE el punto de operacion.

  El disparador heuristico no tenia esta perilla, y por eso quedo clavado en 21% de precision
  sin forma de canjear. Un detector con umbral se mueve por la curva, y reportar un solo
  punto de esa curva es reportar una eleccion, no un resultado. Por eso `barrer` devuelve la
  curva entera.

  Los tres parametros salen de lo medido y no de la intuicion:

  - LARGO MINIMO 5. Los golpes duran 7 cuadros de mediana con p10 en 5. Un tramo activo de
    dos cuadros no es un golpe corto, es ruido.
  - HUECO MAXIMO 2. Un bajon de un cuadro en el medio de un golpe es un error de pose, no el
    final del golpe.
  - CORTE POR B, apagado por defecto. Sirve para separar dos golpes seguidos del MISMO brazo,
    y los carriles ya son por brazo: medido, eso pasa 1 vez en 400 eventos. La B se sigue
    aprendiendo igual porque afila el inicio durante el entrenamiento, que es otra cosa que
    usarla para cortar.

QUE HACE
  logits (T, 3) -> lista de Segmento, y el barrido del umbral.

USO
  segs = decodificar(logits, umbral=0.5)
  curva = barrer(logits, umbrales=np.arange(0.1, 0.95, 0.05))
"""

from __future__ import annotations

import numpy as np

from boxtwin_detector.bio import Segmento

__all__ = ["decodificar", "decodificar_score", "probabilidad_de_golpe", "LARGO_MINIMO",
           "HUECO_MAXIMO"]

LARGO_MINIMO = 5
HUECO_MAXIMO = 2


def probabilidad_de_golpe(logits: np.ndarray) -> np.ndarray:
    """P(B) + P(I) por cuadro, o sea 1 - P(O)."""
    x = np.asarray(logits, np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return (p[:, 1] + p[:, 2]).astype(np.float32)


def _tramos(activo: np.ndarray) -> list[list[int]]:
    if not activo.any():
        return []
    d = np.diff(activo.astype(np.int8))
    ini = ([0] if activo[0] else []) + list(np.where(d == 1)[0] + 1)
    fin = list(np.where(d == -1)[0]) + ([len(activo) - 1] if activo[-1] else [])
    return [[int(a), int(b)] for a, b in zip(ini, fin)]


def decodificar_score(
    score: np.ndarray,
    umbral: float,
    largo_minimo: int = LARGO_MINIMO,
    hueco_maximo: int = HUECO_MAXIMO,
    valido: np.ndarray | None = None,
) -> list[Segmento]:
    """
    Segmentos a partir de una senal cualquiera, no necesariamente una probabilidad.

    Existe para poder pasar la HEURISTICA por el mismo decodificador que el modelo. Si la
    linea de base se decodifica distinto, la comparacion mide la diferencia de decodificador
    y no la de detector, que es el error que hace parecer bueno a cualquier modelo nuevo.
    """
    activo = np.asarray(score) >= umbral
    if valido is not None:
        activo = activo & np.asarray(valido, bool)
    tramos = _tramos(activo)
    if not tramos:
        return []
    unidos = [tramos[0]]
    for a, b in tramos[1:]:
        if a - unidos[-1][1] - 1 <= hueco_maximo:
            unidos[-1][1] = b
        else:
            unidos.append([a, b])
    return [Segmento(a, b, 0) for a, b in unidos if b - a + 1 >= largo_minimo]


def decodificar(
    logits: np.ndarray,
    umbral: float = 0.5,
    largo_minimo: int = LARGO_MINIMO,
    hueco_maximo: int = HUECO_MAXIMO,
    valido: np.ndarray | None = None,
    cortar_en_b: float | None = None,
) -> list[Segmento]:
    """
    Segmentos de golpe a partir de los logits.

    `valido` apaga los cuadros donde no se le pregunta al modelo: fuera del tramo anotado el
    detector no tiene por que opinar, y contar sus marcas ahi ensuciaria la precision con
    metraje que nadie miro.
    """
    if not 0.0 < umbral < 1.0:
        raise ValueError(f"umbral tiene que estar en (0, 1), no {umbral}")
    p = probabilidad_de_golpe(logits)
    activo = p >= umbral
    if valido is not None:
        activo = activo & np.asarray(valido, bool)

    tramos = _tramos(activo)
    if not tramos:
        return []

    unidos = [tramos[0]]
    for a, b in tramos[1:]:
        if a - unidos[-1][1] - 1 <= hueco_maximo:
            unidos[-1][1] = b
        else:
            unidos.append([a, b])

    segs: list[Segmento] = []
    for a, b in unidos:
        cortes = [a]
        if cortar_en_b is not None:
            x = np.asarray(logits, np.float64)[a : b + 1]
            x = x - x.max(axis=1, keepdims=True)
            e = np.exp(x)
            pb = (e[:, 1] / e.sum(axis=1)).astype(np.float32)
            for i in range(1, len(pb) - 1):
                if pb[i] >= cortar_en_b and pb[i] >= pb[i - 1] and pb[i] > pb[i + 1]:
                    if a + i - cortes[-1] >= largo_minimo:
                        cortes.append(a + i)
        cortes.append(b + 1)
        for x, y in zip(cortes, cortes[1:]):
            if y - x >= largo_minimo:
                segs.append(Segmento(x, y - 1, 0))
    return segs
