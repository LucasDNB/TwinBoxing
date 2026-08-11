"""
BoxTwin - Keypoints derivados de guante.

POR QUE EXISTE
  COCO17 termina en la muneca. Para juzgar si un golpe conecto hace falta saber donde esta
  el punio, que en un guante de 16 onzas queda casi diez centimetros mas alla, y la muneca
  es ademas el keypoint menos confiable justo en el instante del golpe, cuando el
  desenfoque de movimiento es mayor.
  Un modelo whole-body no resuelve esto: sus keypoints de mano localizan nudillos y puntas
  de dedos, y un guante es un bulto liso sin dedos. Devolveria 21 puntos con confianza alta
  y posicion arbitraria, que es el peor modo de falla posible.

  Limitacion que hay que decir de frente: este punto derivado es una funcion determinista
  del codo y la muneca, asi que para un clasificador que ya recibe los dos **no aporta
  absolutamente nada**. Sirve para dos cosas reales y ninguna es el modelo: que el anotador
  vea la posicion estimada del guante mientras juzga si el golpe llego, y alimentar los
  heuristicos de conexion, que se reportan como estimacion con margen. Un detector de
  guantes de verdad si seria observacion nueva, y queda como trabajo aparte.

QUE HACE
  Extrapola el guante sobre el vector codo-muneca y le asigna una confianza que no puede
  superar la de sus insumos. Se calcula al leer y nunca se escribe en el npz: el cache
  guarda lo que el modelo observo y nada mas, asi que cambiar k no obliga a reprocesar.

USO
  from boxtwin.core.gloves import derive_gloves, append_gloves
  xy19, sc19 = append_gloves(keypoints, scores, k=0.35)
"""

from __future__ import annotations

import numpy as np

from boxtwin.core.constants import (
    GLOVE_LEFT_IDX,
    GLOVE_RIGHT_IDX,
    LEFT_ELBOW_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ELBOW_IDX,
    RIGHT_WRIST_IDX,
)

__all__ = ["DEFAULT_K", "derive_gloves", "append_gloves"]

# Fraccion del antebrazo que se extiende mas alla de la muneca hasta el centro del guante.
# Sale de geometria de mano y guante (unos 9 cm sobre un antebrazo de 26), NO de una
# medicion. Hay que calibrarlo contra detecciones reales de guante cuando existan.
DEFAULT_K = 0.35


def derive_gloves(
    keypoints: np.ndarray, scores: np.ndarray, k: float = DEFAULT_K
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los dos guantes de cada deteccion.

    keypoints (n, 17, 2) y scores (n, 17) entran; salen (n, 2, 2) y (n, 2), con el guante
    izquierdo en el indice 0 y el derecho en el 1.

    La confianza es el minimo entre codo y muneca: no se puede estar mas seguro de una
    extrapolacion que de los puntos con los que se extrapola.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError(f"keypoints tiene forma {keypoints.shape}, se esperaba (n, 17, 2)")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(f"scores tiene forma {scores.shape}, se esperaba {keypoints.shape[:2]}")

    n = keypoints.shape[0]
    xy = np.empty((n, 2, 2), np.float32)
    sc = np.empty((n, 2), np.float32)

    for salida, codo, muneca in (
        (0, LEFT_ELBOW_IDX, LEFT_WRIST_IDX),
        (1, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX),
    ):
        w = keypoints[:, muneca, :]
        e = keypoints[:, codo, :]
        xy[:, salida, :] = w + k * (w - e)
        sc[:, salida] = np.minimum(scores[:, muneca], scores[:, codo])

    return xy, sc


def append_gloves(
    keypoints: np.ndarray, scores: np.ndarray, k: float = DEFAULT_K
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve los 17 keypoints observados mas los 2 guantes derivados, en los indices 17 y 18.

    El orden coincide con el formato coco17+gloves que declara el esquema, para que el
    overlay y los exports usen los mismos indices.
    """
    xy, sc = derive_gloves(keypoints, scores, k)
    completos_xy = np.concatenate([keypoints, xy], axis=1)
    completos_sc = np.concatenate([scores, sc], axis=1)
    assert completos_xy.shape[1] == GLOVE_RIGHT_IDX + 1 == 19
    assert GLOVE_LEFT_IDX == 17
    return completos_xy, completos_sc
