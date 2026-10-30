"""
BoxTwin - Relleno temporal de keypoints bajo umbral (E10).

POR QUE EXISTE
  Un keypoint con score bajo el umbral saca el cuadro entero de la mascara de `features_de`,
  y a sus vecinos tambien, porque las derivadas cruzan el agujero. En un golpe la muneca se
  borronea justo en la extension: el cuadro que mas importa es el que mas se tira. E10
  (docs/specs/2026-10-30-mejoras-post-e8/02-e10-relleno-temporal.md) prueba rellenar los
  huecos cortos por interpolacion en vez de enmascararlos.

  El tope es 0,10 s y se fijo antes de mirar los huecos: un golpe de brazo extendido dura
  del orden de 0,3 a 0,5 s, e interpolar un hueco que tape la extension entera borra el
  golpe (la muneca iria en linea recta de guardia a guardia).

QUE HACE
  Por articulacion de las que usan las features (nariz, hombros, codos, munecas, caderas),
  sobre la serie de un peleador en el fps original:

  1. Un hueco es una corrida de cuadros con score bajo el umbral de esa articulacion, con un
     cuadro aceptado justo antes y otro justo despues, y con la deteccion del peleador
     presente en todos (los extremos incluidos).
  2. Si dura a lo sumo floor(0,10 s * fps) cuadros, las coordenadas se interpolan
     linealmente entre los extremos y el score queda en el umbral, o sea aceptado.
  3. Huecos mas largos o abiertos quedan como estaban.
  4. Las coordenadas medidas no se tocan nunca: solo cambian los cuadros del hueco, y se
     marcan en `rellenado`.

  Corre antes de `features_de`; las features, las derivadas y la mascara se calculan sobre la
  serie rellenada sin ningun otro cambio.

USO
  kp, sc, rellenado = rellenar(kp, sc, umbrales, fps, presente=valid)
"""

from __future__ import annotations

import numpy as np

from boxtwin_detector.features import resolver_umbrales

__all__ = ["MAX_S", "INDICES", "maximo_cuadros", "rellenar"]

MAX_S = 0.10

# COCO17: izquierda y derecha comparten umbral, pero cada keypoint tiene sus propios huecos.
INDICES = {"nariz": (0,), "hombro": (5, 6), "codo": (7, 8), "muneca": (9, 10), "cadera": (11, 12)}


def maximo_cuadros(fps: float, max_s: float = MAX_S) -> int:
    """El largo maximo de un hueco rellenable, redondeado hacia abajo (3 a 30 fps, 6 a 60)."""
    # el 1e-9 es para que 0,1 * 30 = 3,0000000000000004 no dependa del redondeo de punto flotante
    return int(np.floor(max_s * fps + 1e-9))


def _corridas(x: np.ndarray) -> list[tuple[int, int]]:
    """[inicio, fin) de cada corrida de True."""
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def rellenar(
    kp: np.ndarray, sc: np.ndarray, umbrales, fps: float, max_s: float = MAX_S,
    presente: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    kp (T, K, 2), sc (T, K) de un peleador; `presente` (T,) si hay deteccion asignada.

    Devuelve copias de kp y sc con los huecos cortos rellenados, y `rellenado` (T, K) bool.
    """
    u = resolver_umbrales(umbrales)
    kp = np.array(kp, copy=True)
    sc = np.array(sc, copy=True)
    T = len(kp)
    presente = np.ones(T, bool) if presente is None else np.asarray(presente, bool)
    rellenado = np.zeros(sc.shape, bool)
    tope = maximo_cuadros(fps, max_s)
    if tope < 1:
        return kp, sc, rellenado

    for nombre, indices in INDICES.items():
        umbral = u[nombre]
        # en float32 el umbral puede redondear por debajo de si mismo y el cuadro rellenado
        # quedaria rechazado: se toma el menor valor representable que sigue aceptado
        puesto = sc.dtype.type(umbral)
        if float(puesto) < umbral:
            puesto = np.nextafter(puesto, sc.dtype.type(np.inf))
        for j in indices:
            aceptado = (sc[:, j].astype(np.float64) >= umbral) & presente
            for a, b in _corridas(~aceptado):
                if a == 0 or b == T or b - a > tope or not presente[a:b].all():
                    continue
                t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
                kp[a:b, j] = kp[a - 1, j] + t[:, None] * (kp[b, j] - kp[a - 1, j])
                sc[a:b, j] = puesto
                rellenado[a:b, j] = True
    return kp, sc, rellenado
