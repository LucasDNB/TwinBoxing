"""
BoxTwin - Particiones para evaluar el detector.

POR QUE EXISTE
  Sobre video continuo, una particion al azar por cuadro no mide nada: el cuadro n y el n+1
  son casi la misma imagen, asi que la validacion queda dentro del entrenamiento y el numero
  sale alto por construccion. Ya se pago una version de este error en el proyecto: el
  baseline publico de 84,51% comparte el 96% de sus sujetos entre train y val porque su
  split es por clip y no por sujeto.

  Dos particiones, que miden cosas distintas y las dos hacen falta:

  - DEJANDO UNA FUENTE AFUERA. Es la que importa. El clasificador aprendio en distribucion y
    quedo en o por debajo de su linea de base en los tres folds cruzados, y esa brecha es el
    problema abierto del proyecto. Si el detector no se mide asi, no se entera.

  - EN DISTRIBUCION, sobre un tramo continuo de la misma fuente. No dice nada sobre
    generalizar, pero dice si el modelo aprende algo, que es otra pregunta y hay que poder
    responderla por separado cuando la primera da mal.

  La particion en distribucion es POR TRAMO CONTINUO y con una banda muerta en el medio. Sin
  la banda, una ventana centrada cerca del corte ve cuadros de los dos lados y vuelve a
  filtrar. El ancho de la banda tiene que ser al menos el de la ventana del modelo.

QUE HACE
  Arma los folds. No mueve datos: devuelve indices de cuadro y nombres de fuente.

USO
  folds = leave_one_source_out(["sparring-3", "Sparring", "pacquiao"])
  tr, va = en_distribucion(T=17399, fraccion=0.25, banda=60)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["Fold", "leave_one_source_out", "en_distribucion", "BANDA_POR_DEFECTO"]

# 2 segundos a 30 fps. Tiene que cubrir la ventana del modelo, o la validacion filtra.
BANDA_POR_DEFECTO = 60


@dataclass(frozen=True)
class Fold:
    nombre: str
    train: tuple[str, ...]
    val: tuple[str, ...]


def leave_one_source_out(fuentes: Sequence[str]) -> list[Fold]:
    """Un fold por fuente, con esa fuente entera como validacion."""
    fuentes = list(fuentes)
    if len(fuentes) < 2:
        raise ValueError("hacen falta al menos dos fuentes para dejar una afuera")
    return [
        Fold(nombre=f"sin-{f}", train=tuple(x for x in fuentes if x != f), val=(f,))
        for f in fuentes
    ]


def en_distribucion(
    T: int, fraccion: float = 0.25, banda: int = BANDA_POR_DEFECTO
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mascaras (train, val) sobre una sola fuente, con la validacion en la cola.

    La cola y no un tramo del medio porque parte el video en dos y no en tres, y cada corte
    cuesta una banda muerta. La banda se descuenta del entrenamiento, que es el lado que
    puede permitirselo.
    """
    if not 0.0 < fraccion < 1.0:
        raise ValueError(f"fraccion tiene que estar en (0, 1), no {fraccion}")
    if banda < 0:
        raise ValueError("la banda no puede ser negativa")
    n_val = int(round(T * fraccion))
    if n_val < 1 or T - n_val - banda < 1:
        raise ValueError(
            f"con T={T}, fraccion={fraccion} y banda={banda} no queda entrenamiento"
        )
    train = np.zeros(T, bool)
    val = np.zeros(T, bool)
    corte = T - n_val
    train[: max(corte - banda, 0)] = True
    val[corte:] = True
    return train, val
