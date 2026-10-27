"""
BoxTwin - El producto: de un video crudo a una Fight-Card, sin proyecto de anotacion.

El resto del paquete es la herramienta con la que se arma el dataset. Esto es lo que ve un
entrenador: subir el video, decir cual peleador es cual, y recibir el volumen de golpes
detectados con cada evento enlazado a su instante.

La separacion no es cosmetica. El anotador puede pedir que una persona resuelva lo que el
sistema no sabe; el producto no, salvo en el unico paso donde esta declarado que si (la
siembra de identidad). Todo lo que aca se abstiene tiene que salir dicho en la Fight-Card.
"""

from __future__ import annotations

__all__ = ["ESTADOS", "Sesion", "VERSION_FIGHTCARD"]

from boxtwin.mvp.sesion import ESTADOS, Sesion
from boxtwin.mvp.fightcard import VERSION_FIGHTCARD
