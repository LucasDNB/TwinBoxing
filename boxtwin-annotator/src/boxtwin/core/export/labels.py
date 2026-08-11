"""
BoxTwin - Espacios de clases del export.

POR QUE EXISTE
  El evento guarda tipo, lado y altura como campos independientes, y la clase se arma
  recien aca. Colapsarlos al anotar seria mas comodo y haria imposible auditar: cuando una
  etiqueta esta mal, con la clase colapsada no se sabe si fallo el tipo, el brazo o la
  altura. Con los campos separados el error se localiza en la dimension que lo produjo.

  Dos espacios porque miden cosas distintas, y `side` es el default. Es el lado observado
  y no tiene ningun paso de derivacion: lo que el anotador vio es lo que se exporta. El
  conjunto de 6 queda straight-left, straight-right, hook-left, hook-right, uppercut-left,
  uppercut-right.

  `lead-rear` sigue disponible y es el espacio tactico que usan los datasets del area, pero
  depende de la guardia, que la pone el anotador y puede estar mal o cambiar a mitad del
  combate. Un error de guardia intercambia sistematicamente dos clases sin que nada lo
  delate. En ese espacio el conjunto de 6 coincide con las de BoxingVI (Jab, Cross,
  Lead/Rear Hook, Lead/Rear Uppercut), asi que es el que sirve para concatenar con lo que ya
  esta cortado de ese dataset.

  Solo entran los golpes completos a los espacios de 6 y 12. Un amague no es un recto mal
  hecho: es otra cosa, y meterlo en la clase recto ensucia justamente los ejemplos que
  definen la frontera. En el espacio de 14 tiene clase propia, que es donde corresponde.

QUE HACE
  Arma la lista ordenada de clases de cada espacio y traduce un evento a su indice.

USO
  clases = class_list(LabelSpace.LEAD_REAR, 12)
  idx = class_index(evento, LabelSpace.LEAD_REAR, 12)
"""

from __future__ import annotations

from enum import Enum
from itertools import product

from boxtwin.core.schema import Event
from boxtwin.core.types import ArmRole, Completeness, PunchType, Side

__all__ = [
    "LabelSpace",
    "CLASS_SETS",
    "BACKGROUND",
    "FEINT",
    "class_list",
    "class_name",
    "class_index",
]

BACKGROUND = "background"
FEINT = "feint"

CLASS_SETS = (6, 12, 14)


class LabelSpace(str, Enum):
    SIDE = "side"
    LEAD_REAR = "lead-rear"


def _eje(space: LabelSpace) -> tuple[str, ...]:
    return tuple(s.value for s in (Side if space is LabelSpace.SIDE else ArmRole))


def class_list(space: LabelSpace, classes: int) -> list[str]:
    """
    Clases en orden fijo. El indice es la etiqueta numerica, asi que el orden es contrato:
    si cambiara, los modelos ya entrenados quedarian con las clases permutadas.
    """
    if classes not in CLASS_SETS:
        raise ValueError(f"conjunto de clases invalido: {classes}, se esperaba {CLASS_SETS}")

    tipos = tuple(t.value for t in PunchType)
    eje = _eje(space)

    if classes == 6:
        return [f"{t}-{e}" for t, e in product(tipos, eje)]

    con_altura = [f"{t}-{e}-{a}" for t, e, a in product(tipos, eje, ("head", "body"))]
    if classes == 12:
        return con_altura
    # 14 = las 12 mas amague y fondo, en ese orden y al final para que agregar el espacio
    # de 14 no renumere las 12 anteriores.
    return [*con_altura, FEINT, BACKGROUND]


def class_name(ev: Event, space: LabelSpace, classes: int) -> str | None:
    """
    Clase del evento, o None si no pertenece a este espacio.

    Los abortados quedan afuera de todos los espacios: son movimientos que no llegaron a
    ser un golpe ni a ser un amague completo, y no describen ninguna categoria.
    """
    if ev.completeness is Completeness.ABORTED:
        return None
    if ev.completeness is Completeness.FEINT:
        return FEINT if classes == 14 else None

    eje = ev.side.value if space is LabelSpace.SIDE else ev.arm_role.value
    if classes == 6:
        return f"{ev.punch_type.value}-{eje}"
    return f"{ev.punch_type.value}-{eje}-{ev.target.value}"


def class_index(ev: Event, space: LabelSpace, classes: int) -> int | None:
    nombre = class_name(ev, space, classes)
    if nombre is None:
        return None
    return class_list(space, classes).index(nombre)
