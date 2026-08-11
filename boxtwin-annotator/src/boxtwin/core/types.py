"""
BoxTwin - Vocabulario cerrado del modelo de anotacion.

POR QUE EXISTE
  La leccion de BoxingVI fue que las etiquetas colapsadas no se pueden auditar. Ese
  dataset entrega "Lead Hook" como un unico string y cuando la etiqueta esta mal no hay
  forma de saber si fallo el tipo de golpe, el brazo o la ventana temporal. Aca cada
  dimension del evento es un enum propio y la agregacion a clases se decide recien en el
  export, asi que un error se localiza en la dimension que lo produjo.
  Que el vocabulario sea cerrado tambien evita el otro problema clasico: dos sesiones de
  anotacion escribiendo "uppercut" y "upper" y descubriendolo al armar el dataset.

QUE HACE
  Define los enums de las dimensiones del evento, de la gestion de identidad y del
  formato de keypoints, mas la unica derivacion del modelo: mano adelantada o atrasada a
  partir de la guardia y el lado.

USO
  from boxtwin.core.types import Guard, Side, arm_role
  arm_role(Guard.SOUTHPAW, Side.RIGHT)   # ArmRole.LEAD
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "FighterId",
    "TrackRole",
    "Side",
    "PunchType",
    "Target",
    "Completeness",
    "Landed",
    "Guard",
    "Quality",
    "ArmRole",
    "UnreliableReason",
    "UnreliableSource",
    "AssignmentOp",
    "InterpolationMethod",
    "ComboOp",
    "KeypointFormat",
    "GloveSource",
    "FpsSource",
    "IssueLevel",
    "arm_role",
]


# Se hereda de str para que el valor serializado sea exactamente el string del esquema
# y para que un enum se pueda comparar contra el JSON crudo sin conversiones.
# No se usa enum.StrEnum porque el paquete declara requires-python >= 3.10.


class FighterId(str, Enum):
    """Los dos peleadores anotables. Es lo unico que puede referenciar un evento."""

    A = "fighter_A"
    B = "fighter_B"


class TrackRole(str, Enum):
    """
    Rol asignable a un track_id sobre un rango de frames.

    `ignore` es explicito y sirve para el arbitro, los que miran y los boxeadores
    pintados en la pared del gimnasio, que YOLOv8-pose detecta como personas.
    `unassigned` no existe como valor almacenable: es la ausencia de assignment y el
    resolver de identidad lo representa con None.
    """

    A = "fighter_A"
    B = "fighter_B"
    IGNORE = "ignore"


class Side(str, Enum):
    """Lado del brazo que lanza. Es lo que el modelo observa directamente."""

    LEFT = "left"
    RIGHT = "right"


class PunchType(str, Enum):
    STRAIGHT = "straight"
    HOOK = "hook"
    UPPERCUT = "uppercut"


class Target(str, Enum):
    HEAD = "head"
    BODY = "body"


class Completeness(str, Enum):
    FULL = "full"
    FEINT = "feint"
    ABORTED = "aborted"


class Landed(str, Enum):
    """
    Resultado estimado del golpe.

    Un sistema monocular no establece contacto fisico, asi que esto es siempre una
    estimacion del anotador y se reporta como tal. `unknown` es el default y no es un
    valor de descarte: que el anotador no pueda decidir es informacion.
    """

    LANDED = "landed"
    BLOCKED = "blocked"
    SLIPPED = "slipped"
    MISSED = "missed"
    UNKNOWN = "unknown"


class Guard(str, Enum):
    ORTHODOX = "orthodox"
    SOUTHPAW = "southpaw"


class Quality(str, Enum):
    CLEAN = "clean"
    PARTIAL_OCCLUSION = "partial_occlusion"
    AMBIGUOUS = "ambiguous"


class ArmRole(str, Enum):
    """Derivado de guardia y lado. Nunca se almacena en el archivo de anotacion."""

    LEAD = "lead"
    REAR = "rear"


class UnreliableReason(str, Enum):
    OCCLUDED = "occluded"
    POSE_UNRELIABLE = "pose_unreliable"


class UnreliableSource(str, Enum):
    """Quien marco el tramo. Los automaticos se pueden recalcular, el manual no."""

    MANUAL = "manual"
    AUTO_MANUAL_TRACK = "auto_manual_track"
    AUTO_LOW_SCORE = "auto_low_score"


class AssignmentOp(str, Enum):
    """
    Operacion de la que nacio un assignment.

    El estado que se guarda es el materializado, no un log a reproducir, porque resolver
    el rol de un track en un frame tiene que ser una busqueda en un intervalo. El op_id
    compartido recupera la trazabilidad: los dos assignments que salen de un mismo swap
    lo comparten y en un diff se ve que fueron un solo gesto.
    """

    MANUAL = "manual"
    SWAP = "swap"
    RESEED = "reseed"
    SPLIT = "split"
    INTERPOLATION = "interpolation"


class InterpolationMethod(str, Enum):
    LINEAR = "linear"


class ComboOp(str, Enum):
    """Correccion a la deteccion automatica de combinaciones."""

    SPLIT = "split"
    JOIN = "join"


class KeypointFormat(str, Enum):
    """
    COCO17 es lo que devuelve yolov8l-pose. La variante con guantes agrega los indices
    17 y 18, que no viven en el npz: se derivan al leer o salen de un detector aparte.
    """

    COCO17 = "coco17"
    COCO17_GLOVES = "coco17+gloves"


class GloveSource(str, Enum):
    """
    De donde salen los keypoints 17 y 18.

    `derived` extrapola sobre el antebrazo y por lo tanto no agrega informacion a un
    clasificador que ya ve codo y muneca: sirve para el overlay y para los heuristicos de
    conexion, no para el modelo. `detected` si es observacion nueva y resuelve el caso en
    que la muneca se pierde por desenfoque de movimiento.
    """

    NONE = "none"
    DERIVED = "derived"
    DETECTED = "detected"


class FpsSource(str, Enum):
    """
    Como se obtuvo el fps.

    `measured` es contando frames decodificados. `container_verified` es que el valor
    declarado por el contenedor coincidio con el conteo real. En BoxingVI los nueve
    videos no declaran nb_frames y aparecieron cuatro fps nativos distintos donde el
    paper declaraba uno solo, asi que el default es medir.
    """

    MEASURED = "measured"
    CONTAINER_VERIFIED = "container_verified"


class IssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


def arm_role(guard: Guard, side: Side) -> ArmRole:
    """
    Deriva mano adelantada o atrasada a partir de la guardia y el lado.

    orthodox+left y southpaw+right son la mano adelantada; los otros dos casos son la
    atrasada. La direccion de la relacion no se invierte nunca: se anota el lado, que es
    lo observable en el video, y el rol se calcula. Anotar el rol directamente obligaria
    al anotador a resolver la guardia de cabeza en cada golpe y a reanotar todo el video
    si el peleador cambia de guardia.
    """
    return ArmRole.LEAD if (guard is Guard.ORTHODOX) == (side is Side.LEFT) else ArmRole.REAR
