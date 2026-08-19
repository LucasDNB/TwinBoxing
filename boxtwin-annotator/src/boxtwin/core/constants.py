"""
BoxTwin - Constantes del formato de keypoints.

POR QUE EXISTE
  El orden de los 17 keypoints COCO y las aristas del esqueleto se usan en el overlay,
  en la interpolacion de gaps y en los exports. Repetir la tabla en cada modulo garantiza
  que en algun momento dos copias dejen de coincidir y el esqueleto se dibuje bien pero
  el export salga con los indices cruzados, que es un error silencioso.

QUE HACE
  Define nombres e indices COCO17, las aristas del esqueleto, los indices reservados para
  los guantes y el conteo de keypoints por formato.

USO
  from boxtwin.core.constants import COCO17_NAMES, COCO17_EDGES, GLOVE_LEFT_IDX
"""

from __future__ import annotations

from boxtwin.core.types import KeypointFormat

__all__ = [
    "COCO17_NAMES",
    "COCO17_INDEX",
    "COCO17_EDGES",
    "LEFT_ELBOW_IDX",
    "RIGHT_ELBOW_IDX",
    "LEFT_WRIST_IDX",
    "RIGHT_WRIST_IDX",
    "GLOVE_LEFT_IDX",
    "GLOVE_RIGHT_IDX",
    "GLOVE_EDGES",
    "KEYPOINT_COUNT",
]

# Orden exacto que devuelve yolov8l-pose. El indice es la posicion en la tupla.
COCO17_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO17_INDEX: dict[str, int] = {name: i for i, name in enumerate(COCO17_NAMES)}

# Aristas estandar del esqueleto COCO: piernas, torso, brazos y cara.
COCO17_EDGES: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

LEFT_ELBOW_IDX = COCO17_INDEX["left_elbow"]
RIGHT_ELBOW_IDX = COCO17_INDEX["right_elbow"]
LEFT_WRIST_IDX = COCO17_INDEX["left_wrist"]
RIGHT_WRIST_IDX = COCO17_INDEX["right_wrist"]

# Indices reservados para los guantes. No estan en el npz: los derivados se calculan al
# leer y los detectados van en arrays propios del cache.
GLOVE_LEFT_IDX = 17
GLOVE_RIGHT_IDX = 18

GLOVE_EDGES: tuple[tuple[int, int], ...] = (
    (LEFT_WRIST_IDX, GLOVE_LEFT_IDX),
    (RIGHT_WRIST_IDX, GLOVE_RIGHT_IDX),
)

KEYPOINT_COUNT: dict[KeypointFormat, int] = {
    KeypointFormat.COCO17: 17,
    KeypointFormat.COCO17_GLOVES: 19,
}

# Color por ROL, nunca por track_id. Un track_id no significa nada estable: cambia en cada
# oclusion y en cada reanudacion del preproceso. Si el color siguiera al id, el anotador
# veria cambiar de color al mismo peleador y perderia justo la senal que necesita.
# RGB 0-255. Viven en core porque son parte del contrato de lectura, no de la GUI.
ROLE_COLOR_A: tuple[int, int, int] = (236, 88, 76)  # rojo
ROLE_COLOR_B: tuple[int, int, int] = (74, 158, 235)  # azul
ROLE_COLOR_IGNORE: tuple[int, int, int] = (128, 128, 128)  # gris, se ve poco a proposito
ROLE_COLOR_UNASSIGNED: tuple[int, int, int] = (245, 194, 66)  # ambar, llama la atencion

# Color por tipo de golpe, para las marcas del timeline.
PUNCH_COLORS: dict[str, tuple[int, int, int]] = {
    "straight": (92, 200, 145),
    "hook": (216, 138, 224),
    "uppercut": (240, 166, 80),
}
