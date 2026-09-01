"""
BoxTwin - Features por cuadro y por brazo.

POR QUE EXISTE
  El clasificador PoseConv3D no generalizo a una fuente nueva: los tres folds que dejan una
  fuente afuera quedaron en o por debajo de su linea de base. La sospecha razonable es que
  aprendio la camara y la escala, porque trabaja sobre volumenes de heatmaps en pixeles.

  Aca las features se construyen para que esa via este cerrada por diseno:

  - CENTRADAS en el punto medio de los hombros, asi la posicion en el cuadro no entra.
  - ESCALADAS por el ancho de hombros, asi la distancia a camara no entra.
  - ESPEJADAS por brazo, asi un jab y un cross se ven iguales. Detectar no necesita saber
    que brazo es, y espejar duplica los datos efectivos.

  Es la pieza del sistema con mejor chance de sobrevivir al leave-one-source-out, porque la
  senal que busca es geometrica y normalizable. Lo que NO arregla es la familia del golpe:
  eso esta medido tres veces que no es separable en pose monocular, y el detector no lo
  intenta.

QUE HACE
  De (T, 17, 2) keypoints COCO y (T, 17) scores, saca (T, 20) features y una mascara de
  cuadros usables, para un brazo dado.

  Los angulos que dan la vuelta se codifican como (coseno, seno) y no como el angulo: a
  -pi y a +pi el cuerpo esta en la misma pose, y un escalar que salta de uno a otro le
  ensena al modelo un borde que no existe.

USO
  f, ok = features_de(kp, sc, brazo="left")
"""

from __future__ import annotations

import numpy as np

__all__ = ["NOMBRES", "N_FEATURES", "features_de", "espejar", "MIN_SCORE"]

# COCO17
NARIZ = 0
L_HOM, R_HOM = 5, 6
L_COD, R_COD = 7, 8
L_MUN, R_MUN = 9, 10
L_CAD, R_CAD = 11, 12

MIN_SCORE = 0.3
_EPS = 1e-6

# Un espejo real no es solo negar la x: lo que era el hombro izquierdo de la persona aparece
# del otro lado y sigue siendo su hombro izquierdo, asi que los pares tambien se permutan.
# Sin la permutacion, el vector de hombros queda con el signo cambiado y el brazo derecho no
# se ve como un izquierdo, que es todo el objetivo.
PARES_LR = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def espejar(kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refleja el esqueleto: niega la x y permuta los pares izquierda/derecha."""
    kp, sc = kp.copy(), sc.copy()
    kp[..., 0] *= -1
    for a, b in PARES_LR:
        kp[:, [a, b]] = kp[:, [b, a]]
        sc[:, [a, b]] = sc[:, [b, a]]
    return kp, sc

NOMBRES = [
    "muneca_x", "muneca_y",
    "codo_x", "codo_y",
    "extension",
    "angulo_codo",
    "hombros_cos", "hombros_sin",
    "caderas_cos", "caderas_sin",
    "torsion_cos", "torsion_sin",
    "otra_muneca_x", "otra_muneca_y",
    "nariz_y",
    "d_muneca_x", "d_muneca_y", "d_extension", "d_angulo_codo",
    "dd_extension",
]
N_FEATURES = len(NOMBRES)


def _angulo(v: np.ndarray) -> np.ndarray:
    """Angulo de un vector (T, 2), como (cos, sin). Sin saltos en +-pi."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


def features_de(
    kp: np.ndarray, sc: np.ndarray, brazo: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Features del brazo `brazo` ("left" o "right") y mascara de cuadros usables.

    El brazo derecho se resuelve espejando el esqueleto entero y leyendo despues el
    izquierdo. Asi la invariancia queda por construccion y no por dos ramas de codigo que
    hay que mantener de acuerdo.

    Un cuadro es usable si los dos hombros, el codo y la muneca del brazo que actua tienen
    score suficiente y los hombros no estan colapsados. Los cuadros no usables van en cero,
    pero eso no los hace fondo: la mascara es lo que los saca de la perdida.

    Los cuadros vecinos a uno no usable tambien se marcan, porque las derivadas se calculan
    por diferencias y ahi cruzan el agujero.
    """
    if brazo not in ("left", "right"):
        raise ValueError(f"brazo invalido: {brazo!r}")
    kp = np.asarray(kp, np.float64)
    sc = np.asarray(sc, np.float64)
    if brazo == "right":
        kp, sc = espejar(kp, sc)
    T = len(kp)

    escala = np.linalg.norm(kp[:, L_HOM] - kp[:, R_HOM], axis=-1)
    centro = (kp[:, L_HOM] + kp[:, R_HOM]) / 2

    ok = (
        (np.minimum(sc[:, L_HOM], sc[:, R_HOM]) >= MIN_SCORE)
        & (sc[:, L_COD] >= MIN_SCORE)
        & (sc[:, L_MUN] >= MIN_SCORE)
        & (escala > _EPS)
    )

    rel = (kp - centro[:, None, :]) / np.maximum(escala, _EPS)[:, None, None]

    muneca = rel[:, L_MUN]
    codo = rel[:, L_COD]
    hombro = rel[:, L_HOM]
    extension = np.linalg.norm(muneca, axis=-1)

    # angulo del codo entre hombro->codo y codo->muneca, en [0, pi]. No da la vuelta, asi
    # que va como escalar y no como (cos, sin).
    a = codo - hombro
    b = muneca - codo
    cos = (a * b).sum(-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), _EPS
    )
    angulo_codo = np.arccos(np.clip(cos, -1.0, 1.0))

    hom = _angulo(rel[:, L_HOM] - rel[:, R_HOM])
    cad = _angulo(rel[:, L_CAD] - rel[:, R_CAD])
    # torsion = hombros menos caderas, compuesta como rotacion: (cos(a-b), sin(a-b))
    tor = np.stack(
        [
            hom[:, 0] * cad[:, 0] + hom[:, 1] * cad[:, 1],
            hom[:, 1] * cad[:, 0] - hom[:, 0] * cad[:, 1],
        ],
        axis=-1,
    )

    def d(x: np.ndarray) -> np.ndarray:
        return np.gradient(x) if T > 1 else np.zeros_like(x)

    f = np.stack(
        [
            muneca[:, 0], muneca[:, 1],
            codo[:, 0], codo[:, 1],
            extension,
            angulo_codo,
            hom[:, 0], hom[:, 1],
            cad[:, 0], cad[:, 1],
            tor[:, 0], tor[:, 1],
            rel[:, R_MUN, 0], rel[:, R_MUN, 1],
            rel[:, NARIZ, 1],
            d(muneca[:, 0]), d(muneca[:, 1]), d(extension), d(angulo_codo),
            d(d(extension)),
        ],
        axis=-1,
    )
    f[~ok] = 0.0
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # las derivadas cruzan los agujeros: el vecino de un cuadro no usable tampoco lo es
    usable = ok.copy()
    if T > 1:
        usable[:-1] &= ok[1:]
        usable[1:] &= ok[:-1]
    return f.astype(np.float32), usable
