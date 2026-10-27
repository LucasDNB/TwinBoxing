"""
BoxTwin - Representacion de entrada del modelo de atributos de BoxMind, con poses propias.

POR QUE EXISTE
  El modelo de BoxMind (MFusion, variante PoseOnly+TCN+MoE) recibe por cuadro 28 articulaciones
  (14 del atacante y 14 del rival) con 5 canales: 2D normalizado al recorte del atacante y 3D
  en el marco del cuerpo de cada uno. Su cargador (dataset/dataset_youtube.py@e06add2) arma
  eso desde articulaciones SMPL de 4D-Humans. BoxTwin tiene COCO-17 en 2D (YOLOv8l-pose) y
  H36M-17 en 3D (MotionBERT). Este modulo reproduce la misma representacion desde esas poses,
  y se usa igual para BoxTwin y para BoxingWeb re-estimado, de modo que entrenamiento y prueba
  vean el mismo estimador. Se reimplementa (no se copia) porque el codigo original no tiene
  licencia.

QUE HACE
  - coco_a_14 / h36m_a_14: las 14 articulaciones de BoxMind en su orden (cabeza, cuello,
    hombro, codo y muneca derechos, los mismos izquierdos, cadera, rodilla y tobillo derechos,
    los mismos izquierdos).
  - caja_atacante: el recorte de get_region (cabeza llevada al tope, munecas llevadas al
    guante, 40% de expansion) y la normalizacion cuadrada de prepare_video_data.
  - marco_cuerpo: origen en el medio de las caderas, x de cadera izquierda a derecha, z hacia
    el medio de los hombros (convert_pose_3d). Desvio declarado (pre-registro, seccion 4.1,
    B2): ademas se divide por el largo del torso, porque MotionBERT no da metros y la
    escala cambiaria con la resolucion del video.
  - ejemplo: (T, 28, 5). espejar: el aumento de datos de __getitem__ (permutacion izquierda /
    derecha, 1 - x en 2D, menos el primer eje 3D).

USO
  from representacion import ejemplo, espejar
  No corre solo. Solo usa numpy: se importa desde boxtwin_mmaction y desde twinboxing_env.
"""

from __future__ import annotations

import numpy as np

# Orden de las 14 articulaciones de BoxMind (JOINT_INDEX de dataset_youtube.py).
NOMBRES_14 = ("cabeza", "cuello", "hombro_d", "codo_d", "muneca_d", "hombro_i", "codo_i", "muneca_i",
              "cadera_d", "rodilla_d", "tobillo_d", "cadera_i", "rodilla_i", "tobillo_i")
HOMBRO_D, CODO_D, MUNECA_D, HOMBRO_I, CODO_I, MUNECA_I = 2, 3, 4, 5, 6, 7
CADERA_D, CADERA_I = 8, 11
# Permutacion de espejo sobre las 28 (atacante 0-13, rival 14-27), la de __getitem__.
ESPEJO_28 = (0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10,
             14, 15, 19, 20, 21, 16, 17, 18, 25, 26, 27, 22, 23, 24)

# COCO-17: 0 nariz, 1-2 ojos, 3-4 orejas, 5/6 hombro i/d, 7/8 codo, 9/10 muneca, 11/12 cadera,
# 13/14 rodilla, 15/16 tobillo. Cabeza y cuello se construyen (ver coco_a_14).
COCO_DIRECTAS = {2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15}
COCO_CABEZA = (0, 1, 2, 3, 4)
# H36M-17 (MMPose): 0 raiz, 1-3 cadera/rodilla/pie d, 4-6 los i, 7 columna, 8 torax,
# 9 base del cuello, 10 cabeza, 11-13 hombro/codo/muneca i, 14-16 los d.
H36M_A_14 = (10, 9, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6)

MIN_SCORE = 0.3
EXPANSION = 0.4
EPS = 1e-8


def coco_a_14(kp: np.ndarray, sc: np.ndarray | None = None, min_score: float = MIN_SCORE) -> np.ndarray:
    """
    kp (T, 17, 2) COCO -> (T, 14, 2). Cabeza: promedio de nariz, ojos y orejas con score
    suficiente (la nariz sola si ninguno pasa). Cuello: medio de los hombros. SMPL tiene un
    cuello algo mas alto; la diferencia es la misma en entrenamiento y prueba.
    """
    kp = np.asarray(kp, np.float32)
    T = kp.shape[0]
    out = np.zeros((T, 14, 2), np.float32)
    for dst, src in COCO_DIRECTAS.items():
        out[:, dst] = kp[:, src]
    out[:, 1] = 0.5 * (kp[:, 5] + kp[:, 6])
    if sc is None:
        out[:, 0] = kp[:, list(COCO_CABEZA)].mean(1)
    else:
        sc = np.asarray(sc, np.float32)
        w = (sc[:, list(COCO_CABEZA)] >= min_score).astype(np.float32)
        sin = w.sum(1) == 0
        w[sin, 0] = 1.0
        out[:, 0] = (kp[:, list(COCO_CABEZA)] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    return out


def h36m_a_14(P: np.ndarray) -> np.ndarray:
    """P (T, 17, 3) H36M -> (T, 14, 3)."""
    return np.asarray(P, np.float32)[:, list(H36M_A_14)]


def caja_atacante(p14: np.ndarray, ancho: int, alto: int, expansion: float = EXPANSION) -> tuple[int, int, int, int] | None:
    """
    Recorte de get_region sobre toda la ventana: cabeza llevada al tope (2 cabeza - cuello),
    munecas llevadas al guante (2 muneca - codo), caja de cabeza, cuello, brazos y caderas,
    expandida, recortada a la imagen. None si queda degenerada.
    """
    q = np.array(p14, np.float32, copy=True)
    q[:, MUNECA_I] = 2 * q[:, MUNECA_I] - q[:, CODO_I]
    q[:, MUNECA_D] = 2 * q[:, MUNECA_D] - q[:, CODO_D]
    q[:, 0] = 2 * q[:, 0] - q[:, 1]
    parte = q[:, [0, 1, HOMBRO_D, CODO_D, MUNECA_D, HOMBRO_I, CODO_I, MUNECA_I, CADERA_D, CADERA_I]]
    x0, x1 = float(parte[..., 0].min()), float(parte[..., 0].max())
    y0, y1 = float(parte[..., 1].min()), float(parte[..., 1].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + expansion), (y1 - y0) * (1 + expansion)
    nx0, nx1 = int(max(cx - w / 2, 0)), int(min(cx + w / 2, ancho))
    ny0, ny1 = int(max(cy - h / 2, 0)), int(min(cy + h / 2, alto))
    if nx0 >= nx1 or ny0 >= ny1:
        return None
    return nx0, ny0, nx1, ny1


def norm2d(p14: np.ndarray, caja: tuple[int, int, int, int]) -> np.ndarray:
    """(T, 14, 2) en pixeles -> coordenadas del recorte cuadrado de prepare_video_data."""
    x0, y0, x1, y1 = caja
    tam = np.array([x1 - x0, y1 - y0], np.float32)
    lado = float(tam.max())
    pad = (lado - tam) // 2
    return ((np.asarray(p14, np.float32) - np.array([x0, y0], np.float32) + pad) / lado).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > EPS, v / np.maximum(n, EPS), v)


def marco_cuerpo(P14: np.ndarray, escalar: bool = True) -> np.ndarray:
    """
    (T, 14, 3) -> (T, 14, 3) en el marco del propio cuerpo, cuadro por cuadro, como
    convert_pose_3d: origen medio de caderas, z hacia el medio de los hombros, y = z x x,
    x = y x z. Con `escalar`, dividido por la mediana del largo del torso en la ventana.
    """
    P = np.asarray(P14, np.float32)
    origen = 0.5 * (P[:, CADERA_D] + P[:, CADERA_I])
    x = P[:, CADERA_D] - P[:, CADERA_I]
    medio_hombros = 0.5 * (P[:, HOMBRO_D] + P[:, HOMBRO_I])
    z = _unit(medio_hombros - origen)
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    R = np.stack([x, y, z], axis=-1)                                  # (T, 3, 3), columnas x, y, z
    out = np.einsum("tjk,tkl->tjl", P - origen[:, None], R).astype(np.float32)
    if escalar:
        torso = np.linalg.norm(medio_hombros - origen, axis=-1)
        torso = torso[torso > EPS]
        if torso.size:
            out = out / float(np.median(torso))
    return out


def ejemplo(kp_at: np.ndarray, sc_at: np.ndarray | None, kp_ri: np.ndarray, sc_ri: np.ndarray | None,
            P3_at: np.ndarray | None, P3_ri: np.ndarray | None, ancho: int, alto: int) -> np.ndarray:
    """
    Un golpe -> (T, 28, 5): 2D del atacante y del rival en el recorte del atacante, y 3D de cada
    uno en su propio marco. Sin 3D, esos canales van en cero; sin recorte valido, todo en cero
    (el golpe queda en la evaluacion igual, como en E4).
    """
    a14 = coco_a_14(kp_at, sc_at)
    r14 = coco_a_14(kp_ri, sc_ri)
    T = a14.shape[0]
    x = np.zeros((T, 28, 5), np.float32)
    caja = caja_atacante(a14, ancho, alto)
    if caja is not None:
        x[:, :14, :2] = norm2d(a14, caja)
        x[:, 14:, :2] = norm2d(r14, caja)
    if P3_at is not None:
        x[:, :14, 2:] = marco_cuerpo(h36m_a_14(P3_at))
    if P3_ri is not None:
        x[:, 14:, 2:] = marco_cuerpo(h36m_a_14(P3_ri))
    return x


def espejar(x: np.ndarray) -> np.ndarray:
    """El espejo de __getitem__: (T, 28, 5) -> (T, 28, 5). Cambia izquierda por derecha."""
    y = np.array(x, np.float32, copy=True)[:, list(ESPEJO_28)]
    y[..., 0] = 1 - y[..., 0]
    y[..., 2] = -y[..., 2]
    return y
