"""
BoxTwin - El detector sobre un video que nadie anoto.

POR QUE EXISTE
  Todo lo que corre el detector hoy entra por `Fuente`, que se construye desde un export de
  un proyecto de anotacion y trae etiquetas, cobertura y mascara de amagues. En produccion
  no hay nada de eso: hay keypoints, una identidad resuelta y un checkpoint congelado. Sin
  este modulo, poner el sistema en un producto obliga a fabricar una anotacion vacia para
  poder pedirle una prediccion, y esa clase de rodeo es la que despues hace que el camino
  de produccion y el de medicion se separen sin que nadie lo note.

  Hay una cosa que el camino de produccion NO puede saltear y es el remuestreo. El modelo
  aprendio en un timebase de 30 fps y su campo receptivo esta en cuadros, no en segundos:
  alimentarlo a 60 fps le cambia la escala temporal de todo lo que vio. El video de un
  celular puede venir a 24, 30 o 60, asi que el remuestreo es la diferencia entre medir y
  no medir. `demo_vivo.py` se lo salteaba porque sus tres fuentes eran de 30.

QUE HACE
  Arma los cuatro carriles -dos peleadores por dos brazos- desde los keypoints ya
  identificados, los lleva al timebase del modelo, corre el ensamble congelado, decodifica a
  segmentos y devuelve cada uno con su cuadro original y su segundo.

USO
  from boxtwin_detector.inferencia import carriles_de, detectar
  lotes = carriles_de(kp, sc, valid, fps_origen=59.94)
  golpes = detectar(ens, lotes, umbral=0.8, device=torch.device("cuda"))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxtwin_detector.dataset import FPS_DESTINO, indice_remuestreo
from boxtwin_detector.decodificacion import decodificar_score, probabilidad_de_golpe
from boxtwin_detector.entrenamiento import predecir_secuencia
from boxtwin_detector.features import N_FEATURES, features_de

__all__ = ["Carriles", "GolpeDetectado", "carriles_de", "detectar"]

PELEADORES = ("A", "B")
BRAZOS = ("left", "right")


@dataclass
class Carriles:
    """Los cuatro carriles listos para el modelo, y como volver al video original."""

    features: np.ndarray       # (4, T, F) float32
    usable: np.ndarray         # (4, T) bool
    nombres: list[str]         # "A-left", "A-right", "B-left", "B-right"
    indice: np.ndarray         # (T,) cuadro original de cada cuadro remuestreado
    fps_origen: float
    fps: float                 # el timebase del modelo

    @property
    def T(self) -> int:
        return self.features.shape[1]


@dataclass
class GolpeDetectado:
    """Un golpe, en el timebase del modelo y en el del video."""

    peleador: str              # "A" o "B"
    brazo: str                 # "left" o "right"
    inicio: int                # cuadro del video original
    fin: int                   # cuadro del video original, inclusive
    t_inicio: float            # segundos
    t_fin: float
    score: float               # probabilidad maxima adentro del segmento
    cuadros_modelo: tuple[int, int] = (0, 0)

    def a_dict(self) -> dict:
        return {
            "peleador": self.peleador,
            "brazo": self.brazo,
            "inicio": self.inicio,
            "fin": self.fin,
            "t_inicio": round(self.t_inicio, 3),
            "t_fin": round(self.t_fin, 3),
            "score": round(self.score, 4),
            "cuadros_modelo": list(self.cuadros_modelo),
        }


def carriles_de(
    kp: np.ndarray,
    sc: np.ndarray,
    valid: np.ndarray,
    fps_origen: float,
    fps_destino: float = FPS_DESTINO,
) -> Carriles:
    """
    De (2, T, 17, 2) keypoints identificados a (4, T', 20) features en el timebase del modelo.

    El remuestreo va sobre los KEYPOINTS y las features se calculan despues, igual que en el
    armado del dataset. Al reves las derivadas quedarian en cuadros del video original y no
    habria arreglo posterior.

    `valid` es la mascara de identidad resuelta: un cuadro donde no se sabe quien es quien no
    es un cuadro sin golpe, es un cuadro sin dato, y sale de la mascara para que el
    decodificador no pueda abrir un segmento ahi.
    """
    kp = np.asarray(kp)
    sc = np.asarray(sc)
    valid = np.asarray(valid, bool)
    if kp.ndim != 4 or kp.shape[0] != 2 or kp.shape[3] != 2:
        raise ValueError(f"keypoints tiene que ser (2, T, K, 2) y es {kp.shape}")
    if sc.shape[:2] != kp.shape[:2] or valid.shape != kp.shape[:2]:
        raise ValueError("keypoints, scores y mascara tienen que compartir (2, T)")
    if fps_origen <= 0:
        raise ValueError("el fps del video tiene que ser positivo")

    T = kp.shape[1]
    idx = indice_remuestreo(T, fps_origen, fps_destino)
    kp_r, sc_r, valid_r = kp[:, idx], sc[:, idx], valid[:, idx]

    T_nuevo = len(idx)
    F = np.zeros((4, T_nuevo, N_FEATURES), np.float32)
    U = np.zeros((4, T_nuevo), bool)
    nombres = []
    for p, peleador in enumerate(PELEADORES):
        for b, brazo in enumerate(BRAZOS):
            i = p * len(BRAZOS) + b
            nombres.append(f"{peleador}-{brazo}")
            f, ok = features_de(kp_r[p], sc_r[p], brazo)
            F[i] = f
            U[i] = ok & valid_r[p]
    return Carriles(
        features=F, usable=U, nombres=nombres, indice=idx,
        fps_origen=float(fps_origen), fps=float(fps_destino),
    )


def detectar(
    ens,
    carriles: Carriles,
    umbral: float,
    device=None,
    devolver_probabilidades: bool = False,
):
    """
    Corre el ensamble congelado sobre los cuatro carriles y decodifica a golpes.

    El promedio de probabilidades es el mismo que usa la evaluacion por folds, y el umbral
    tambien es el mismo parametro: el punto de operacion se elige una vez y se declara, no
    se reajusta por video. Mover el umbral en produccion y despues comparar contra la
    precision medida en la tesis seria comparar dos sistemas distintos.
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    golpes: list[GolpeDetectado] = []
    probs = []
    for i, nombre in enumerate(carriles.nombres):
        x = ens.estandarizador.aplicar(carriles.features[i])
        acum = np.zeros(carriles.T, np.float64)
        for m in ens.modelos:
            acum += probabilidad_de_golpe(predecir_secuencia(m, x, ens.config, device))
        prob = (acum / ens.n).astype(np.float32)
        probs.append(prob)
        peleador, brazo = nombre.split("-")
        for s in decodificar_score(prob, umbral, valido=carriles.usable[i]):
            ini = int(carriles.indice[s.inicio])
            fin = int(carriles.indice[min(s.fin, carriles.T - 1)])
            golpes.append(
                GolpeDetectado(
                    peleador=peleador,
                    brazo=brazo,
                    inicio=ini,
                    fin=fin,
                    t_inicio=ini / carriles.fps_origen,
                    t_fin=(fin + 1) / carriles.fps_origen,
                    score=float(prob[s.inicio : s.fin + 1].max()),
                    cuadros_modelo=(int(s.inicio), int(s.fin)),
                )
            )
    golpes.sort(key=lambda g: (g.inicio, g.peleador, g.brazo))
    if devolver_probabilidades:
        return golpes, probs
    return golpes
