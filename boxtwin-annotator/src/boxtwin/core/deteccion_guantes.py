"""
BoxTwin - Deteccion de guantes sobre recortes de persona, y su color.

POR QUE EXISTE
  El rol de un track lo asigna hoy una persona, y es la ultima pieza enteramente manual del
  sistema: se lleva el 73% del tiempo de anotacion medido. El guante es lo que permite
  automatizarla, por dos razones distintas que conviene no mezclar.

  Separa PELEADOR de NO PELEADOR, porque el arbitro es el unico adentro del ring sin guantes
  de boxeo. El filtro por altura saca al publico pero no a la gente a distancia de ring, y
  quedan entre 20 y 43 tracks del tamano de un peleador que no lo son. Medido sobre seis
  fuentes: los tracks de peleador dan fraccion de guante entre 0,59 y 0,96 contra 0,22 a 0,33
  del resto, y las medianas no se tocan en ninguna.

  Y su COLOR separa A de B, que es donde esta el trabajo: de los 390 relevos de track
  medidos, el 65% tiene al otro peleador como candidato. Sobre 8150 guantes de seis fuentes,
  el color acierta el 87,7% por guante suelto y 112 de 113 tracks por voto de mayoria.

QUE HACE
  Corre un detector de guantes sobre el recorte de una persona y muestrea el color del
  parche. Dos cuidados que no son cosmeticos y que se midieron:

  El tono se promedia CIRCULARMENTE. Es un angulo, y el rojo vive en los dos extremos de la
  escala: la media aritmetica de 5 y 175 daria verde, que es el color que ninguno de los dos
  tiene.

  El parche se toma del centro del 60% de la caja. Los bordes traen fondo, y en clinch traen
  guante del rival.

USO
  det = DetectorGuantes("modelos/guantes.pt")
  cajas = det.detectar(frame, (x0, y0, x1, y1))
  color = color_de_parche(parche)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Color", "DetectorGuantes", "color_de_parche", "perfil_de", "distancia_color",
    "ACROMATICO",
]

# Tono de un parche sin pixeles con color. No es un tono valido: es la marca de que el
# guante es negro o blanco y que preguntar por su tono no tiene sentido.
ACROMATICO = -1.0

Color = tuple[float, float, float]  # (tono 0-179, saturacion 0-255, valor 0-255)


@dataclass(frozen=True)
class CajaGuante:
    """Un guante detectado, en coordenadas del cuadro completo."""

    xyxy: tuple[float, float, float, float]
    conf: float


def color_de_parche(bgr: np.ndarray, sat_min: int = 40, val_min: int = 40) -> Color | None:
    """
    Tono, saturacion y valor del parche, sobre los pixeles que tienen color.

    Devuelve None si el parche esta vacio. Si no hay suficientes pixeles con color devuelve
    el tono ACROMATICO, que es distinto de no devolver nada: un guante negro es un dato, y la
    distancia sabe compararlo.
    """
    import cv2

    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if h < 3 or w < 3:
        return None
    # El centro del 60%: los bordes traen fondo, y en clinch traen guante del rival.
    centro = bgr[int(h * 0.2):max(int(h * 0.8), 1), int(w * 0.2):max(int(w * 0.8), 1)]
    if centro.size == 0:
        return None
    hsv = cv2.cvtColor(centro, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(float)
    S = hsv[..., 1].astype(float)
    V = hsv[..., 2].astype(float)
    con_color = (S > sat_min) & (V > val_min)
    if con_color.sum() < 12:
        return (ACROMATICO, float(np.median(S)), float(np.median(V)))
    ang = H[con_color] * 2 * np.pi / 180.0
    medio = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    tono = (np.degrees(medio) % 360) / 2.0
    return (float(tono), float(np.median(S[con_color])), float(np.median(V[con_color])))


def perfil_de(colores: list[Color]) -> Color:
    """Color representativo de un conjunto. El tono, otra vez, promediado circularmente."""
    if not colores:
        raise ValueError("no se puede hacer un perfil de cero colores")
    arr = np.asarray(colores, dtype=float)
    con_tono = arr[arr[:, 0] >= 0]
    if len(con_tono):
        ang = con_tono[:, 0] * 2 * np.pi / 180.0
        tono = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2.0
    else:
        tono = ACROMATICO
    return (float(tono), float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))


def distancia_color(a: Color, b: Color) -> float:
    """
    Distancia entre dos colores. El tono pesa el doble, pero solo si los dos lo tienen.

    Si uno es acromatico y el otro no, se suma una penalidad: un guante negro y uno rojo son
    distintos aunque comparar sus tonos no se pueda.
    """
    ha, sa, va = a
    hb, sb, vb = b
    d_sv = (abs(sa - sb) + abs(va - vb)) / 255.0
    if ha < 0 or hb < 0:
        return d_sv + (0.5 if (ha < 0) != (hb < 0) else 0.0)
    dh = abs(ha - hb)
    dh = min(dh, 180.0 - dh) / 90.0
    return 2.0 * dh + d_sv


class DetectorGuantes:
    """
    Envoltorio del detector. Carga el modelo una sola vez y perezosamente.

    El modelo se entreno sobre RECORTES DE PERSONA y no sobre cuadros completos, asi que hay
    que darle de comer lo mismo: un guante mide siempre cerca de 0,167 del alto de un cuerpo
    sin importar la distancia a la camara, y es el recorte lo que vuelve esa proporcion
    constante. Pasarle el cuadro entero lo saca de distribucion.
    """

    def __init__(self, pesos: str, conf: float = 0.25, imgsz: int = 320,
                 device: str | None = None, margen: float = 0.08) -> None:
        self.pesos = str(pesos)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.margen = margen
        self._modelo = None

    @property
    def modelo(self):
        if self._modelo is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:  # pragma: no cover - depende del extra
                raise RuntimeError(
                    "la deteccion de guantes necesita ultralytics: "
                    'instalar con pip install -e ".[preprocess]"'
                ) from e
            self._modelo = YOLO(self.pesos)
        return self._modelo

    def recorte(self, frame: np.ndarray, bbox) -> tuple[np.ndarray, int, int]:
        """El recorte de una persona con margen, y el origen del recorte en el cuadro."""
        alto, ancho = frame.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in bbox)
        m = self.margen
        cx0 = max(0, int(x0 - m * (x1 - x0)))
        cy0 = max(0, int(y0 - m * (y1 - y0)))
        cx1 = min(ancho, int(x1 + m * (x1 - x0)))
        cy1 = min(alto, int(y1 + m * (y1 - y0)))
        return frame[cy0:cy1, cx0:cx1], cx0, cy0

    def detectar(self, frame: np.ndarray, bbox) -> list[CajaGuante]:
        """Guantes de esa persona, en coordenadas del cuadro completo."""
        rec, ox, oy = self.recorte(frame, bbox)
        if rec.size == 0:
            return []
        kw = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False}
        if self.device is not None:
            kw["device"] = self.device
        r = self.modelo.predict(rec, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        confs = r.boxes.conf.cpu().numpy()
        salida = []
        for (gx0, gy0, gx1, gy1), c in zip(r.boxes.xyxy.cpu().numpy(), confs):
            salida.append(CajaGuante(
                xyxy=(float(gx0 + ox), float(gy0 + oy), float(gx1 + ox), float(gy1 + oy)),
                conf=float(c),
            ))
        return salida

    def detectar_con_color(self, frame: np.ndarray, bbox) -> list[tuple[CajaGuante, Color]]:
        """Los guantes de esa persona y el color de cada uno."""
        salida = []
        for caja in self.detectar(frame, bbox):
            x0, y0, x1, y1 = (int(v) for v in caja.xyxy)
            color = color_de_parche(frame[max(0, y0):y1, max(0, x0):x1])
            if color is not None:
                salida.append((caja, color))
        return salida
