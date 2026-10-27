"""
BoxTwin - A la cabeza o al cuerpo: la altura del golpe por geometria (B11, ampliacion 03).

POR QUE EXISTE
  `01-producto.md` dejo la altura afuera porque el clasificador da familia y no altura. Hay
  otro camino que no pasa por el clasificador: la mano del que pega contra el cuerpo del que
  recibe, en el mismo cuadro, con los keypoints que la sesion ya guardo en `poses.npz`. Es
  geometria normalizada, que es lo que ya demostro que transfiere entre fuentes (el detector
  generaliza por eso; el clasificador, que dependia de pixeles, no).

  Dice "a donde fue dirigido", no "a donde llego": con la camara de costado un golpe que pasa
  por delante del pecho del rival sin llegar cuenta como cuerpo. La regla no ve profundidad.

QUE HACE (la regla de 03 §2, escrita antes de medir)
  1. Cuadro pico: dentro de [inicio, fin], el cuadro en que la muneca del brazo del golpe esta
     mas lejos de su propio hombro, en anchos de hombro del que pega. Sin cajas de guante
     guardadas por cuadro (ninguna cache las tiene hoy) se usa la muneca; si se pasan, el
     centro del guante.
  2. En ese cuadro, la linea de hombros del rival (punto medio) y el largo de su torso (de
     punto medio de hombros a punto medio de caderas).
  3. h = (y_mano - y_hombros_rival) / torso_rival. Negativa es por encima de los hombros.
  4. cabeza si h < tau, cuerpo si h >= tau. Si el rival no tiene pose valida en el pico ni en
     los +-2 cuadros vecinos, None con motivo `sin_rival`; si el que pega no tiene ningun
     cuadro valido en el golpe, None con motivo `sin_pose`.
  5. Confianza: |h - tau| por una logistica de pendiente fija (PENDIENTE), llevada a 0..1.
     Sirve para ordenar y marcar dudosos, no es una probabilidad calibrada.

  Un keypoint cuenta si su score es >= MIN_SCORE (el umbral del anotador).

  marcar_fightcard  pone objetivo, confianza y motivo en cada golpe de la Fight-Card y cuenta
                    cabeza / cuerpo / sin dato por peleador y por round (contrato 0.3)
  contar            recuenta, para despues de una correccion

  C8 se midio el 13-10 y NO paso (docs/experiments/2026-10-13-cabeza-cuerpo.md): exactitud
  balanceada 0,726 contra 0,75. Los campos se escriben igual, con el resultado medido adentro
  del documento, y la bandera `objetivo` queda apagada: ninguna pantalla los muestra.

USO
  from boxtwin.mvp.objetivo import objetivo_de
  r = objetivo_de(kp, sc, valid, peleador=0, brazo="izq", inicio=120, fin=128, tau=0.2)
  r.objetivo, r.h, r.confianza, r.motivo
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GRILLA_TAU", "MEDIDO", "MIN_SCORE", "PENDIENTE", "REGLA", "TAU", "Resultado", "contar",
           "marcar_fightcard", "objetivo_de"]

REGLA = "altura_relativa_v1"
GRILLA_TAU = (-0.10, 0.00, 0.10, 0.20, 0.30, 0.40)
# Elegido en la calibracion sobre 01 y 02 (balanceada 0,698), antes de medir.
TAU = 0.20
# C8, nivel 1, una sola corrida sobre 806 golpes de cuatro fuentes no usadas para calibrar.
MEDIDO = {"exactitud_balanceada": 0.726, "recall_cuerpo": 0.818, "golpes": 806, "pasa": False,
          "fecha": "2026-10-13", "detalle": "docs/experiments/2026-10-13-cabeza-cuerpo.md"}
MIN_SCORE = 0.3
PENDIENTE = 10.0
VECINOS = 2

# COCO-17
HOMBRO = {"izq": 5, "der": 6}
MUNECA = {"izq": 9, "der": 10}
CADERAS = (11, 12)
_BRAZO = {"izq": "izq", "der": "der", "left": "izq", "right": "der"}


@dataclass
class Resultado:
    objetivo: str | None
    h: float | None
    confianza: float | None
    motivo: str | None
    cuadro_pico: int | None = None


def _ok(sc: np.ndarray, *idx: int) -> bool:
    return all(float(sc[i]) >= MIN_SCORE for i in idx)


def _pico(kp, sc, valid, p: int, brazo: str, inicio: int, fin: int, guantes=None) -> tuple[int, np.ndarray] | None:
    h_i, m_i = HOMBRO[brazo], MUNECA[brazo]
    mejor, mejor_d, mano = None, -1.0, None
    for f in range(max(inicio, 0), min(fin, kp.shape[1] - 1) + 1):
        if not valid[p, f] or not _ok(sc[p, f], 5, 6, h_i, m_i):
            continue
        ancho = float(np.linalg.norm(kp[p, f, 5] - kp[p, f, 6]))
        if ancho <= 1e-6:
            continue
        punto = kp[p, f, m_i]
        if guantes is not None and (f, brazo) in guantes:
            punto = np.asarray(guantes[(f, brazo)], dtype=np.float32)
        d = float(np.linalg.norm(punto - kp[p, f, h_i])) / ancho
        if d > mejor_d:
            mejor, mejor_d, mano = f, d, punto
    return None if mejor is None else (mejor, mano)


def _rival_en(kp, sc, valid, r: int, f: int) -> tuple[float, float] | None:
    """(y de la linea de hombros, largo del torso) del rival, en el cuadro o en los vecinos."""
    T = kp.shape[1]
    for df in sorted(range(-VECINOS, VECINOS + 1), key=abs):
        g = f + df
        if not 0 <= g < T or not valid[r, g] or not _ok(sc[r, g], 5, 6, *CADERAS):
            continue
        hombros = (kp[r, g, 5] + kp[r, g, 6]) / 2.0
        caderas = (kp[r, g, CADERAS[0]] + kp[r, g, CADERAS[1]]) / 2.0
        torso = float(np.linalg.norm(caderas - hombros))
        if torso > 1e-6:
            return float(hombros[1]), torso
    return None


def confianza(h: float, tau: float) -> float:
    return round(2.0 / (1.0 + math.exp(-PENDIENTE * abs(h - tau))) - 1.0, 3)


def objetivo_de(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray, peleador: int, brazo: str,
                inicio: int, fin: int, tau: float, guantes: dict | None = None) -> Resultado:
    """
    La altura de un golpe. `kp` es (2, T, 17, 2) en pixeles, `sc` (2, T, 17), `valid` (2, T);
    `peleador` es 0 (A) o 1 (B) y el rival es el otro.
    """
    brazo = _BRAZO.get(brazo, brazo)
    pico = _pico(kp, sc, valid, peleador, brazo, int(inicio), int(fin), guantes)
    if pico is None:
        return Resultado(None, None, None, "sin_pose")
    f, mano = pico
    rival = _rival_en(kp, sc, valid, 1 - peleador, f)
    if rival is None:
        return Resultado(None, None, None, "sin_rival", f)
    y_hombros, torso = rival
    h = (float(mano[1]) - y_hombros) / torso
    return Resultado("cabeza" if h < tau else "cuerpo", round(h, 4), confianza(h, tau), None, f)


def marcar_fightcard(fc: dict, kp, sc, valid, tau: float = TAU) -> dict:
    """
    Escribe la altura de cada golpe. Sin rival (bolsa, sombra) no se calcula (DA-5): motivo
    `individual`. Una correccion de altura que ya exista se vuelve a aplicar despues, en
    `aplicar_correcciones`; aca se pisa el valor del modelo, no el de la persona.
    """
    individual = "B" not in fc["peleadores"]
    for lado, datos in fc["peleadores"].items():
        p = 0 if lado == "A" else 1
        for g in datos["golpes"]:
            if individual:
                r = Resultado(None, None, None, "individual")
            else:
                r = objetivo_de(kp, sc, valid, p, g["brazo"], g["cuadro_inicio"], g["cuadro_fin"], tau)
            g["objetivo"] = r.objetivo
            g["objetivo_confianza"] = r.confianza
            g["objetivo_motivo"] = r.motivo
            g["objetivo_h"] = r.h
            g.setdefault("objetivo_corregido", None)
    fc.setdefault("metodos", {})["objetivo"] = {"regla": REGLA, "tau": tau, "medido": dict(MEDIDO)}
    return contar(fc)


def contar(fc: dict) -> dict:
    """
    cabeza + cuerpo + objetivo_sin_dato = total, por peleador y por round (CA-14.1). Una
    Fight-Card que no paso por `enriquecer` no se toca: no trae altura y no se inventa.
    """
    if not (fc.get("metodos") or {}).get("objetivo"):
        return fc
    for datos in fc["peleadores"].values():
        golpes = datos["golpes"]
        datos["total"].update(_tres(golpes))
        for r in datos["por_round"]:
            dentro = [g for g in golpes if r["inicio_s"] <= g["t_inicio"] < r["fin_s"]]
            r.update(_tres(dentro))
    return fc


def _tres(golpes: list[dict]) -> dict:
    cab = sum(1 for g in golpes if g.get("objetivo") == "cabeza")
    cue = sum(1 for g in golpes if g.get("objetivo") == "cuerpo")
    return {"cabeza": cab, "cuerpo": cue, "objetivo_sin_dato": len(golpes) - cab - cue}
