"""
BoxTwin - Verificador de apariencia: guante, remera y pantalon (IE7).

POR QUE EXISTE
  IE6 (3-11) dio "no estable": el 98% de los cuadros con rol equivocado que nunca llegan a la
  cola esta en 03-sparring, donde los perfiles del guante separan 0,348 y el piso de 0,55
  apaga todos los detectores de sospecha por color. El diagnostico de senales sobre las seis
  fuentes de C1 mostro que la remera y el pantalon separan donde el guante no (03: 0,919 por
  muestra, 0,879 en contacto) y que ninguna senal sirve en todas. Pre-registro:
  docs/specs/2026-11-02-identidad-estable/02-ie7-apariencia.md.

QUE HACE
  - Mide el color de la remera y del pantalon de una deteccion desde su pose.
  - Arma, con las dos semillas, un modelo por video: perfiles A y B de cada senal y un peso
    por senal, logit de su acierto en bloques de tiempo de las propias semillas (sin verdad).
    Una senal que no separa en ese video pesa cero.
  - Da, por cuadro de un tramo, v en [-1, 1]: el voto ponderado de las senales presentes
    (positivo es "mas A"), y los dos motivos de sospecha que salen de ahi.
  No asigna nada: solo dice que tramo hay que mirar.

USO
  modelo = ajustar(evidencia, (semilla_a, semilla_b))
  motivo = sospecha(modelo, evidencia[track], "A", desde, hasta, cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from boxtwin.core.deteccion_guantes import Color, color_de_parche, distancia_color, perfil_de

__all__ = ["ModeloApariencia", "SENALES", "ajustar", "medir_ropa", "serie", "sospecha",
           "zona_pantalon", "zona_remera"]

HOMBROS, CADERAS, RODILLAS = (5, 6), (11, 12), (13, 14)
KP_MIN = 0.3                     # puntaje minimo de cada keypoint que define una zona
FRACCION_PANTALON = 0.4          # de la cadera hacia la rodilla: el short, no la pierna
BLOQUES = 4                      # bloques de tiempo para estimar la confiabilidad
R_MIN, R_MAX = 0.5, 0.99         # recorte de la confiabilidad antes del logit
SENALES = ("guante", "remera", "pantalon")


# ---------------------------------------------------------------------------
# Medir
# ---------------------------------------------------------------------------


def zona_remera(kp: np.ndarray, sc: np.ndarray) -> tuple[int, int, int, int] | None:
    """Caja (x0, y0, x1, y1) de hombros a caderas, o None si la pose no la sostiene."""
    idx = list(HOMBROS + CADERAS)
    if (np.asarray(sc)[idx] < KP_MIN).any():
        return None
    kp = np.asarray(kp)
    x0, x1 = kp[idx, 0].min(), kp[idx, 0].max()
    y0, y1 = kp[list(HOMBROS), 1].mean(), kp[list(CADERAS), 1].mean()
    if y1 - y0 < 8 or x1 - x0 < 6:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def zona_pantalon(kp: np.ndarray, sc: np.ndarray) -> tuple[int, int, int, int] | None:
    """Caja desde las caderas hasta FRACCION_PANTALON del camino a las rodillas."""
    idx = list(CADERAS + RODILLAS)
    if (np.asarray(sc)[idx] < KP_MIN).any():
        return None
    kp = np.asarray(kp)
    yc, yr = kp[list(CADERAS), 1].mean(), kp[list(RODILLAS), 1].mean()
    y1 = yc + FRACCION_PANTALON * (yr - yc)
    x0, x1 = kp[idx, 0].min(), kp[idx, 0].max()
    if y1 - yc < 6 or x1 - x0 < 6:
        return None
    return int(x0), int(yc), int(x1), int(y1)


def medir_ropa(frame: np.ndarray, kp: np.ndarray, sc: np.ndarray) -> tuple[Color | None, Color | None]:
    """Color de la remera y del pantalon de una deteccion, o None donde la pose no alcanza."""
    salida = []
    for zona in (zona_remera, zona_pantalon):
        z = zona(kp, sc)
        salida.append(color_de_parche(frame[max(z[1], 0):z[3], max(z[0], 0):z[2]]) if z else None)
    return salida[0], salida[1]


# ---------------------------------------------------------------------------
# Modelo por video
# ---------------------------------------------------------------------------


@dataclass
class ModeloApariencia:
    """Perfiles A y B y peso de cada senal, sacados de las semillas de un video."""

    perfiles: dict[str, tuple[Color, Color]] = field(default_factory=dict)
    confiabilidad: dict[str, float] = field(default_factory=dict)
    pesos: dict[str, float] = field(default_factory=dict)

    @property
    def util(self) -> bool:
        return any(w > 0 for w in self.pesos.values())

    def a_dict(self) -> dict:
        return {"confiabilidad": {k: round(v, 3) for k, v in self.confiabilidad.items()},
                "pesos": {k: round(v, 3) for k, v in self.pesos.items()}}


def _muestras(e, senal: str, solo_aisladas: bool) -> list[tuple[int, Color]]:
    """(cuadro, color) de una senal en la evidencia de un track."""
    if e is None:
        return []
    if senal == "guante":
        return [(f, tuple(c)) for f, c in e.colores]
    k = 1 if senal == "remera" else 2
    return [(r[0], tuple(r[k])) for r in getattr(e, "ropa", [])
            if r[k] is not None and (r[3] or not solo_aisladas)]


def _signo(c: Color, pa: Color, pb: Color) -> float:
    return float(np.sign(distancia_color(c, pb) - distancia_color(c, pa)))


def confiabilidad(ma: list[tuple[int, Color]], mb: list[tuple[int, Color]],
                  bloques: int = BLOQUES) -> float | None:
    """
    Acierto de una senal sobre las propias semillas, en bloques de tiempo: perfiles con los
    otros bloques, acierto en el que queda. Bloques y no cuadros alternados, porque cuadros
    vecinos se parecen y el acierto saldria inflado.
    """
    ma, mb = sorted(ma), sorted(mb)
    bien = total = 0
    for k in range(bloques):
        def parte(ms, dentro):
            n = len(ms)
            return [c for i, (_, c) in enumerate(ms) if ((i * bloques // n) == k) == dentro]
        tra, trb = parte(ma, False), parte(mb, False)
        if not tra or not trb:
            continue
        pa, pb = perfil_de(tra), perfil_de(trb)
        for c in parte(ma, True):
            s = _signo(c, pa, pb)
            total += s != 0
            bien += s > 0
        for c in parte(mb, True):
            s = _signo(c, pa, pb)
            total += s != 0
            bien += s < 0
    return bien / total if total else None


def ajustar(evidencia: dict, semillas: tuple[int, int]) -> ModeloApariencia:
    """El modelo del video con las dos semillas (A, B). Solo muestras aisladas."""
    modelo = ModeloApariencia()
    ea, eb = evidencia.get(semillas[0]), evidencia.get(semillas[1])
    for s in SENALES:
        ma, mb = _muestras(ea, s, True), _muestras(eb, s, True)
        if not ma or not mb:
            continue
        r = confiabilidad(ma, mb)
        if r is None:
            continue
        modelo.perfiles[s] = (perfil_de([c for _, c in ma]), perfil_de([c for _, c in mb]))
        modelo.confiabilidad[s] = r
        rc = min(max(r, R_MIN), R_MAX)
        modelo.pesos[s] = math.log(rc / (1 - rc))
    return modelo


# ---------------------------------------------------------------------------
# Senal y sospecha
# ---------------------------------------------------------------------------


def serie(modelo: ModeloApariencia, e, desde: int, hasta: int) -> tuple[list[int], list[float]]:
    """
    Por cuadro del rango [desde, hasta): v = suma de peso * signo / suma de pesos presentes.
    Con dos guantes en el mismo cuadro vale el signo de la suma de los dos.
    """
    por_cuadro: dict[int, dict[str, float]] = {}
    for s in SENALES:
        w = modelo.pesos.get(s, 0.0)
        if w <= 0:
            continue
        pa, pb = modelo.perfiles[s]
        for f, c in _muestras(e, s, False):
            if desde <= f < hasta:
                d = por_cuadro.setdefault(f, {})
                d[s] = d.get(s, 0.0) + distancia_color(c, pb) - distancia_color(c, pa)
    frames, v = [], []
    for f in sorted(por_cuadro):
        d = por_cuadro[f]
        peso = sum(modelo.pesos[s] for s in d)
        frames.append(f)
        v.append(sum(modelo.pesos[s] * float(np.sign(x)) for s, x in d.items()) / peso)
    return frames, v


def sospecha(modelo: ModeloApariencia | None, e, rol: str, desde: int, hasta: int,
             cfg) -> str | None:
    """
    `apariencia_contradice` o `apariencia_mezcla` para un tramo con rol "A" o "B", o None.
    Las mismas reglas y parametros que `color_contradice` y `posible_mezcla`, sobre v.
    """
    from boxtwin.core.identidad_auto import _mejor_corte

    if modelo is None or not modelo.util or rol not in ("A", "B"):
        return None
    frames, v = serie(modelo, e, desde, hasta)
    vals = [(f, x) for f, x in zip(frames, v) if x != 0]
    if len(vals) < cfg.min_guantes_voto:
        return None
    a = sum(1 for _, x in vals if x > 0)
    b = len(vals) - a
    if (rol == "A" and b > a) or (rol == "B" and a > b):
        return "apariencia_contradice"
    if len(vals) >= 2 * cfg.min_guantes_voto and _mejor_corte(
            [f for f, _ in vals], [x for _, x in vals], cfg.min_guantes_voto,
            cfg.margen_orientacion / 2) is not None:
        return "apariencia_mezcla"
    return None
