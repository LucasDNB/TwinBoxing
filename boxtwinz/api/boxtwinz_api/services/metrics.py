"""
BoxTwinZ - Metricas por sesion (spec 6.2) y stat general (spec 6.3).

POR QUE EXISTE
  Todo numero que BoxTwinZ muestra sale de aca, y algunos van a la tesis (costo por modalidad,
  valoraciones contra volumen). Por eso son funciones puras, sin base ni FastAPI, con tests
  contra casos calculados a mano (CZ5).

  Ninguna metrica dice "recibio" ni "conecto": se cuentan golpes DETECTADOS, con el recall del
  detector por debajo del real (0,456 como corre el producto).

QUE HACE
  Segmentos, volumen, balance, mano adelantada, combinaciones, sosten del ritmo, iniciativa,
  curva por bloques de 10 s, exposicion por lado y tendencia sobre las ultimas N sesiones.

USO
  from boxtwinz_api.services.metrics import calcular_sesion, segmentos, tendencia
  m = calcular_sesion(golpes, duracion_s=240, guardia="ortodoxa")
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "GolpeT", "segmentos", "volumen", "combinaciones", "sosten_segmento", "curva",
    "metricas_de", "calcular_sesion", "iniciativa", "exposicion", "tendencia", "mano_adelantada",
]

SEGMENTO_S = 180.0
RESTO_MINIMO_S = 60.0
SOSTEN_MINIMO_S = 90.0
BLOQUE_CURVA_S = 10.0


@dataclass(frozen=True)
class GolpeT:
    """Lo minimo que las metricas necesitan de un golpe."""
    t_inicio: float
    mano: str  # "izquierda" | "derecha"


def mano_adelantada(guardia: str | None) -> str:
    """Izquierda en ortodoxa, derecha en zurda. Sin guardia se supone ortodoxa."""
    return "derecha" if guardia == "zurda" else "izquierda"


def segmentos(duracion_s: float, rounds_declarados: int | None = None) -> list[tuple[float, float]]:
    """
    Los segmentos de la sesion como (inicio, fin) en segundos.

    Con rounds declarados, partes iguales. Sin ellos, bloques de 180 s, y si el ultimo dura
    menos de 60 s se suma al anterior: misma regla que A (rounds_origen "segmentos_180s").
    """
    if duracion_s <= 0:
        return []
    if rounds_declarados and rounds_declarados > 0:
        paso = duracion_s / rounds_declarados
        return [(i * paso, duracion_s if i == rounds_declarados - 1 else (i + 1) * paso)
                for i in range(rounds_declarados)]
    cortes: list[tuple[float, float]] = []
    t = 0.0
    while t < duracion_s:
        cortes.append((t, min(t + SEGMENTO_S, duracion_s)))
        t += SEGMENTO_S
    if len(cortes) > 1 and cortes[-1][1] - cortes[-1][0] < RESTO_MINIMO_S:
        ultimo = cortes.pop()
        cortes[-1] = (cortes[-1][0], ultimo[1])
    return cortes


def volumen(n_golpes: int, duracion_s: float) -> float:
    """Golpes detectados por minuto."""
    return n_golpes / (duracion_s / 60.0) if duracion_s > 0 else 0.0


def combinaciones(tiempos: Sequence[float], gap_s: float = 0.5) -> tuple[int, int]:
    """
    (cantidad de combinaciones, golpes que forman parte de una).

    Golpes consecutivos con t_inicio[i+1] - t_inicio[i] < gap forman un grupo; un grupo de 2 o
    mas es UNA combinacion por larga que sea la serie.
    """
    ts = sorted(tiempos)
    if not ts:
        return 0, 0
    combos, en_combo, largo = 0, 0, 1
    for a, b in zip(ts, ts[1:]):
        if b - a < gap_s:
            largo += 1
        else:
            if largo >= 2:
                combos += 1
                en_combo += largo
            largo = 1
    if largo >= 2:
        combos += 1
        en_combo += largo
    return combos, en_combo


def sosten_segmento(tiempos: Iterable[float], inicio: float, fin: float) -> float | None:
    """
    volumen(resto del segmento) / volumen(primer minuto). None si el segmento dura menos de
    90 s o si el primer minuto no tiene golpes (el cociente no esta definido).
    """
    largo = fin - inicio
    if largo < SOSTEN_MINIMO_S:
        return None
    ts = [t for t in tiempos if inicio <= t < fin]
    primero = sum(1 for t in ts if t < inicio + 60.0)
    resto = len(ts) - primero
    if primero == 0:
        return None
    return volumen(resto, largo - 60.0) / volumen(primero, 60.0)


def curva(tiempos: Iterable[float], duracion_s: float) -> list[int]:
    """Golpes por bloque de 10 s. El ultimo bloque puede ser parcial."""
    n = max(1, math.ceil(duracion_s / BLOQUE_CURVA_S)) if duracion_s > 0 else 0
    bins = [0] * n
    for t in tiempos:
        if n and 0 <= t <= duracion_s:
            bins[min(int(t // BLOQUE_CURVA_S), n - 1)] += 1
    return bins


def metricas_de(golpes: Sequence[GolpeT], inicio: float, fin: float, guardia: str | None,
                gap_s: float = 0.5, con_curva: bool = True) -> dict:
    """Las metricas de un tramo [inicio, fin): un segmento o la sesion entera."""
    g = sorted((x for x in golpes if inicio <= x.t_inicio < fin), key=lambda x: x.t_inicio)
    dur = fin - inicio
    n = len(g)
    izq = sum(1 for x in g if x.mano == "izquierda")
    adel = mano_adelantada(guardia)
    combos, en_combo = combinaciones([x.t_inicio for x in g], gap_s)
    return {
        "golpes": n,
        "volumen": volumen(n, dur),
        "pct_izquierda": izq / n if n else None,
        "pct_mano_adelantada": sum(1 for x in g if x.mano == adel) / n if n else None,
        "combinaciones_min": combos / (dur / 60.0) if dur > 0 else 0.0,
        "pct_golpes_en_combinacion": en_combo / n if n else None,
        "sosten_ritmo": sosten_segmento([x.t_inicio for x in g], inicio, fin),
        "curva": curva([x.t_inicio - inicio for x in g], dur) if con_curva else [],
    }


def calcular_sesion(golpes: Sequence[GolpeT], duracion_s: float, guardia: str | None = "ortodoxa",
                    rounds_declarados: int | None = None, gap_s: float = 0.5) -> dict:
    """
    Total y por segmento. El sosten del total es el promedio de los sostenes de los segmentos
    que lo tienen definido (spec 6.2), no el sosten de la sesion entera vista como un tramo.
    """
    segs = segmentos(duracion_s, rounds_declarados)
    por_seg = [metricas_de(golpes, a, b, guardia, gap_s, con_curva=False) for a, b in segs]
    total = metricas_de(golpes, 0.0, duracion_s + 1e-9, guardia, gap_s)
    total["curva"] = curva([x.t_inicio for x in golpes], duracion_s)
    sostenes = [s["sosten_ritmo"] for s in por_seg if s["sosten_ritmo"] is not None]
    total["sosten_ritmo"] = sum(sostenes) / len(sostenes) if sostenes else None
    for s, (a, b) in zip(por_seg, segs):
        s["inicio_s"], s["fin_s"] = a, b
    return {"total": total, "segmentos": por_seg}


def iniciativa(n_usuario: int, n_rival: int) -> float | None:
    tot = n_usuario + n_rival
    return n_usuario / tot if tot else None


def exposicion(golpes_rival: Iterable[GolpeT]) -> dict:
    """
    Golpes DIRIGIDOS del rival por mano y por lado del usuario. Enfrentados, la izquierda del
    rival llega por el lado derecho del usuario y la derecha por el izquierdo. No dice que
    hayan llegado: un sistema monocular no establece contacto.
    """
    izq = der = 0
    for g in golpes_rival:
        if g.mano == "izquierda":
            izq += 1
        elif g.mano == "derecha":
            der += 1
    return {"rival_izquierda": izq, "rival_derecha": der,
            "lado_izquierdo_usuario": der, "lado_derecho_usuario": izq}


def tendencia(valores: Sequence[float], banda: float = 0.05) -> str | None:
    """
    "subiendo", "estable" o "bajando" sobre la serie en orden cronologico.

    Pendiente de minimos cuadrados sobre el indice de sesion, llevada a cambio relativo a lo
    largo de la ventana: pendiente * (n - 1) / |media|. Dentro de +-banda es estable. Con menos
    de dos valores no hay tendencia.
    """
    n = len(valores)
    if n < 2:
        return None
    xm = (n - 1) / 2
    ym = sum(valores) / n
    den = sum((i - xm) ** 2 for i in range(n))
    pendiente = sum((i - xm) * (v - ym) for i, v in enumerate(valores)) / den
    if ym == 0:
        return "estable" if pendiente == 0 else ("subiendo" if pendiente > 0 else "bajando")
    rel = pendiente * (n - 1) / abs(ym)
    if rel > banda:
        return "subiendo"
    if rel < -banda:
        return "bajando"
    return "estable"
