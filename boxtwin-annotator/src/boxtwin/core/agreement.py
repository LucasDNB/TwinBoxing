"""
BoxTwin - Reporte de acuerdo entre la anotacion y su reanotacion ciega.

POR QUE EXISTE
  Es el numero que hace defendible un dataset propio. Sin el, "anote 800 golpes" es una
  afirmacion sobre el esfuerzo y no sobre la calidad.

  Se reporta kappa de Cohen y no porcentaje de acuerdo. Con seis clases desbalanceadas, un
  anotador que responda siempre la clase mas frecuente puede sacar 40% de acuerdo sin saber
  nada; kappa descuenta el acuerdo esperado por azar y ese caso da cero. El porcentaje se
  reporta igual, al lado, porque es lo que la gente entiende de un vistazo, pero no es la
  medida.

  Se reporta por dimension y no sobre la clase colapsada. Si el acuerdo baja, hace falta
  saber si baja en el tipo de golpe, en el brazo o en la altura, porque cada uno se corrige
  de manera distinta: el brazo con mejor overlay, el tipo con mejor definicion operacional.

  De las fronteras se reporta el error absoluto medio y tambien el SESGO con signo. Son dos
  problemas distintos: ruido simetrico significa que la definicion es dificil de aplicar con
  precision; un sesgo consistente significa que la definicion se esta interpretando distinto,
  y eso se arregla cambiando la definicion, no esforzandose mas.

QUE HACE
  Arma matriz de confusion y kappa por dimension, y estadisticas de error de frontera.

USO
  reporte = comparar(doc, doc_re)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from boxtwin.core.reanno import ReannoDoc
from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import Completeness, PunchType, Side, Target

__all__ = ["DIMENSIONES", "Acuerdo", "cohen_kappa", "matriz_confusion", "comparar"]

# Las cuatro que pide el enunciado. landed y quality quedan afuera del reporte principal
# porque son estimaciones del anotador sobre algo que un sistema monocular no observa, y
# mezclarlas bajaria el kappa por una razon que no es calidad de anotacion.
DIMENSIONES: dict[str, Sequence] = {
    "side": list(Side),
    "punch_type": list(PunchType),
    "target": list(Target),
    "completeness": list(Completeness),
}

# Por debajo de esto, kappa es demasiado inestable para reportarlo sin advertencia.
N_MINIMO = 20


@dataclass
class Acuerdo:
    dimension: str
    etiquetas: list[str]
    matriz: list[list[int]]
    n: int
    acuerdo_observado: float
    acuerdo_esperado: float
    kappa: float | None
    nota: str = ""


@dataclass
class Reporte:
    n_intentos: int
    n_ciegos: int
    n_revelados: int
    dimensiones: dict[str, Acuerdo] = field(default_factory=dict)
    fronteras: dict[str, Any] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


def matriz_confusion(
    a: Sequence[str], b: Sequence[str], etiquetas: Sequence[str]
) -> list[list[int]]:
    """Filas: la anotacion original. Columnas: la reanotacion."""
    indice = {e: i for i, e in enumerate(etiquetas)}
    m = [[0] * len(etiquetas) for _ in etiquetas]
    for x, y in zip(a, b):
        m[indice[x]][indice[y]] += 1
    return m


def cohen_kappa(matriz: list[list[int]]) -> tuple[float, float, float | None, str]:
    """
    (acuerdo observado, acuerdo esperado, kappa, nota).

    kappa queda en None cuando el acuerdo esperado es 1, que pasa si los dos anotadores
    usaron una sola categoria. Ahi la formula divide por cero y el valor no existe: no es
    acuerdo perfecto ni nulo, es que la medida no aplica.
    """
    n = sum(sum(fila) for fila in matriz)
    if n == 0:
        return 0.0, 0.0, None, "sin intentos"

    po = sum(matriz[i][i] for i in range(len(matriz))) / n
    filas = [sum(f) for f in matriz]
    columnas = [sum(matriz[i][j] for i in range(len(matriz))) for j in range(len(matriz))]
    pe = sum(filas[i] * columnas[i] for i in range(len(matriz))) / (n * n)

    if abs(1.0 - pe) < 1e-12:
        return po, pe, None, "kappa no definida: una sola categoria usada por ambos"
    return po, pe, (po - pe) / (1 - pe), ""


def _valor(obj, campo: str) -> str:
    v = getattr(obj, campo)
    return v.value if hasattr(v, "value") else str(v)


def comparar(doc: AnnotationDoc, re_doc: ReannoDoc, *, solo_ciegos: bool = True) -> Reporte:
    """
    Compara la anotacion original contra su reanotacion.

    Por defecto solo entran los intentos ciegos. Un intento donde el reanotador miro la
    etiqueta previa mide otra cosa: mide si acepta lo que ya habia, no si llega a lo mismo
    por su cuenta.
    """
    intentos = re_doc.trials
    ciegos = [t for t in intentos if not t.revealed]
    usados = ciegos if solo_ciegos else intentos

    reporte = Reporte(
        n_intentos=len(intentos),
        n_ciegos=len(ciegos),
        n_revelados=len(intentos) - len(ciegos),
    )

    pares = []
    for t in usados:
        original = doc.event_by_id(t.event_id)
        if original is None:
            reporte.avisos.append(
                f"el evento {t.event_id} ya no existe en la anotacion; se omite del reporte"
            )
            continue
        pares.append((original, t))

    if not pares:
        reporte.avisos.append("no hay intentos comparables")
        return reporte

    for dim, valores in DIMENSIONES.items():
        etiquetas = [v.value for v in valores]
        a = [_valor(o, dim) for o, _ in pares]
        b = [_valor(t.labels, dim) for _, t in pares]
        m = matriz_confusion(a, b, etiquetas)
        po, pe, kappa, nota = cohen_kappa(m)
        reporte.dimensiones[dim] = Acuerdo(
            dimension=dim, etiquetas=etiquetas, matriz=m, n=len(pares),
            acuerdo_observado=round(po, 4), acuerdo_esperado=round(pe, 4),
            kappa=None if kappa is None else round(kappa, 4), nota=nota,
        )

    # -- fronteras
    d_ini = [t.labels.start_frame - o.start_frame for o, t in pares]
    d_fin = [t.labels.end_frame - o.end_frame for o, t in pares]
    d_dur = [
        (t.labels.end_frame - t.labels.start_frame) - (o.end_frame - o.start_frame)
        for o, t in pares
    ]
    reporte.fronteras = {
        "n": len(pares),
        "start": _resumen_error(d_ini),
        "end": _resumen_error(d_fin),
        "duracion": _resumen_error(d_dur),
    }

    if len(pares) < N_MINIMO:
        reporte.avisos.append(
            f"solo {len(pares)} intentos comparados: kappa es inestable con n bajo y el "
            "intervalo de confianza es amplio. No conviene reportar el valor sin el n."
        )
    if reporte.n_revelados:
        reporte.avisos.append(
            f"{reporte.n_revelados} intentos tuvieron la etiqueta revelada y no entran en "
            "el reporte: no fueron ciegos"
        )
    for dim, ac in reporte.dimensiones.items():
        if ac.kappa is not None and ac.kappa < 0.6:
            reporte.avisos.append(
                f"kappa de {dim} es {ac.kappa:.2f}: acuerdo bajo, revisar la definicion "
                "operacional de esa dimension antes de seguir anotando"
            )
    return reporte


def _resumen_error(deltas: list[int]) -> dict[str, Any]:
    """
    Error absoluto medio y sesgo con signo.

    El sesgo es lo que distingue una definicion dificil de aplicar de una definicion
    entendida distinto: ruido simetrico da sesgo cerca de cero, interpretacion distinta da
    un sesgo consistente.
    """
    if not deltas:
        return {"mae": None, "sesgo": None}
    absolutos = [abs(d) for d in deltas]
    return {
        "mae": round(statistics.fmean(absolutos), 2),
        "sesgo": round(statistics.fmean(deltas), 2),
        "mediana_abs": statistics.median(absolutos),
        "max_abs": max(absolutos),
        "exactos": sum(1 for d in deltas if d == 0),
    }


def a_dict(r: Reporte) -> dict[str, Any]:
    return {
        "n_intentos": r.n_intentos,
        "n_ciegos": r.n_ciegos,
        "n_revelados": r.n_revelados,
        "dimensiones": {
            k: {
                "etiquetas": v.etiquetas,
                "matriz": v.matriz,
                "n": v.n,
                "acuerdo_observado": v.acuerdo_observado,
                "acuerdo_esperado": v.acuerdo_esperado,
                "kappa": v.kappa,
                "nota": v.nota,
            }
            for k, v in r.dimensiones.items()
        },
        "fronteras": r.fronteras,
        "avisos": r.avisos,
    }


def a_texto(r: Reporte) -> str:
    """Resumen legible. Es lo que se pega en el capitulo, asi que declara siempre el n."""
    lineas = [
        "BoxTwin - acuerdo intra-anotador",
        f"intentos: {r.n_intentos} ({r.n_ciegos} ciegos, {r.n_revelados} revelados)",
        "",
    ]
    if not r.dimensiones:
        return "\n".join(lineas + ["sin intentos comparables", ""])

    lineas.append(f"{'dimension':14s} {'n':>4s} {'acuerdo':>8s} {'azar':>6s} {'kappa':>7s}")
    for dim, ac in r.dimensiones.items():
        k = "n/d" if ac.kappa is None else f"{ac.kappa:.3f}"
        lineas.append(
            f"{dim:14s} {ac.n:4d} {ac.acuerdo_observado:8.3f} "
            f"{ac.acuerdo_esperado:6.3f} {k:>7s}"
        )
        if ac.nota:
            lineas.append(f"  ({ac.nota})")

    lineas += ["", "fronteras, en cuadros"]
    for campo in ("start", "end", "duracion"):
        d = r.fronteras.get(campo, {})
        if d.get("mae") is None:
            continue
        lineas.append(
            f"  {campo:9s} mae {d['mae']:6.2f}  sesgo {d['sesgo']:+6.2f}  "
            f"mediana {d['mediana_abs']:4.1f}  max {d['max_abs']:3d}  "
            f"exactos {d['exactos']}/{r.fronteras['n']}"
        )

    if r.avisos:
        lineas += ["", "observaciones"]
        lineas += [f"  - {a}" for a in r.avisos]

    lineas += [
        "",
        "matrices de confusion (filas: anotacion original, columnas: reanotacion)",
    ]
    for dim, ac in r.dimensiones.items():
        lineas.append(f"  {dim}")
        ancho = max(len(e) for e in ac.etiquetas)
        lineas.append("    " + " " * ancho + " " + " ".join(f"{e[:6]:>6s}" for e in ac.etiquetas))
        for i, e in enumerate(ac.etiquetas):
            lineas.append(
                f"    {e:>{ancho}s} " + " ".join(f"{v:6d}" for v in ac.matriz[i])
            )
    lineas.append("")
    return "\n".join(lineas)
