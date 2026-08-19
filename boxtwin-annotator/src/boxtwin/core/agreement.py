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

  El emparejamiento entre lo anotado y lo reanotado es un paso EXPLICITO con su umbral
  declarado, y no una correspondencia dada por el protocolo. Tiene que serlo: cada intento
  pide todos los golpes de un peleador en una ventana, asi que cuantos hay y cual va con cual
  es parte de lo que se mide, no un dato de entrada.

  De ahi salen dos familias de numeros que no hay que confundir. La DETECCION dice si el
  golpe se encontro: uno de la anotacion sin pareja es una omision, uno de la reanotacion sin
  pareja es un golpe que la primera pasada no marco. La CLASIFICACION dice si, sobre los que
  las dos pasadas encontraron, la etiqueta coincide. Un kappa alto con recall bajo describe a
  alguien consistente en lo que ve y que ve poco, y eso no se arregla igual que lo contrario.

QUE HACE
  Empareja por solapamiento temporal, arma matriz de confusion y kappa por dimension,
  estadisticas de error de frontera, y precision y recall de deteccion.

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

__all__ = [
    "DIMENSIONES", "Acuerdo", "cohen_kappa", "matriz_confusion", "comparar",
    "emparejar", "iou_temporal", "IOU_MINIMO",
]

# Solapamiento temporal minimo para considerar que dos marcas son el mismo golpe.
#
# 0,3 y no 0,5. Los golpes duran unos 10 cuadros, asi que exigir 0,5 descartaria parejas que
# difieren en 3 o 4 cuadros de frontera, y ese error de frontera es justo lo que se quiere
# MEDIR, no excluir. Un umbral que descarta los casos dificiles reporta el error de los
# faciles. Con 0,3 entra todo lo que un humano llamaria "el mismo golpe".
IOU_MINIMO = 0.3

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
    deteccion: dict[str, Any] = field(default_factory=dict)
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


def iou_temporal(a1: int, a2: int, b1: int, b2: int) -> float:
    """Solapamiento sobre union de dos rangos inclusivos de cuadros."""
    inter = max(0, min(a2, b2) - max(a1, b1) + 1)
    if inter == 0:
        return 0.0
    union = (a2 - a1 + 1) + (b2 - b1 + 1) - inter
    return inter / union if union > 0 else 0.0


def emparejar(
    originales: Sequence, reanotados: Sequence, *, min_iou: float = IOU_MINIMO
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Empareja golpes por solapamiento temporal. Devuelve (parejas, omitidos, agregados).

    Codicioso por solapamiento descendente y uno a uno. Codicioso y no optimo global a
    proposito: con dos o tres golpes por ventana las dos soluciones coinciden, y el codicioso
    se explica en una linea en el capitulo, que para un numero que hay que defender vale mas
    que un decimal.

    Uno a uno importa: sin esa restriccion un golpe reanotado largo se emparejaria con los
    dos golpes de un 1-2 y el recall saldria inflado.
    """
    candidatos = []
    for i, o in enumerate(originales):
        for j, r in enumerate(reanotados):
            v = iou_temporal(o.start_frame, o.end_frame, r.start_frame, r.end_frame)
            if v >= min_iou:
                candidatos.append((v, i, j))
    candidatos.sort(key=lambda x: (-x[0], x[1], x[2]))

    parejas: list[tuple[int, int, float]] = []
    usados_o: set[int] = set()
    usados_r: set[int] = set()
    for v, i, j in candidatos:
        if i in usados_o or j in usados_r:
            continue
        parejas.append((i, j, v))
        usados_o.add(i)
        usados_r.add(j)

    return (
        parejas,
        [i for i in range(len(originales)) if i not in usados_o],
        [j for j in range(len(reanotados)) if j not in usados_r],
    )


def comparar(doc: AnnotationDoc, re_doc: ReannoDoc, *, solo_ciegos: bool = True) -> Reporte:
    """
    Compara la anotacion original contra su reanotacion.

    Por defecto solo entran los intentos ciegos. Un intento donde el reanotador miro la
    etiqueta previa mide otra cosa: mide si acepta lo que ya habia, no si llega a lo mismo
    por su cuenta.

    Dos reglas que parecen detalles y no lo son.

    Un golpe que asoma por el borde de la ventana no se puede reanotar bien, asi que no se
    cuenta como omision. Pero TAMPOCO se lo trata como inexistente: si el reanotador lo marco,
    su marca se empareja con el y no se cuenta como agregada. La primera version solo hacia lo
    primero, y sobre la corrida real 8 de 12 "agregados" eran eso: golpes anotados que asomaban
    por el borde, marcados correctamente y contados como inventados. La precision salia 0,83
    cuando era 0,94.

    Y cada evento anotado entra UNA sola vez, aunque caiga en varias ventanas. Las ventanas se
    solapan, asi que sin deduplicar el mismo golpe aporta dos observaciones: sobre la corrida
    real, 60 parejas eran 52 eventos distintos. Contar dos veces no agrega informacion y
    estrecha el intervalo de kappa mas de lo que corresponde.
    """
    intentos = re_doc.trials
    ciegos = [t for t in intentos if not t.revealed]
    usados = ciegos if solo_ciegos else intentos

    reporte = Reporte(
        n_intentos=len(intentos),
        n_ciegos=len(ciegos),
        n_revelados=len(intentos) - len(ciegos),
    )

    # La deteccion se cuenta por evento unico y la clasificacion tambien, pero la precision
    # se cuenta por marca del reanotador: son denominadores distintos y mezclarlos hace que
    # las cuentas no cierren.
    mejor_por_evento: dict[str, tuple[float, Any, Any]] = {}
    exigibles: set[str] = set()      # eventos que caen enteros en alguna ventana
    encontrados: set[str] = set()    # de esos, los que el reanotador marco
    n_reanotados = n_marcas_emparejadas = n_de_borde = 0
    ventanas_vacias = 0
    de_v1 = 0

    for t in usados:
        if getattr(t, "protocolo_v1", False):
            de_v1 += 1
        if doc.event_by_id(t.event_id) is None:
            reporte.avisos.append(
                f"el evento ancla {t.event_id} ya no existe en la anotacion; se omite"
            )
            continue
        try:
            v_ini, v_fin = re_doc.ventana(t.event_id)
        except KeyError:
            reporte.avisos.append(f"el intento {t.event_id} no tiene ventana; se omite")
            continue

        candidatos = sorted(
            (
                e for e in doc.events
                if e.fighter == t.fighter and e.start_frame <= v_fin and e.end_frame >= v_ini
            ),
            key=lambda e: e.start_frame,
        )
        # Enteros dentro de la ventana: son los unicos exigibles.
        enteros = {
            e.id for e in candidatos if v_ini <= e.start_frame and e.end_frame <= v_fin
        }
        parejas, omitidos, agregados = emparejar(candidatos, t.punches)

        exigibles |= enteros
        n_reanotados += len(t.punches)
        n_marcas_emparejadas += len(parejas)
        if not t.punches:
            ventanas_vacias += 1
        for i, j, v in parejas:
            o = candidatos[i]
            if o.id not in enteros:
                # Asomaba por el borde: se empareja para no contarlo como agregado, pero no
                # entra al acuerdo, porque el reanotador no lo vio completo.
                n_de_borde += 1
                continue
            encontrados.add(o.id)
            # Un evento en dos ventanas es una sola observacion: se queda la mejor.
            previo = mejor_por_evento.get(o.id)
            if previo is None or v > previo[0]:
                mejor_por_evento[o.id] = (v, o, t.punches[j])

    pares: list[tuple[Any, Any]] = [(o, r) for _, o, r in mejor_por_evento.values()]
    omitidos = sorted(exigibles - encontrados)
    agregadas = n_reanotados - n_marcas_emparejadas

    reporte.deteccion = {
        "golpes_exigibles": len(exigibles),
        "golpes_encontrados": len(encontrados),
        "omitidos": len(omitidos),
        "ids_omitidos": omitidos,
        "marcas_del_reanotador": n_reanotados,
        "marcas_sin_correspondencia": agregadas,
        "marcas_en_el_borde": n_de_borde,
        # Un evento que cae en dos ventanas se marca dos veces. Las dos marcas son correctas
        # y cuentan para la precision, pero al acuerdo entra una sola.
        "marcas_repetidas": n_marcas_emparejadas - len(encontrados) - n_de_borde,
        "recall": round(len(encontrados) / len(exigibles), 4) if exigibles else None,
        "precision": (
            round(n_marcas_emparejadas / n_reanotados, 4) if n_reanotados else None
        ),
        "iou_minimo": IOU_MINIMO,
        "ventanas_sin_ningun_golpe": ventanas_vacias,
    }

    if not pares:
        reporte.avisos.append("no hay golpes emparejados; no se puede calcular acuerdo")
        return reporte

    for dim, valores in DIMENSIONES.items():
        etiquetas = [v.value for v in valores]
        a = [_valor(o, dim) for o, _ in pares]
        b = [_valor(r, dim) for _, r in pares]
        m = matriz_confusion(a, b, etiquetas)
        po, pe, kappa, nota = cohen_kappa(m)
        reporte.dimensiones[dim] = Acuerdo(
            dimension=dim, etiquetas=etiquetas, matriz=m, n=len(pares),
            acuerdo_observado=round(po, 4), acuerdo_esperado=round(pe, 4),
            kappa=None if kappa is None else round(kappa, 4), nota=nota,
        )

    # -- fronteras
    d_ini = [r.start_frame - o.start_frame for o, r in pares]
    d_fin = [r.end_frame - o.end_frame for o, r in pares]
    d_dur = [(r.end_frame - r.start_frame) - (o.end_frame - o.start_frame) for o, r in pares]
    reporte.fronteras = {
        "n": len(pares),
        "start": _resumen_error(d_ini),
        "end": _resumen_error(d_fin),
        "duracion": _resumen_error(d_dur),
    }

    if len(pares) < N_MINIMO:
        reporte.avisos.append(
            f"solo {len(pares)} golpes emparejados: kappa es inestable con n bajo y el "
            "intervalo de confianza es amplio. No conviene reportar el valor sin el n."
        )
    if de_v1:
        reporte.avisos.append(
            f"{de_v1} intentos vienen del protocolo v1, que pedia reanotar un evento sin "
            "decir cual. Sobre material con combinaciones eso hace que se reanote el golpe "
            "de al lado, y ademas cada intento podia marcar un solo golpe por construccion: "
            "sus numeros de deteccion no significan nada. Rehacer la muestra antes de "
            "reportar."
        )
    d = reporte.deteccion
    if d.get("recall") is not None and d["recall"] < 0.9:
        reporte.avisos.append(
            f"recall de deteccion {d['recall']:.2f}: {d['omitidos']} golpes de la anotacion "
            "no aparecieron en la reanotacion. Eso se mira antes que kappa: una etiqueta "
            "consistente sobre los golpes que se ven no dice nada de los que no se ven."
        )
    if d.get("precision") is not None and d["precision"] < 0.9:
        reporte.avisos.append(
            f"precision de deteccion {d['precision']:.2f}: "
            f"{d['marcas_sin_correspondencia']} golpes marcados "
            "en la reanotacion no estaban en la anotacion. Pueden ser golpes que la primera "
            "pasada se perdio, y en ese caso el dataset esta incompleto."
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
    d = r.deteccion
    if d:
        # La deteccion va PRIMERO. Un kappa alto sobre los golpes que las dos pasadas
        # encontraron no dice nada de los que una de las dos no vio, y leerlo sin el recall
        # al lado invita a esa confusion.
        rec = "n/d" if d.get("recall") is None else f"{d['recall']:.3f}"
        pre = "n/d" if d.get("precision") is None else f"{d['precision']:.3f}"
        lineas += [
            "deteccion (emparejado por solapamiento temporal >= "
            f"{d['iou_minimo']:.2f})",
            f"  golpes exigibles         {d['golpes_exigibles']:4d}   "
            "(anotados y enteros dentro de alguna ventana)",
            f"  encontrados              {d['golpes_encontrados']:4d}",
            f"  omitidos                 {d['omitidos']:4d}   (estaban y no se marcaron)",
            "",
            f"  marcas del reanotador    {d['marcas_del_reanotador']:4d}",
            f"  sin correspondencia      {d['marcas_sin_correspondencia']:4d}   "
            "(se marcaron y no estaban)",
            f"  en el borde              {d['marcas_en_el_borde']:4d}   "
            "(anotadas, pero asomaban: no entran al acuerdo)",
            f"  repetidas                {d['marcas_repetidas']:4d}   "
            "(el mismo golpe visto en dos ventanas)",
            "",
            f"  recall {rec}   precision {pre}",
            "",
        ]

    if not r.dimensiones:
        return "\n".join(lineas + ["sin golpes emparejados", ""])

    lineas.append("clasificacion, sobre los golpes emparejados")
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
