"""
BoxTwin - Evaluacion del detector, por evento.

POR QUE EXISTE
  La exactitud por cuadro no mide nada acá: decir siempre O acierta 95,3%. Lo que hay que
  saber es cuantos golpes encuentra y cuantas de sus marcas son un golpe, que es una pregunta
  sobre eventos y no sobre cuadros.

  Se usa `boxtwin.core.agreement.emparejar`, el MISMO emparejador con que se midio el acuerdo
  intra-anotador: codicioso por solapamiento descendente, uno a uno, IoU minimo 0,3. Con otro
  emparejador los numeros del detector no se podrian poner al lado de los 0,911 de recall y
  0,903 de precision del humano, y esa comparacion es el punto.

  EL EMPAREJAMIENTO ES POR CARRIL. Dos golpes simultaneos de peleadores distintos no son el
  mismo golpe, y emparejarlos globalmente inflaria el recall justo en los intercambios, que
  es donde mas golpes hay.

  Se reporta ademas el recall sobre los golpes ALCANZABLES, que son los que tienen algun
  cuadro con pose usable. Los otros el sistema no los puede encontrar por definicion, y la
  diferencia entre los dos numeros dice cuanto del error es del modelo y cuanto del
  preproceso.

  Y el recall separado por si el golpe esta sobre pose MEDIDA o RELLENADA. Cuando la
  identidad tiene un hueco, el anotador lo interpola linealmente; eso es la mejor estimacion
  disponible pero no es un dato observado, y una recta entre dos puntos no tiene la firma
  temporal que el detector busca. Pesa muy desparejo -el 23,6% de los golpes de Sparring
  contra 0% en las cuatro fuentes nuevas- asi que un promedio sobre todo mezcla el error del
  detector con el de la anotacion. Separarlos dice cual de los dos se esta midiendo.

QUE HACE
  Compara segmentos predichos contra los anotados y devuelve recall, precision, error de
  fronteras e IoU. Y barre el umbral, que es lo unico honesto cuando hay una perilla.

USO
  r = evaluar(predichos_por_carril, anotados_por_carril)
  curva = barrer(logits_por_carril, fuente)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from boxtwin.core.agreement import IOU_MINIMO, emparejar

from boxtwin_detector.bio import Segmento, segmentos_de
from boxtwin_detector.decodificacion import decodificar, decodificar_score

__all__ = ["evaluar", "barrer", "segmentos_anotados", "alcanzables", "sobre_pose_medida"]


def segmentos_anotados(
    labels: np.ndarray, cobertura: tuple[int, int] | None = None
) -> list[list[Segmento]]:
    """
    Los segmentos de la anotacion, un listado por carril.

    Con `cobertura`, solo los que caen ENTEROS adentro. Es imprescindible al evaluar una
    particion en distribucion: ahi el tensor de etiquetas es el de la fuente completa y solo
    la mascara distingue train de val, asi que sin recortar se contarian como no encontrados
    los golpes de la mitad que ni siquiera se evaluo. Un golpe a caballo del borde tampoco
    entra: no se lo puede detectar entero, y castigarlo seria medir el corte y no el modelo.
    """
    todos = [segmentos_de(labels[c]) for c in range(labels.shape[0])]
    if cobertura is None:
        return todos
    a, b = cobertura
    return [[s for s in segs if s.inicio >= a and s.fin <= b] for segs in todos]


def alcanzables(segs_por_carril: Sequence[Sequence[Segmento]], usable: np.ndarray) -> list[bool]:
    """Por cada golpe anotado, si tiene algun cuadro con pose usable."""
    out: list[bool] = []
    for c, segs in enumerate(segs_por_carril):
        for s in segs:
            out.append(bool(usable[c, s.inicio : s.fin + 1].any()))
    return out


def sobre_pose_medida(
    segs_por_carril: Sequence[Sequence[Segmento]], interpolado: np.ndarray
) -> list[bool]:
    """
    Por cada golpe anotado, si NINGUNO de sus cuadros tiene pose rellenada.

    Basta un cuadro interpolado para contaminarlo: el golpe dura 7 cuadros de mediana, asi
    que uno solo ya es una fraccion grande de su trayectoria.
    """
    out: list[bool] = []
    for c, segs in enumerate(segs_por_carril):
        for s in segs:
            out.append(not bool(interpolado[c, s.inicio : s.fin + 1].any()))
    return out


def evaluar(
    predichos: Sequence[Sequence[Segmento]],
    anotados: Sequence[Sequence[Segmento]],
    usable: np.ndarray | None = None,
    min_iou: float = IOU_MINIMO,
    interpolado: np.ndarray | None = None,
) -> dict:
    """Recall, precision, error de fronteras e IoU, emparejando carril por carril."""
    if len(predichos) != len(anotados):
        raise ValueError("predichos y anotados tienen que tener los mismos carriles")

    n_gt = n_pred = n_par = 0
    d_ini: list[int] = []
    d_fin: list[int] = []
    ious: list[float] = []
    encontrado: list[bool] = []

    for c, (pred, gt) in enumerate(zip(predichos, anotados)):
        parejas, omitidos, _ = emparejar(list(gt), list(pred), min_iou=min_iou)
        n_gt += len(gt)
        n_pred += len(pred)
        n_par += len(parejas)
        hallados = {i for i, _, _ in parejas}
        encontrado += [i in hallados for i in range(len(gt))]
        for i, j, v in parejas:
            d_ini.append(pred[j].inicio - gt[i].inicio)
            d_fin.append(pred[j].fin - gt[i].fin)
            ious.append(v)

    out = {
        "golpes": n_gt,
        "predichos": n_pred,
        "emparejados": n_par,
        "recall": round(n_par / n_gt, 4) if n_gt else 0.0,
        "precision": round(n_par / n_pred, 4) if n_pred else 0.0,
        "iou_medio": round(float(np.mean(ious)), 4) if ious else 0.0,
        "error_inicio": round(float(np.mean(np.abs(d_ini))), 2) if d_ini else None,
        "error_fin": round(float(np.mean(np.abs(d_fin))), 2) if d_fin else None,
        "sesgo_inicio": round(float(np.mean(d_ini)), 2) if d_ini else None,
        "sesgo_fin": round(float(np.mean(d_fin)), 2) if d_fin else None,
        "min_iou": min_iou,
    }
    if usable is not None:
        alc = alcanzables(anotados, usable)
        n_alc = sum(alc)
        hall_alc = sum(1 for e, a in zip(encontrado, alc) if a and e)
        out["golpes_alcanzables"] = n_alc
        out["recall_alcanzables"] = round(hall_alc / n_alc, 4) if n_alc else 0.0
    if interpolado is not None:
        med = sobre_pose_medida(anotados, interpolado)
        n_med = sum(med)
        n_rel = len(med) - n_med
        h_med = sum(1 for e, m in zip(encontrado, med) if m and e)
        h_rel = sum(1 for e, m in zip(encontrado, med) if not m and e)
        out["golpes_medidos"] = n_med
        out["golpes_rellenados"] = n_rel
        out["recall_medidos"] = round(h_med / n_med, 4) if n_med else None
        out["recall_rellenados"] = round(h_rel / n_rel, 4) if n_rel else None
    return out


def barrer(
    logits_por_carril: Sequence[np.ndarray],
    labels: np.ndarray,
    usable: np.ndarray,
    umbrales: Sequence[float],
    cobertura: tuple[int, int] | None = None,
    como_score: bool = False,
    interpolado: np.ndarray | None = None,
    **kw,
) -> list[dict]:
    """
    Un renglon por umbral. Reportar un solo punto de esta curva es reportar una eleccion.

    Con `como_score`, lo que llega no son logits sino una senal cruda por carril: es el
    camino por el que pasa la heuristica, para que la linea de base y el modelo compartan
    decodificador, emparejador y region evaluada.
    """
    gt = segmentos_anotados(labels, cobertura)
    valido = None
    if cobertura is not None:
        valido = np.zeros(labels.shape[1], bool)
        valido[cobertura[0] : cobertura[1] + 1] = True
    filas = []
    for u in umbrales:
        if como_score:
            pred = [decodificar_score(l, u, valido=valido, **kw) for l in logits_por_carril]
        else:
            pred = [decodificar(l, umbral=u, valido=valido, **kw) for l in logits_por_carril]
        fila = evaluar(pred, gt, usable, interpolado=interpolado)
        fila["umbral"] = round(float(u), 3)
        filas.append(fila)
    return filas
