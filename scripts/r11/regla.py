"""
BoxTwin - R-11: las reglas del benchmark de perfiles, sin GPU (3-11-2026).

POR QUE EXISTE
  Los perfiles A, B y C no se declaran, se miden (CLAUDE.md §6.1). Las reglas estan fijadas en
  docs/specs/2026-11-03-r11-perfiles/00-preregistro.md antes de medir. Si quedan mezcladas con
  el cronometraje no se pueden probar, y un numero para la tesis no puede quedar sin test.

QUE HACE
  - elegir_tramos: los cuadros del benchmark desde el cache (§4), sin mirar tiempos.
  - estadisticos: mediana, p95 y % sobre presupuesto (§4).
  - pasa: una celda pasa un presupuesto si su p95 de t_total entra en los dos videos (§6).
  - celdas_etapa2: que se mide en la etapa 2 segun la etapa 1 (§3).
  - asignar: los perfiles A, B y C (§6).
  - concordancia: error de muneca contra la base (§5).

USO
  from regla import elegir_tramos, estadisticos, pasa, celdas_etapa2, asignar
  No corre solo.
"""

from __future__ import annotations

import numpy as np

PRESUPUESTOS = (40.0, 66.0)
TAMANOS = ("n", "s", "m", "l")
RESOLUCIONES = (416, 480, 640)
FORMATOS = ("engine", "onnx")
EMPATE_MAP = 0.005          # 0,5 puntos de mAP, en escala 0 a 1
MIN_TRAMO = 150
CUADROS = 2200
WARMUP = 200
MUNECAS = (9, 10)           # coco17
MIN_SCORE = 0.3


def clave(formato: str, tamano: str, imgsz: int) -> str:
    return f"{formato}-{tamano}-{imgsz}"


def elegir_tramos(ok: np.ndarray, cortes, min_largo: int = MIN_TRAMO,
                  total: int = CUADROS) -> list[tuple[int, int]]:
    """Tramos [ini, fin) contiguos con ok, sin corte adentro, desde el principio hasta juntar total."""
    ok = np.asarray(ok, bool)
    bordes = set(int(c) for c in cortes)
    tramos: list[tuple[int, int]] = []
    juntados, i, n = 0, 0, len(ok)
    while i < n and juntados < total:
        if not ok[i]:
            i += 1
            continue
        j = i + 1
        while j < n and ok[j] and j not in bordes:
            j += 1
        if j - i >= min_largo:
            fin = min(j, i + total - juntados)
            tramos.append((i, fin))
            juntados += fin - i
        i = j
    if juntados < total:
        raise ValueError(f"solo hay {juntados} cuadros de {total} con la regla de tramos")
    return tramos


def estadisticos(ms, presupuestos=PRESUPUESTOS) -> dict:
    ms = np.asarray(ms, float)
    d = {"n": int(ms.size), "mediana": float(np.median(ms)), "p95": float(np.percentile(ms, 95))}
    for p in presupuestos:
        d[f"pct_sobre_{int(p)}"] = float((ms > p).mean() * 100)
    return d


def pasa(por_video: dict[str, dict], presupuesto: float) -> bool:
    """por_video: {video: estadisticos de t_total}. Gana el peor video; falta un video, no pasa."""
    return bool(por_video) and all(e["p95"] <= presupuesto for e in por_video.values())


def celdas_etapa2(etapa1: dict[str, dict[str, dict]], presupuesto: float = PRESUPUESTOS[0]) -> list[str]:
    """etapa1: {clave a 640: {video: estadisticos}}. Devuelve las claves a medir en la etapa 2."""
    salida: list[str] = []
    for formato in FORMATOS:
        fallan = [t for t in TAMANOS if not pasa(etapa1.get(clave(formato, t, 640), {}), presupuesto)]
        for t in fallan or ["l"]:
            salida += [clave(formato, t, 480), clave(formato, t, 416)]
    return salida


def _orden_sin_map(c: dict) -> tuple:
    return (TAMANOS.index(c["tamano"]), c["imgsz"])


def mejor(celdas: list[dict]) -> dict | None:
    """Mayor calidad; dentro de 0,5 puntos del mejor, la de menor mediana. Sin mAP: tamano y resolucion."""
    if not celdas:
        return None
    if all(c.get("map") is not None for c in celdas):
        tope = max(c["map"] for c in celdas)
        return min((c for c in celdas if tope - c["map"] < EMPATE_MAP), key=lambda c: c["mediana"])
    return max(celdas, key=_orden_sin_map)


def asignar(celdas: list[dict]) -> dict:
    """
    celdas: {clave, formato, tamano, imgsz, por_video, mediana, map}. Mediana: la peor de los
    videos. Devuelve cada perfil con su clave o None, si colapsa y si es provisional.
    """
    def elegibles(p, formatos):
        return [c for c in celdas if c["formato"] in formatos and pasa(c["por_video"], p)]

    a = mejor(elegibles(40.0, FORMATOS))
    b = mejor(elegibles(66.0, FORMATOS))
    c = mejor(elegibles(66.0, ("onnx",)))
    k = {n: (x["clave"] if x else None) for n, x in (("A", a), ("B", b), ("C", c))}
    usadas = [x for x in (a, b, c) if x]
    return {
        **k,
        "B_colapsa_en_A": k["B"] is not None and k["B"] == k["A"],
        "C_colapsa_en": next((n for n in ("A", "B") if k["C"] is not None and k["C"] == k[n]), None),
        "provisional": any(x.get("map") is None for x in celdas) if usadas else False,
    }


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def concordancia(cuadros) -> dict:
    """
    cuadros: iterable de (cajas_base, kp_base, sc_base, cajas, kp, sc). Empareja por IoU >= 0,5,
    con la mayor primero. Error de muneca sobre el alto de la caja base, solo con puntaje >= 0,3
    en las dos. Fraccion de munecas con puntaje >= 0,3 en la celda, sobre las emparejadas.
    """
    errores: list[float] = []
    muestras = visibles = emparejadas = base = 0
    for cb, kb, sb, cc, kc, sc in cuadros:
        base += len(cb)
        pares = sorted(((_iou(x, y), i, j) for i, x in enumerate(cb) for j, y in enumerate(cc)),
                       reverse=True)
        usados_i, usados_j = set(), set()
        for v, i, j in pares:
            if v < 0.5 or i in usados_i or j in usados_j:
                continue
            usados_i.add(i)
            usados_j.add(j)
            emparejadas += 1
            alto = max(float(cb[i][3] - cb[i][1]), 1.0)
            for m in MUNECAS:
                muestras += 1
                visibles += int(sc[j][m] >= MIN_SCORE)
                if sb[i][m] >= MIN_SCORE and sc[j][m] >= MIN_SCORE:
                    errores.append(float(np.linalg.norm(kb[i][m] - kc[j][m])) / alto)
    return {
        "cajas_base": base,
        "emparejadas": emparejadas,
        "fraccion_emparejada": emparejadas / base if base else None,
        "fraccion_munecas_visibles": visibles / muestras if muestras else None,
        "error_muneca_mediana": float(np.median(errores)) if errores else None,
        "error_muneca_p95": float(np.percentile(errores, 95)) if errores else None,
        "n_error": len(errores),
    }
