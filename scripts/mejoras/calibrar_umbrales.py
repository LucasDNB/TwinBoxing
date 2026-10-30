"""
BoxTwin - Umbrales de score por articulacion para YOLO26 (E8b, T2, 30-10-2026).

POR QUE EXISTE
  E8 midio YOLO26 con el umbral de 0,3 pensado para YOLOv8, y K3 mostro que YOLO26 reparte sus
  scores mas abajo: munecas con score >= 0,3 de 0,896 a 0,750. E8b
  (docs/specs/2026-10-30-mejoras-post-e8/01-e8b-umbral-calibrado.md §2) separa el estimador del
  umbral: para cada articulacion, el umbral de YOLO26 es el que deja aceptada la misma fraccion
  de keypoints que v8 con 0,3.

QUE HACE
  Poblacion: los keypoints de las detecciones asignadas a A o a B, sobre el denominador comun de
  P12c (los cuadros donde las dos condiciones tienen emparejado a ese peleador, como
  medir_trackeo.py), en las siete fuentes juntas, sin etiquetas de golpe. Por articulacion
  (nariz, hombros, codos, munecas, caderas; izquierda y derecha juntas): la fraccion de v8 con
  0,3, el umbral de YOLO26 que la iguala, y K-cal (la fraccion de YOLO26 con ese umbral queda a
  +-0,005). Escribe umbrales.json con los umbrales, las fracciones, los n y el sha256 de cada
  cache leido. Si K-cal falla, sale con codigo 2.

USO
  python scripts/mejoras/calibrar_umbrales.py
  python scripts/mejoras/calibrar_umbrales.py --out ~/boxtwin-mejoras/e8b/umbrales.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rutas  # noqa: E402

UMBRAL_V8 = 0.3
TOLERANCIA = 0.005
INDICES = {"nariz": (0,), "hombro": (5, 6), "codo": (7, 8), "muneca": (9, 10), "cadera": (11, 12)}


def puntajes(cache, asign: dict, frames: dict) -> dict[str, np.ndarray]:
    """Scores por articulacion de las detecciones asignadas en los cuadros dados."""
    salida: dict[str, list] = {a: [] for a in INDICES}
    for p, fs in frames.items():
        for f in fs:
            dets = cache.detections(f)
            i = dets.index_of_track(asign[p][f])
            if i is None:
                continue
            sc = np.asarray(dets.kp_score[i], np.float64)
            for a, idx in INDICES.items():
                salida[a].extend(sc[list(idx)].tolist())
    return {a: np.asarray(v, np.float64) for a, v in salida.items()}


def umbral_que_iguala(y: np.ndarray, fraccion: float) -> float:
    """El umbral u tal que la fraccion de y >= u es la mas cercana a `fraccion`."""
    ys = np.sort(np.asarray(y, np.float64))[::-1]
    k = int(round(fraccion * len(ys)))
    if k <= 0:
        return float(np.nextafter(ys[0], np.inf))
    return float(ys[min(k, len(ys)) - 1])


def calibrar(v8: dict, y26: dict, umbral_v8: float = UMBRAL_V8) -> dict:
    """La regla de §2: por articulacion, igualar la fraccion aceptada de v8 con 0,3."""
    r: dict = {"umbrales": {}, "fraccion_v8": {}, "fraccion_y26_con_0_3": {}, "n": {}}
    for a in INDICES:
        frac = float((v8[a] >= umbral_v8).mean())
        r["fraccion_v8"][a] = frac
        r["fraccion_y26_con_0_3"][a] = float((y26[a] >= umbral_v8).mean())
        r["umbrales"][a] = umbral_que_iguala(y26[a], frac)
        r["n"][a] = {"v8": int(len(v8[a])), "y26": int(len(y26[a]))}
    return r


def k_cal(y26: dict, umbrales: dict, objetivo: dict, tol: float = TOLERANCIA) -> dict:
    """La fraccion aceptada de YOLO26 con los umbrales calibrados, contra la de v8."""
    por = {}
    for a in INDICES:
        frac = float((y26[a] >= umbrales[a]).mean())
        por[a] = {"fraccion": frac, "objetivo": objetivo[a], "diferencia": frac - objetivo[a]}
    return {"tolerancia": tol, "por_articulacion": por,
            "pasa": all(abs(v["diferencia"]) <= tol for v in por.values())}


def main(argv=None) -> int:
    import comun
    from boxtwin.core.posecache import PoseCache
    from puente_identidad import PELEADORES, _original, emparejar, verdad

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raiz-original", type=Path, default=rutas.REPO_A)
    ap.add_argument("--raiz-v8", type=Path, default=rutas.RAIZ_V8)
    ap.add_argument("--raiz-y26", type=Path, default=rutas.RAIZ_Y26)
    ap.add_argument("--out", type=Path, default=rutas.MEJORAS / "e8b" / "umbrales.json")
    args = ap.parse_args(argv)

    v8: dict[str, list] = {a: [] for a in INDICES}
    y26: dict[str, list] = {a: [] for a in INDICES}
    shas: dict = {"v8": {}, "y26": {}}
    for fuente, d in comun.FUENTES.items():
        doc, cache = _original(args.raiz_original, fuente)
        npz_v8 = args.raiz_v8 / d / "cache" / f"{fuente}.pose.npz"
        npz_y26 = args.raiz_y26 / d / "cache" / f"{fuente}.pose.npz"
        c8, c26 = PoseCache.open(npz_v8), PoseCache.open(npz_y26)
        shas["v8"][fuente], shas["y26"][fuente] = rutas.sha256(npz_v8), rutas.sha256(npz_y26)
        gt = verdad(doc, cache)
        a8, _, _, _ = emparejar(gt, c8)
        a26, _, _, _ = emparejar(gt, c26)
        comun_ = {p: set(a8[p]) & set(a26[p]) for p in PELEADORES}
        p8, p26 = puntajes(c8, a8, comun_), puntajes(c26, a26, comun_)
        for a in INDICES:
            v8[a].append(p8[a])
            y26[a].append(p26[a])
        print(f"{fuente}: denominador {[len(comun_[p]) for p in PELEADORES]}", flush=True)

    v8c = {a: np.concatenate(v) for a, v in v8.items()}
    y26c = {a: np.concatenate(v) for a, v in y26.items()}
    r = calibrar(v8c, y26c)
    r["k_cal"] = k_cal(y26c, r["umbrales"], r["fraccion_v8"])
    r["regla"] = "01-e8b-umbral-calibrado.md §2: igualar la fraccion aceptada de v8 con 0,3"
    r["caches_sha256"] = shas
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2) + "\n")
    for a in INDICES:
        print(f"  {a}: v8 {r['fraccion_v8'][a]:.4f}, y26 con 0,3 {r['fraccion_y26_con_0_3'][a]:.4f}"
              f" -> umbral {r['umbrales'][a]:.4f}")
    print(f"K-cal {'pasa' if r['k_cal']['pasa'] else 'FALLA'} -> {args.out}")
    return 0 if r["k_cal"]["pasa"] else 2


if __name__ == "__main__":
    sys.exit(main())
