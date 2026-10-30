"""
BoxTwin - P14d: cuanto duran los huecos de muneca y codo bajo umbral (E10, S7, 30-10-2026).

POR QUE EXISTE
  El tope de relleno de E10 (0,10 s) se fijo antes de mirar los huecos
  (docs/specs/2026-10-30-mejoras-post-e8/02-e10-relleno-temporal.md §2, decision 4). P14d mide,
  sin etiquetas, que parte de la perdida cae debajo de ese tope: si menos del 30% de los cuadros
  perdidos de muneca esta en huecos de <= 0,10 s, el relleno actua sobre la minoria de la
  perdida. El tope no se mueve con esto.

QUE HACE
  Por export `sequence` (un peleador a la vez, en el fps original) y por articulacion, los
  largos de los huecos cerrados con la definicion de E10 §2 (bajo umbral, aceptado antes y
  despues, deteccion presente en todos, sin pose interpolada por la identidad) y los cuadros
  perdidos en total (con los huecos abiertos). Escribe un JSON con el histograma en segundos
  y la fraccion de perdidos en huecos de <= 0,10 s, por fuente y total.

USO
  python scripts/mejoras/huecos.py ~/boxtwin-gemelo/e8/base/exports/*.sequence.npz --out v8.json
  python scripts/mejoras/huecos.py ~/boxtwin-gemelo/e8/y26/exports/*.sequence.npz \
    --umbrales ~/boxtwin-mejoras/e8b/umbrales.json --out y26cal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rutas  # noqa: E402,F401  (deja boxtwin_detector a mano si se corre con PYTHONPATH)

ARTICULACIONES = {"muneca": (9, 10), "codo": (7, 8)}
BORDES_S = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
TOPE_S = 0.10


def largos(sc: np.ndarray, presente: np.ndarray, umbrales: dict) -> dict:
    """Largo en cuadros de cada hueco cerrado, y los cuadros perdidos, por articulacion."""
    from boxtwin_detector.relleno import _corridas

    T = len(sc)
    salida: dict = {}
    for nombre, idx in ARTICULACIONES.items():
        ls, perdidos = [], 0
        for j in idx:
            aceptado = (np.asarray(sc[:, j], np.float64) >= umbrales[nombre]) & presente
            perdidos += int((~aceptado & presente).sum())
            for a, b in _corridas(~aceptado):
                if a > 0 and b < T and presente[a:b].all():
                    ls.append(int(b - a))
        salida[nombre], salida[f"{nombre}_perdidos"] = ls, perdidos
    return salida


def resumen(l: dict, fps: float) -> dict:
    from boxtwin_detector.relleno import maximo_cuadros
    tope = maximo_cuadros(fps, TOPE_S)
    salida = {}
    for nombre in ARTICULACIONES:
        ls = np.asarray(l[nombre], np.float64)
        s = ls / fps
        hist, previo = {}, 0.0
        for b in BORDES_S:
            hist[str(b)] = int(((s > previo) & (s <= b + 1e-9)).sum())
            previo = b
        hist[f">{BORDES_S[-1]}"] = int((s > BORDES_S[-1]).sum())
        perd = l[f"{nombre}_perdidos"]
        corto = float(ls[ls <= tope].sum())
        salida[nombre] = {"huecos": int(len(ls)), "cuadros_perdidos": perd,
                          "fraccion_perdidos_en_huecos_cortos": corto / perd if perd else None,
                          "histograma_s": hist}
    return salida


def main(argv=None) -> int:
    from boxtwin_detector.dataset import leer_umbrales
    from boxtwin_detector.features import resolver_umbrales

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("exports", type=Path, nargs="+")
    ap.add_argument("--umbrales", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    u = leer_umbrales(args.umbrales)[0] if args.umbrales else resolver_umbrales(None)
    por, acum = {}, {}
    for npz in args.exports:
        meta = json.loads(npz.with_name(npz.name.replace(".npz", ".meta.json")).read_text())
        fps = float(meta["video"]["fps"])
        d = np.load(npz)
        l: dict = {}
        for p in range(d["kp_score"].shape[0]):
            x = largos(d["kp_score"][p], d["valid"][p] & ~d["interpolated"][p], u)
            for k, v in x.items():
                l[k] = l.get(k, 0 if isinstance(v, int) else []) + v
        nombre = npz.name.replace(".sequence.npz", "")
        por[nombre] = {"fps": fps, **resumen(l, fps)}
        # el total se suma en segundos: las fuentes no comparten fps
        for k in ARTICULACIONES:
            acum.setdefault(k, []).extend((np.asarray(l[k]) / fps * 30).tolist())
            acum[f"{k}_perdidos"] = acum.get(f"{k}_perdidos", 0) + l[f"{k}_perdidos"]
    salida = {"umbrales": u, "fuentes": por, "total_en_cuadros_de_30": resumen(acum, 30.0),
              "regla": "02-e10 §2; menos de 0,30 en huecos cortos = el relleno actua sobre la minoria"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(salida, indent=2) + "\n")
    print(json.dumps({k: v["fraccion_perdidos_en_huecos_cortos"]
                      for k, v in salida["total_en_cuadros_de_30"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
