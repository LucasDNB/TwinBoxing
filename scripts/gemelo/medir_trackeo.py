"""
BoxTwin - Cuanto le cuesta a cada pose seguir a cada peleador (E8, P12c, 27-10-2026).

POR QUE EXISTE
  El 20-10, 18 de los 20 errores de C1 salieron de tracks mixtos. E8 pregunta si YOLO26 sigue
  mejor a cada peleador (docs/specs/2026-10-27-gemelo-yolo26, 02-preregistro-e8.md §4, P12c).
  tools/comparar_tracker.py hace la cuenta, pero sobre Sparring y por pantalla. Esto es lo
  mismo para las siete fuentes, con salida JSON y con un denominador que no favorece a nadie.

QUE HACE
  La verdad es la caja observada de cada peleador en la anotacion original. Cada condicion
  (base y y26) se empareja con esa verdad como en el puente (IoU >= 0,5). Las cuentas se
  hacen solo sobre los cuadros donde las DOS condiciones tienen emparejado a ese peleador:
  - cambios de track por peleador, y por minuto de esos cuadros;
  - tracks distintos por peleador;
  - tracks mixtos (un mismo track de A en unos cuadros y de B en otros) y cuadros dentro de
    tracks mixtos.
  Aparte, sin verdad: tracks totales de cada cache. Escribe trackeo.json con cada fuente y el
  total.

USO
  python scripts/gemelo/medir_trackeo.py
  python scripts/gemelo/medir_trackeo.py --raiz-base ~/boxtwin-gemelo/raiz-v8-sin-reid   # rama D-H2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "clasificador"))
import guardas  # noqa: E402
from puente_identidad import PELEADORES, _original, emparejar, verdad  # noqa: E402


def _por_minuto(cambios: int, cuadros: int, fps: float) -> float | None:
    return cambios / (cuadros / fps / 60) if cuadros else None


def metricas(asign: dict[str, dict[int, int]], frames: dict[str, set[int]], fps: float) -> dict:
    """Las cuentas de P12c de una condicion, restringidas a `frames` por peleador."""
    salida: dict = {}
    de: dict[str, dict[int, int]] = {}
    for p in PELEADORES:
        fs = sorted(f for f in frames[p] if f in asign[p])
        de[p] = {f: asign[p][f] for f in fs}
        cambios = sum(1 for x, y in zip(fs, fs[1:]) if asign[p][x] != asign[p][y])
        salida[p] = {"cuadros": len(fs), "cambios": cambios,
                     "cambios_por_minuto": _por_minuto(cambios, len(fs), fps),
                     "tracks_distintos": len(set(de[p].values()))}
    mixtos = set(de["A"].values()) & set(de["B"].values())
    salida["tracks_mixtos"] = len(mixtos)
    salida["cuadros_en_tracks_mixtos"] = sum(1 for p in PELEADORES for t in de[p].values() if t in mixtos)
    return salida


def comparar(asign_base, asign_y26, fps: float) -> dict:
    """Las dos condiciones sobre el denominador comun: la interseccion por peleador."""
    comun = {p: set(asign_base[p]) & set(asign_y26[p]) for p in PELEADORES}
    base, y26 = metricas(asign_base, comun, fps), metricas(asign_y26, comun, fps)
    dif = {f"{p}_cambios": y26[p]["cambios"] - base[p]["cambios"] for p in PELEADORES}
    dif.update({f"{p}_tracks_distintos": y26[p]["tracks_distintos"] - base[p]["tracks_distintos"]
                for p in PELEADORES})
    dif["tracks_mixtos"] = y26["tracks_mixtos"] - base["tracks_mixtos"]
    dif["cuadros_en_tracks_mixtos"] = y26["cuadros_en_tracks_mixtos"] - base["cuadros_en_tracks_mixtos"]
    return {"denominador": {p: len(comun[p]) for p in PELEADORES}, "fps": fps,
            "base": base, "y26": y26, "diferencia": dif}


def medir_fuente(doc, cache_orig, cache_base, cache_y26) -> dict:
    gt = verdad(doc, cache_orig)
    ab, _, cb, _ = emparejar(gt, cache_base)
    ay, _, cy, _ = emparejar(gt, cache_y26)
    r = comparar(ab, ay, doc.video.fps)
    r["cuadros_con_verdad"] = {p: len(gt[p]) for p in PELEADORES}
    r["emparejados"] = {"base": {p: len(ab[p]) for p in PELEADORES},
                        "y26": {p: len(ay[p]) for p in PELEADORES}}
    r["conflictos"] = {"base": cb, "y26": cy}
    r["tracks_totales"] = {"base": len(cache_base.track_ids()), "y26": len(cache_y26.track_ids())}
    return r


def total(fuentes: list[dict], fps_ref: float | None = None) -> dict:
    """Suma de las fuentes. Los cambios por minuto salen de las sumas, con el fps de cada una."""
    salida: dict = {"denominador": {p: sum(f["denominador"][p] for f in fuentes) for p in PELEADORES}}
    for cond in ("base", "y26"):
        c: dict = {}
        for p in PELEADORES:
            cambios = sum(f[cond][p]["cambios"] for f in fuentes)
            minutos = sum(f[cond][p]["cuadros"] / f["fps"] / 60 for f in fuentes)
            c[p] = {"cuadros": sum(f[cond][p]["cuadros"] for f in fuentes), "cambios": cambios,
                    "cambios_por_minuto": cambios / minutos if minutos else None,
                    "tracks_distintos": sum(f[cond][p]["tracks_distintos"] for f in fuentes)}
        c["tracks_mixtos"] = sum(f[cond]["tracks_mixtos"] for f in fuentes)
        c["cuadros_en_tracks_mixtos"] = sum(f[cond]["cuadros_en_tracks_mixtos"] for f in fuentes)
        salida[cond] = c
    return salida


def main(argv=None) -> int:
    import comun
    from boxtwin.core.posecache import PoseCache

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=list(comun.FUENTES), choices=list(comun.FUENTES))
    ap.add_argument("--raiz-original", type=Path, default=guardas.REPO_A)
    ap.add_argument("--raiz-base", type=Path, default=None,
                    help="caches de la base; por defecto los originales (sin D-H2)")
    ap.add_argument("--raiz-y26", type=Path, default=guardas.GEMELO / "raiz-y26")
    ap.add_argument("--out", type=Path, default=guardas.GEMELO / "e8" / "trackeo.json")
    args = ap.parse_args(argv)
    out = guardas.exigir_out(args.out)
    por_fuente = {}
    for fuente in args.fuentes:
        doc, cache = _original(args.raiz_original, fuente)
        base = cache
        if args.raiz_base:
            base = PoseCache.open(args.raiz_base / comun.FUENTES[fuente] / "cache" / f"{fuente}.pose.npz")
        y26 = PoseCache.open(args.raiz_y26 / comun.FUENTES[fuente] / "cache" / f"{fuente}.pose.npz")
        por_fuente[fuente] = medir_fuente(doc, cache, base, y26)
        d = por_fuente[fuente]["diferencia"]
        print(f"{fuente}: denominador {por_fuente[fuente]['denominador']}, y26 - base: {d}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fuentes": por_fuente, "total": total(list(por_fuente.values()))},
                              indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
