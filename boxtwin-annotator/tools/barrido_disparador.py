#!/usr/bin/env python3
"""
BoxTwin - Barrido del disparador por extension de muneca, contra la anotacion.

POR QUE EXISTE
  El disparador de tools/demo_vivo.py se calibro a ojo, buscando que la pantalla del demo no
  se llenara de marcas. Antes de construir el salto a candidatos sobre el, habia que saber
  a que umbral encuentra los golpes y cuantas paradas cuesta, que es un punto de operacion
  completamente distinto: para navegar, un candidato de mas cuesta un segundo de mirar y un
  golpe perdido cuesta el golpe.

  La respuesta fue que no sirve, y el numero que lo dice no es el recall sino la comparacion
  contra el azar. Con 42 golpes por minuto, casi la mitad de la linea de tiempo esta a menos
  de medio segundo de un golpe POR CONSTRUCCION, asi que una precision de 0,52 es tirar
  dardos. Sin la linea de base, el barrido se lee al reves.

  Ver docs/experiments/2026-09-01-disparador-vs-azar.md.

QUE HACE
  Tres mediciones sobre cada fuente, restringidas al tramo anotado:

  1. Barrido del umbral, por region y por pico, con recall y precision de parada.
  2. Linea de base: que fraccion del video esta cerca de un golpe, o sea que precision
     tendria poner las paradas al azar, y cuanto tiempo muerto hay para saltear.
  3. AUC de la extension y de su velocidad como discriminadores por cuadro.

  La ventana y el refractario se expresan en segundos y no en cuadros: Pacquiao corre a
  59,94 fps y las otras fuentes a 30, y en cuadros el mismo gesto dura el doble.

USO
  python tools/barrido_disparador.py proyecto/videos/spar.mp4 [otro/videos/pelea.mp4 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxtwin.core.annotations import load as load_doc     # noqa: E402
from boxtwin.core.identity import IdentityResolver       # noqa: E402
from boxtwin.core.posecache import PoseCache             # noqa: E402
from boxtwin.core.project import project_paths           # noqa: E402
from boxtwin.core.types import FighterId, Side           # noqa: E402

L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
TOL_S = 0.5      # a que distancia una parada "encuentra" el golpe, para navegar
VENTANA_S = 0.167
REFRACTARIO_S = 0.667
MIN_SCORE = 0.3


def extension(kp: np.ndarray, sc: np.ndarray) -> dict[str, np.ndarray]:
    """Extension de cada muneca respecto del centro de hombros, en anchos de hombro."""
    ok = np.minimum(sc[:, L_SH], sc[:, R_SH]) >= MIN_SCORE
    esc = np.linalg.norm(kp[:, L_SH] - kp[:, R_SH], axis=-1)
    ok &= esc > 1e-3
    centro = (kp[:, L_SH] + kp[:, R_SH]) / 2
    out = {}
    for lado, wr in (("left", L_WR), ("right", R_WR)):
        e = np.linalg.norm(kp[:, wr] - centro, axis=-1) / np.maximum(esc, 1e-6)
        e[~ok | (sc[:, wr] < MIN_SCORE)] = 0.0
        out[lado] = e
    return out


def por_region(e: np.ndarray, umbral: float, junte: int) -> list[int]:
    """Una parada por tramo continuo sobre el umbral, ubicada en el maximo del tramo."""
    sobre = e >= umbral
    if not sobre.any():
        return []
    d = np.diff(sobre.astype(np.int8))
    ini = ([0] if sobre[0] else []) + list(np.where(d == 1)[0] + 1)
    fin = list(np.where(d == -1)[0]) + ([len(e) - 1] if sobre[-1] else [])
    unidos = [list(t) for t in zip(ini, fin)][:1]
    for a, b in zip(ini[1:], fin[1:]):
        if a - unidos[-1][1] <= junte:
            unidos[-1][1] = b
        else:
            unidos.append([a, b])
    return [int(a + np.argmax(e[a:b + 1])) for a, b in unidos]


def por_pico(e: np.ndarray, umbral: float, ventana: int, refractario: int) -> list[int]:
    """Maximo local por encima del umbral, con refractario. Es lo que hace el demo."""
    r = ventana // 2
    vent = np.lib.stride_tricks.sliding_window_view(
        np.pad(e, (r, r), constant_values=-np.inf), ventana
    )
    sel, ultimo = [], -10**9
    for t in np.where((e >= umbral) & (e >= vent.max(axis=1)))[0]:
        if t - ultimo >= refractario:
            sel.append(int(t))
            ultimo = int(t)
    return sel


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney). Sin sklearn, que no es dependencia del paquete."""
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    orden = np.argsort(x, kind="mergesort")
    xs, r = x[orden], np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def keypoints_por_peleador(doc, cache, lo: int, hi: int):
    """(kp, sc) por peleador sobre [lo, hi], con la identidad ya resuelta."""
    resolver = IdentityResolver(doc, cache)
    T = hi - lo + 1
    peleadores = [FighterId.A, FighterId.B]
    kp = np.zeros((2, T, 17, 2), np.float32)
    sc = np.zeros((2, T, 17), np.float32)
    for f in range(lo, hi + 1):
        resuelto = resolver.by_fighter(f)
        for p, fid in enumerate(peleadores):
            pose = resuelto[fid]
            if pose is not None:
                kp[p, f - lo] = pose.keypoints
                sc[p, f - lo] = pose.kp_score
    return kp, sc


def medir(video: Path, umbrales: list[float]) -> None:
    paths = project_paths(video)
    if not paths.npz.is_file() or not paths.annot.is_file():
        print(f"error: falta cache o anotacion para {video.name}", file=sys.stderr)
        return
    cache = PoseCache.open(paths.npz)
    doc, _ = load_doc(paths.annot)
    eventos = doc.events
    if not eventos:
        print(f"{video.name}: sin eventos anotados", file=sys.stderr)
        return

    fps = doc.video.fps
    lo = min(e.start_frame for e in eventos)
    hi = max(e.end_frame for e in eventos)
    T = hi - lo + 1
    dur_min = T / fps / 60
    tol = int(round(TOL_S * fps))
    ventana = max(3, int(round(VENTANA_S * fps)) | 1)
    refractario = max(1, int(round(REFRACTARIO_S * fps)))
    junte = max(1, int(round(VENTANA_S * fps)))

    kp, sc = keypoints_por_peleador(doc, cache, lo, hi)
    ext = {fid: extension(kp[p], sc[p]) for p, fid in enumerate([FighterId.A, FighterId.B])}
    # los cuadros van relativos a lo, igual que ext
    rangos = [(e.start_frame - lo, e.end_frame - lo) for e in eventos]

    print(f"\n### {video.stem} - {len(eventos)} eventos, {dur_min:.1f} min, {fps:.2f} fps")

    # -- linea de base -----------------------------------------------------
    ocup = np.zeros(T, bool)
    ocup_tol = np.zeros(T, bool)
    for a, b in rangos:
        ocup[max(a, 0):b + 1] = True
        ocup_tol[max(a - tol, 0):min(b + tol + 1, T)] = True
    d = np.diff((~ocup_tol).astype(np.int8))
    ini = ([0] if not ocup_tol[0] else []) + list(np.where(d == 1)[0] + 1)
    fin = list(np.where(d == -1)[0]) + ([T - 1] if not ocup_tol[-1] else [])
    ahorro = sum(b - a for a, b in zip(ini, fin) if (b - a) / fps > 2.0) / T
    print(f"  golpes/min {len(eventos)/dur_min:.1f} | dentro de un golpe {ocup.mean():.0%} | "
          f"a menos de {TOL_S}s {ocup_tol.mean():.0%} (= precision de tirar dardos) | "
          f"tramos muertos >2s {ahorro:.0%}")

    # -- barrido -----------------------------------------------------------
    for modo, disparar in (("region", por_region), ("pico", por_pico)):
        print(f"\n  por {modo}")
        print(f"  {'umbral':>7} {'paradas':>8} {'recall':>8} {'precis':>8} "
              f"{'par/min':>9} {'s entre':>8}")
        for u in umbrales:
            stops = []
            for fid in (FighterId.A, FighterId.B):
                for lado in ("left", "right"):
                    stops += (disparar(ext[fid][lado], u, junte) if modo == "region"
                              else disparar(ext[fid][lado], u, ventana, refractario))
            stops.sort()
            grupos: list[list[int]] = []
            for t in stops:
                if grupos and t - grupos[-1][-1] <= junte:
                    grupos[-1].append(t)
                else:
                    grupos.append([t])
            n = len(grupos)
            cub = sum(1 for a, b in rangos
                      if any(any(a - tol <= t <= b + tol for t in g) for g in grupos))
            util = sum(1 for g in grupos
                       if any(any(a - tol <= t <= b + tol for a, b in rangos) for t in g))
            print(f"  {u:7.1f} {n:8d} {cub/len(rangos):8.3f} {(util/n if n else 0):8.3f} "
                  f"{n/dur_min:9.1f} {dur_min*60/max(n,1):8.1f}")

    # -- AUC ---------------------------------------------------------------
    pe, ne, pv, nv = [], [], [], []
    for p, fid in enumerate([FighterId.A, FighterId.B]):
        for lado, side in (("left", Side.LEFT), ("right", Side.RIGHT)):
            e = ext[fid][lado]
            v = np.abs(np.gradient(e))
            m = e > 0  # solo cuadros con pose utilizable
            etiq = np.zeros(T, bool)
            for ev in eventos:
                if ev.fighter is fid and ev.side is side:
                    etiq[ev.start_frame - lo:ev.end_frame - lo + 1] = True
            pe.append(e[m & etiq]); ne.append(e[m & ~etiq])
            pv.append(v[m & etiq]); nv.append(v[m & ~etiq])
    pe, ne = np.concatenate(pe), np.concatenate(ne)
    pv, nv = np.concatenate(pv), np.concatenate(nv)
    print(f"\n  AUC por cuadro: extension {auc(pe, ne):.3f} | velocidad {auc(pv, nv):.3f}"
          f"  (0,50 es azar; hook contra straight dio 0,64)")
    print(f"  mediana de la extension: {np.median(pe):.2f} con golpe, "
          f"{np.median(ne):.2f} sin golpe")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--umbrales", type=float, nargs="+",
                   default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    args = p.parse_args()
    for v in args.videos:
        medir(v, args.umbrales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
