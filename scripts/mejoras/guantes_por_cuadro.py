"""
BoxTwin - El guante detectado, cuadro por cuadro y por mano (E12, T8, 30-10-2026).

POR QUE EXISTE
  El detector de golpes ve la mano solo por el keypoint de muneca, y en la extension del golpe
  la muneca se borronea y su score cae (K3 de E8). El sistema ya tiene otra fuente de la
  posicion de la mano: el detector de guantes, que hoy solo se usa para la identidad y en uno
  de cada cinco cuadros. E12 (docs/specs/2026-10-30-mejoras-post-e8/04-e12-guantes-en-detector.md
  §2) lo suma como features del detector, y para eso hace falta el guante en cada cuadro.

QUE HACE
  Por fuente, con la anotacion y el cache de raiz-v8 (la base de E8): en cada cuadro y para cada
  peleador con deteccion (no interpolada), `detectar_con_color` sobre su caja, sin saltear
  cuadros y sin el filtro de aislamiento (es para el color, no para la posicion); despues
  `_solo_los_propios` con max_distancia_muneca 1,2, el de A. Cada guante que queda va a la mano
  cuya posicion esperada (`derive_gloves`, con score >= 0,3) esta mas cerca; si hay una sola
  esperada, a esa; si ninguna, se descarta; dos a la misma mano, queda el de mayor confianza.
  Escribe <out>/<fuente>.npz con guantes (2, T, mano, [x, y, conf]) en pixeles (conf 0 = sin
  guante; mano 0 izquierda, 1 derecha) y ms por cuadro. Reanudable por fuente. Al final,
  resumen.json con P16c (cobertura por mano y fuente, sobre los cuadros con el peleador) y P16d (mediana y p95 de ms por cuadro),
  y kmano.json con los 200 cuadros de K-mano de sparring-3-rounds (semilla 0).

USO
  python scripts/mejoras/guantes_por_cuadro.py --device 0
  python scripts/mejoras/guantes_por_cuadro.py --fuentes Sparring --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rutas  # noqa: E402

UMBRAL_ESPERADA = 0.3
FUENTE_K_MANO = "sparring-3-rounds"


def asignar_mano(guantes, esperadas, ok) -> np.ndarray:
    """guantes [(x, y, conf)], esperadas [xy izq, xy der], ok [bool, bool] -> (2, 3)."""
    salida = np.zeros((2, 3), np.float32)
    manos = [i for i in (0, 1) if ok[i]]
    if not manos:
        return salida
    for x, y, c in guantes:
        mano = min(manos, key=lambda i: float(np.hypot(x - esperadas[i][0], y - esperadas[i][1])))
        if c > salida[mano, 2]:
            salida[mano] = (x, y, c)
    return salida


def extraer_cuadro(frame, poses, detector, cfg=None) -> np.ndarray:
    """poses = [(bbox, kp (17, 2), sc (17,)) o None] por peleador -> (2, 2, 3)."""
    from boxtwin.core.gloves import derive_gloves
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, _solo_los_propios

    cfg = cfg or ConfigIdentidadAuto()
    salida = np.zeros((2, 2, 3), np.float32)
    for p, pose in enumerate(poses):
        if pose is None:
            continue
        bbox, kp, sc = pose
        dets = detector.detectar_con_color(frame, np.asarray(bbox, np.float64))
        propios = _solo_los_propios(dets, kp, sc, cfg)
        centros = [((c.xyxy[0] + c.xyxy[2]) / 2, (c.xyxy[1] + c.xyxy[3]) / 2, c.conf) for c, _ in propios]
        xy, conf = derive_gloves(np.asarray(kp, np.float32)[None], np.asarray(sc, np.float32)[None])
        salida[p] = asignar_mano(centros, xy[0], conf[0] >= UMBRAL_ESPERADA)
    return salida


def lista_k_mano(guantes: np.ndarray, n: int = 200, semilla: int = 0) -> list[dict]:
    """n cuadros con guante asignado, al azar con semilla fija, para que Lucas los marque."""
    p, t, m = np.nonzero(guantes[..., 2] > 0)
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(p), size=min(n, len(p)), replace=False)
    return [{"peleador": int(p[i]), "cuadro": int(t[i]), "mano": int(m[i]),
             "xy": [float(v) for v in guantes[p[i], t[i], m[i], :2]]} for i in sorted(idx)]


def extraer_fuente(fuente: str, dir_: str, raiz: Path, detector, destino: Path) -> dict:
    import cv2

    from boxtwin.core.annotations import load
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import FighterId

    ruta = raiz / dir_ / "annotations" / f"{fuente}.annot.json"
    doc, _ = load(ruta)
    cache = PoseCache.open(raiz / dir_ / "cache" / f"{fuente}.pose.npz")
    video = raiz / dir_ / "videos" / Path(doc.video.path).name
    T = int(doc.video.total_frames)
    resolver = IdentityResolver(doc, cache)
    guantes = np.zeros((2, T, 2, 3), np.float32)
    ms = np.full(T, np.nan, np.float32)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir {video}")
    try:
        for f in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            porf = resolver.by_fighter(f)
            poses = []
            for fid in (FighterId.A, FighterId.B):
                pose = porf.get(fid)
                poses.append(None if pose is None or pose.interpolated else
                             (pose.bbox, pose.keypoints[:17], pose.kp_score[:17]))
            guantes[:, f] = extraer_cuadro(frame, poses, detector)
            ms[f] = (time.perf_counter() - t0) * 1000
            if f % 3000 == 0:
                print(f"  {fuente}: {f}/{T}", flush=True)
    finally:
        cap.release()
    np.savez_compressed(destino, guantes=guantes, ms=ms)
    return {"cuadros": T}


def presencia_de_exports(exports: Path, fuentes: list[str]) -> dict[str, np.ndarray]:
    """(2, T) por fuente: el peleador tiene deteccion (valid y no interpolated en el export)."""
    salida = {}
    for f in fuentes:
        p = exports / f"{f}.sequence.npz"
        if p.is_file():
            with np.load(p) as z:
                salida[f] = z["valid"].astype(bool) & ~z["interpolated"].astype(bool)
    return salida


def resumir(destino: Path, fuentes: list[str], presente: dict[str, np.ndarray] | None = None) -> dict:
    """
    P16c sobre los cuadros donde el peleador tiene deteccion, no sobre el video entero: de
    Pacquiao solo esta anotado el round 1, y contar los 87 minutos daba 4%. Sin `presente`
    de una fuente, se cuenta el video entero y se avisa.
    """
    por, todos = {}, []
    for fuente in fuentes:
        p = destino / f"{fuente}.npz"
        if not p.is_file():
            continue
        with np.load(p) as z:
            g, ms = z["guantes"], z["ms"]
        pres = (presente or {}).get(fuente)
        if pres is None:
            pres = np.ones(g.shape[:2], bool)
        # P16d: el costo en los cuadros con algun peleador, que es el caso del producto; sin
        # peleador la extraccion no corre el detector y el cuadro cuesta casi nada
        todos.append(ms[np.isfinite(ms) & pres.any(axis=0)])
        cob = {}
        for a in (0, 1):
            n = int(pres[a].sum())
            for m in (0, 1):
                cob[f"{'AB'[a]}-{'izq der'.split()[m]}"] = (
                    float((g[a, :, m, 2][pres[a]] > 0).mean()) if n else None)
        por[fuente] = {"cobertura_por_mano": cob,
                       "cuadros_con_peleador": {"AB"[a]: int(pres[a].sum()) for a in (0, 1)}}
        if presente is None or fuente not in presente:
            por[fuente]["aviso"] = "sin presencia del export: cobertura sobre el video entero"
    ms = np.concatenate(todos) if todos else np.array([])
    return {"P16c": por, "P16d": {"mediana_ms": float(np.median(ms)) if len(ms) else None,
                                  "p95_ms": float(np.percentile(ms, 95)) if len(ms) else None,
                                  "cuadros": int(len(ms))}}


def main(argv=None) -> int:
    import comun

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=list(comun.FUENTES), choices=list(comun.FUENTES))
    ap.add_argument("--raiz", type=Path, default=rutas.RAIZ_V8)
    ap.add_argument("--modelo", type=Path, default=rutas.REPO_A / "boxtwin-guantes" / "modelos" / "guantes-v2.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=rutas.MEJORAS / "e12" / "guantes")
    ap.add_argument("--exports", type=Path, default=rutas.E8 / "base" / "exports",
                    help="exports sequence de la base, para contar solo cuadros con el peleador")
    args = ap.parse_args(argv)
    from boxtwin.core.deteccion_guantes import DetectorGuantes

    args.out.mkdir(parents=True, exist_ok=True)
    detector = None
    for fuente in args.fuentes:
        destino = args.out / f"{fuente}.npz"
        if destino.is_file():
            print(f"{fuente}: ya extraido, se saltea")
            continue
        detector = detector or DetectorGuantes(str(args.modelo), device=args.device)
        print(f"{fuente}: extrayendo", flush=True)
        extraer_fuente(fuente, comun.FUENTES[fuente], args.raiz, detector, destino)
    pres = presencia_de_exports(args.exports, args.fuentes)
    (args.out / "resumen.json").write_text(json.dumps(resumir(args.out, args.fuentes, pres), indent=2) + "\n")
    k = args.out / f"{FUENTE_K_MANO}.npz"
    if k.is_file():
        with np.load(k) as z:
            (args.out / "kmano.json").write_text(json.dumps(lista_k_mano(z["guantes"]), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
