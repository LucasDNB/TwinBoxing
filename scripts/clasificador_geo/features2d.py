"""
BoxTwin - Features geometricas 2D por golpe para E4 (20 del brazo y 6 del rival).

POR QUE EXISTE
  El detector de golpes generaliza a fuentes nuevas y PoseConv3D no. La diferencia mas
  visible es la entrada: el detector ve 20 features centradas, escaladas y espejadas
  (boxtwin_detector.features), PoseConv3D ve mapas de calor en pixeles. E4 prueba el mismo
  modelo chico sobre esas features. La altura necesita saber donde esta el rival, asi que se
  suman 6 features relativas a su torso: la primera es la altura relativa de la regla de C8,
  como serie en vez de un solo cuadro.

QUE HACE
  Por golpe: `features_de` sobre el video entero del atacante (no se copia la funcion, se
  importa), la mascara `usable` como mascara por cuadro, las 6 del rival en pixeles de imagen
  (diseno 2.2), el corte de la ventana y el remuestreo lineal a 32 pasos (la mascara, con
  vecino mas cercano). Escribe <trabajo>/datos/golpes.npz con X (N, 32, 26), M (N, 32), las
  etiquetas como enteros y frame_dir, y golpes2d.meta.json con los nombres de las features.

  Dos opciones de la campana posterior a E8 (docs/specs/2026-10-30-mejoras-post-e8/), por
  entorno y apagadas por defecto; apagadas, golpes.npz es el de antes:
  BOXTWIN_UMBRALES=<umbrales.json> (E8b) pone un umbral de score por articulacion en las 20
  del brazo y en las 6 del rival; BOXTWIN_RELLENO=1 (E10) rellena los huecos cortos de cada
  serie antes de las features y suma R (N, 32), los pasos con pose rellenada. El meta registra
  el sha256 de los umbrales y si hubo relleno.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/features2d.py
  BOXTWIN_UMBRALES=~/boxtwin-mejoras/e8b/umbrales.json BOXTWIN_RELLENO=1 \
    ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/features2d.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import ALTURAS, PASOS, ROLES, TIPOS, TRABAJO  # noqa: E402

MIN_SCORE = 0.3
HOMBROS, CADERAS = (5, 6), (11, 12)
MUNECA = {"left": 9, "right": 10}
NOMBRES_RIVAL = ["rival_h", "rival_dx", "rival_d", "rival_incl_cos", "rival_incl_sin", "rival_ok"]


def nombres_brazo() -> list[str]:
    from boxtwin_detector.features import NOMBRES
    return list(NOMBRES)


def leer_umbrales(ruta) -> tuple[dict, str]:
    """Los umbrales por articulacion de un umbrales.json (E8b) y su sha256."""
    from boxtwin_detector.dataset import leer_umbrales as leer
    return leer(Path(ruta).expanduser())


def opciones_de_entorno() -> tuple[dict | None, str | None, bool]:
    """(umbrales, sha256 de umbrales.json, relleno) segun BOXTWIN_UMBRALES y BOXTWIN_RELLENO."""
    ruta = os.environ.get("BOXTWIN_UMBRALES")
    u, sha = leer_umbrales(ruta) if ruta else (None, None)
    return u, sha, os.environ.get("BOXTWIN_RELLENO") == "1"


def features_brazo(kp_p: np.ndarray, sc_p: np.ndarray, side: str,
                   umbrales: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Las 20 del detector y su mascara, sobre el video entero del peleador."""
    from boxtwin_detector.features import features_de
    f, usable = features_de(kp_p, sc_p, brazo=side, umbrales=umbrales)
    return f.astype(np.float32), usable.astype(bool)


def features_rival(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray, p: int, side: str,
                   umbrales: dict | None = None) -> np.ndarray:
    """
    (T, 6) en pixeles de imagen, sin espejar. kp (2, T, 17, 2), sc (2, T, 17), valid (2, T).
    Donde el rival no tiene hombros y caderas confiables, rival_ok = 0 y las otras cinco en 0.
    Con `umbrales`, hombros y caderas usan cada uno el suyo.
    """
    u_hom = MIN_SCORE if umbrales is None else umbrales["hombro"]
    u_cad = MIN_SCORE if umbrales is None else umbrales["cadera"]
    r = 1 - p
    kr, sr = kp[r].astype(np.float64), sc[r]
    s = kr[:, list(HOMBROS)].mean(1)
    c = kr[:, list(CADERAS)].mean(1)
    v = s - c
    torso = np.linalg.norm(v, axis=-1)
    ok = ((sr[:, list(HOMBROS)].min(1) >= u_hom) & (sr[:, list(CADERAS)].min(1) >= u_cad)
          & valid[r].astype(bool) & (torso > 1e-6))
    t = np.where(ok, torso, 1.0)
    m = kp[p][:, MUNECA[side]].astype(np.float64)
    h = (m[:, 1] - s[:, 1]) / t
    dx = np.abs(m[:, 0] - s[:, 0]) / t
    d = np.linalg.norm(m - s, axis=-1) / t
    # angulo de c -> s respecto de la vertical hacia arriba (0, -1) de la imagen
    incl_cos = -v[:, 1] / t
    incl_sin = v[:, 0] / t
    f = np.stack([h, dx, d, incl_cos, incl_sin, ok.astype(np.float64)], axis=-1)
    f[~ok, :5] = 0.0
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def remuestrear(x: np.ndarray, pasos: int = PASOS) -> np.ndarray:
    """(L, F) o (L,) -> (pasos, F) o (pasos,) con interpolacion lineal sobre el eje temporal."""
    L = len(x)
    pos = np.linspace(0, L - 1, pasos)
    if x.ndim == 1:
        return np.interp(pos, np.arange(L), x).astype(np.float32)
    return np.stack([np.interp(pos, np.arange(L), x[:, j]) for j in range(x.shape[1])], -1).astype(np.float32)


def remuestrear_mascara(m: np.ndarray, pasos: int = PASOS) -> np.ndarray:
    """Vecino mas cercano: un paso es usable si su cuadro mas cercano lo era."""
    pos = np.rint(np.linspace(0, len(m) - 1, pasos)).astype(int)
    return np.asarray(m, bool)[pos]


def rellenar_serie(s: dict, umbrales: dict | None) -> dict:
    """La serie de una fuente con los huecos cortos de cada peleador rellenados (E10)."""
    from boxtwin_detector.relleno import rellenar
    kp, sc = s["kp"].copy(), s["sc"].copy()
    rell = np.zeros(sc.shape[:2], bool)
    for p in range(kp.shape[0]):
        kp[p], sc[p], r = rellenar(kp[p], sc[p], umbrales, float(s["fps"]),
                                   presente=s["valid"][p].astype(bool))
        rell[p] = r.any(axis=1)
    return {**s, "kp": kp, "sc": sc, "rellenado": rell}


def construir(golpes: list[dict], series: dict[str, dict], umbrales: dict | None = None,
              relleno: bool = False) -> dict:
    """Junta X (N, 32, 26), M (N, 32) y las etiquetas de los golpes, en el orden de golpes."""
    if relleno:
        series = {k: rellenar_serie(s, umbrales) for k, s in series.items()}
    cache: dict = {}
    X, M, R = [], [], []
    for g in golpes:
        s = series[g["fuente"]]
        clave = (g["fuente"], g["p"], g["side"])
        if clave not in cache:
            fb, us = features_brazo(s["kp"][g["p"]], s["sc"][g["p"]], g["side"], umbrales)
            fr = features_rival(s["kp"], s["sc"], s["valid"], g["p"], g["side"], umbrales)
            cache[clave] = (np.concatenate([fb, fr], -1), us)
        f, us = cache[clave]
        a, b = g["ini"], g["fin"] + 1
        X.append(remuestrear(f[a:b]))
        M.append(remuestrear_mascara(us[a:b]))
        if relleno:
            R.append(remuestrear_mascara(s["rellenado"][g["p"]][a:b]))
    opcional = {"R": np.stack(R)} if relleno else {}
    return {**opcional,
        "X": np.stack(X).astype(np.float32), "M": np.stack(M),
        "y_tipo": np.array([TIPOS.index(g["tipo"]) for g in golpes]),
        "y_altura": np.array([ALTURAS.index(g["altura"]) for g in golpes]),
        "y_rol": np.array([ROLES.index(g["rol"]) for g in golpes]),
        "frame_dir": np.array([g["frame_dir"] for g in golpes]),
        "fuente": np.array([g["fuente"] for g in golpes]),
    }


def cargar_golpes(trabajo: Path = None):
    """golpes.npz como dict de arrays."""
    trabajo = trabajo or TRABAJO
    with np.load(trabajo / "datos" / "golpes.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    from datos import leer_csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    d = TRABAJO / "datos"
    golpes = leer_csv(d / "golpes.csv")
    umbrales, sha, relleno = opciones_de_entorno()
    series = {}
    for f in dict.fromkeys(g["fuente"] for g in golpes):
        with np.load(d / "series" / f"{f}.npz") as z:
            series[f] = {k: z[k] for k in ("kp", "sc", "valid", "fps")}
    datos = construir(golpes, series, umbrales, relleno)
    np.savez_compressed(d / "golpes.npz", **datos)
    nombres = nombres_brazo() + NOMBRES_RIVAL
    meta = {"features": nombres, "F": len(nombres), "pasos": PASOS, "golpes": len(golpes)}
    if umbrales is not None:     # apagado, el meta es el de antes
        meta.update(umbrales=umbrales, umbrales_sha256=sha)
    if relleno:
        meta["relleno"] = True
    (d / "golpes2d.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    M = datos["M"]
    print(f"X {datos['X'].shape}, pasos usables {M.mean():.3f}, golpes sin ningun paso usable "
          f"{int((M.sum(1) == 0).sum())}, rival_ok medio {datos['X'][..., 25].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
