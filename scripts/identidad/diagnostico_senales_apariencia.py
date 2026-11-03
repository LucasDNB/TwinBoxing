"""
BoxTwin - Identidad: cuanto separa a A de B cada senal de apariencia (diagnostico, no decide).

POR QUE EXISTE
  IE6 (3-11) dio "no estable": el 98% de los errores silenciosos esta en 03-sparring, donde
  los perfiles de color del GUANTE no separan (0,348 contra el piso de 0,55) y ningun
  detector de sospecha puede dispararse. La propuesta de Lucas (3-11) es seguir a cada
  peleador con un perfil fijo de guantes + pantalon + remera + altura. Lucas confirma que en
  sparring de gimnasio la ropa de los dos suele ser distinta y no cambia entre rounds. Antes
  de pre-registrar un diseno hay que saber si esas senales separan en las seis fuentes de
  C1, y en particular donde el guante no.

QUE HACE
  Por fuente, con la misma siembra simulada de C1 y de IE6:
  - recorre el video (un cuadro de cada `paso`) y, por deteccion con pose, mide el color de
    la REMERA (hombros a caderas) y del PANTALON (caderas hacia rodillas) con la misma
    funcion que el guante (color_de_parche), y el alto de la caja;
  - perfiles A y B de cada senal con las muestras de las semillas (las aisladas, como el
    guante), separacion con distancia_color (comparable con el piso 0,55);
  - acierto por muestra contra la verdad de C1 (perfil mas cercano), aisladas y en
    contacto por separado, para guante, remera, pantalon y remera+pantalon;
  - altura: en los cuadros donde A y B de la verdad coexisten, cociente de altos y en cuantos
    el mas alto es el que dicen las semillas.
  Guarda las muestras por fuente en --cache-dir (reanudable) y escribe un JSON.

USO
  python scripts/identidad/diagnostico_senales_apariencia.py --raiz ~/Proyectos/TwinBoxing \\
      --evidencia-dir ~/boxtwin/mediciones/c1 \\
      --out ~/boxtwin/mediciones/identidad-ropa/senales.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import diagnostico_identidad_estable as diag  # noqa: E402  (arma sys.path del repo)

A, B = diag.A, diag.B
# Las zonas y el color de la ropa son las del producto (IE7), para que las muestras que se
# guardan aca sirvan de evidencia en la medicion de IE7 sin volver a leer el video.
from boxtwin.core.apariencia import medir_ropa, zona_pantalon, zona_remera  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Partes puras
# ---------------------------------------------------------------------------


def senal(c, pa, pb) -> float:
    """Positiva es "mas A": d(c, perfil B) - d(c, perfil A), como en segmentar (E11)."""
    from boxtwin.core.deteccion_guantes import distancia_color
    return distancia_color(c, pb) - distancia_color(c, pa)


def acierto(muestras: list[tuple[str, float]]) -> dict:
    """`muestras` = [(rol verdad, senal)]. Acierta si el signo dice el rol; 0 no cuenta."""
    n = bien = 0
    for rol, s in muestras:
        if rol not in (A, B) or s == 0:
            continue
        n += 1
        bien += int((s > 0) == (rol == A))
    return {"n": n, "acierto": round(bien / n, 4) if n else None}


def altura_relativa(pares: list[tuple[float, float]], a_mas_alto_semillas: bool | None) -> dict:
    """`pares` = [(alto A, alto B)] en cuadros donde coexisten. Coincidencia con las semillas."""
    if not pares:
        return {"n": 0}
    r = np.asarray([a / b for a, b in pares if b > 0])
    mas_alto_a = float((r > 1).mean())
    salida = {"n": int(len(r)), "cociente_mediana": round(float(np.median(r)), 3),
              "a_mas_alto": round(mas_alto_a, 4)}
    if a_mas_alto_semillas is not None:
        salida["acierto_con_semillas"] = round(mas_alto_a if a_mas_alto_semillas
                                               else 1 - mas_alto_a, 4)
    return salida


# ---------------------------------------------------------------------------
# Medicion
# ---------------------------------------------------------------------------


def muestrear(video: Path, cache, cfg, fin: int) -> list[dict]:
    """Una muestra por deteccion con pose, un cuadro de cada cfg.paso."""
    import cv2
    from boxtwin.core.identidad_auto import _iou

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir {video}")
    salida, siguiente = [], 0
    try:
        for f in range(0, fin, cfg.paso):
            while siguiente < f and cap.grab():
                siguiente += 1
            ok, frame = cap.read() if siguiente == f else (False, None)
            siguiente += 1
            if not ok:
                break
            dets = cache.detections(f)
            cajas = [np.asarray(dets.bbox[i], dtype=np.float64) for i in range(len(dets))]
            for i in range(len(dets)):
                x0, y0, x1, y1 = cajas[i]
                alto = y1 - y0
                # El mismo criterio de aislamiento que el color del guante (analizar).
                aislado = all(
                    j == i or (cajas[j][3] - cajas[j][1]) < cfg.fraccion_alto_relevante * alto
                    or _iou(cajas[i], cajas[j]) <= cfg.iou_max_aislado
                    for j in range(len(dets)))
                m = {"t": int(dets.track_id[i]), "f": f, "alto": float(alto), "aislado": aislado}
                remera, pantalon = medir_ropa(frame, dets.keypoints[i], dets.kp_score[i])
                m["remera"] = list(remera) if remera else None
                m["pantalon"] = list(pantalon) if pantalon else None
                salida.append(m)
    finally:
        cap.release()
    return salida


def medir_fuente(video: Path, evd: Path, cache_dir: Path) -> dict:
    from boxtwin.core.deteccion_guantes import distancia_color, perfil_de
    from medir_e11 import _serie

    p = diag.preparar(video, evd)
    res: dict = {"fuente": video.stem, "separacion_guante_ie6": p["separacion"]}
    if not p["semillas"]:
        res["motivo"] = "sin siembra posible"
        return res
    sa, sb = p["semillas"]
    arch = cache_dir / f"{video.stem}.muestras.json"
    if arch.is_file():
        muestras = json.loads(arch.read_text())
    else:
        muestras = muestrear(video, p["cache"], p["cfg"], min(p["T"], len(p["cache"])))
        arch.write_text(json.dumps(muestras))
    verdad = p["verdad"]

    def rol(t: int, f: int) -> str:
        return str(_serie(verdad.get(t, []), np.asarray([f]))[0])

    for m in muestras:
        m["rol"] = rol(m["t"], m["f"])

    # Guante: las muestras de la evidencia de C1 (ya aisladas), con su cuadro.
    guante = [(rol(t, f), c, True, t) for t, e in p["ev"].items() for f, c in e.colores]
    senales = {"guante": guante}
    for nombre in ("remera", "pantalon"):
        senales[nombre] = [(m["rol"], tuple(m[nombre]), m["aislado"], m["t"])
                           for m in muestras if m[nombre] is not None]

    perfiles, sal = {}, {}
    for nombre, ms in senales.items():
        de = {s: [c for _, c, ais, t in ms if t == s and ais] for s in (sa, sb)}
        if not de[sa] or not de[sb]:
            sal[nombre] = {"motivo": "semilla sin muestras aisladas"}
            continue
        pa, pb = perfil_de(de[sa]), perfil_de(de[sb])
        perfiles[nombre] = (pa, pb)
        sal[nombre] = {
            "separacion": round(distancia_color(pa, pb), 3),
            "perfiles": [list(np.round(pa, 1)), list(np.round(pb, 1))],
            "aisladas": acierto([(r, senal(c, pa, pb)) for r, c, ais, _ in ms if ais]),
            "en_contacto": acierto([(r, senal(c, pa, pb)) for r, c, ais, _ in ms if not ais]),
        }
    if "remera" in perfiles and "pantalon" in perfiles:
        (ra, rb), (pa_, pb_) = perfiles["remera"], perfiles["pantalon"]
        comb = {True: [], False: []}
        for m in muestras:
            if m["remera"] is None or m["pantalon"] is None:
                continue
            s = senal(tuple(m["remera"]), ra, rb) + senal(tuple(m["pantalon"]), pa_, pb_)
            comb[m["aislado"]].append((m["rol"], s))
        sal["remera+pantalon"] = {"aisladas": acierto(comb[True]),
                                  "en_contacto": acierto(comb[False])}

    # Altura: cuadros donde la verdad tiene a A y a B, el mas grande de cada rol.
    por_cuadro: dict[int, dict[str, float]] = {}
    for m in muestras:
        if m["rol"] in (A, B):
            d = por_cuadro.setdefault(m["f"], {})
            d[m["rol"]] = max(d.get(m["rol"], 0.0), m["alto"])
    pares = [(d[A], d[B]) for d in por_cuadro.values() if A in d and B in d]
    # Lo que sabe el producto: el alto de las semillas cuando coexisten.
    alto_sem = {}
    for m in muestras:
        if m["t"] in (sa, sb):
            alto_sem.setdefault(m["f"], {})[m["t"]] = m["alto"]
    sem = [d[sa] / d[sb] for d in alto_sem.values() if sa in d and sb in d and d[sb] > 0]
    sal["altura"] = altura_relativa(pares, (float(np.median(sem)) > 1) if sem else None)
    sal["altura"]["cuadros_semillas"] = len(sem)

    res["senales"] = sal
    res["muestras"] = {"detecciones": len(muestras),
                       "remera": len(senales["remera"]), "pantalon": len(senales["pantalon"])}
    return res


def main(argv=None) -> int:
    from medir_c1 import FUENTES

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raiz", type=Path, default=Path.home() / "Proyectos" / "TwinBoxing")
    ap.add_argument("--evidencia-dir", type=Path,
                    default=Path.home() / "boxtwin" / "mediciones" / "c1")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "boxtwin" / "mediciones" / "identidad-ropa" / "muestras")
    ap.add_argument("--out", type=Path, default=Path.home() / "boxtwin" / "mediciones"
                    / "identidad-ropa" / "senales.json")
    args = ap.parse_args(argv)
    raiz, evd = args.raiz.expanduser(), args.evidencia_dir.expanduser()
    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    salida = {"etiqueta": "diagnostico de senales de apariencia, desarrollo, no decide nada",
              "fuentes": []}
    for f in FUENTES:
        r = medir_fuente(raiz / f, evd, cache_dir)
        salida["fuentes"].append(r)
        print(r["fuente"], json.dumps(r.get("senales", r.get("motivo")), ensure_ascii=False))
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
