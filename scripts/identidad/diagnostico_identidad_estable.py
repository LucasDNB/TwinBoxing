"""
BoxTwin - Identidad estable: diagnostico de IE6 (no pre-registrado, no decide nada).

POR QUE EXISTE
  IE6 (medir_identidad_estable.py, regla de §6) dio "no estable": errores silenciosos en 1,54%
  y precision tras revisar 0,970. Para leer eso hace falta saber de donde salen: que forma
  tienen los errores silenciosos (rol intercambiado en todo el tramo, tramo mixto, o un tercero
  con rol de peleador), por que quedan errores adentro de items que el usuario simulado ya
  contesto, y que pasa si la cola se recalcula despues de la primera ronda de correcciones,
  como hace el producto (los conflictos se buscan en los corregidos). Nada de esto cambia la
  regla ni el veredicto de IE6.

QUE HACE
  Repite la preparacion de IE6 (misma evidencia, misma siembra) y, por fuente y condicion:
  - errores con rol A o B equivocado, por forma (intercambiado, mixto, tercero = verdad
    "ignore" o sin anotar) y por si caen adentro o afuera de la cola;
  - lo mismo despues del usuario simulado (ronda 1);
  - ronda 2: la cola recalculada con las correcciones de la ronda 1, cuantos cuadros mal
    destapa, cuantas decisiones mas cuesta y la precision y cobertura despues.
  Escribe un JSON.

USO
  python scripts/identidad/diagnostico_identidad_estable.py --raiz ~/Proyectos/TwinBoxing \\
      --evidencia-dir ~/boxtwin/mediciones/c1 \\
      --out ~/boxtwin/mediciones/identidad-estable/diagnostico.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import medir_identidad_estable as m  # noqa: E402  (arma sys.path del repo)

A, B = m.A, m.B


def formas_de_error(verdad: dict, tramos, cola: list[dict], frames_de) -> dict:
    """Cuadros con A o B equivocado, por '<cola|fuera>/<intercambiado|mixto|tercero>'."""
    zonas: dict[int, list] = {}
    for it in cola:
        zonas.setdefault(int(it["track"]), []).append((it["desde"], it["hasta"]))
    salida: dict[str, int] = {}
    for t in tramos:
        if t.rol not in ("A", "B"):
            continue
        fs = frames_de(t.track)
        if fs is None or not len(fs):
            continue
        fs = np.asarray(fs)
        fs = fs[(fs >= t.desde) & (fs < t.hasta)]
        if not len(fs):
            continue
        v = m._serie(verdad.get(int(t.track), []), fs)
        rol = m.ROL_LARGO[t.rol]
        otro = B if rol == A else A
        mal = v != rol
        if not mal.any():
            continue
        # La forma es del tramo: si la verdad nunca es su rol, esta intercambiado entero.
        if (v == otro).any():
            forma = "intercambiado" if not (v == rol).any() else "mixto"
        else:
            forma = "tercero"
        en = np.zeros(len(fs), bool)
        for d, h in zonas.get(int(t.track), []):
            en |= (fs >= d) & (fs < h)
        # Los cuadros de tercero adentro de un tramo con el otro rol cuentan como tercero.
        for lado, sel in (("cola", mal & en), ("fuera", mal & ~en)):
            n_otro = int((sel & (v == otro)).sum())
            n_terc = int((sel & (v != otro)).sum())
            if n_otro:
                k = f"{lado}/{forma if forma != 'tercero' else 'mixto'}"
                salida[k] = salida.get(k, 0) + n_otro
            if n_terc:
                salida[f"{lado}/tercero"] = salida.get(f"{lado}/tercero", 0) + n_terc
    return salida


def preparar(video: Path, evidencia_dir: Path) -> dict:
    """Lo mismo que medir_fuente de IE6 hasta antes de las condiciones."""
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, cargar_evidencia, proponer
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths
    from boxtwin.mvp.candidatos import elegir
    from medir_c1 import sembrar, verdad_de
    from medir_e11 import rangos_verdad

    paths = project_paths(video)
    doc, _ = cargar_doc(paths.annot)                       # solo lectura
    ev = cargar_evidencia(evidencia_dir / f"{video.stem}.evidencia.json")
    cache = PoseCache.open(paths.npz)
    cfg = ConfigIdentidadAuto()
    T, fps = doc.video.total_frames, doc.video.fps
    auto = proponer(ev, cfg, T, fps)
    pareja, cands = elegir(ev, auto.diagnostico.get("nucleo_tracks", []),
                           min_coexistencia=cfg.min_coexistencia, fps=fps,
                           separacion_minima=cfg.separacion_minima)
    verdad_t, _ = verdad_de(doc.identity.assignments, cache.frames_of_track)
    semillas = sembrar(pareja, [c.track for c in cands], verdad_t)["semillas"]
    return {"ev": ev, "cfg": cfg, "T": T, "fps": fps, "semillas": semillas, "cache": cache,
            "cuadro": {c.track: c.cuadro for c in cands if semillas and c.track in semillas},
            "verdad": rangos_verdad(doc.identity.assignments),
            "separacion": auto.diagnostico.get("separacion_de_perfiles")}


def diagnosticar(p: dict, cortar: bool, estricta: bool) -> dict:
    from boxtwin.core.identidad_auto import proponer
    from boxtwin.mvp.revision import (
        aplicar_correcciones, cola_de_revision, degradar_conflictos, tramos_de_propuesta,
    )

    ev, fps, verdad, fd = p["ev"], p["fps"], p["verdad"], p["cache"].frames_of_track
    c = replace(p["cfg"], cortar_mixtos=cortar)
    prop = proponer(ev, c, p["T"], fps, semillas=p["semillas"], cache=p["cache"],
                    cuadros_semilla=p["cuadro"])
    auto = tramos_de_propuesta(prop, ev, c)

    def final(ts):
        return degradar_conflictos(ts, ev, c)[0] if estricta else ts

    def pendientes(corregidos):
        return [i for i in cola_de_revision(auto, corregidos, ev, prop.perfiles, c, fps,
                                            todos=False, estricta=estricta)
                if i["revision_pendiente"]]

    cola1 = pendientes(auto)
    corr1, dec1 = m.oraculo(cola1, verdad, fd)
    r1 = aplicar_correcciones(auto, corr1)
    cola2 = pendientes(r1)
    corr2, dec2 = m.oraculo(cola2, verdad, fd)
    r2 = aplicar_correcciones(auto, corr1 + corr2)
    destapados = m.errores(verdad, m.en_rangos(final(r1)), cola2, fd)["visibles"]
    return {
        "antes": formas_de_error(verdad, final(auto), cola1, fd),
        "ronda1": formas_de_error(verdad, final(r1), cola1, fd),
        "ronda2": {"items": len(cola2), "decisiones": dec2, "destapados": destapados,
                   "decisiones_total": dec1 + dec2,
                   **m.por_cuadro(verdad, m.en_rangos(final(r2)), fd)},
    }


def main(argv=None) -> int:
    from medir_c1 import FUENTES

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raiz", type=Path, default=Path.home() / "Proyectos" / "TwinBoxing")
    ap.add_argument("--evidencia-dir", type=Path,
                    default=Path.home() / "boxtwin" / "mediciones" / "c1")
    ap.add_argument("--out", type=Path, default=Path.home() / "boxtwin" / "mediciones"
                    / "identidad-estable" / "diagnostico.json")
    args = ap.parse_args(argv)
    raiz, evd = args.raiz.expanduser(), args.evidencia_dir.expanduser()
    salida = {"etiqueta": "diagnostico de IE6, no pre-registrado: no cambia la regla ni el veredicto",
              "fuentes": []}
    for f in FUENTES:
        p = preparar(raiz / f, evd)
        r = {"fuente": (raiz / f).stem, "separacion_de_perfiles": p["separacion"]}
        if p["semillas"]:
            for nombre, (cortar, estricta) in m.CONDICIONES.items():
                r[nombre] = diagnosticar(p, cortar, estricta)
                d = r[nombre]
                print(f"{r['fuente']:>18} {nombre:>15}: antes {d['antes']}  ronda1 {d['ronda1']}  "
                      f"ronda2 +{d['ronda2']['items']} items, destapa {d['ronda2']['destapados']}, "
                      f"precision {d['ronda2']['precision']:.3f}")
        salida["fuentes"].append(r)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
