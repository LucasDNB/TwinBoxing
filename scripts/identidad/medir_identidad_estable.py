"""
BoxTwin - Identidad estable: errores silenciosos, cola de revision y usuario simulado (IE5).

POR QUE EXISTE
  La identidad automatica no llega al 100% (C1 0,810 por track; E11 0,923 por cuadro en
  desarrollo). Lo que hace estable al producto es otra cosa: que ningun cuadro con el rol
  equivocado llegue a la Fight-Card sin haber pasado por los ojos de la persona
  (docs/specs/2026-11-02-identidad-estable/00-README.md). Este script mide eso, y cuanto le
  cuesta a la persona, sin GPU: lee la evidencia de guantes ya guardada de C1.

QUE HACE
  Por fuente (las seis de C1, con la siembra simulada de C1 y E11) y por condicion (base,
  corte, corte+estricta):
  - precision y cobertura por cuadro de lo automatico (como E11);
  - error silencioso: cuadro con rol A o B equivocado FUERA de todo item de la cola;
    error visible: el mismo error ADENTRO de un item;
  - la cola: items, segundos a mirar, items por minuto de video;
  - usuario simulado: contesta cada item con la verdad (un item con A y B, al menos 5 cuadros
    cada uno, son dos decisiones: partir y asignar) y se recalculan precision y cobertura.
  Despues aplica la regla de §6 (E1, E2, eleccion de condicion) y escribe un JSON.

USO
  python scripts/identidad/medir_identidad_estable.py
  python scripts/identidad/medir_identidad_estable.py --raiz ~/Proyectos/TwinBoxing \\
      --evidencia-dir ~/boxtwin/mediciones/c1 --out ~/boxtwin/mediciones/identidad-estable/desarrollo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
for _r in (REPO / "scripts" / "mejoras", REPO / "boxtwin-annotator" / "scripts",
           REPO / "boxtwin-annotator" / "src"):
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

A, B = "fighter_A", "fighter_B"
ROL_LARGO = {"A": A, "B": B, "ninguno": "ninguno"}
MIN_LADO = 5                      # cuadros de cada rol para que un item cuente como mixto
CONDICIONES = {"base": (False, False), "corte": (True, False), "corte+estricta": (True, True)}
# Regla de §6, fija desde el commit de la spec.
E1_TOTAL = 0.005
E1_FUENTE = 0.02
E2_PRECISION = 0.99
EMPATE_DECISIONES = 0.1


# ---------------------------------------------------------------------------
# Partes puras
# ---------------------------------------------------------------------------


def _serie(rangos, frames: np.ndarray) -> np.ndarray:
    r = np.full(len(frames), "", dtype=object)
    for ini, fin, rol in rangos:
        r[(frames >= ini) & (frames < fin)] = rol
    return r


def _por_track(propuesta) -> dict[int, list]:
    salida: dict[int, list] = {}
    for t, ini, fin, rol in propuesta:
        salida.setdefault(int(t), []).append((ini, fin, ROL_LARGO.get(rol, rol)))
    return salida


def errores(verdad: dict, propuesta, cola: list[dict], frames_de) -> dict:
    """
    Cuadros con rol A o B equivocado, separados en silenciosos (fuera de la cola) y visibles.
    `verdad` es track -> [(ini, fin, rol largo)]; `propuesta` es [(track, ini, fin, "A"|"B"|...)];
    `cola` son los items de cola_de_revision (con track, desde, hasta).
    """
    prop = _por_track(propuesta)
    zonas: dict[int, list] = {}
    for it in cola:
        zonas.setdefault(int(it["track"]), []).append((it["desde"], it["hasta"]))
    silenciosos = visibles = peleador = 0
    for t in set(verdad) | set(prop):
        fs = frames_de(t)
        if fs is None or not len(fs):
            continue
        fs = np.asarray(fs)
        v = _serie(verdad.get(t, []), fs)
        p = _serie(prop.get(t, []), fs)
        mal = ((p == A) | (p == B)) & (p != v)
        en_cola = np.zeros(len(fs), bool)
        for ini, fin in zonas.get(t, []):
            en_cola |= (fs >= ini) & (fs < fin)
        silenciosos += int((mal & ~en_cola).sum())
        visibles += int((mal & en_cola).sum())
        peleador += int(((v == A) | (v == B)).sum())
    return {"silenciosos": silenciosos, "visibles": visibles, "peleador": peleador,
            "fraccion_silenciosa": silenciosos / peleador if peleador else None}


def _corridas(fs: np.ndarray, v: np.ndarray, hasta: int) -> list[tuple[int, int, str]]:
    """Tramos contiguos de la verdad adentro de un item: (desde, hasta, rol corto)."""
    corto = np.where(v == A, "A", np.where(v == B, "B", "ninguno"))
    salida = []
    i = 0
    while i < len(fs):
        j = i
        while j + 1 < len(fs) and corto[j + 1] == corto[i]:
            j += 1
        fin = int(fs[j + 1]) if j + 1 < len(fs) else hasta
        salida.append((int(fs[i]), fin, str(corto[i])))
        i = j + 1
    return salida


def oraculo(cola: list[dict], verdad: dict, frames_de) -> tuple[list, int]:
    """
    El usuario simulado: contesta cada item que pide revision con la verdad. Devuelve las
    correcciones (track, desde, hasta, rol corto) y cuantas decisiones costaron.
    Un item con A y B (MIN_LADO cuadros cada uno) se parte donde dice la verdad: dos
    decisiones aunque tenga mas de un cambio (es generoso con la maquina; se declara).
    """
    from boxtwin.mvp.revision import Correccion

    correcciones, decisiones = [], 0
    for it in cola:
        if not it.get("revision_pendiente"):
            continue
        t = int(it["track"])
        fs = frames_de(t)
        fs = np.asarray(fs if fs is not None else [])
        fs = fs[(fs >= it["desde"]) & (fs < it["hasta"])]
        if not len(fs):
            continue
        v = _serie(verdad.get(t, []), fs)
        na, nb = int((v == A).sum()), int((v == B).sum())
        if na >= MIN_LADO and nb >= MIN_LADO:
            decisiones += 2
            for d, h, rol in _corridas(fs, v, it["hasta"]):
                correcciones.append(Correccion(t, d, h, rol))
            continue
        decisiones += 1
        nn = len(fs) - na - nb
        rol = max((("A", na), ("B", nb), ("ninguno", nn)), key=lambda x: x[1])[0]
        correcciones.append(Correccion(t, int(it["desde"]), int(it["hasta"]), rol))
    return correcciones, decisiones


def por_cuadro(verdad: dict, propuesta, frames_de) -> dict:
    prop = _por_track(propuesta)
    bien = asignados = peleador = 0
    for t in set(verdad) | set(prop):
        fs = frames_de(t)
        if fs is None or not len(fs):
            continue
        fs = np.asarray(fs)
        v = _serie(verdad.get(t, []), fs)
        p = _serie(prop.get(t, []), fs)
        con_rol = (p == A) | (p == B)
        bien += int((con_rol & (p == v)).sum())
        asignados += int(con_rol.sum())
        peleador += int(((v == A) | (v == B)).sum())
    return {"bien": bien, "asignados": asignados, "peleador": peleador,
            "precision": bien / asignados if asignados else None,
            "cobertura": bien / peleador if peleador else None}


def en_rangos(tramos) -> list[tuple[int, int, int, str]]:
    return [(t.track, t.desde, t.hasta, t.rol) for t in tramos if t.rol in ("A", "B")]


def sumar(fuentes: list[dict], cond: str) -> dict:
    """Totales de una condicion sobre las fuentes medidas."""
    ms = [f[cond] for f in fuentes if cond in f]
    s = lambda campo, k: sum(m[campo][k] for m in ms)  # noqa: E731
    minutos = sum(f["minutos"] for f in fuentes if cond in f)
    sil, pel = s("errores", "silenciosos"), s("errores", "peleador")
    ob, oa, op = s("oraculo", "bien"), s("oraculo", "asignados"), s("oraculo", "peleador")
    ab, aa, ap = s("auto", "bien"), s("auto", "asignados"), s("auto", "peleador")
    decisiones = sum(m["decisiones"] for m in ms)
    return {
        "fuentes": len(ms),
        "auto": {"precision": ab / aa if aa else None, "cobertura": ab / ap if ap else None},
        "errores": {"silenciosos": sil, "visibles": s("errores", "visibles"), "peleador": pel,
                    "fraccion_silenciosa": sil / pel if pel else None,
                    "peor_fuente": max((m["errores"]["fraccion_silenciosa"] or 0.0 for m in ms),
                                       default=None)},
        "cola": {"items": sum(m["cola"]["items"] for m in ms),
                 "segundos": round(sum(m["cola"]["segundos"] for m in ms), 1)},
        "decisiones": decisiones,
        "decisiones_por_minuto": decisiones / minutos if minutos else None,
        "oraculo": {"precision": ob / oa if oa else None, "cobertura": ob / op if op else None},
    }


def regla(totales: dict[str, dict]) -> dict:
    """§6: E1, E2 y que condicion se enciende."""
    veredictos = {}
    for cond, t in totales.items():
        fs = t["errores"]["fraccion_silenciosa"]
        peor = t["errores"]["peor_fuente"]
        e1 = fs is not None and fs <= E1_TOTAL and (peor or 0.0) <= E1_FUENTE
        e2 = (t["oraculo"]["precision"] or 0.0) >= E2_PRECISION
        veredictos[cond] = {"E1": e1, "E2": e2}
    pasan = [c for c in totales if veredictos[c]["E1"] and veredictos[c]["E2"]]
    banderas = {c: sum(CONDICIONES[c]) for c in totales}
    if pasan:
        mejor = min(totales[c]["decisiones_por_minuto"] or 0.0 for c in pasan)
        empatados = [c for c in pasan
                     if (totales[c]["decisiones_por_minuto"] or 0.0) - mejor <= EMPATE_DECISIONES]
        elegida = min(empatados, key=lambda c: (banderas[c], totales[c]["decisiones_por_minuto"]))
        estable = True
    else:
        elegida = min(totales, key=lambda c: (totales[c]["errores"]["silenciosos"], banderas[c]))
        estable = False
    return {"veredictos": veredictos, "elegida": elegida, "estable": estable,
            "banderas": {"BOXTWIN_CORTE_MIXTOS": int(CONDICIONES[elegida][0]),
                         "BOXTWIN_IDENTIDAD_ESTRICTA": int(CONDICIONES[elegida][1])}}


# ---------------------------------------------------------------------------
# Medicion
# ---------------------------------------------------------------------------


def medir_condicion(ev, cfg, T, fps, semillas, cache, cuadro, verdad, frames_de,
                    cortar: bool, estricta: bool) -> dict:
    from boxtwin.core.identidad_auto import proponer
    from boxtwin.mvp.revision import (
        aplicar_correcciones, cola_de_revision, degradar_conflictos, tramos_de_propuesta,
    )

    c = replace(cfg, cortar_mixtos=cortar)
    prop = proponer(ev, c, T, fps, semillas=semillas, cache=cache, cuadros_semilla=cuadro)
    auto = tramos_de_propuesta(prop, ev, c)
    final = degradar_conflictos(auto, ev, c)[0] if estricta else auto
    cola = cola_de_revision(auto, auto, ev, prop.perfiles, c, fps, todos=False, estricta=estricta)
    pendientes = [i for i in cola if i["revision_pendiente"]]
    corr, decisiones = oraculo(pendientes, verdad, frames_de)
    corregidos = aplicar_correcciones(auto, corr)
    final_o = degradar_conflictos(corregidos, ev, c)[0] if estricta else corregidos
    por_motivo: dict[str, int] = {}
    for i in pendientes:
        por_motivo[i["motivo"]] = por_motivo.get(i["motivo"], 0) + 1
    return {
        "auto": por_cuadro(verdad, en_rangos(final), frames_de),
        "errores": errores(verdad, en_rangos(final), pendientes, frames_de),
        "cola": {"items": len(pendientes), "por_motivo": por_motivo,
                 "segundos": round(sum((i["hasta"] - i["desde"]) / fps for i in pendientes), 1)},
        "decisiones": decisiones,
        "oraculo": por_cuadro(verdad, en_rangos(final_o), frames_de),
        "tracks_partidos": prop.diagnostico.get("tracks_partidos", 0),
    }


def medir_fuente(video: Path, evidencia_dir: Path) -> dict:
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, cargar_evidencia, proponer
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths
    from boxtwin.mvp.candidatos import elegir
    from medir_c1 import sembrar, verdad_de
    from medir_e11 import rangos_verdad

    paths = project_paths(video)
    doc, _ = cargar_doc(paths.annot)                       # solo lectura
    ev_path = evidencia_dir / f"{video.stem}.evidencia.json"
    if not ev_path.is_file():
        raise FileNotFoundError(f"falta la evidencia {ev_path}: es la que guardo C1")
    ev = cargar_evidencia(ev_path)
    cache = PoseCache.open(paths.npz)
    cfg = ConfigIdentidadAuto()
    T, fps = doc.video.total_frames, doc.video.fps
    auto = proponer(ev, cfg, T, fps)
    pareja, cands = elegir(ev, auto.diagnostico.get("nucleo_tracks", []),
                           min_coexistencia=cfg.min_coexistencia, fps=fps,
                           separacion_minima=cfg.separacion_minima)
    verdad_t, _ = verdad_de(doc.identity.assignments, cache.frames_of_track)
    siembra = sembrar(pareja, [c.track for c in cands], verdad_t)
    res: dict = {"fuente": video.stem, "minutos": T / fps / 60.0 if fps else 0.0,
                 "siembra": {"caso": siembra["caso"],
                             "semillas": list(siembra["semillas"] or [])}}
    if not siembra["semillas"]:
        return res
    semillas = siembra["semillas"]
    cuadro = {c.track: c.cuadro for c in cands if c.track in semillas}
    verdad = rangos_verdad(doc.identity.assignments)
    for nombre, (cortar, estricta) in CONDICIONES.items():
        res[nombre] = medir_condicion(ev, cfg, T, fps, semillas, cache, cuadro, verdad,
                                      cache.frames_of_track, cortar, estricta)
    return res


def main(argv=None) -> int:
    from medir_c1 import FUENTES

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raiz", type=Path,
                    default=Path(os.environ.get("BOXTWIN_REPO_A", "~/Proyectos/TwinBoxing")))
    ap.add_argument("--evidencia-dir", type=Path,
                    default=Path(os.environ.get("BOXTWIN_C1", "~/boxtwin/mediciones/c1")))
    ap.add_argument("--out", type=Path,
                    default=Path("~/boxtwin/mediciones/identidad-estable/desarrollo.json"))
    args = ap.parse_args(argv)
    raiz, evd = args.raiz.expanduser(), args.evidencia_dir.expanduser()
    fuentes = []
    for f in FUENTES:
        r = medir_fuente(raiz / f, evd)
        fuentes.append(r)
        if "base" not in r:
            print(f"{r['fuente']}: sin siembra ({r['siembra']['caso']})", flush=True)
            continue
        for cond in CONDICIONES:
            m = r[cond]
            print(f"{r['fuente']:>18} {cond:>15}: silenciosos {m['errores']['silenciosos']:6d} "
                  f"({(m['errores']['fraccion_silenciosa'] or 0):.3%}), cola {m['cola']['items']:4d}, "
                  f"decisiones {m['decisiones']:4d}, precision tras revisar "
                  f"{(m['oraculo']['precision'] or 0):.3f}", flush=True)
    totales = {c: sumar(fuentes, c) for c in CONDICIONES}
    salida = {
        "etiqueta": "desarrollo: material donde se diagnostico C1 y se diseno E11; no confirma",
        "spec": "docs/specs/2026-11-02-identidad-estable/00-README.md",
        "regla": {"E1_total": E1_TOTAL, "E1_fuente": E1_FUENTE, "E2_precision": E2_PRECISION},
        "fuentes": fuentes, "totales": totales, "lectura": regla(totales),
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float) + "\n")
    lec = salida["lectura"]
    print(f"\nelegida: {lec['elegida']} ({'estable' if lec['estable'] else 'NO estable'}); "
          f"banderas {lec['banderas']}\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
