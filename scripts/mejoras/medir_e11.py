"""
BoxTwin - E11: identidad por cuadro con y sin el corte de tracks mixtos (T7, 30-10-2026).

POR QUE EXISTE
  E11 (docs/specs/2026-10-30-mejoras-post-e8/03-e11-tracks-mixtos.md) parte los tracks donde
  el color dice que cambio de persona. Con cortes, "un track" deja de ser la unidad, asi que
  C1 (por track) no sirve para decidir: la medida primaria es por cuadro (§3). Y la idea de
  cortar salio de mirar los errores de C1, asi que el material de C1 es de desarrollo y lo
  que decide es material nuevo (§4).

QUE HACE
  Por video: la siembra simulada con la regla fija de C1 (medir_c1: elegir, verdad_de,
  sembrar; el cuadro del recorte de cada semilla es el que muestra la pantalla), la propuesta
  sin corte y con corte, y por cada una:
  - precision por cuadro: cuadros con rol correcto / cuadros con rol A o B asignado;
  - cobertura por cuadro: cuadros con rol correcto / cuadros de peleador segun la verdad;
  contados en los cuadros donde el track tiene deteccion en el cache.
  Al lado: C1 por track sin corte (medir_c1.puntuar, tiene que reproducir el 85/105 de C1),
  la misma cuenta por segmento con corte (P15c, referencia), y P15d (tracks partidos, cuantos
  son mixtos segun la verdad, cuantos mixtos se partieron).
  `desarrollo` usa las seis fuentes de C1 con la evidencia guardada (sin GPU);
  `confirmacion` usa anotaciones nuevas con su evidencia. Escribe <mejoras>/e11/<modo>.json.

USO
  python scripts/mejoras/medir_e11.py desarrollo
  python scripts/mejoras/medir_e11.py confirmacion --video a.mp4 b.mp4 --evidencia-dir ~/boxtwin-mejoras/e11/evidencia
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rutas  # noqa: E402

sys.path.insert(0, str(rutas.REPO / "boxtwin-annotator" / "scripts"))

A, B = "fighter_A", "fighter_B"


def _rol(r) -> str:
    return getattr(r, "value", r)


# ---------------------------------------------------------------------------
# Partes puras
# ---------------------------------------------------------------------------


def rangos_verdad(asignaciones) -> dict[int, list[tuple[int, int, str]]]:
    """track -> [(inicio, fin, rol)] de la anotacion, con cualquier rol."""
    salida: dict[int, list] = {}
    for a in asignaciones:
        salida.setdefault(int(a.track_id), []).append(
            (int(a.start_frame), int(a.end_frame_excl), _rol(a.role)))
    return salida


def _serie(rangos, frames: np.ndarray) -> np.ndarray:
    """El rol de cada cuadro dado, o '' donde ningun rango lo cubre. El ultimo rango gana."""
    r = np.full(len(frames), "", dtype=object)
    for ini, fin, rol in rangos:
        r[(frames >= ini) & (frames < fin)] = rol
    return r


def por_cuadro(verdad: dict, propuesta: list[tuple[int, int, int, str]], frames_de) -> dict:
    """
    `propuesta` = [(track, inicio, fin, rol)]. Precision y cobertura por cuadro, en los
    cuadros donde cada track tiene deteccion (`frames_de(track)`).
    """
    prop: dict[int, list] = {}
    for t, ini, fin, rol in propuesta:
        prop.setdefault(int(t), []).append((ini, fin, _rol(rol)))
    bien = asignados = peleador = 0
    for t in set(verdad) | set(prop):
        fs = frames_de(t)
        if fs is None or not len(fs):
            continue
        fs = np.asarray(fs)
        v = _serie(verdad.get(t, []), fs)
        p = _serie(prop.get(t, []), fs)
        es_pel = (v == A) | (v == B)
        con_rol = (p == A) | (p == B)
        bien += int((con_rol & (p == v)).sum())
        asignados += int(con_rol.sum())
        peleador += int(es_pel.sum())
    return {"bien": bien, "asignados": asignados, "peleador": peleador,
            "precision": bien / asignados if asignados else None,
            "cobertura": bien / peleador if peleador else None}


def por_unidad(verdad: dict, propuesta: list[tuple[int, int, int, str]], frames_de) -> dict:
    """P15c con cortes: cada segmento contra el rol mayoritario (A o B) dentro de su rango."""
    evaluables = aciertos = 0
    for t, ini, fin, rol in propuesta:
        fs = np.asarray(frames_de(t) if frames_de(t) is not None else [])
        fs = fs[(fs >= ini) & (fs < fin)]
        v = _serie(verdad.get(int(t), []), fs)
        na, nb = int((v == A).sum()), int((v == B).sum())
        if na + nb == 0 or _rol(rol) not in (A, B):
            continue
        evaluables += 1
        aciertos += int(_rol(rol) == (A if na >= nb else B))
    return {"evaluables": evaluables, "aciertos": aciertos}


def p15d(partidos: set[int], verdad_cuadros: dict[int, dict]) -> dict:
    """Tracks partidos, cuantos de esos son mixtos segun la verdad, y cuantos mixtos se partieron."""
    mixtos = {t for t, d in verdad_cuadros.items() if d.get(A) and d.get(B)}
    pm = partidos & mixtos
    return {"partidos": len(partidos), "mixtos": len(mixtos), "partidos_mixtos": len(pm),
            "precision_del_corte": len(pm) / len(partidos) if partidos else None,
            "recall_del_corte": len(pm) / len(mixtos) if mixtos else None}


def total(videos: list[dict]) -> dict:
    """Suma por cuadro de los videos, con Wilson (los cuadros no son independientes: se declara)."""
    from medir_c1 import wilson
    bien = sum(v["bien"] for v in videos)
    asig = sum(v["asignados"] for v in videos)
    pel = sum(v["peleador"] for v in videos)
    return {"bien": bien, "asignados": asig, "peleador": pel,
            "precision": bien / asig if asig else None, "cobertura": bien / pel if pel else None,
            "wilson95_precision": wilson(bien, asig), "wilson95_cobertura": wilson(bien, pel),
            "advertencia": "los cuadros no son independientes: el intervalo es optimista"}


def propuesta_en_rangos(prop) -> list[tuple[int, int, int, str]]:
    return [(prop.tracks.get(k, k), *prop.rangos[k], _rol(r)) for k, r in prop.roles.items()
            if _rol(r) in (A, B)]


# ---------------------------------------------------------------------------
# Medicion
# ---------------------------------------------------------------------------


def medir(doc, cache, ev, nombre: str) -> dict:
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, proponer
    from boxtwin.mvp.candidatos import elegir
    from medir_c1 import puntuar, sembrar, verdad_de

    cfg = ConfigIdentidadAuto()
    T, fps = doc.video.total_frames, doc.video.fps
    auto = proponer(ev, cfg, T, fps)
    pareja, cands = elegir(ev, auto.diagnostico.get("nucleo_tracks", []),
                           min_coexistencia=cfg.min_coexistencia, fps=fps,
                           separacion_minima=cfg.separacion_minima)
    verdad_t, cuadros = verdad_de(doc.identity.assignments, cache.frames_of_track)
    siembra = sembrar(pareja, [c.track for c in cands], verdad_t)
    res: dict = {"fuente": nombre, "siembra": {"caso": siembra["caso"],
                                               "semillas": list(siembra["semillas"] or [])}}
    if not siembra["semillas"]:
        return res
    semillas = siembra["semillas"]
    cuadro = {c.track: c.cuadro for c in cands if c.track in semillas}
    verdad = rangos_verdad(doc.identity.assignments)
    frames_de = cache.frames_of_track

    sin = proponer(ev, cfg, T, fps, semillas=semillas)
    con = proponer(ev, replace(cfg, cortar_mixtos=True), T, fps, semillas=semillas,
                   cache=cache, cuadros_semilla=cuadro)
    res["sin_corte"] = {**por_cuadro(verdad, propuesta_en_rangos(sin), frames_de),
                        "c1_por_track": puntuar({t: r.value for t, r in sin.roles.items()},
                                                verdad_t, cuadros)}
    res["con_corte"] = {**por_cuadro(verdad, propuesta_en_rangos(con), frames_de),
                        "c1_por_segmento": por_unidad(verdad, propuesta_en_rangos(con), frames_de),
                        "cortes": con.diagnostico.get("cortes", [])}
    res["p15d"] = p15d(set(con.tracks.values()), cuadros)
    return res


def _video(v: Path, evidencia_dir: Path) -> dict:
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identidad_auto import cargar_evidencia
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    paths = project_paths(v)
    doc, _ = cargar_doc(paths.annot)        # solo lectura
    ev_path = evidencia_dir / f"{v.stem}.evidencia.json"
    if not ev_path.is_file():
        raise FileNotFoundError(f"falta la evidencia {ev_path}: correr analizar antes (S11)")
    return medir(doc, PoseCache.open(paths.npz), cargar_evidencia(ev_path), v.stem)


def resumen(videos: list[dict], modo: str) -> dict:
    medidos = [v for v in videos if "sin_corte" in v]
    salida = {"modo": modo, "videos": videos}
    for cond in ("sin_corte", "con_corte"):
        salida[cond] = {"total": total([v[cond] for v in medidos])}
    salida["sin_corte"]["c1"] = {k: sum(v["sin_corte"]["c1_por_track"][k] for v in medidos)
                                 for k in ("aciertos", "evaluables")}
    salida["con_corte"]["c1_por_segmento"] = {
        k: sum(v["con_corte"]["c1_por_segmento"][k] for v in medidos) for k in ("aciertos", "evaluables")}
    salida["p15d"] = {k: sum(v["p15d"][k] for v in medidos)
                      for k in ("partidos", "mixtos", "partidos_mixtos")}
    if modo == "desarrollo":
        salida["etiqueta"] = "desarrollo: material donde se diseno el corte, no decide P15a ni P15b"
    return salida


def main(argv=None) -> int:
    from medir_c1 import FUENTES

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="modo", required=True)
    d = sub.add_parser("desarrollo", help="las seis fuentes de C1 con su evidencia guardada")
    d.add_argument("--raiz", type=Path, default=rutas.REPO_A)
    d.add_argument("--evidencia-dir", type=Path, default=rutas.C1)
    c = sub.add_parser("confirmacion", help="videos nuevos anotados a mano (L3)")
    c.add_argument("--video", type=Path, nargs="+", required=True)
    c.add_argument("--evidencia-dir", type=Path, default=rutas.MEJORAS / "e11" / "evidencia")
    for x in (d, c):
        x.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    videos = [args.raiz / f for f in FUENTES] if args.modo == "desarrollo" else args.video
    res = []
    for v in videos:
        r = _video(Path(v).expanduser(), args.evidencia_dir.expanduser())
        res.append(r)
        s, k = r.get("sin_corte"), r.get("con_corte")
        if s:
            print(f"{r['fuente']}: precision {s['precision']:.3f} -> {k['precision']:.3f}, "
                  f"cobertura {s['cobertura']:.3f} -> {k['cobertura']:.3f}, partidos {r['p15d']['partidos']}",
                  flush=True)
        else:
            print(f"{r['fuente']}: {r['siembra']['caso']}", flush=True)
    out = args.out or rutas.MEJORAS / "e11" / f"{args.modo}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resumen(res, args.modo), indent=2, ensure_ascii=False) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
