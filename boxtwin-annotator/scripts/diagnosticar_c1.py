"""
BoxTwin - Diagnostico de C1: que mecanismo de la identidad decidio cada track, y cual fallo.

POR QUE EXISTE
  C1 dio 85/105 (0,810) con siembra humana, y la spec del MVP decia 99,1% "con perfiles
  sembrados". El 99,1% sale de color_guante.py del 18-09: perfiles armados con TODOS los
  guantes de la anotacion (dejando el track afuera), solo tracks con >= 10 guantes, y el
  color decidiendo cada track solo. La identidad del producto no funciona asi: decide
  primero por coexistencia, despues orienta componentes por color, despues vota sueltos y
  rescata fragmentos. Este script separa los 20 errores por mecanismo para saber cual se
  equivoca, y compara cada error con lo que habria dicho el color con perfiles de oraculo.

QUE HACE
  Por fuente, lee la evidencia ya guardada por medir_c1.py (no corre el detector de
  guantes), rearma la propuesta con las mismas semillas y, para cada track evaluable:
  1. Mecanismo: ancla (componente de la semilla A), semilla_b (componente de la semilla B),
     comp_color (componente orientado por color contra el ancla), voto (suelto por color),
     rescate (fuera del nucleo, enganchado por coexistencia o color).
  2. Color de oraculo: voto contra perfiles A y B armados con los guantes de los otros
     tracks segun la anotacion, como el 18-09.
  3. Aristas de coexistencia entre dos tracks del mismo rol segun la anotacion (no deberian
     existir: dos tracks en el mismo cuadro son dos personas) y contra tracks que no son
     peleadores.
  Escribe un JSON por fuente y una tabla agregada. Solo lectura sobre la anotacion.

USO
  python scripts/diagnosticar_c1.py --raiz ~/Proyectos/TwinBoxing \
      --mediciones ~/boxtwin/mediciones/c1 -o ~/boxtwin/mediciones/c1-diagnostico
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medir_c1 import A, B, FUENTES, verdad_de  # noqa: E402

MIN_GUANTES_ORACULO = 10   # el mismo piso que color_guante.py del 18-09


# ---------------------------------------------------------------------------
# Partes puras (con pruebas)
# ---------------------------------------------------------------------------


def aristas_espurias(ady: dict[int, set[int]], frames: dict[int, set[int]],
                     verdad: dict[int, str]) -> tuple[list, list]:
    """
    Coexistencias que la geometria toma como "dos personas distintas" y no lo son segun la
    anotacion: entre dos tracks del mismo rol (con los cuadros compartidos), y entre un
    peleador y un track que no lo es.
    """
    mismo_rol, con_no_peleador = [], []
    for t1 in sorted(ady):
        for t2 in sorted(ady[t1]):
            if t2 <= t1:
                continue
            r1, r2 = verdad.get(t1), verdad.get(t2)
            if r1 and r2 and r1 == r2:
                mismo_rol.append([t1, t2, len(frames[t1] & frames[t2])])
            elif bool(r1) != bool(r2):
                con_no_peleador.append([t1, t2])
    return mismo_rol, con_no_peleador


def voto_oraculo(propios, de_a, de_b) -> str | None:
    """Voto por guante contra perfiles A y B armados desde la anotacion. Empate: None."""
    from boxtwin.core.identidad_auto import distancia_color, perfil_de

    if not propios or not de_a or not de_b:
        return None
    pa, pb = perfil_de(de_a), perfil_de(de_b)
    a = sum(1 for c in propios if distancia_color(c, pa) < distancia_color(c, pb))
    b = len(propios) - a
    return None if a == b else (A if a > b else B)


def diagnosticar(video: Path, mediciones: Path) -> dict:
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identidad_auto import (
        ConfigIdentidadAuto, _componentes, _pintar, cargar_evidencia, proponer,
    )
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    nombre = video.stem
    medido = json.loads((mediciones / f"{nombre}.c1.json").read_text())
    semillas = tuple(medido["siembra"]["semillas"])
    paths = project_paths(video)
    doc, _ = cargar_doc(paths.annot)
    cache = PoseCache.open(paths.npz)
    ev = cargar_evidencia(mediciones / f"{nombre}.evidencia.json")
    cfg = ConfigIdentidadAuto()
    fps, total = doc.video.fps, doc.video.total_frames

    verdad, cuadros = verdad_de(doc.identity.assignments, cache.frames_of_track)
    prop = proponer(ev, cfg, total, fps, semillas=semillas)
    roles = {t: r.value for t, r in prop.roles.items()}
    nucleo = set(prop.diagnostico.get("nucleo_tracks", [])) | set(semillas)
    votos = prop.diagnostico.get("votos", {})

    # Los mismos componentes que arma particionar sobre el nucleo.
    frames = {t: {f for f, _ in ev[t].colores} for t in nucleo}
    frames = {t: fs for t, fs in frames.items() if fs}
    nodos = sorted(frames)
    ady: dict[int, set[int]] = {t: set() for t in nodos}
    for i, t1 in enumerate(nodos):
        for t2 in nodos[i + 1:]:
            if len(frames[t1] & frames[t2]) >= cfg.min_coexistencia:
                ady[t1].add(t2)
                ady[t2].add(t1)
    comp_de: dict[int, int] = {}
    con_dos: set[int] = set()
    for i, c in enumerate(_componentes(ady, nodos)):
        for t in c:
            comp_de[t] = i
        if len(set(_pintar(c, ady).values())) == 2:
            con_dos.add(i)
    sa, sb = semillas
    comp_a, comp_b = comp_de.get(sa), comp_de.get(sb)

    def mecanismo(t: int) -> str:
        if t not in nucleo:
            return "rescate"
        c = comp_de.get(t)
        if c is not None and c in con_dos:
            if c == comp_a:
                return "ancla"
            if c == comp_b:
                return "semilla_b"
            return "comp_color"
        if t in votos:
            return "voto"
        return "otro"

    mismo_rol, con_no_peleador = aristas_espurias(ady, frames, verdad)

    # Color de oraculo: perfiles desde la anotacion, dejando el track afuera.
    cols = {t: [c for _, c in ev[t].colores] for t in ev}

    def oraculo(t: int) -> str | None:
        if len(cols.get(t, [])) < MIN_GUANTES_ORACULO:
            return None
        de_a = [c for u, r in verdad.items() if r == A and u != t for c in cols.get(u, [])]
        de_b = [c for u, r in verdad.items() if r == B and u != t for c in cols.get(u, [])]
        return voto_oraculo(cols[t], de_a, de_b)

    tracks = []
    for t in sorted(verdad):
        if roles.get(t) not in (A, B):
            continue
        o = oraculo(t)
        tracks.append({
            "track": t, "verdad": verdad[t], "propuesto": roles[t],
            "acierto": roles[t] == verdad[t], "mecanismo": mecanismo(t),
            "componente": comp_de.get(t), "guantes": len(cols.get(t, [])),
            "mixto": bool(cuadros[t][A] and cuadros[t][B]),
            "oraculo": o, "oraculo_acierta": None if o is None else o == verdad[t],
        })
    return {
        "fuente": nombre, "semillas": list(semillas),
        "separacion_de_perfiles": prop.diagnostico.get("separacion_de_perfiles"),
        "componentes_con_dos_lados": len(con_dos),
        "aristas_mismo_rol": mismo_rol, "aristas_con_no_peleador": con_no_peleador,
        "tracks": tracks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--mediciones", default="~/boxtwin/mediciones/c1")
    ap.add_argument("-o", "--salida", default=None, help="carpeta para los JSON (opcional)")
    args = ap.parse_args()
    raiz = Path(args.raiz).expanduser()
    med = Path(args.mediciones).expanduser()
    salida = Path(args.salida).expanduser() if args.salida else None
    if salida:
        salida.mkdir(parents=True, exist_ok=True)

    total_mec: Counter = Counter()
    error_mec: Counter = Counter()
    for f in FUENTES:
        d = diagnosticar(raiz / f, med)
        if salida:
            (salida / f"{d['fuente']}.diagnostico.json").write_text(json.dumps(d, indent=1))
        errores = [t for t in d["tracks"] if not t["acierto"]]
        for t in d["tracks"]:
            total_mec[t["mecanismo"]] += 1
            if not t["acierto"]:
                error_mec[t["mecanismo"]] += 1
        print(f"\n{d['fuente']}  semillas {d['semillas']}  separacion {d['separacion_de_perfiles']}"
              f"  componentes con dos lados {d['componentes_con_dos_lados']}")
        print(f"  aristas entre tracks del mismo rol: {len(d['aristas_mismo_rol'])}"
              f"  {d['aristas_mismo_rol'][:8]}")
        print(f"  aristas con no peleador: {len(d['aristas_con_no_peleador'])}")
        for t in errores:
            print(f"  ERROR track {t['track']:>4}  verdad {t['verdad'][-1]}  mec {t['mecanismo']:<10}"
                  f" comp {t['componente']}  guantes {t['guantes']:>4}  mixto {t['mixto']}"
                  f"  oraculo {t['oraculo'] and t['oraculo'][-1]}")
    print("\nmecanismo     tracks  errores")
    for m in sorted(total_mec):
        print(f"{m:<12} {total_mec[m]:>7} {error_mec[m]:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
