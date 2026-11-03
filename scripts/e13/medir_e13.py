"""
BoxTwin - E13: P18a, P18b y P18c por tracker (3-11-2026).

POR QUE EXISTE
  La regla de E13 esta escrita antes de medir (docs/specs/2026-10-30-mejoras-post-e8/06-e13-tracker.md
  §4). Reusa las cuentas de E8 (medir_trackeo) y de E11 (medir_e11) para que lo unico que cambie
  sea el tracker. Si H13-2 no paso, no lee nada: comparar contra una base que no se reproduce no
  mide el algoritmo de asociacion.

QUE HACE
  - evidencia: por condicion y fuente de gimnasio, la evidencia de guante sobre el cache de esa
    condicion, con guantes-v2.pt y el candado de GPU. La de ~/boxtwin/mediciones/c1 no sirve: es de
    los track_id del cache de A. Una evidencia guardada no se repite.
  - medir: la verdad es la de raiz-v8 (verdad de puente_identidad) y cada condicion se empareja con
    ella como en el puente. P18a y P18b salen de comparar(base, tracker) en cada fuente, sobre el
    denominador comun de ese par, sumado sobre las siete fuentes. Cambios por minuto: A y B juntos.
    P18c: el puente pasa la anotacion a los ids del tracker y medir_e11.medir da C1 por track sin
    corte y por segmento con corte, seis fuentes. Escribe <e13>/resultado.json.

USO
  python scripts/e13/medir_e13.py evidencia --condiciones base ocsort bytetrack
  python scripts/e13/medir_e13.py medir --condiciones ocsort bytetrack
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
for r in ("scripts/mejoras", "scripts/gemelo", "scripts/clasificador", "boxtwin-annotator/scripts"):
    sys.path.insert(0, str(REPO / r))
import comun_e13 as c  # noqa: E402

GUANTES = Path("~/Proyectos/TwinBoxing/boxtwin-guantes/modelos/guantes-v2.pt").expanduser()


def exigir_h13_2() -> None:
    p = c.E13 / "h13-2.json"
    if not p.is_file() or not json.loads(p.read_text())["pasa"]:
        raise SystemExit("H13-2 no paso: E13 se detiene y no lee resultados")


def abrir(condicion: str, fuente: str):
    from boxtwin.core.posecache import PoseCache

    return PoseCache.open(c.cache_de(fuente) if condicion == "base" else c.E13 / condicion / f"{fuente}.pose.npz")


def cmd_evidencia(a) -> int:
    exigir_h13_2()
    from boxtwin.core.annotations import load
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, analizar, guardar_evidencia

    candado = open(c.CANDADO, "a+")
    det = None
    cfg = ConfigIdentidadAuto()
    for cond in a.condiciones:
        for f in c.SEIS:
            ruta = c.E13 / "evidencia" / cond / f"{f}.evidencia.json"
            if ruta.is_file():
                continue
            ruta.parent.mkdir(parents=True, exist_ok=True)
            c.esperar_colas()
            fcntl.flock(candado, fcntl.LOCK_EX)
            try:
                det = det or DetectorGuantes(str(GUANTES), device=None)
                doc, _ = load(c.RAIZ_V8 / c.FUENTES[f] / "annotations" / f"{f}.annot.json")
                ev = analizar(abrir(cond, f), c.video_de(f), det, cfg, alto_imagen=doc.video.height,
                              fps=doc.video.fps, total_frames=doc.video.total_frames,
                              progreso=lambda *_: None)
                tmp = ruta.with_name(ruta.name + ".tmp")
                guardar_evidencia(ev, tmp)
                tmp.rename(ruta)
                c.log(f"evidencia {cond} {f}")
            finally:
                fcntl.flock(candado, fcntl.LOCK_UN)
    return 0


def p18c(cond: str, f: str, doc, cache_orig) -> dict:
    from boxtwin.core.identidad_auto import cargar_evidencia
    from medir_e11 import medir
    from puente_identidad import puente

    cache = abrir(cond, f)
    informe = None
    if cond != "base":
        doc, informe, _ = puente(doc, cache_orig, cache)
    ev = cargar_evidencia(c.E13 / "evidencia" / cond / f"{f}.evidencia.json")
    r = medir(doc, cache, ev, f)
    r["puente"] = informe and {k: informe[k] for k in ("conflictos", "iou_mediana", "tracks_nuevos_usados")}
    return r


def resumen_c1(fuentes: list[dict]) -> dict:
    medidas = [v for v in fuentes if "sin_corte" in v]

    def suma(ruta):
        a, b = ruta
        return {k: sum(v[a][b][k] for v in medidas) for k in ("aciertos", "evaluables")}

    return {"fuentes_con_siembra": len(medidas), "sin_siembra": [v["fuente"] for v in fuentes if "sin_corte" not in v],
            "sin_corte": suma(("sin_corte", "c1_por_track")), "con_corte": suma(("con_corte", "c1_por_segmento"))}


def cmd_medir(a) -> int:
    exigir_h13_2()
    from puente_identidad import _original, emparejar, verdad
    from medir_trackeo import comparar, total

    pares: dict[str, list[dict]] = {t: [] for t in a.condiciones}
    c1: dict[str, list[dict]] = {t: [] for t in ["base", *a.condiciones]}
    for f in c.ORDEN:
        doc, orig = _original(c.RAIZ_V8, f)
        gt = verdad(doc, orig)
        ab = emparejar(gt, orig)[0]
        for t in a.condiciones:
            r = comparar(ab, emparejar(gt, abrir(t, f))[0], doc.video.fps)
            r["fuente"] = f
            pares[t].append(r)
        if f in c.SEIS:
            for t in c1:
                c1[t].append(p18c(t, f, doc, orig))
        c.log(f"medido {f}")

    salida: dict = {"p18": {}, "p18c": {t: {"total": resumen_c1(v), "fuentes": v} for t, v in c1.items()}}
    for t, fuentes in pares.items():
        tot = total(fuentes)
        cpm_b, cpm_t = c.cambios_por_minuto(tot["base"]), c.cambios_por_minuto(tot["y26"])
        salida["p18"][t] = {
            "regla": c.regla_p18a(tot["base"]["tracks_mixtos"], tot["y26"]["tracks_mixtos"], cpm_b, cpm_t),
            "base": {**tot["base"], "cambios_por_minuto": cpm_b}, "tracker": {**tot["y26"], "cambios_por_minuto": cpm_t},
            "denominador": tot["denominador"], "fuentes": fuentes}
    (c.E13 / "resultado.json").write_text(json.dumps(salida, indent=1, default=str))
    print(json.dumps({t: v["regla"] for t, v in salida["p18"].items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("evidencia", cmd_evidencia), ("medir", cmd_medir)):
        p = sub.add_parser(n)
        p.add_argument("--condiciones", nargs="+", required=True)
        p.set_defaults(f=f)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
