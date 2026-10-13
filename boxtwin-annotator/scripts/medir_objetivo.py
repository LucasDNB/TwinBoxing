"""
BoxTwin - Calibracion y medicion de C8: la altura del golpe por la regla geometrica.

POR QUE EXISTE
  C8 (ampliacion 03 §4) decide si la Fight-Card separa cabeza y cuerpo. Esta pre-registrado:
  tau se elige sobre 01-sparring y 02-sparring (166 golpes) y se mide UNA vez sobre
  03-sparring, 04-sparring, sparring-3 y pacquiao_margarito (806 golpes), con exactitud
  balanceada >= 0,75 y recall de cuerpo >= 0,60. La exactitud simple no sirve: con 83% de
  golpes a la cabeza, "todo es cabeza" ya da 0,83 sin saber nada (su balanceada es 0,50).

  Esto es el nivel 1 (regla sola): golpes anotados, identidad anotada y los keypoints de la
  cache de cada proyecto de anotacion, pasados por el MISMO camino que usa el producto
  (`series_por_peleador` y `objetivo_de`). Aisla el error de la regla. El nivel 2 (punta a
  punta, sobre lo que el detector encontro) necesita correr el sistema sobre esas fuentes y no
  lo hace este script.

  Lo que el spec no fijaba y se fija aca antes de correr (doc del experimento, §2):
  - Golpes: todos los anotados de la fuente (fintas y abortados incluidos), como los cuenta
    el spec (166 y 806).
  - Un golpe sin dato (`sin_rival`, `sin_pose`) cuenta como ERROR en la metrica del criterio
    (estricta). La metrica sobre los golpes con dato se reporta al lado, sin umbral.
  - Desempate de tau en la calibracion: mayor exactitud balanceada; si empata, mayor recall de
    cuerpo; si empata, el tau menor.

QUE HACE
  calibrar  la tabla de la grilla de tau sobre las fuentes de calibracion y el tau elegido
  medir     con un tau dado, las metricas sobre las fuentes de medicion, por fuente y juntas
  Escribe JSON en --salida. Lee las anotaciones y las caches, no escribe nada en ellas.

USO
  python scripts/medir_objetivo.py calibrar --salida calibracion.json
  python scripts/medir_objetivo.py medir --tau 0.2 --salida medicion.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FUENTES = {
    "01-sparring": "anotacion-spar-01/annotations/01-sparring.annot.json",
    "02-sparring": "anotacion-spar-02/annotations/02-sparring.annot.json",
    "03-sparring": "anotacion-spar-03/annotations/03-sparring.annot.json",
    "04-sparring": "anotacion-spar-04/annotations/04-sparring.annot.json",
    "sparring-3": "anotacion-sparring-3/annotations/sparring-3-rounds.annot.json",
    "pacquiao_margarito": "anotacion-pacquiao/annotations/pacquiao_margarito.annot.json",
}
CALIBRACION = ("01-sparring", "02-sparring")
MEDICION = ("03-sparring", "04-sparring", "sparring-3", "pacquiao_margarito")
UMBRAL_BALANCEADA = 0.75
UMBRAL_RECALL_CUERPO = 0.60


# -- metricas (puras, con prueba) ---------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - medio), min(1.0, centro + medio))


def metricas(pares: list[tuple[str, str | None]]) -> dict:
    """
    `pares` es (verdad, prediccion) con verdad en {cabeza, cuerpo} y prediccion en
    {cabeza, cuerpo, None}. Estricta: None cuenta como error. Con dato: sin los None.
    """
    def _calc(ps):
        n_cab = sum(1 for v, _ in ps if v == "cabeza")
        n_cue = sum(1 for v, _ in ps if v == "cuerpo")
        ok_cab = sum(1 for v, p in ps if v == "cabeza" and p == "cabeza")
        ok_cue = sum(1 for v, p in ps if v == "cuerpo" and p == "cuerpo")
        r_cab = ok_cab / n_cab if n_cab else None
        r_cue = ok_cue / n_cue if n_cue else None
        bal = (r_cab + r_cue) / 2 if r_cab is not None and r_cue is not None else None
        return {
            "golpes": len(ps), "cabeza": n_cab, "cuerpo": n_cue,
            "recall_cabeza": _r(r_cab), "recall_cuerpo": _r(r_cue),
            "exactitud_balanceada": _r(bal),
            "recall_cuerpo_wilson95": [_r(x) for x in wilson(ok_cue, n_cue)],
            "exactitud_simple": _r(sum(1 for v, p in ps if v == p) / len(ps)) if ps else None,
        }

    confusion = {v: {str(p): 0 for p in ("cabeza", "cuerpo", None)} for v in ("cabeza", "cuerpo")}
    for v, p in pares:
        confusion[v][str(p)] += 1
    con_dato = [(v, p) for v, p in pares if p is not None]
    return {
        "estricta": _calc(pares),
        "con_dato": _calc(con_dato),
        "sin_dato": len(pares) - len(con_dato),
        "confusion": confusion,
        "linea_de_base_todo_cabeza": _calc([(v, "cabeza") for v, _ in pares])["exactitud_balanceada"],
    }


def _r(x):
    return None if x is None else round(x, 4)


def elegir_tau(tabla: dict[float, dict]) -> float:
    """Mayor balanceada estricta; desempata mayor recall de cuerpo; despues el tau menor."""
    def clave(t):
        e = tabla[t]["estricta"]
        return (e["exactitud_balanceada"] or 0.0, e["recall_cuerpo"] or 0.0, -t)
    return max(tabla, key=clave)


def pasa(m: dict) -> bool:
    e = m["estricta"]
    return ((e["exactitud_balanceada"] or 0) >= UMBRAL_BALANCEADA
            and (e["recall_cuerpo"] or 0) >= UMBRAL_RECALL_CUERPO)


# -- datos (leen las anotaciones y las caches) --------------------------------------------


def series_de_fuente(nombre: str):
    """kp, sc, valid por peleador con la identidad ANOTADA, y los golpes anotados."""
    from boxtwin.core.annotations import load
    from boxtwin.core.posecache import PoseCache
    from boxtwin.mvp.orquesta import series_por_peleador

    ruta = RAIZ / FUENTES[nombre]
    crudo = json.loads(ruta.read_text())
    doc, _ = load(ruta)
    npz = Path(crudo["pose"]["npz_path"])
    if not npz.is_file():          # el proyecto se pudo haber movido: la cache esta al lado
        npz = ruta.parents[1] / "cache" / npz.name
    cache = PoseCache.open(npz)
    kp, sc, valid, _ = series_por_peleador(doc, cache)
    return kp, sc, valid, crudo["events"]


def pares_de_fuente(nombre: str, taus) -> dict[float, list[tuple[str, str | None]]]:
    from boxtwin.mvp.objetivo import objetivo_de

    kp, sc, valid, eventos = series_de_fuente(nombre)
    salida = {t: [] for t in taus}
    for e in eventos:
        verdad = {"head": "cabeza", "body": "cuerpo"}.get(e.get("target"))
        if verdad is None:
            continue
        p = 0 if e["fighter"] == "fighter_A" else 1
        for t in taus:
            r = objetivo_de(kp, sc, valid, p, e["side"], e["start_frame"], e["end_frame"], tau=t)
            salida[t].append((verdad, r.objetivo))
    return salida


def main() -> int:
    from boxtwin.mvp.objetivo import GRILLA_TAU, REGLA

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ca = sub.add_parser("calibrar")
    ca.add_argument("--salida", type=Path, required=True)
    me = sub.add_parser("medir")
    me.add_argument("--tau", type=float, required=True)
    me.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "calibrar":
        pares = {t: [] for t in GRILLA_TAU}
        for f in CALIBRACION:
            for t, ps in pares_de_fuente(f, GRILLA_TAU).items():
                pares[t].extend(ps)
        tabla = {t: metricas(ps) for t, ps in pares.items()}
        elegido = elegir_tau(tabla)
        out = {"regla": REGLA, "fuentes": list(CALIBRACION),
               "grilla": {str(t): tabla[t] for t in GRILLA_TAU}, "tau_elegido": elegido}
    else:
        juntas: list = []
        por_fuente = {}
        for f in MEDICION:
            ps = pares_de_fuente(f, [a.tau])[a.tau]
            por_fuente[f] = metricas(ps)
            juntas.extend(ps)
        total = metricas(juntas)
        out = {"regla": REGLA, "tau": a.tau, "fuentes": list(MEDICION), "total": total,
               "por_fuente": por_fuente, "C8_pasa": pasa(total),
               "umbral": {"exactitud_balanceada": UMBRAL_BALANCEADA,
                          "recall_cuerpo": UMBRAL_RECALL_CUERPO}}
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("grilla", "por_fuente")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
