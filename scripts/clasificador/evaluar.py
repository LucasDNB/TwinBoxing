"""
BoxTwin - Metricas de la campana del clasificador, dejando una fuente afuera.

POR QUE EXISTE
  Las salidas de mmaction son un pickle por corrida con las probabilidades de cada muestra.
  El numero que va al Cap 7 sale de juntarlas como dice el pre-registro (spec, seccion 6) y
  de ningun otro modo: ensamble de las tres semillas por fold (promedio de probabilidades),
  folds concatenados, familia y altura sumando las probabilidades de las clases que las
  comparten. La dispersion entre semillas se reporta al lado porque el ruido entre semillas
  en este problema ya se midio del orden de 0,10 de F1.

  El orden de las muestras en el pickle de salida no viene con nombre. Se verifica contra
  el fold: la etiqueta real de cada posicion tiene que coincidir con la del pickle de
  prueba. Si no coincide, se frena: un desorden ahi da numeros plausibles y falsos.

QUE HACE
  Para cada experimento: metricas por fold, por semilla y del ensamble; total y sobre las
  cuatro fuentes de C8; lineas de base (siempre straight, siempre cabeza). Escribe
  <trabajo>/resultados/<exp>.json y <trabajo>/resultados/resumen.md.

USO (en boxtwin_mmaction: los pickles traen tensores de torch)
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador/evaluar.py --exp E1 E2 E3
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (ALTURAS, EXPERIMENTOS, FUENTES, FUENTES_C8, SEMILLAS, TIPOS, TRABAJO,  # noqa: E402
                   balanceada, exactitud, partes_de_clase, recall_por_clase, redondear, wilson)


def _arr(v) -> np.ndarray:
    if isinstance(v, dict) and "item" in v:
        v = v["item"]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def leer_pred(pkl: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Probabilidades (n, C) y etiquetas reales (n,) si el dump las trae."""
    with pkl.open("rb") as fh:
        items = pickle.load(fh)
    scores, gts = [], []
    for d in items:
        for k in ("pred_score", "pred_scores"):
            if k in d:
                scores.append(_arr(d[k]).reshape(-1))
                break
        else:
            raise SystemExit(f"{pkl}: el dump no trae pred_score (claves: {sorted(d)})")
        g = d.get("gt_label", d.get("gt_labels"))
        gts.append(None if g is None else int(_arr(g).reshape(-1)[0]))
    gt = None if any(g is None for g in gts) else np.array(gts)
    return np.vstack(scores), gt


def esperado(fold_pkl: Path) -> tuple[list[str], np.ndarray]:
    """Nombres y etiquetas de prueba en el orden en que PoseDataset los filtra."""
    with fold_pkl.open("rb") as fh:
        d = pickle.load(fh)
    test = set(d["split"]["test"])
    ms = [m for m in d["annotations"] if m["frame_dir"] in test]
    return [m["frame_dir"] for m in ms], np.array([m["label"] for m in ms])


def agrupar(P: np.ndarray, clases: list[str], campo: str) -> tuple[np.ndarray, list[str]]:
    """Suma las probabilidades de las clases que comparten tipo, eje o altura."""
    valores = [partes_de_clase(c)[campo] for c in clases]
    grupos = [v for v in dict.fromkeys(valores) if v is not None]
    if not grupos or None in valores:
        return None, []
    M = np.zeros((P.shape[1], len(grupos)))
    for i, v in enumerate(valores):
        M[i, grupos.index(v)] = 1
    return P @ M, grupos


def metricas(P: np.ndarray, y: np.ndarray, clases: list[str]) -> dict:
    n = len(y)
    pred = P.argmax(1)
    out = {"golpes": int(n), "top1": exactitud(list(y), list(pred))}
    real_n = [clases[i] for i in y]
    for campo, orden in (("tipo", TIPOS), ("altura", ALTURAS), ("eje", None)):
        G, grupos = agrupar(P, clases, campo)
        if G is None:
            continue
        r = [partes_de_clase(c)[campo] for c in real_n]
        p = [grupos[i] for i in G.argmax(1)]
        orden = orden or grupos
        bloque = {"exactitud": exactitud(r, p), "balanceada": balanceada(r, p, orden),
                  "recall": recall_por_clase(r, p, orden)}
        if campo == "altura":
            bloque["recall_cuerpo"] = bloque["recall"]["body"]["recall"]
            bloque["recall_cuerpo_wilson95"] = bloque["recall"]["body"]["wilson95"]
        aciertos = sum(1 for a, b in zip(r, p) if a == b)
        bloque["wilson95"] = wilson(aciertos, n)
        out["familia" if campo == "tipo" else campo] = bloque
    return out


def lineas_de_base(y: np.ndarray, clases: list[str]) -> dict:
    tipos = [partes_de_clase(clases[i])["tipo"] for i in y]
    base = {"familia_siempre_straight": exactitud(tipos, ["straight"] * len(tipos)),
            "familia_balanceada_azar": 1 / 3}
    if partes_de_clase(clases[0])["altura"]:
        base["altura_balanceada_siempre_cabeza"] = 0.5
    return base


def evaluar(exp: str, verificar_orden: bool = True) -> dict:
    manifest = json.loads((TRABAJO / "folds" / exp / "manifest.json").read_text())
    clases = manifest["clases"]
    por_semilla = {s: {"P": [], "y": [], "fuente": []} for s in SEMILLAS}
    ens = {"P": [], "y": [], "fuente": []}
    faltan, folds = [], {}
    for fuente in FUENTES:
        _, y = esperado(TRABAJO / "folds" / exp / f"fold_{fuente}.pkl")
        probs = []
        for s in SEMILLAS:
            pkl = TRABAJO / "corridas" / exp / fuente / f"s{s}" / "pred.pkl"
            if not pkl.exists():
                faltan.append(f"{fuente}/s{s}")
                continue
            P, gt = leer_pred(pkl)
            if P.shape[0] != len(y):
                raise SystemExit(f"{pkl}: {P.shape[0]} predicciones para {len(y)} muestras de prueba")
            if verificar_orden:
                if gt is None:
                    raise SystemExit(f"{pkl}: el dump no trae gt_label y no se puede verificar el orden")
                if not np.array_equal(gt, y):
                    raise SystemExit(f"{pkl}: las etiquetas del dump no siguen el orden del fold")
            probs.append(P)
            for k, v in (("P", P), ("y", y), ("fuente", [fuente] * len(y))):
                por_semilla[s][k].append(v)
        if probs:
            Pm = np.mean(probs, axis=0)
            folds[fuente] = {"semillas": len(probs), **metricas(Pm, y, clases)}
            ens["P"].append(Pm); ens["y"].append(y); ens["fuente"].append([fuente] * len(y))

    def juntar(b):
        return np.vstack(b["P"]), np.concatenate(b["y"]), [f for fs in b["fuente"] for f in fs]

    salida = {"exp": exp, "que": manifest["que"], "clases": clases, "faltan": faltan,
              "completo": not faltan, "por_fold": folds}
    if ens["P"]:
        P, y, fu = juntar(ens)
        salida["ensamble_total"] = metricas(P, y, clases)
        sel = np.array([f in FUENTES_C8 for f in fu])
        if sel.any():
            salida["ensamble_fuentes_c8"] = metricas(P[sel], y[sel], clases)
        salida["lineas_de_base"] = lineas_de_base(y, clases)
    semillas = {}
    for s, b in por_semilla.items():
        if len(b["P"]) == len(FUENTES):
            P, y, _ = juntar(b)
            semillas[f"s{s}"] = metricas(P, y, clases)
    salida["por_semilla_total"] = semillas
    if len(semillas) >= 2:
        disp = {}
        for clave in ("top1",):
            vals = [m[clave] for m in semillas.values()]
            disp[clave] = {"media": statistics.mean(vals), "desvio": statistics.stdev(vals)}
        for bloque, clave in (("familia", "exactitud"), ("altura", "balanceada"), ("eje", "exactitud")):
            vals = [m[bloque][clave] for m in semillas.values() if bloque in m]
            if len(vals) >= 2:
                disp[f"{bloque}_{clave}"] = {"media": statistics.mean(vals), "desvio": statistics.stdev(vals)}
        salida["dispersion_entre_semillas"] = disp
    return redondear(salida)


def resumen_md(resultados: list[dict]) -> str:
    filas = ["| Exp | Completo | Familia (ensamble) | Familia balanceada | Altura balanceada | Recall cuerpo | Altura balanceada en fuentes C8 | Eje |",
             "|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        t = r.get("ensamble_total", {})
        c8 = r.get("ensamble_fuentes_c8", {})
        f = lambda m, b, k: (m.get(b, {}) or {}).get(k, "")  # noqa: E731
        filas.append(f"| {r['exp']} | {'sí' if r['completo'] else 'no, faltan ' + str(len(r['faltan']))} | "
                     f"{f(t, 'familia', 'exactitud')} | {f(t, 'familia', 'balanceada')} | "
                     f"{f(t, 'altura', 'balanceada')} | {f(t, 'altura', 'recall_cuerpo')} | "
                     f"{f(c8, 'altura', 'balanceada')} | {f(t, 'eje', 'exactitud')} |")
    return "\n".join(filas) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", nargs="+", default=list(EXPERIMENTOS))
    args = ap.parse_args(argv)
    destino = TRABAJO / "resultados"
    destino.mkdir(parents=True, exist_ok=True)
    resultados = []
    for exp in args.exp:
        r = evaluar(exp)
        (destino / f"{exp}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
        resultados.append(r)
    md = resumen_md(resultados)
    (destino / "resumen.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
