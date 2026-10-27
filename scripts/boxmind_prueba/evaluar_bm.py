"""
BoxTwin - Metricas y veredictos de la prueba BM, contra su pre-registro.

POR QUE EXISTE
  Los numeros de BM se leen con las mismas reglas y las mismas funciones que E4 (misma
  diferencia minima, mismos folds, mismo freno por dispersion entre semillas), importadas de
  scripts/clasificador_geo/evaluar.py, para que la comparacion contra E4 no dependa de una
  segunda implementacion. Lo propio de BM (la validez de la representacion en BoxingWeb, la
  comparacion de datos externos contra arquitectura y el gatillo de la propuesta de dataset
  propio) sale de funciones con prueba, no de mirar una tabla.

QUE HACE
  Lee las corridas de BM en <trabajo_bm>/corridas y las de E4 en el trabajo de E4. Calcula:
  V0 (BM-0b, F1 macro de tecnica y objetivo sobre el test de BoxingWeb), P7 (BM-1b contra E4),
  P8 (BM-2 contra BM-1b), P9 (BM-2 contra E4), P10 (altura de BM-2 contra la regla de C8),
  sub-hipotesis 1, la curva de aprendizaje sin umbral y el gatillo. Escribe
  <trabajo_bm>/resultados/{<exp>.json, veredictos.json, resumen.md}.

USO (en twinboxing_env)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/evaluar_bm.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import SEMILLAS, TRABAJO_BM, fuera_del_repo, geo  # noqa: E402

sys.path.insert(0, str(Path(geo.__file__).resolve().parent))
import evaluar as ev4  # noqa: E402  (scripts/clasificador_geo/evaluar.py)

# Pre-registro, seccion 5.
REFERENCIA_BOXMIND = {"tecnica": 0.692, "objetivo": 0.747}   # Tabla S3, fila TCN (pose-only)
TOLERANCIA_V0 = 0.05
CURVA = ("BM-1b", "BM-2-25", "BM-2-50", "BM-2")


def f1_macro(real: np.ndarray, pred: np.ndarray) -> float:
    """F1 macro sobre las clases presentes en real o en pred (sklearn f1_score average='macro')."""
    clases = sorted(set(real.tolist()) | set(pred.tolist()))
    f1s = []
    for c in clases:
        tp = int(((pred == c) & (real == c)).sum())
        fp = int(((pred == c) & (real != c)).sum())
        fn = int(((pred != c) & (real == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def v0(trabajo: Path) -> dict:
    """BM-0b: F1 macro por semilla sobre el test de BoxingWeb y su mediana."""
    por_sem = {}
    for s in SEMILLAS:
        p = trabajo / "corridas" / "BM-0b" / "boxingweb" / f"s{s}" / "pred.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            por_sem[f"s{s}"] = {"tecnica": f1_macro(z["y_tipo"], z["p_tipo"].argmax(1)),
                                "objetivo": f1_macro(z["y_altura"], z["p_altura"].argmax(1)),
                                "golpes": int(len(z["y_tipo"]))}
    if not por_sem:
        return {"veredicto": "no realizado"}
    med = {k: statistics.median(v[k] for v in por_sem.values()) for k in ("tecnica", "objetivo")}
    piso = REFERENCIA_BOXMIND["tecnica"] - TOLERANCIA_V0
    return {"por_semilla": por_sem, "mediana": med, "referencia_boxmind": REFERENCIA_BOXMIND,
            "piso_tecnica": piso, "veredicto": "valida" if med["tecnica"] >= piso - ev4.EPS else "no valida"}


def desvio(r: dict) -> float | None:
    return r.get("dispersion_entre_semillas", {}).get("familia_exactitud", {}).get("desvio")


def comparar(nuevo: dict, base: dict, que: str) -> dict:
    if not nuevo or "ensamble_total" not in nuevo or not base or "ensamble_total" not in base:
        return {"compara": que, "veredicto": "no realizado"}
    ds = [d for d in (desvio(nuevo), desvio(base)) if d is not None]
    return {"compara": que, **ev4.veredicto_mejora(
        nuevo["ensamble_total"]["familia"]["exactitud"], base["ensamble_total"]["familia"]["exactitud"],
        ev4.familia_por_fold(nuevo), ev4.familia_por_fold(base), ev4.FOLDS_MIN, ds or None),
        "recall_hook": {"nuevo": nuevo["ensamble_total"]["familia"]["recall"]["hook"],
                        "base": base["ensamble_total"]["familia"]["recall"]["hook"]}}


def altura_contra_regla(r: dict, regla: dict | None) -> dict:
    if not regla or not r or "ensamble_fuentes_c8" not in r:
        return {"veredicto": "no realizado"}
    c8 = r["ensamble_fuentes_c8"]["altura"]
    folds = {f: r["por_fold"][f]["altura"]["balanceada"] for f in geo.FUENTES_C8 if f in r["por_fold"]}
    folds_r = {f: regla["por_fuente"][f]["balanceada"] for f in geo.FUENTES_C8}
    out = ev4.veredicto_mejora(c8["balanceada"], regla["total"]["balanceada"], folds, folds_r, ev4.FOLDS_MIN_ALTURA)
    out["C8b"] = {"balanceada": c8["balanceada"], "recall_cuerpo": c8["recall_cuerpo"],
                  "cumple": c8["balanceada"] >= ev4.C8B["balanceada"] and c8["recall_cuerpo"] >= ev4.C8B["recall_cuerpo"]}
    return {"compara": "BM-2 (fuentes de C8) contra la regla de C8", **out}


def gatillo(v0_: dict, p8: dict, ocho_bm2: float | None) -> dict:
    """
    Pre-registro, seccion 6: la propuesta de dataset propio se presenta como respaldada si la
    representacion es valida (V0) y sumar datos externos del mismo esquema mejora a la misma
    arquitectura (P8), o si BM-2 alcanza la sub-hipotesis 1. Si no, se presenta como no
    respaldada por esta prueba.
    """
    if v0_.get("veredicto") != "valida":
        return {"resultado": "no respaldada", "motivo": "V0 no valida: la representacion no reproduce a BoxMind en su dominio"}
    if ocho_bm2 is not None and ocho_bm2 >= ev4.UMBRAL_OCHO - ev4.EPS:
        return {"resultado": "respaldada", "motivo": "BM-2 alcanza 0,80 en ocho clases"}
    v = p8.get("veredicto")
    if v == "mejora":
        return {"resultado": "respaldada", "motivo": "P8: los datos externos mejoran a la misma arquitectura"}
    if v == "no concluyente":
        return {"resultado": "no concluyente", "motivo": "P8 no concluyente por dispersion entre semillas"}
    return {"resultado": "no respaldada", "motivo": f"P8 {v}"}


def curva(res: dict, n_bw: dict) -> list[dict]:
    filas = []
    for exp in CURVA:
        r = res.get(exp)
        if not r or "ensamble_total" not in r:
            continue
        filas.append({"exp": exp, "golpes_boxingweb": n_bw.get(exp, 0),
                      "familia": r["ensamble_total"]["familia"]["exactitud"],
                      "ocho_clases": r["ensamble_total"]["ocho_clases"]["exactitud"],
                      "desvio_familia": desvio(r)})
    return filas


def main(argv=None) -> int:
    from datos import leer_csv
    golpes = leer_csv(geo.TRABAJO / "datos" / "golpes.csv")
    salida = fuera_del_repo(TRABAJO_BM / "resultados")
    salida.mkdir(parents=True, exist_ok=True)
    res, n_bw = {}, {}
    for exp in ("BM-1", "BM-1b", "BM-2", "BM-2-25", "BM-2-50"):
        lc = ev4.leer_corridas(exp, golpes, TRABAJO_BM, SEMILLAS)
        if lc["preds"]:
            res[exp] = ev4.evaluar_probs(lc["preds"], golpes, SEMILLAS)
            res[exp]["faltan"] = lc["faltan"]
            cfg = next((TRABAJO_BM / "corridas" / exp).glob("*/s*/config.json"), None)
            n_bw[exp] = json.loads(cfg.read_text())["entrenamiento_boxingweb"] if cfg else None
    e4 = ev4.evaluar_probs(ev4.leer_corridas("E4", golpes, geo.TRABAJO, geo.SEMILLAS)["preds"], golpes, geo.SEMILLAS)
    regla_p = geo.TRABAJO / "resultados" / "regla_c8.json"
    regla = json.loads(regla_p.read_text()) if regla_p.exists() else None
    for k, r in {**res, "E4": e4}.items():
        (salida / f"{k}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")

    ver = {"V0": v0(TRABAJO_BM),
           "P7": comparar(res.get("BM-1b"), e4, "BM-1b contra E4 (arquitectura, mismos datos)"),
           "P8": comparar(res.get("BM-2"), res.get("BM-1b"), "BM-2 contra BM-1b (datos externos, misma arquitectura)"),
           "P9": comparar(res.get("BM-2"), e4, "BM-2 contra E4 (lo mejor de BM contra lo que corre)"),
           "P10": altura_contra_regla(res.get("BM-2"), regla)}
    ochos = {k: r["ensamble_total"]["ocho_clases"]["exactitud"] for k, r in res.items() if "ensamble_total" in r}
    ver["subhipotesis_1"] = ev4.lectura_ocho(ochos)
    ver["curva"] = curva(res, n_bw)
    ver["BM-1_sin_ver_boxtwin"] = ({"familia": res["BM-1"]["ensamble_total"]["familia"]["exactitud"],
                                    "altura_balanceada": res["BM-1"]["ensamble_total"]["altura"]["balanceada"]}
                                   if "BM-1" in res and "ensamble_total" in res["BM-1"] else {"veredicto": "no realizado"})
    ver["D9_sospechosos"] = [k for k, r in res.items() if "ensamble_total" in r and ev4.sospechoso(r["ensamble_total"])]
    ver["gatillo_dataset_propio"] = gatillo(ver["V0"], ver["P8"], ochos.get("BM-2"))
    ver["exploratorio"] = ver["V0"].get("veredicto") != "valida"
    ver["E4_recalculado_familia"] = e4.get("ensamble_total", {}).get("familia", {}).get("exactitud")
    (salida / "veredictos.json").write_text(json.dumps(ver, indent=1, default=float) + "\n")
    (salida / "resumen.md").write_text(resumen_md(res, e4, ver))
    print(json.dumps({k: ver[k].get("veredicto", ver[k]) if isinstance(ver[k], dict) else ver[k]
                      for k in ("V0", "P7", "P8", "P9", "P10", "gatillo_dataset_propio")}, indent=1, default=float))
    return 0


def resumen_md(res: dict, e4: dict, ver: dict) -> str:
    f = ev4._f
    filas = ["# Resumen BM (generado por evaluar_bm.py)", "",
             "| Exp | Familia | Recall hook | Altura bal. C8 | Ocho clases | Desvío familia |",
             "|---|---|---|---|---|---|"]
    for k, r in [("E4", e4), *res.items()]:
        if "ensamble_total" not in r:
            continue
        t = r["ensamble_total"]
        c8 = r.get("ensamble_fuentes_c8", {}).get("altura", {}).get("balanceada")
        filas.append(f"| {k} | {f(t['familia']['exactitud'])} | {f(t['familia']['recall']['hook']['recall'])} | "
                     f"{f(c8)} | {f(t['ocho_clases']['exactitud'])} | {f(desvio(r))} |")
    filas += ["", f"V0: {ver['V0'].get('veredicto')} · P7: {ver['P7'].get('veredicto')} · "
              f"P8: {ver['P8'].get('veredicto')} · P9: {ver['P9'].get('veredicto')} · "
              f"P10: {ver['P10'].get('veredicto')} · sub-hipótesis 1: {ver['subhipotesis_1'].get('lectura')} · "
              f"propuesta de dataset propio: {ver['gatillo_dataset_propio']['resultado']}", ""]
    if ver["exploratorio"]:
        filas.append("V0 no es válida: todo lo de arriba se informa como exploratorio.")
    return "\n".join(filas) + "\n"


if __name__ == "__main__":
    sys.exit(main())
