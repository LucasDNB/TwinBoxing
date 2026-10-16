"""
BoxTwin - Metricas y veredictos de E4, E4a, E5 y la fusion F, contra el pre-registro.

POR QUE EXISTE
  Los numeros de E4 a E5 se leen contra reglas escritas antes de medir (01-preregistro.md,
  seccion 3): una diferencia minima de 0,03 y una cantidad minima de folds ganados, con la
  dispersion entre semillas como freno. Si el veredicto se escribe a mano mirando una tabla,
  la regla se termina interpretando despues de ver el numero. Aca el veredicto sale de una
  funcion con prueba, y el documento lo copia.

QUE HACE
  Junta los pred.npz de cada experimento. Ensamble de las cinco semillas por fold (promedio de
  probabilidades), folds concatenados y por semilla. Metricas con scripts/clasificador/comun.py:
  familia (exactitud, balanceada, recall por tipo con Wilson), altura (balanceada, recall de
  cuerpo con Wilson), ocho clases (tipo x altura, recto partido por el rol anotado), por fold,
  total y sobre las cuatro fuentes de C8. Fusion F: promedio de E4 con E3 (pred.pkl del 15-10,
  probabilidades sumadas por tipo y por altura). Veredictos de P5 (familia contra E1, altura
  contra la regla de C8), P6 (E5 contra E4) y sub-hipotesis 1. Escribe
  <trabajo>/resultados/{E4,E4a,E5,F}.json, veredictos.json y resumen.md.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/evaluar.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import (ALTURAS, FUENTES, FUENTES_C8, REPO, SEMILLAS, TIPOS, TRABAJO, TRABAJO_E1E3,  # noqa: E402
                 comun)

DIF_MIN = 0.03          # pre-registro, seccion 3, puntos 1 a 3
FOLDS_MIN = 5           # de 7
FOLDS_MIN_ALTURA = 3    # de 4
DESVIO_MAX = 0.03       # punto 6
UMBRAL_OCHO = 0.80      # sub-hipotesis 1
C8B = {"balanceada": 0.75, "recall_cuerpo": 0.60}
SOSPECHA = {"familia": 0.90, "ocho": 0.80}     # D9
EPS = 1e-9
DATOS_15_10 = REPO / "docs" / "experiments" / "datos"


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------


def clases_ocho(tipos: list[str], alturas: list[str], roles: list[str]) -> list[str]:
    return [comun.clase_ocho(f"{t}-{a}", r) for t, a, r in zip(tipos, alturas, roles)]


def metricas(pt: np.ndarray, pa: np.ndarray, golpes: list[dict]) -> dict:
    """pt (n, 3), pa (n, 2) y los golpes en el mismo orden."""
    n = len(golpes)
    rt, ra, rol = [g["tipo"] for g in golpes], [g["altura"] for g in golpes], [g["rol"] for g in golpes]
    qt = [TIPOS[i] for i in pt.argmax(1)]
    qa = [ALTURAS[i] for i in pa.argmax(1)]
    out = {"golpes": n}
    ac = sum(a == b for a, b in zip(rt, qt))
    out["familia"] = {"exactitud": comun.exactitud(rt, qt), "wilson95": comun.wilson(ac, n),
                      "balanceada": comun.balanceada(rt, qt, TIPOS), "recall": comun.recall_por_clase(rt, qt, TIPOS)}
    ac = sum(a == b for a, b in zip(ra, qa))
    rec = comun.recall_por_clase(ra, qa, ALTURAS)
    out["altura"] = {"exactitud": comun.exactitud(ra, qa), "wilson95": comun.wilson(ac, n),
                     "balanceada": comun.balanceada(ra, qa, ALTURAS), "recall": rec,
                     "recall_cuerpo": rec["body"]["recall"], "recall_cuerpo_wilson95": rec["body"]["wilson95"]}
    r8, q8 = clases_ocho(rt, ra, rol), clases_ocho(qt, qa, rol)
    ac = sum(a == b for a, b in zip(r8, q8))
    out["ocho_clases"] = {"exactitud": comun.exactitud(r8, q8), "wilson95": comun.wilson(ac, n),
                          "balanceada": comun.balanceada(r8, q8, comun.CLASES_OCHO),
                          "recall": comun.recall_por_clase(r8, q8, comun.CLASES_OCHO)}
    return out


# ---------------------------------------------------------------------------
# Veredictos (pre-registro, seccion 3)
# ---------------------------------------------------------------------------


def veredicto_mejora(total_nuevo: float, total_base: float, folds_nuevo: dict, folds_base: dict,
                     folds_min: int = FOLDS_MIN, desvios: list[float] | None = None) -> dict:
    """
    El nuevo mejora a la base si supera el total por DIF_MIN o mas y gana en al menos
    folds_min folds. Si algun desvio entre semillas de la familia pasa de DESVIO_MAX, una
    mejora se informa como no concluyente (punto 6).
    """
    comunes = [f for f in folds_nuevo if f in folds_base]
    ganados = [f for f in comunes if folds_nuevo[f] > folds_base[f] + EPS]
    dif = total_nuevo - total_base
    cumple = dif >= DIF_MIN - EPS and len(ganados) >= folds_min
    desvio_max = max(desvios) if desvios else None
    if cumple and desvio_max is not None and desvio_max > DESVIO_MAX + EPS:
        v = "no concluyente"
    else:
        v = "mejora" if cumple else "no demostrada"
    return {"veredicto": v, "diferencia": dif, "folds_ganados": len(ganados), "folds": len(comunes),
            "folds_min": folds_min, "ganados": ganados, "desvio_familia_max": desvio_max}


def lectura_ocho(exactitudes: dict[str, float]) -> dict:
    """Sub-hipotesis 1, lectura a/b/c del 14-10 (d359247)."""
    if not exactitudes:
        return {"lectura": "no realizado"}
    debajo = all(v < UMBRAL_OCHO for v in exactitudes.values())
    texto = ("a: por debajo de 0,80 en todas; la parte de golpes no alcanza el umbral" if debajo else
             "b: alguna en 0,80 o mas, y eso no la confirma (faltan defensas, finta y fondo; segmentos ya anotados)")
    return {"lectura": "a" if debajo else "b", "texto": texto, "exactitudes": exactitudes,
            "c": "jab y cross salen del rol anotado: supone un detector que acierta el brazo siempre"}


def sospechoso(m: dict) -> bool:
    """D9: familia >= 0,90 u ocho clases >= 0,80 obliga a buscar fuga antes de escribir."""
    return (m["familia"]["exactitud"] >= SOSPECHA["familia"]) or (m["ocho_clases"]["exactitud"] >= SOSPECHA["ocho"])


# ---------------------------------------------------------------------------
# Lectura de corridas
# ---------------------------------------------------------------------------


def leer_corridas(exp: str, golpes: list[dict], trabajo: Path = None, semillas=SEMILLAS) -> dict:
    """
    {fuente: {semilla: (pt, pa)}} en el orden de los golpes de esa fuente en golpes.csv.
    Frena si un pred.npz no tiene exactamente los golpes de su fuente.
    """
    trabajo = trabajo or TRABAJO
    salida, faltan = {}, []
    for fuente in FUENTES:
        esperado = [g["frame_dir"] for g in golpes if g["fuente"] == fuente]
        for s in semillas:
            p = trabajo / "corridas" / exp / fuente / f"s{s}" / "pred.npz"
            if not p.exists():
                faltan.append(f"{fuente}/s{s}")
                continue
            with np.load(p) as z:
                fd, pt, pa = list(z["frame_dir"]), z["p_tipo"], z["p_altura"]
            if sorted(fd) != sorted(esperado):
                raise SystemExit(f"{p}: los golpes no son los de la fuente {fuente}")
            pos = {n: i for i, n in enumerate(fd)}
            orden = [pos[n] for n in esperado]
            salida.setdefault(fuente, {})[s] = (pt[orden], pa[orden])
    return {"preds": salida, "faltan": faltan}


def evaluar_probs(probs: dict, golpes: list[dict], semillas=SEMILLAS) -> dict:
    """probs {fuente: {semilla: (pt, pa)}} -> metricas por fold, total, C8, por semilla y dispersion."""
    por_fuente = {f: [g for g in golpes if g["fuente"] == f] for f in FUENTES}
    folds, ens = {}, {}
    for f, ps in probs.items():
        pt = np.mean([v[0] for v in ps.values()], 0)
        pa = np.mean([v[1] for v in ps.values()], 0)
        ens[f] = (pt, pa)
        folds[f] = {"semillas": len(ps), **metricas(pt, pa, por_fuente[f])}
    out = {"por_fold": folds}
    if not ens:
        return out
    orden = [f for f in FUENTES if f in ens]
    junta = lambda fs, d: (np.vstack([d[f][0] for f in fs]), np.vstack([d[f][1] for f in fs]),  # noqa: E731
                           [g for f in fs for g in por_fuente[f]])
    out["ensamble_total"] = metricas(*junta(orden, ens))
    c8 = [f for f in orden if f in FUENTES_C8]
    if c8:
        out["ensamble_fuentes_c8"] = metricas(*junta(c8, ens))
    por_sem = {}
    for s in semillas:
        if all(s in probs[f] for f in orden):
            d = {f: probs[f][s] for f in orden}
            por_sem[f"s{s}"] = metricas(*junta(orden, d))
    out["por_semilla_total"] = {k: {b: {"exactitud": v[b]["exactitud"], "balanceada": v[b]["balanceada"]}
                                    for b in ("familia", "altura", "ocho_clases")} for k, v in por_sem.items()}
    if len(por_sem) >= 2:
        disp = {}
        for bloque, clave in (("familia", "exactitud"), ("altura", "balanceada"), ("ocho_clases", "exactitud")):
            vals = [m[bloque][clave] for m in por_sem.values()]
            disp[f"{bloque}_{clave}"] = {"media": statistics.mean(vals), "desvio": statistics.stdev(vals)}
        out["dispersion_entre_semillas"] = disp
    return out


# ---------------------------------------------------------------------------
# Fusion F con E3
# ---------------------------------------------------------------------------


def _evaluar_e1e3():
    """El evaluar.py de la campana anterior, cargado por ruta para no taparse con este."""
    spec = importlib.util.spec_from_file_location("evaluar_e1e3", REPO / "scripts" / "clasificador" / "evaluar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def agrupar_e3(P: np.ndarray, clases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades de las 6 clases tipo-altura de E3 sumadas por tipo y por altura."""
    pt = np.zeros((len(P), len(TIPOS)))
    pa = np.zeros((len(P), len(ALTURAS)))
    for j, c in enumerate(clases):
        p = comun.partes_de_clase(c)
        pt[:, TIPOS.index(p["tipo"])] += P[:, j]
        pa[:, ALTURAS.index(p["altura"])] += P[:, j]
    return pt, pa


def probs_e3(golpes: list[dict]) -> dict | None:
    """{fuente: (pt, pa)} de E3 s0 en el orden de golpes.csv, o None si faltan pickles (D4)."""
    base = TRABAJO_E1E3
    man = base / "folds" / "E3" / "manifest.json"
    if not man.exists():
        return None
    clases = json.loads(man.read_text())["clases"]
    ev = _evaluar_e1e3()
    salida = {}
    for f in FUENTES:
        pkl = base / "corridas" / "E3" / f / "s0" / "pred.pkl"
        if not pkl.exists():
            return None
        nombres, y = ev.esperado(base / "folds" / "E3" / f"fold_{f}.pkl")
        P, gt = ev.leer_pred(pkl)
        if gt is None or not np.array_equal(gt, y) or len(P) != len(nombres):
            raise SystemExit(f"{pkl}: el orden no coincide con el fold de E3")
        pos = {n: i for i, n in enumerate(nombres)}
        orden = [pos[g["frame_dir"]] for g in golpes if g["fuente"] == f]
        salida[f] = agrupar_e3(P[orden], clases)
    return salida


def fusion(ens_e4: dict, e3: dict) -> dict:
    """{fuente: {0: (pt, pa)}}: promedio simple de las probabilidades de E4 (ensamble) y E3."""
    return {f: {0: ((ens_e4[f][0] + e3[f][0]) / 2, (ens_e4[f][1] + e3[f][1]) / 2)} for f in ens_e4 if f in e3}


# ---------------------------------------------------------------------------
# Lineas de base del 15-10 y veredictos
# ---------------------------------------------------------------------------


def base_e1() -> dict:
    d = json.loads((DATOS_15_10 / "2026-10-15-clasificador-E1.json").read_text())
    return {"total": d["ensamble_total"]["familia"]["exactitud"],
            "por_fold": {f: v["familia"]["exactitud"] for f, v in d["por_fold"].items()}}


def familia_por_fold(r: dict) -> dict:
    return {f: v["familia"]["exactitud"] for f, v in r["por_fold"].items()}


def veredictos(res: dict, regla: dict | None, e1: dict) -> dict:
    v = {}
    e4 = res.get("E4")
    if e4 and "ensamble_total" in e4:
        desv = e4.get("dispersion_entre_semillas", {}).get("familia_exactitud", {}).get("desvio")
        v["P5_familia"] = {"compara": "E4 contra E1", **veredicto_mejora(
            e4["ensamble_total"]["familia"]["exactitud"], e1["total"], familia_por_fold(e4), e1["por_fold"],
            FOLDS_MIN, [desv] if desv is not None else None)}
        if regla:
            c8 = e4["ensamble_fuentes_c8"]["altura"]
            folds_e4 = {f: e4["por_fold"][f]["altura"]["balanceada"] for f in FUENTES_C8 if f in e4["por_fold"]}
            folds_r = {f: regla["por_fuente"][f]["balanceada"] for f in FUENTES_C8}
            alt = veredicto_mejora(c8["balanceada"], regla["total"]["balanceada"], folds_e4, folds_r, FOLDS_MIN_ALTURA)
            alt["C8b"] = {"balanceada": c8["balanceada"], "recall_cuerpo": c8["recall_cuerpo"],
                          "cumple": c8["balanceada"] >= C8B["balanceada"] and c8["recall_cuerpo"] >= C8B["recall_cuerpo"]}
            v["P5_altura"] = {"compara": "E4 (fuentes de C8) contra la regla de C8", **alt}
        else:
            v["P5_altura"] = {"veredicto": "no realizado", "motivo": "falta resultados/regla_c8.json"}
    e5 = res.get("E5")
    if e5 and "ensamble_total" in e5 and e4:
        fe5 = familia_por_fold(e5)
        if set(fe5) == set(FUENTES):
            tot5, tot4, fmin = e5["ensamble_total"]["familia"]["exactitud"], e4["ensamble_total"]["familia"]["exactitud"], FOLDS_MIN
        else:   # D7: E5 solo sobre las fuentes de C8, regla con 3 de 4
            tot5 = e5["ensamble_fuentes_c8"]["familia"]["exactitud"]
            tot4 = e4["ensamble_fuentes_c8"]["familia"]["exactitud"]
            fmin = FOLDS_MIN_ALTURA
        desv = [r.get("dispersion_entre_semillas", {}).get("familia_exactitud", {}).get("desvio") for r in (e4, e5)]
        desv = [d for d in desv if d is not None]
        v["P6"] = {"compara": "E5 contra E4", **veredicto_mejora(tot5, tot4, fe5, familia_por_fold(e4), fmin, desv or None),
                   "recall_hook": {k: res[k]["ensamble_total"]["familia"]["recall"]["hook"] for k in ("E4", "E5")}}
    else:
        v["P6"] = {"veredicto": "no realizado", "motivo": "no hay corridas de E5"}
    v["subhipotesis_1"] = lectura_ocho({k: res[k]["ensamble_total"]["ocho_clases"]["exactitud"]
                                        for k in ("E4", "E5") if res.get(k) and "ensamble_total" in res[k]})
    v["D9_sospechosos"] = [k for k, r in res.items() if r and "ensamble_total" in r and sospechoso(r["ensamble_total"])]
    return v


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def resumen_md(res: dict, ver: dict, regla: dict | None) -> str:
    lineas = ["# Resumen E4 a E6 (generado por evaluar.py)", "",
              "| Exp | Folds | Familia | Familia bal. | Recall hook | Altura bal. | Recall cuerpo | Altura bal. C8 | Ocho clases | Desvío familia |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if not r or "ensamble_total" not in r:
            lineas.append(f"| {k} | no realizado | | | | | | | | |")
            continue
        t, c8 = r["ensamble_total"], r.get("ensamble_fuentes_c8", {})
        dv = r.get("dispersion_entre_semillas", {}).get("familia_exactitud", {}).get("desvio")
        lineas.append(f"| {k} | {len(r['por_fold'])} | {_f(t['familia']['exactitud'])} | {_f(t['familia']['balanceada'])} | "
                      f"{_f(t['familia']['recall']['hook']['recall'])} | {_f(t['altura']['balanceada'])} | "
                      f"{_f(t['altura']['recall_cuerpo'])} | {_f(c8.get('altura', {}).get('balanceada'))} | "
                      f"{_f(t['ocho_clases']['exactitud'])} | {_f(dv)} |")
    if regla:
        lineas += ["", f"Regla de C8 sobre {regla['total']['golpes']} golpes: balanceada {_f(regla['total']['balanceada'])}, "
                       f"recall de cuerpo {_f(regla['total']['recall_cuerpo'])}, sin dato {regla['total']['sin_dato']}."]
    lineas += ["", "## Veredictos", ""]
    for k, x in ver.items():
        if k in ("subhipotesis_1",):
            lineas.append(f"- {k}: lectura {x.get('lectura')} ({x.get('texto', '')})")
        elif k == "D9_sospechosos":
            lineas.append(f"- D9, números a revisar por fuga: {x or 'ninguno'}")
        else:
            extra = ""
            if "diferencia" in x:
                extra = f" (diferencia {_f(x['diferencia'])}, folds ganados {x['folds_ganados']} de {x['folds']}, mínimo {x['folds_min']})"
            lineas.append(f"- {k}, {x.get('compara', '')}: **{x['veredicto']}**{extra}")
    return "\n".join(lineas) + "\n"


def main(argv=None) -> int:
    from datos import leer_csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", nargs="+", default=["E4", "E4a", "E5"])
    args = ap.parse_args(argv)
    golpes = leer_csv(TRABAJO / "datos" / "golpes.csv")
    dest = TRABAJO / "resultados"
    dest.mkdir(parents=True, exist_ok=True)
    res, ens_e4 = {}, None
    for exp in args.exp:
        c = leer_corridas(exp, golpes)
        if not c["preds"]:
            res[exp] = None
            continue
        r = {"exp": exp, "faltan": c["faltan"], "completo": not c["faltan"], **evaluar_probs(c["preds"], golpes)}
        res[exp] = comun.redondear(r)
        (dest / f"{exp}.json").write_text(json.dumps(res[exp], indent=2, ensure_ascii=False) + "\n")
        if exp == "E4":
            ens_e4 = {f: (np.mean([v[0] for v in ps.values()], 0), np.mean([v[1] for v in ps.values()], 0))
                      for f, ps in c["preds"].items()}
    # F: sin umbral, seccion aparte
    e3 = probs_e3(golpes) if ens_e4 else None
    if e3 is None:
        res["F"] = None
        (dest / "F.json").write_text(json.dumps({"exp": "F", "veredicto": "no realizado",
                                                 "motivo": "faltan E4 o los pred.pkl de E3 (D4)"}, indent=2) + "\n")
    else:
        rf = {"exp": "F", "que": "promedio de probabilidades de E4 (ensamble) y E3 (s0, 15-10)",
              **evaluar_probs(fusion(ens_e4, e3), golpes, semillas=(0,))}
        rf["E3_recalculado"] = evaluar_probs({f: {0: v} for f, v in e3.items()}, golpes, semillas=(0,))["ensamble_total"]
        res["F"] = comun.redondear(rf)
        (dest / "F.json").write_text(json.dumps(res["F"], indent=2, ensure_ascii=False) + "\n")
    regla_p = dest / "regla_c8.json"
    regla = json.loads(regla_p.read_text()) if regla_p.exists() else None
    ver = comun.redondear(veredictos({k: v for k, v in res.items() if k in ("E4", "E5")}, regla, base_e1()))
    (dest / "veredictos.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False) + "\n")
    (dest / "resumen.md").write_text(resumen_md(res, ver, regla))
    print(f"escrito: {sorted(p.name for p in dest.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
