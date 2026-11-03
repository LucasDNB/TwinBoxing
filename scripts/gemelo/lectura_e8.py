"""
BoxTwin - Lectura de E8 contra su pre-registro (S15, 29-10-2026).

POR QUE EXISTE
  E8 deja catorce caches, dos reportes del detector, setenta corridas de E4, el trackeo, C1 y
  el humo, cada uno en su JSON. Las reglas de lectura estan escritas antes de medir
  (docs/specs/2026-10-27-gemelo-yolo26, 02-preregistro-e8.md §3 y §4). Si la lectura se hace
  a mano, la regla termina aplicandose a ojo despues de ver el numero. Aca se aplica como
  esta escrita, con pruebas, y el auditor la recalcula por su lado (§7, punto 6).

QUE HACE
  Controles: K1 y K4 del puente por fuente, K2 del detector (F1 del ensamble en su mejor
  umbral, media de los seis folds comparables, a +-0,02 de 0,635) y de E4 (familia a +-0,02
  de 0,727), K3 (munecas con puntaje, se declara si difiere en mas de 0,10).
  P12a: F1 del ensamble en el umbral 0,80, media de los seis folds comparables; mejora con
  +0,03 y 5 de 6 folds ganados; no concluyente si K2 del detector falla. La curva de 0,70 a
  0,90, recall, precision, recall medido y rellenado y las corridas de una semilla, al lado.
  P12b: la regla de E7 (evaluar.veredicto_mejora con 5 de 7 y el desvio entre semillas).
  P12c: cambios de track y tracks mixtos, con la direccion declarada. P12d: el resumen de
  C1. P12e: H3 del humo. SH1: ocho clases sobre YOLO26, lectura a/b/c.
  Escribe e8/veredictos.json y los tres JSON de docs/experiments/datos/.

USO
  python scripts/gemelo/lectura_e8.py --fecha 2026-10-29
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "clasificador_geo"))
sys.path.append(str(AQUI.parent / "clasificador"))   # comun; al final: tiene otro evaluar.py
import guardas  # noqa: E402

UMBRAL_P12A = 0.80
CURVA = (0.70, 0.75, 0.80, 0.85, 0.90)
F1_PUBLICADO, FAMILIA_PUBLICADA, TOL_K2 = 0.635, 0.727, 0.02
FOLDS_MIN_DETECTOR = 5          # de 6: la proporcion de 5 de 7, redondeada hacia arriba
K3_DECLARAR = 0.10


def _evaluar():
    import evaluar
    return evaluar


def folds_comparables(reporte: dict) -> list[dict]:
    """La misma regla del reporte publicado: fuera el fold cuya validacion esta anotada bajo
    una boundary_definitions_version que el entrenamiento no tiene."""
    salida = []
    for r in reporte["resultados"]:
        if r["fold"].startswith("en-distribucion"):
            continue
        val, train = set(r.get("boundary_version_val") or []), set(r.get("boundary_version_train") or [])
        if val and train and val - train:
            continue
        salida.append(r)
    return salida


def _en_umbral(r: dict, u: float) -> dict:
    for x in r["ensamble_curva"]:
        if abs(x["umbral"] - u) < 1e-6:
            return x
    raise KeyError(f"{r['fold']}: no esta el umbral {u} en ensamble_curva")


def _resumen_detector(reporte: dict) -> dict:
    fs = folds_comparables(reporte)
    una = [c["f1"] for r in fs for c in r["corridas"]]
    return {
        "folds": [r["fold"] for r in fs],
        "f1_080": statistics.mean(_en_umbral(r, UMBRAL_P12A)["f1"] for r in fs),
        "f1_080_por_fold": {r["fold"]: _en_umbral(r, UMBRAL_P12A)["f1"] for r in fs},
        "recall_080": statistics.mean(_en_umbral(r, UMBRAL_P12A)["recall"] for r in fs),
        "precision_080": statistics.mean(_en_umbral(r, UMBRAL_P12A)["precision"] for r in fs),
        "curva": {str(u): statistics.mean(_en_umbral(r, u)["f1"] for r in fs) for u in CURVA},
        "f1_mejor_umbral": statistics.mean(r["ensamble"]["f1"] for r in fs),
        "recall_medidos_mejor_umbral": [r["ensamble"].get("recall_medidos") for r in fs],
        "recall_rellenados_mejor_umbral": [r["ensamble"].get("recall_rellenados") for r in fs],
        "una_semilla": {"media": statistics.mean(una) if una else None,
                        "desvio": statistics.stdev(una) if len(una) > 1 else None},
        "no_comparables": [r["fold"] for r in reporte["resultados"] if r not in fs],
    }


def k2_detector(base: dict) -> dict:
    v = _resumen_detector(base)["f1_mejor_umbral"]
    return {"f1": v, "publicado": F1_PUBLICADO, "pasa": abs(v - F1_PUBLICADO) <= TOL_K2 + 1e-9}


def p12a(base: dict, y26: dict, k2_pasa: bool) -> dict:
    b, y = _resumen_detector(base), _resumen_detector(y26)
    v = _evaluar().veredicto_mejora(y["f1_080"], b["f1_080"], y["f1_080_por_fold"], b["f1_080_por_fold"],
                                    FOLDS_MIN_DETECTOR, None)
    if not k2_pasa:
        v["veredicto"] = "no concluyente"
        v["motivo"] = "K2 del detector fuera de +-0,02"
    return {**v, "base": b, "y26": y}


def k2_e4(base_e4: dict) -> dict:
    v = base_e4["ensamble_total"]["familia"]["exactitud"]
    return {"familia": v, "publicado": FAMILIA_PUBLICADA, "pasa": abs(v - FAMILIA_PUBLICADA) <= TOL_K2 + 1e-9}


def _fam_por_fold(r: dict) -> dict:
    return {f: v["familia"]["exactitud"] for f, v in r["por_fold"].items()}


def _desvio(r: dict):
    return r.get("dispersion_entre_semillas", {}).get("familia_exactitud", {}).get("desvio")


def _extras(r: dict) -> dict:
    t = r["ensamble_total"]
    return {"familia": t["familia"]["exactitud"], "recall_hook": t["familia"]["recall"].get("hook"),
            "recall_uppercut": t["familia"]["recall"].get("uppercut"),
            "altura_balanceada": t["altura"]["balanceada"], "recall_cuerpo": t["altura"]["recall"].get("body")}


def p12b(base_e4: dict, y26_e4: dict) -> dict:
    desv = [d for d in (_desvio(base_e4), _desvio(y26_e4)) if d is not None]
    v = _evaluar().veredicto_mejora(y26_e4["ensamble_total"]["familia"]["exactitud"],
                                    base_e4["ensamble_total"]["familia"]["exactitud"],
                                    _fam_por_fold(y26_e4), _fam_por_fold(base_e4), 5, desv or None)
    return {**v, "base": _extras(base_e4), "y26": _extras(y26_e4)}


def p12c(trackeo: dict) -> dict:
    t = trackeo["total"]
    cb, cy = (sum(t[c][p]["cambios"] for p in ("A", "B")) for c in ("base", "y26"))
    mb, my = t["base"]["tracks_mixtos"], t["y26"]["tracks_mixtos"]
    return {"denominador": t["denominador"],
            "cambios": {"base": cb, "y26": cy, "en_la_direccion": cy < cb},
            "tracks_mixtos": {"base": mb, "y26": my, "en_la_direccion": my < mb},
            "total": t, "por_fuente": {f: v.get("diferencia") for f, v in trackeo.get("fuentes", {}).items()}}


def k3(trackeo: dict) -> dict:
    k = trackeo["total"].get("k3") or {}
    d = k.get("diferencia")
    return {**k, "declarar": d is not None and abs(d) > K3_DECLARAR}


def k1_k4(por_fuente: dict) -> dict:
    medidas = {f: v["k1"] for f, v in por_fuente.items() if v.get("k1") is not None}
    fallan = [f for f, k in medidas.items() if not k["pasa"]]
    debiles = [f for f, v in por_fuente.items() if v["puente"]["puente_debil"]]
    return {"K1": {"pasa": bool(medidas) and not fallan, "fallan": fallan, "fuentes": len(medidas)},
            "K4": {"puente_debil": debiles,
                   "iou_mediana": {f: v["puente"]["iou_mediana"] for f, v in por_fuente.items()}}}


def json_e4(base: dict, y26: dict) -> dict:
    """La estructura de la decision 11, la que lee entrenar_final.py --exp E4-y26."""
    return {"exp": "E8", "base": base, "y26": y26}


def _leer(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else None


def main(argv=None) -> int:
    import comun

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fecha", required=True, help="la de la corrida, para los nombres de docs/")
    ap.add_argument("--e8", type=Path, default=guardas.GEMELO / "e8")
    ap.add_argument("--raices", type=Path, nargs=2, default=[guardas.GEMELO / "raiz-y26", guardas.GEMELO / "raiz-v8"],
                    help="raiz-y26 y la de la base re-cacheada (D-H2)")
    ap.add_argument("--datos", type=Path, default=guardas.REPO / "docs" / "experiments" / "datos")
    args = ap.parse_args(argv)
    e8 = guardas.exigir_bajo_gemelo(args.e8)

    det = {c: _leer(e8 / c / "folds.json") for c in ("base", "y26")}
    e4 = {c: {x: _leer(e8 / c / "clasificador-geo" / "resultados" / f"{x}.json") for x in ("E4", "E4a")}
          for c in ("base", "y26")}
    trackeo = _leer(e8 / "trackeo.json")
    humo = _leer(guardas.GEMELO / "humo" / "humo.json")
    puentes = {}
    for nombre, raiz in zip(("y26", "base"), args.raices):
        for fuente, d in comun.FUENTES.items():
            k = _leer(raiz / d / "k1.json")
            p = _leer(raiz / d / "puente.json")
            if p is not None:
                # K1 es del algoritmo (el original contra si mismo) y se guarda una vez, en raiz-y26
                puentes[f"{nombre}/{fuente}"] = {"k1": k, "puente": p}

    ver: dict = {"fecha": args.fecha, "controles": {}}
    if puentes:
        ver["controles"].update(k1_k4(puentes))
    if det["base"]:
        ver["controles"]["K2_detector"] = k2_detector(det["base"])
    if e4["base"]["E4"]:
        ver["controles"]["K2_E4"] = k2_e4(e4["base"]["E4"])
    if trackeo:
        ver["controles"]["K3"] = k3(trackeo)
        ver["P12c"] = p12c(trackeo)
    if det["base"] and det["y26"]:
        ver["P12a"] = p12a(det["base"], det["y26"], ver["controles"]["K2_detector"]["pasa"])
    if e4["base"]["E4"] and e4["y26"]["E4"]:
        ver["P12b"] = p12b(e4["base"]["E4"], e4["y26"]["E4"])
        ver["SH1"] = _evaluar().lectura_ocho({"E4-y26": e4["y26"]["E4"]["ensamble_total"]["ocho_clases"]["exactitud"]})
    c1 = sorted((e8 / "y26" / "c1").glob("*resumen*.json")) if (e8 / "y26" / "c1").is_dir() else []
    ver["P12d"] = {"archivos": [str(p) for p in c1], "resumen": _leer(c1[0]) if c1 else None}
    if humo:
        ver["P12e"] = humo.get("H3")
    (e8 / "veredictos.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False) + "\n")

    args.datos.mkdir(parents=True, exist_ok=True)
    pref = args.datos / f"{args.fecha}-e8-yolo26"
    if det["base"] and det["y26"]:
        Path(f"{pref}-detector.json").write_text(json.dumps({"base": det["base"], "y26": det["y26"],
                                                             "P12a": ver["P12a"]}, indent=2) + "\n")
    if e4["base"]["E4"] and e4["y26"]["E4"]:
        Path(f"{pref}-E4.json").write_text(json.dumps(json_e4(e4["base"], e4["y26"]), indent=2) + "\n")
    if trackeo:
        Path(f"{pref}-trackeo.json").write_text(json.dumps(trackeo, indent=2) + "\n")
    print(json.dumps({k: (v.get("veredicto") if isinstance(v, dict) else v) for k, v in ver.items()
                      if k.startswith(("P12", "SH"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
