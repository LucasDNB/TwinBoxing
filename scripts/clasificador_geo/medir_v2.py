"""
BoxTwin - M1: el tipo de golpe v2 de punta a punta sobre las sesiones de C8 nivel 2.

POR QUE EXISTE
  E4 midio la familia con la ventana anotada; E6, el rol con el brazo del detector. El producto
  junta las dos cosas sobre los golpes que encontro el detector, con sus fronteras, y con la
  guardia que tenga cargada el boxeador, que puede no estar. M1 describe cuanto acierta eso
  (docs/specs/2026-10-16-fightcard-tipo-v2/01-requisitos.md seccion 4, pre-registrado). Sin
  umbral: es lo que el producto haria, no un criterio.

QUE HACE
  1. Copia de cada sesion de C8 nivel 2 los archivos que lee el producto (segmentos.json,
     poses.npz, fightcard.json, sesion.json) a <trabajo>/m1/<sesion>/<lectura>/. Las
     originales no se tocan.
  2. Por sesion y lectura de guardia (anotada: la mas frecuente de los eventos anotados de ese
     peleador en la fuente; ortodoxos: todos ortodoxos), corre `boxtwin-annotator clasificar`
     con el modelo final de E4 entrenado sin esa fuente (sin_<fuente>) y `tipos`.
  3. Empareja los golpes de la Fight-Card con los anotados completos como E6 (por peleador,
     sin separar por brazo, IoU 0,3; importado de rol_por_brazo.py).
  4. Exactitud de familia y de cuatro clases (jab, cross, hook, uppercut) con Wilson, total y
     por fuente, recall por familia, y al lado la familia del ensamble de E4 sobre los mismos
     golpes con la ventana anotada (pred.npz de las cinco semillas).
  Escribe <trabajo>/resultados/m1.json y su copia en docs/experiments/datos/.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/medir_v2.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import REPO, TIPOS, comun  # noqa: E402
from geo import TRABAJO as TRABAJO_E4  # noqa: E402
from rol_por_brazo import IOU, SESION_FUENTE, SESIONES, emparejar_por_peleador  # noqa: E402

TRABAJO = Path("~/boxtwin/tipo-v2").expanduser()
MODELOS = Path("~/boxtwin/modelos/clasificador_geo_v1").expanduser()
CMD = [str(Path("~/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator").expanduser())]
ARCHIVOS = ("segmentos.json", "poses.npz", "fightcard.json", "sesion.json")
LECTURAS = ("anotada", "ortodoxos")
CUATRO = ("jab", "cross", "hook", "uppercut")
A_GUARDIA = {"orthodox": "ortodoxa", "southpaw": "zurda"}


def guardia_mas_frecuente(eventos: list[dict], fighter: str) -> str | None:
    """La guardia mas frecuente de los eventos completos de ese peleador ('fighter_A'), o None."""
    c = Counter(e["guard"] for e in eventos if e.get("completeness") == "full" and e["fighter"] == fighter)
    if not c:
        return None
    # empate: la ortodoxa, que es lo que el producto supone sin dato
    g = max(sorted(c), key=lambda k: (c[k], k == "orthodox"))
    return A_GUARDIA[g]


def guardias_por_lado(eventos: list[dict], invertido: bool) -> dict[str, str | None]:
    """Lado de la Fight-Card -> guardia anotada del peleador que le corresponde."""
    anotado = {"A": "fighter_B" if invertido else "fighter_A", "B": "fighter_A" if invertido else "fighter_B"}
    return {lado: guardia_mas_frecuente(eventos, f) for lado, f in anotado.items()}


def cuatro_clases(punch_type: str, rol: str) -> str:
    if punch_type == "straight":
        return "jab" if rol == "lead" else "cross"
    return punch_type


def filas_de_pares(pares, fuente: str) -> list[dict]:
    """Una fila por golpe emparejado: verdad y prediccion de familia y de cuatro clases."""
    from boxtwin.core.types import Guard, Side, arm_role
    filas = []
    for g, e, p in pares:
        rol_v = arm_role(Guard(e["guard"]), Side(e["side"])).value
        filas.append({"frame_dir": f"{fuente}_{e['id']}", "fuente": fuente, "peleador": p,
                      "familia": e["punch_type"], "familia_pred": g.get("tipo_clasificador"),
                      "cuatro": cuatro_clases(e["punch_type"], rol_v), "cuatro_pred": g.get("tipo")})
    return filas


def exactitud(filas: list[dict], verdad: str, pred: str) -> dict:
    n = len(filas)
    a = sum(1 for f in filas if f[pred] == f[verdad])
    return {"n": n, "aciertos": a, "exactitud": a / n if n else None, "wilson95": comun.wilson(a, n)}


def recall_por_clase(filas: list[dict], verdad: str, pred: str, clases) -> dict:
    out = {}
    for c in clases:
        sub = [f for f in filas if f[verdad] == c]
        out[c] = exactitud(sub, verdad, pred) if sub else {"n": 0, "aciertos": 0, "exactitud": None, "wilson95": None}
    return out


def metricas(filas: list[dict]) -> dict:
    return {"golpes": len(filas),
            "familia": {**exactitud(filas, "familia", "familia_pred"),
                        "recall": recall_por_clase(filas, "familia", "familia_pred", TIPOS)},
            "cuatro_clases": {**exactitud(filas, "cuatro", "cuatro_pred"),
                              "recall": recall_por_clase(filas, "cuatro", "cuatro_pred", CUATRO)}}


def familia_e4(fuente: str, trabajo: Path = TRABAJO_E4) -> dict[str, str]:
    """frame_dir -> familia del ensamble de E4 (promedio de las cinco semillas) con la ventana anotada."""
    ps, nombres = [], None
    for s in range(5):
        z = np.load(trabajo / "corridas" / "E4" / fuente / f"s{s}" / "pred.npz")
        if nombres is None:
            nombres = z["frame_dir"]
        elif not np.array_equal(nombres, z["frame_dir"]):
            raise SystemExit(f"E4/{fuente}: las semillas no siguen el mismo orden")
        ps.append(z["p_tipo"])
    P = np.mean(ps, 0)
    return {str(n): TIPOS[int(i)] for n, i in zip(nombres, P.argmax(1))}


def copiar(sesion: str, lectura: str, origen: Path = SESIONES, destino: Path = TRABAJO / "m1") -> Path:
    d = destino / sesion / lectura
    d.mkdir(parents=True, exist_ok=True)
    for a in ARCHIVOS:
        shutil.copy2(origen / sesion / a, d / a)
    return d


def correr(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"fallo {' '.join(cmd)}:\n{r.stderr or r.stdout}")


def main(argv=None) -> int:
    from boxtwin.mvp.medicion import medir_sesion
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    dest = TRABAJO / "resultados"
    dest.mkdir(parents=True, exist_ok=True)
    faltan = [s for s in SESION_FUENTE if not (SESIONES / s / "segmentos.json").exists()]
    if faltan:   # T5
        (dest / "m1.json").write_text(json.dumps({"medicion": "M1", "estado": "no realizado",
                                                  "motivo": f"faltan sesiones de C8 nivel 2: {faltan}"}, indent=2) + "\n")
        print(f"M1 no realizado: faltan {faltan}")
        return 0
    filas = {l: [] for l in LECTURAS}
    por_fuente: dict = {}
    guardias_usadas = {}
    for sesion, fuente in SESION_FUENTE.items():
        modelo = MODELOS / f"sin_{fuente}"
        if not (modelo / "meta.json").is_file():
            raise SystemExit(f"falta el modelo {modelo}: correr entrenar_final.py --excluir {fuente}")
        annot = json.loads(comun.ruta_annot(fuente).read_text())
        fc0 = json.loads((SESIONES / sesion / "fightcard.json").read_text())
        inv = medir_sesion(fc0, annot).invertido
        g_anot = guardias_por_lado(annot["events"], inv)
        guardias_usadas[fuente] = {"invertido": inv, "anotada": g_anot}
        por_fuente[fuente] = {}
        for lectura in LECTURAS:
            d = copiar(sesion, lectura)
            ga, gb = (g_anot["A"], g_anot["B"]) if lectura == "anotada" else ("ortodoxa", "ortodoxa")
            correr([*CMD, "clasificar", str(d), "--modelo", str(modelo),
                    "--guardia-a", ga or "desconocida", "--guardia-b", gb or "desconocida"])
            correr([*CMD, "tipos", str(d)])
            fc = json.loads((d / "fightcard.json").read_text())
            golpes = {p: x["golpes"] for p, x in fc["peleadores"].items()}
            pares = emparejar_por_peleador(golpes, annot["events"], inv)
            fs = filas_de_pares(pares, fuente)
            filas[lectura].extend(fs)
            por_fuente[fuente][lectura] = {"golpes_fightcard": sum(len(v) for v in golpes.values()),
                                           "anotados_completos": sum(1 for e in annot["events"] if e.get("completeness") == "full"),
                                           **metricas(fs)}
    # la familia no depende de la guardia: las dos lecturas tienen que coincidir
    assert [f["familia_pred"] for f in filas["anotada"]] == [f["familia_pred"] for f in filas["ortodoxos"]]
    e4 = {}
    for fuente in SESION_FUENTE.values():
        e4.update(familia_e4(fuente))
    comp = [{**f, "familia_e4": e4.get(f["frame_dir"])} for f in filas["anotada"]]
    sin_e4 = sum(1 for f in comp if f["familia_e4"] is None)
    comp = [f for f in comp if f["familia_e4"] is not None]
    spec = (TRABAJO / "spec.txt").read_text().strip() if (TRABAJO / "spec.txt").is_file() else None
    out = {
        "medicion": "M1", "que": "tipo v2 de punta a punta sobre golpes del detector (C8 nivel 2)",
        "spec_commit": spec, "iou": IOU, "modelo": "clasificador_geo_v1/sin_<fuente>, 5 semillas, CPU",
        "advertencia": "el detector de produccion se entreno con estas fuentes (como en C8 nivel 2 y E6): "
                       "los golpes encontrados son optimistas; el clasificador si las tiene afuera",
        "guardias": guardias_usadas,
        "total": {l: metricas(filas[l]) for l in LECTURAS},
        "por_fuente": por_fuente,
        "comparacion_e4_ventana_anotada": {
            "golpes": len(comp), "sin_prediccion_e4": sin_e4,
            "familia_v2_detector": exactitud(comp, "familia", "familia_pred"),
            "familia_e4_ventana_anotada": exactitud(comp, "familia", "familia_e4"),
        },
        "filas": filas["anotada"],
    }
    out = comun.redondear(out)
    texto = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    (dest / "m1.json").write_text(texto)
    copia = REPO / "docs" / "experiments" / "datos" / "2026-10-16-tipo-v2-m1.json"
    copia.write_text(texto)
    t = out["total"]
    print(f"M1: {t['anotada']['golpes']} golpes emparejados; familia {t['anotada']['familia']['exactitud']}, "
          f"cuatro clases {t['anotada']['cuatro_clases']['exactitud']} (guardia anotada) y "
          f"{t['ortodoxos']['cuatro_clases']['exactitud']} (todos ortodoxos) -> {dest / 'm1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
