"""
BoxTwin - La regla de altura de C8 recalculada sobre los mismos golpes que E4.

POR QUE EXISTE
  El pre-registro (seccion 3, punto 2) compara la altura de E4 con la regla de produccion
  (altura_relativa_v1, tau 0,20). El numero de C8 del 13-10 no sirve para eso: contaba amagues,
  abortados y golpes sin dato sobre 806 eventos. La comparacion tiene que ser golpe contra
  golpe, asi que la regla se recalcula sobre los 775 golpes completos de las cuatro fuentes de
  C8 que tambien evalua E4, con la misma regla: sin dato es error.

QUE HACE
  Para cada golpe completo de las fuentes de C8 llama a boxtwin.mvp.objetivo.objetivo_de con
  la ventana anotada (start_frame, end_frame, no la ventana con margen de E4), tau 0,20, y
  compara con la altura anotada. Escribe <trabajo>/resultados/regla_c8.json con exactitud,
  balanceada, recall de cuerpo con Wilson y los golpes sin dato, total y por fuente, y la
  prediccion por frame_dir.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/regla_c8.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import ALTURAS, FUENTES_C8, TRABAJO, comun  # noqa: E402

TAU = 0.20
A_INGLES = {"cabeza": "head", "cuerpo": "body"}
SIN_DATO = "sin_dato"


def metricas_altura(real: list[str], pred: list[str]) -> dict:
    """Altura con sin dato como error: una prediccion fuera de head/body nunca acierta."""
    rec = comun.recall_por_clase(real, pred, ALTURAS)
    aciertos = sum(1 for a, b in zip(real, pred) if a == b)
    return {"golpes": len(real), "al_cuerpo": rec["body"]["n"],
            "exactitud": comun.exactitud(real, pred), "wilson95": comun.wilson(aciertos, len(real)),
            "balanceada": comun.balanceada(real, pred, ALTURAS),
            "recall_cuerpo": rec["body"]["recall"], "recall_cuerpo_wilson95": rec["body"]["wilson95"],
            "recall_cabeza": rec["head"]["recall"], "sin_dato": sum(1 for p in pred if p == SIN_DATO)}


def predecir_regla(golpes: list[dict], series: dict, tau: float = TAU, objetivo_de=None) -> list[str]:
    if objetivo_de is None:
        from boxtwin.mvp.objetivo import objetivo_de
    salida = []
    for g in golpes:
        s = series[g["fuente"]]
        r = objetivo_de(s["kp"], s["sc"], s["valid"], g["p"], g["side"], g["start_frame"], g["end_frame"], tau=tau)
        salida.append(A_INGLES.get(r.objetivo, SIN_DATO))
    return salida


def main(argv=None) -> int:
    from datos import leer_csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    d = TRABAJO / "datos"
    golpes = [g for g in leer_csv(d / "golpes.csv") if g["fuente"] in FUENTES_C8]
    series = {}
    for f in FUENTES_C8:
        with np.load(d / "series" / f"{f}.npz") as z:
            series[f] = {k: z[k] for k in ("kp", "sc", "valid")}
    pred = predecir_regla(golpes, series)
    real = [g["altura"] for g in golpes]
    out = {"regla": "altura_relativa_v1", "tau": TAU, "sin_dato": "error",
           "fuentes": list(FUENTES_C8), "total": metricas_altura(real, pred), "por_fuente": {},
           "pred": {g["frame_dir"]: p for g, p in zip(golpes, pred)}}
    for f in FUENTES_C8:
        sel = [i for i, g in enumerate(golpes) if g["fuente"] == f]
        out["por_fuente"][f] = metricas_altura([real[i] for i in sel], [pred[i] for i in sel])
    out = comun.redondear(out)
    dest = TRABAJO / "resultados"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "regla_c8.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"regla_c8: {out['total']['golpes']} golpes, {out['total']['al_cuerpo']} al cuerpo -> resultados/regla_c8.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
