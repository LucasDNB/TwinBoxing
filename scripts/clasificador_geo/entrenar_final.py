"""
BoxTwin - Modelo final de E4 para el producto (tipo de golpe v2, 16-10-2026).

POR QUE EXISTE
  E4 se midio dejando una fuente afuera: cada corrida guardo sus predicciones y no sus pesos.
  El producto necesita pesos, entrenados con la misma receta (02-diseno.md seccion 1 de
  docs/specs/2026-10-16-fightcard-tipo-v2). Si este script reimplementara el entrenamiento,
  el modelo del producto dejaria de ser el que se midio sin que nadie lo note. Por eso no
  toca entrenar.py, modelo.py, features2d.py ni datos.py: los importa. CA2 verifica que con
  --excluir Sparring --semillas 0 sale el mismo modelo que E4 s0 de ese fold.

QUE HACE
  Con la configuracion E4 (columnas 0 a 25 de golpes.npz) y entrenar.entrenar_modelo:
  - sin --excluir, entrena con los 1047 golpes y guarda en <salida>/ s0.pt ... s4.pt
    (state_dict) y meta.json;
  - con --excluir <fuente>, lo mismo sin esa fuente, en <salida>/sin_<fuente>/ (CA2 y M1).
  meta.json lleva features, pasos, margen, la estandarizacion de cada semilla, los
  hiperparametros, el commit, el sha256 de cada .pt y de golpes.npz, y las metricas de E4 que
  muestra la Fight-Card, copiadas del json de E4. Reanudable: una semilla con .pt se saltea.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --excluir Sparring --semillas 0
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --excluir 03-sparring 04-sparring
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import FUENTES, PAD, PASOS, REPO, SEMILLAS, TIPOS, TRABAJO  # noqa: E402
import entrenar  # noqa: E402

SALIDA = Path("~/boxtwin/modelos/clasificador_geo_v1").expanduser()
METRICAS_E4 = REPO / "docs" / "experiments" / "datos" / "2026-10-16-clasificador-geo-E4.json"
REFERENCIA = "docs/experiments/2026-10-16-clasificador-geometrico.md"
EXP = "E4"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def metricas_e4(ruta: Path = METRICAS_E4) -> dict:
    """Lo que la Fight-Card muestra del clasificador: familia del ensamble con una fuente afuera."""
    if not ruta.is_file():
        return {}
    fam = json.loads(ruta.read_text())["ensamble_total"]["familia"]
    return {
        "familia_fuente_no_vista": fam["exactitud"],
        "familia_wilson95": fam.get("wilson95"),
        "familia_balanceada": fam.get("balanceada"),
        "recall_por_familia": fam.get("recall"),
        "golpes": json.loads(ruta.read_text())["ensamble_total"]["golpes"],
        "referencia": REFERENCIA,
        "de": str(ruta.relative_to(REPO)) if ruta.is_relative_to(REPO) else str(ruta),
    }


def entrenar_final(d: dict, salida: Path, excluir: str | None = None, semillas=SEMILLAS,
                   dispositivo: str = "cpu", epocas: int = entrenar.HIPER["epocas"],
                   golpes_npz: Path | None = None, metricas: dict | None = None) -> Path:
    """
    Entrena las semillas pedidas sobre todo menos `excluir` y escribe .pt y meta.json en
    `salida` (o salida/sin_<fuente>). Devuelve el directorio.
    """
    destino = Path(salida) / f"sin_{excluir}" if excluir else Path(salida)
    destino.mkdir(parents=True, exist_ok=True)
    idx_train = np.flatnonzero(d["fuente"] != excluir) if excluir else np.arange(len(d["X"]))
    meta_previo = destino / "meta.json"
    meta = json.loads(meta_previo.read_text()) if meta_previo.is_file() else {}
    por_semilla = dict(meta.get("semillas", {}))
    for s in semillas:
        pt = destino / f"s{s}.pt"
        if pt.is_file() and f"s{s}" in por_semilla:
            continue
        modelo, media, desvio = entrenar.entrenar_modelo(
            d["X"], d["M"], d["y_tipo"], d["y_altura"], idx_train, s, dispositivo, epocas=epocas)
        torch.save({k: v.detach().cpu() for k, v in modelo.state_dict().items()}, pt)
        por_semilla[f"s{s}"] = {"archivo": pt.name, "sha256": sha256(pt),
                                "media": media.tolist(), "desvio": desvio.tolist()}
    meta = {
        "nombre": "clasificador_geo_v1" if not excluir else f"clasificador_geo_v1_sin_{excluir}",
        "metodo": "geometrico-v1",
        "exp": EXP,
        "n_features": int(d["X"].shape[-1]),
        "features": list(d["nombres"]),
        "pasos": PASOS,
        "margen": PAD,
        "clases_familia": list(TIPOS),
        "excluir": excluir,
        "n_train": int(len(idx_train)),
        "fuentes_train": sorted(set(d["fuente"][idx_train].tolist())),
        "hiper": entrenar.HIPER if epocas == entrenar.HIPER["epocas"] else {**entrenar.HIPER, "epocas": epocas},
        "dispositivo": dispositivo,
        "commit": entrenar._commit(),
        "golpes_npz_sha256": sha256(golpes_npz) if golpes_npz and Path(golpes_npz).is_file() else None,
        "metricas_e4": metricas if metricas is not None else metricas_e4(),
        "semillas": dict(sorted(por_semilla.items())),
    }
    meta_previo.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return destino


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--excluir", nargs="*", default=None, choices=FUENTES,
                    help="una o mas fuentes: un modelo por fuente, entrenado sin ella")
    ap.add_argument("--semillas", nargs="*", type=int, default=list(SEMILLAS))
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                    help="E4 se entreno en cuda; para reproducirlo (CA2) hace falta el mismo")
    args = ap.parse_args(argv)
    dispositivo = args.dispositivo
    if dispositivo == "auto":
        dispositivo = "cuda" if entrenar.gpu_libre() else "cpu"
    d = entrenar.cargar_entrada(EXP)
    npz = TRABAJO / "datos" / "golpes.npz"
    for excluir in (args.excluir or [None]):
        out = entrenar_final(d, args.salida, excluir, args.semillas, dispositivo, golpes_npz=npz)
        print(f"{excluir or 'completo'} ({dispositivo}): {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
