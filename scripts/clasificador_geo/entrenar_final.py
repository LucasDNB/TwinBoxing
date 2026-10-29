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

  Con --exp E4a (16-10, bolsa y sombra) usa las columnas 0 a 19, las del brazo: en una sesion
  de una sola persona no hay rival y E4 veria sus 6 features en cero todo el golpe. Guarda en
  ~/boxtwin/modelos/clasificador_geo_v1_sin_rival con las metricas de E4a (familia 0,724 con
  una fuente afuera). La receta es la misma; solo cambian las columnas.

USO
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --exp E4a --dispositivo cpu
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --excluir Sparring --semillas 0
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --excluir 03-sparring 04-sparring
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --exp E7 --dispositivo cuda
    (25-10, modo academico: lee el 3D de ~/boxtwin/e7-4dhumans, guarda en ~/boxtwin/modelos/clasificador_e7_v1)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --exp E4-y26 --dispositivo cpu
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/entrenar_final.py --exp E4a-y26 --dispositivo cpu
    (gemelo C, docs/specs/2026-10-27-gemelo-yolo26: golpes.npz de E8 sobre YOLO26, metricas de
    docs/experiments/datos/<fecha>-e8-yolo26-E4.json, salida en ~/boxtwin-gemelo/modelos.
    Sin ese json se niega: C no muestra numeros que no midio)
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
GEMELO = Path("~/boxtwin-gemelo").expanduser()


def _json_e8(patron: str = "*-e8-yolo26-E4.json") -> Path:
    """El json de E8 mas reciente; si todavia no existe, un nombre que no existe."""
    hechos = sorted((REPO / "docs" / "experiments" / "datos").glob(patron))
    return hechos[-1] if hechos else REPO / "docs" / "experiments" / "datos" / "pendiente-e8-yolo26-E4.json"


# Por experimento: nombre del modelo, salida por defecto y de donde salen las metricas que
# muestra la Fight-Card. E4a es el de bolsa y sombra (sin rival).
VARIANTES = {
    "E4": {"nombre": "clasificador_geo_v1", "salida": SALIDA, "metricas": METRICAS_E4},
    "E4a": {"nombre": "clasificador_geo_v1_sin_rival",
            "salida": Path("~/boxtwin/modelos/clasificador_geo_v1_sin_rival").expanduser(),
            "metricas": REPO / "docs" / "experiments" / "datos" / "2026-10-16-clasificador-geo-E4a.json"},
    # E7 (25-10, docs/specs/2026-10-25-e7-modo-academico): modo academico de la app, apagado por
    # defecto. 26 de E4 + 16 features 3D de HMR2.0b. Depende de SMPL, solo investigacion.
    "E7": {"nombre": "clasificador_e7_v1", "metodo": "geometrico-e7",
           "salida": Path("~/boxtwin/modelos/clasificador_e7_v1").expanduser(),
           "metricas": REPO / "docs" / "experiments" / "datos" / "2026-10-24-e7-4dhumans-E7.json",
           "referencia": "docs/experiments/2026-10-24-e7-4dhumans.md",
           "licencia": ("modo academico: 3D de 4D-Humans con SMPL, licencia de SMPLify "
                        "(Max-Planck-Gesellschaft): uso no comercial y sin fines de lucro, solo "
                        "investigacion y educacion")},
    # Gemelo C (27-10, docs/specs/2026-10-27-gemelo-yolo26, T4): la misma receta de E4 y E4a
    # sobre las poses de YOLO26. `columnas` es el experimento de entrenar.py del que salen las
    # columnas; `trabajo`, el directorio de E8 con golpes.npz; `seccion`, donde estan sus
    # metricas dentro del json de E8, que trae las dos condiciones y las dos variantes.
    "E4-y26": {"nombre": "clasificador_geo_y26_v1", "columnas": "E4",
               "salida": GEMELO / "modelos" / "clasificador_geo_y26_v1",
               "trabajo": GEMELO / "e8" / "y26" / "clasificador-geo",
               "metricas": _json_e8(), "seccion": ("y26", "E4")},
    "E4a-y26": {"nombre": "clasificador_geo_y26_v1_sin_rival", "columnas": "E4a",
                "salida": GEMELO / "modelos" / "clasificador_geo_y26_v1_sin_rival",
                "trabajo": GEMELO / "e8" / "y26" / "clasificador-geo",
                "metricas": _json_e8(), "seccion": ("y26", "E4a")},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def metricas_e4(ruta: Path = METRICAS_E4, referencia: str = REFERENCIA, seccion: tuple = ()) -> dict:
    """Lo que la Fight-Card muestra del clasificador: familia del ensamble con una fuente afuera.

    `seccion` es el camino dentro del json cuando trae mas de un experimento (el de E8).
    """
    if not ruta.is_file():
        return {}
    crudo = json.loads(ruta.read_text())
    for clave in seccion:
        crudo = crudo[clave]
    fam = crudo["ensamble_total"]["familia"]
    return {
        "familia_fuente_no_vista": fam["exactitud"],
        "familia_wilson95": fam.get("wilson95"),
        "familia_balanceada": fam.get("balanceada"),
        "recall_por_familia": fam.get("recall"),
        "golpes": crudo["ensamble_total"]["golpes"],
        "referencia": referencia,
        "de": str(ruta.relative_to(REPO)) if ruta.is_relative_to(REPO) else str(ruta),
    }


def entrenar_final(d: dict, salida: Path, excluir: str | None = None, semillas=SEMILLAS,
                   dispositivo: str = "cpu", epocas: int = entrenar.HIPER["epocas"],
                   golpes_npz: Path | None = None, metricas: dict | None = None, exp: str = EXP,
                   comando: str | None = None) -> Path:
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
    base = VARIANTES[exp]["nombre"]
    meta = {
        "nombre": base if not excluir else f"{base}_sin_{excluir}",
        "metodo": VARIANTES[exp].get("metodo", "geometrico-v1"),
        "exp": exp,
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
        # La clave sigue siendo metricas_e4 porque es la que lee el producto; con --exp E4a
        # trae las de E4a.
        "metricas_e4": metricas if metricas is not None else _metricas_de(exp),
        "semillas": dict(sorted(por_semilla.items())),
    }
    if comando:
        meta["comando"] = comando
    if VARIANTES[exp].get("licencia"):
        meta["licencia"] = VARIANTES[exp]["licencia"]
    meta_previo.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return destino


def _metricas_de(exp: str) -> dict:
    v = VARIANTES[exp]
    return metricas_e4(v["metricas"], v.get("referencia", REFERENCIA), v.get("seccion", ()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--excluir", nargs="*", default=None, choices=FUENTES,
                    help="una o mas fuentes: un modelo por fuente, entrenado sin ella")
    ap.add_argument("--semillas", nargs="*", type=int, default=list(SEMILLAS))
    ap.add_argument("--exp", default=EXP, choices=list(VARIANTES),
                    help="E4 (con rival, sparring y combate), E4a (sin rival, bolsa y sombra) o E7 "
                         "(modo academico, con el 3D de HMR2; solo investigacion)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="por defecto la de la variante (~/boxtwin/modelos/clasificador_geo_v1[_sin_rival])")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                    help="E4 se entreno en cuda; para reproducirlo (CA2) hace falta el mismo")
    args = ap.parse_args(argv)
    dispositivo = args.dispositivo
    if dispositivo == "auto":
        dispositivo = "cuda" if entrenar.gpu_libre() else "cpu"
    v = VARIANTES[args.exp]
    if "seccion" in v and not _metricas_de(args.exp):
        print(f"{args.exp}: no estan las metricas de E8 en {v['metricas']}; no se arma un modelo "
              "de C sin sus numeros (CG5)", file=sys.stderr)
        return 2
    trabajo = v.get("trabajo", TRABAJO)
    d = entrenar.cargar_entrada(v.get("columnas", args.exp), trabajo)
    salida = args.salida or v["salida"]
    npz = trabajo / "datos" / "golpes.npz"
    comando = " ".join(["entrenar_final.py", *(argv if argv is not None else sys.argv[1:])])
    for excluir in (args.excluir or [None]):
        out = entrenar_final(d, salida, excluir, args.semillas, dispositivo, golpes_npz=npz, exp=args.exp,
                             comando=comando)
        print(f"{excluir or 'completo'} ({dispositivo}): {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
