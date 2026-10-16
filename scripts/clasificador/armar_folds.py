"""
BoxTwin - Particiones dejando una fuente afuera para cada configuracion.

POR QUE EXISTE
  El 0,745 de familia de produccion se midio con una particion 75/25 dentro de las mismas
  siete fuentes: el clasificador vio partes de cada video en entrenamiento y por eso el
  numero esta inflado. La pregunta de la sub-hipotesis 1 es sobre sesiones no vistas, y la
  unica particion que la contesta con estos datos es dejar una fuente entera afuera por vez.
  Partir por evento ademas filtra (dos golpes de una combinacion comparten cuadros), y el
  exportador lo advierte en su meta.

  E2 y E3 colapsan las 12 clases con lado a 6 sin lado. El lado lo da el detector (un carril
  por brazo), asi que el clasificador no tiene por que adivinarlo, y sin lado en la etiqueta
  el Flip de la aumentacion deja de contradecirla (spec, seccion 3).

QUE HACE
  Lee los pickles por fuente de la variante del experimento, remapea etiquetas si hace falta y
  escribe un pickle por fold con split train/test (el formato que PoseDataset filtra por
  `split`). Verifica que ninguna muestra de la fuente de prueba quede en entrenamiento.
  Escribe <trabajo>/folds/<exp>/manifest.json con clases y conteos por fold.

USO
  python scripts/clasificador/armar_folds.py --exp E1 E2 E3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES_TIPO_ALTURA, EXPERIMENTOS, FUENTES, TRABAJO, clase_tipo_altura  # noqa: E402


def cargar_fuente(directorio: Path, fuente: str) -> tuple[list[dict], list[str]]:
    pkl = directorio / f"{fuente}.mmaction.pkl"
    meta = directorio / f"{fuente}.mmaction.meta.json"
    with pkl.open("rb") as fh:
        datos = pickle.load(fh)
    clases = json.loads(meta.read_text())["classes"]
    return datos["annotations"], clases


def remapear(muestras: list[dict], clases: list[str], remap: str | None) -> tuple[list[dict], list[str]]:
    if remap is None:
        return muestras, clases
    if remap != "tipo_altura":
        raise ValueError(remap)
    destino = {n: CLASES_TIPO_ALTURA.index(clase_tipo_altura(n)) for n in clases}
    nuevas = []
    for m in muestras:
        m = dict(m)
        m["label"] = destino[clases[m["label"]]]
        nuevas.append(m)
    return nuevas, list(CLASES_TIPO_ALTURA)


def armar(exp: str, exports: Path, salida: Path, fuentes=tuple(FUENTES)) -> dict:
    cfg = EXPERIMENTOS[exp]
    por_fuente, clases_ref = {}, None
    for f in fuentes:
        muestras, clases = cargar_fuente(exports / cfg["variante"], f)
        if clases_ref is not None and clases != clases_ref:
            raise SystemExit(f"{f}: las clases del export no coinciden con las de las otras fuentes")
        clases_ref = clases
        por_fuente[f], clases_fin = remapear(muestras, clases, cfg["remap"])
    nombres = [m["frame_dir"] for ms in por_fuente.values() for m in ms]
    if len(nombres) != len(set(nombres)):
        raise SystemExit("frame_dir repetido entre fuentes")

    todas = [m for f in fuentes for m in por_fuente[f]]
    manifest = {"exp": exp, "que": cfg["que"], "variante": cfg["variante"], "clases": clases_fin, "folds": {}}
    salida.mkdir(parents=True, exist_ok=True)
    for prueba in fuentes:
        test = [m["frame_dir"] for m in por_fuente[prueba]]
        train = [m["frame_dir"] for f in fuentes if f != prueba for m in por_fuente[f]]
        assert not set(test) & set(train), "fuga: una muestra de la fuente de prueba esta en train"
        assert all(n.startswith(prueba + "_") for n in test)
        with (salida / f"fold_{prueba}.pkl").open("wb") as fh:
            pickle.dump({"split": {"train": train, "test": test}, "annotations": todas}, fh, protocol=4)
        cuenta = lambda ns: dict(sorted(Counter(clases_fin[m["label"]] for m in todas if m["frame_dir"] in set(ns)).items()))  # noqa: E731
        manifest["folds"][prueba] = {"train": len(train), "test": len(test),
                                     "por_clase_train": cuenta(train), "por_clase_test": cuenta(test)}
    (salida / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", nargs="+", default=list(EXPERIMENTOS), choices=list(EXPERIMENTOS))
    args = ap.parse_args(argv)
    for exp in args.exp:
        m = armar(exp, TRABAJO / "exports", TRABAJO / "folds" / exp)
        print(f"{exp}: {len(m['clases'])} clases, " +
              ", ".join(f"{f} {v['train']}/{v['test']}" for f, v in m["folds"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
