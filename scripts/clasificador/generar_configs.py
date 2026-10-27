"""
BoxTwin - Configs de mmaction por experimento, fold y semilla.

POR QUE EXISTE
  Las 63 corridas de la campana tienen que ser identicas a produccion salvo en lo que el
  pre-registro dice que cambia: el pickle del fold, el numero de clases y la semilla. Editar
  63 configs a mano es como se cuela una diferencia que nadie ve. Se parte siempre de
  `modelos/poseC3D_7fuentes.py` (el de produccion, sin pesos de clase) y se tocan solo esas
  claves, mas dos que la particion nueva obliga:

  - Sin validacion durante el entrenamiento. El config de produccion valida cada epoca y
    guarda el mejor (`save_best='auto'`). Con una fuente afuera, el unico conjunto que no se
    vio es el de prueba, y elegir la epoca mirandolo es medir sobre lo que se uso para elegir.
    Se entrena el numero fijo de epocas del config y se evalua el ultimo checkpoint.
  - `load_from` a ruta absoluta, porque las corridas no se lanzan desde la raiz del repo.

QUE HACE
  Escribe <trabajo>/configs/<exp>/<fuente>/s<semilla>.py y <trabajo>/corridas.tsv con una
  linea por corrida (exp, fuente, semilla, config, work_dir) en el orden en que corre la
  campana: primero todas las de semilla 0, asi una campana cortada a la mitad deja un
  resultado completo de una semilla y no tres a medias.

USO (en boxtwin_mmaction: necesita mmengine)
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador/generar_configs.py
  ... --load-from /ruta/al/best_acc_top1_epoch_8.pth   si no lo encuentra solo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import EXPERIMENTOS, FUENTES, REPO, SEMILLAS, TRABAJO  # noqa: E402

BASE = REPO / "modelos" / "poseC3D_7fuentes.py"
MMACTION = REPO / "test-poseConvBoxing" / "mmaction2"


def resolver_load_from(rel: str, override: Path | None) -> Path:
    if override:
        p = override.expanduser()
        if not p.is_file():
            raise SystemExit(f"--load-from no existe: {p}")
        return p
    for cand in (REPO / rel, MMACTION / rel, Path(rel)):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(f"no encuentro el checkpoint inicial {rel}; pasalo con --load-from")


def ajustar(cfg, clases: int, fold_pkl: Path, work_dir: Path, semilla: int, load_from: Path):
    """Las unicas claves que cambian respecto de produccion. Devuelve el mismo cfg."""
    cfg.model.cls_head.num_classes = clases
    cfg.train_dataloader.dataset.dataset.ann_file = str(fold_pkl)
    cfg.train_dataloader.dataset.dataset.split = "train"
    cfg.test_dataloader.dataset.ann_file = str(fold_pkl)
    cfg.test_dataloader.dataset.split = "test"
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=cfg.train_cfg.max_epochs)
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=cfg.train_cfg["max_epochs"],
                                        save_best=None, max_keep_ckpts=1, save_last=True)
    cfg.randomness = dict(seed=semilla, deterministic=False)
    cfg.work_dir = str(work_dir)
    cfg.load_from = str(load_from)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", nargs="+", default=list(EXPERIMENTOS))
    ap.add_argument("--load-from", type=Path, default=None)
    args = ap.parse_args(argv)

    from mmengine.config import Config

    base = Config.fromfile(str(BASE))
    load_from = resolver_load_from(base.load_from, args.load_from)
    lineas = []
    for semilla in SEMILLAS:
        for exp in args.exp:
            manifest = json.loads((TRABAJO / "folds" / exp / "manifest.json").read_text())
            for fuente in FUENTES:
                cfg = Config.fromfile(str(BASE))
                pkl = TRABAJO / "folds" / exp / f"fold_{fuente}.pkl"
                wd = TRABAJO / "corridas" / exp / fuente / f"s{semilla}"
                ajustar(cfg, len(manifest["clases"]), pkl, wd, semilla, load_from)
                destino = TRABAJO / "configs" / exp / fuente / f"s{semilla}.py"
                destino.parent.mkdir(parents=True, exist_ok=True)
                cfg.dump(str(destino))
                lineas.append("\t".join([exp, fuente, str(semilla), str(destino), str(wd)]))
    (TRABAJO / "corridas.tsv").write_text("\n".join(lineas) + "\n")
    print(f"{len(lineas)} corridas -> {TRABAJO / 'corridas.tsv'}  (load_from {load_from})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
