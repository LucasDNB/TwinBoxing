"""
BoxTwin - Detector final de C sobre las poses de YOLO26 (S17, 29-10-2026).

POR QUE EXISTE
  C muestra la precision y el recall de SU detector, no los de A (CG5), y su detector tiene
  que ser la receta de `detector_7fuentes.ens.pt` cambiando solo las poses
  (docs/specs/2026-10-27-gemelo-yolo26, 03-instancia-c.md §5). El comando con que se armo el de
  A no quedo en el repo. Este script es esa receta escrita: ensamble de cinco semillas (42, 1,
  2, 3, 4), la Config por defecto, las siete fuentes en entrenamiento, y como validacion de
  la parada temprana las mismas siete (no hay fuente afuera en un modelo final).

QUE HACE
  - entrenar: boxtwin_detector.ensamble.entrenar_ensamble sobre los .det.npz dados; guarda el
    .ens.pt, su .metricas.json (precision y recall de E8 en el umbral 0,80, de
    e8/veredictos.json) y su .meta.json (commit, sha256 de cada tensor, semillas, comando).
  - control: la misma receta sobre otros tensores (los de A), y las probabilidades contra las
    de un checkpoint de referencia sobre una fuente. Sin umbral: la diferencia se declara.

USO
  python scripts/gemelo/detector_final.py entrenar ~/boxtwin-gemelo/e8/y26/det/*.det.npz
  python scripts/gemelo/detector_final.py control ~/Proyectos/TwinBoxing/boxtwin-detector/data/*.det.npz \
      --referencia ~/Proyectos/TwinBoxing/modelos/detector_7fuentes.ens.pt --fuente 01-sparring
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import guardas  # noqa: E402

SEMILLAS = [42, 1, 2, 3, 4]
DESTINO = guardas.GEMELO / "modelos" / "detector_7fuentes_y26.ens.pt"


def exigir_destino(ruta: Path) -> Path:
    return guardas.exigir_fuera_del_repo(guardas.exigir_bajo_gemelo(ruta))


def metricas_de_e8(veredictos: Path) -> dict:
    p = json.loads(Path(veredictos).read_text()).get("P12a")
    if not p or "y26" not in p:
        raise ValueError(f"{veredictos} no tiene P12a: no se arma un detector de C sin sus numeros (CG5)")
    y = p["y26"]
    return {"precision_medida": round(y["precision_080"], 4), "recall_medido": round(y["recall_080"], 4),
            "medido_sobre": (f"E8 sobre YOLO26, ensamble de cinco semillas, umbral 0,80, media de los "
                             f"seis folds comparables con una fuente afuera ({len(y['folds'])} folds)")}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meta(tensores: list[Path], comando: list[str], semillas=SEMILLAS, val: str = "todas") -> dict:
    commit = subprocess.run(["git", "-C", str(guardas.REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    return {"commit": commit, "tensores_sha256": {Path(t).name: _sha(t) for t in tensores},
            "semillas": list(semillas), "val": val, "comando": " ".join(comando)}


def comparar(a: list[np.ndarray], b: list[np.ndarray]) -> dict:
    dif = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    return {"carriles": len(a), "max_abs": dif, "iguales": dif == 0.0}


def _entrenar(tensores: list[Path]):
    import torch
    from boxtwin_detector.dataset import leer
    from boxtwin_detector.ensamble import entrenar_ensamble
    from boxtwin_detector.entrenamiento import Config

    fuentes = [leer(t) for t in sorted(tensores)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return entrenar_ensamble(fuentes, fuentes, Config(), SEMILLAS, device), fuentes, device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("tensores", type=Path, nargs="+")
    e.add_argument("--out", type=Path, default=DESTINO)
    e.add_argument("--veredictos", type=Path, default=guardas.GEMELO / "e8" / "veredictos.json")
    c = sub.add_parser("control")
    c.add_argument("tensores", type=Path, nargs="+")
    c.add_argument("--referencia", type=Path, required=True)
    c.add_argument("--fuente", default="01-sparring")
    c.add_argument("--out", type=Path, default=guardas.GEMELO / "modelos" / "control-receta-detector.json")
    args = ap.parse_args(argv)

    from boxtwin_detector.ensamble import cargar, guardar, probabilidad

    if args.cmd == "entrenar":
        out = exigir_destino(args.out)
        metricas = metricas_de_e8(args.veredictos)
        ens, _, _ = _entrenar(args.tensores)
        guardar(ens, out)
        out.with_suffix(".metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n")
        out.with_suffix(".meta.json").write_text(json.dumps(
            {**meta(args.tensores, sys.argv), "sha256": _sha(out), "metricas": metricas}, indent=2,
            ensure_ascii=False) + "\n")
        print(f"{out} ({_sha(out)[:12]}): {metricas}")
        return 0

    out = exigir_destino(args.out)
    ens, fuentes, device = _entrenar(args.tensores)
    ref = cargar(args.referencia, device)
    (f,) = [x for x in fuentes if x.nombre == args.fuente]
    r = comparar(probabilidad(ens, f, device), probabilidad(ref, f, device))
    r.update({"fuente": args.fuente, "referencia": str(args.referencia), **meta(args.tensores, sys.argv)})
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"control: max |dP| = {r['max_abs']:.6f} sobre {r['carriles']} carriles de {args.fuente}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
