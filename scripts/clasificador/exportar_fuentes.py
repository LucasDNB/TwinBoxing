"""
BoxTwin - Exports de mmaction por fuente para las tres configuraciones de la campana.

POR QUE EXISTE
  El exportador del anotador ya arma los pickles de PoseConv3D con cualquiera de los espacios
  de clase y con o sin el rival. Lo que faltaba es correrlo igual para las siete fuentes y
  dejar todo en un directorio fuera del repo, con la metadata que dice de que anotacion salio
  cada pickle (el exportador guarda el sha256). El pickle de la particion de produccion se
  armo con un script que no quedo en el repo; este si queda.

QUE HACE
  Para cada variante y cada fuente llama a `boxtwin-annotator export --format mmaction` y
  deja <trabajo>/exports/<variante>/<fuente>.mmaction.pkl y su meta. Saltea lo que ya existe.
  No toca las anotaciones: el exportador solo las lee (las huellas lo comprueban despues).

USO
  python scripts/clasificador/exportar_fuentes.py
  python scripts/clasificador/exportar_fuentes.py --variantes v12s_at --cli ~/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import FUENTES, REPO, TRABAJO, ruta_annot  # noqa: E402

# Variante de export -> opciones. E1 usa v6lr_at (la de produccion), E2 v12s_at, E3 v12s_amb.
VARIANTES: dict[str, dict] = {
    "v6lr_at": {"classes": 6, "label_space": "lead-rear", "persons": "attacker"},
    "v12s_at": {"classes": 12, "label_space": "side", "persons": "attacker"},
    "v12s_amb": {"classes": 12, "label_space": "side", "persons": "both"},
}

CLI_DEFECTO = Path("~/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator").expanduser()


def video_de(fuente: str) -> Path:
    v = Path(json.loads(ruta_annot(fuente).read_text())["video"]["path"])
    if v.is_file():
        return v
    alternativa = REPO / FUENTES[fuente] / "videos" / v.name
    return alternativa


def comando(cli: Path, fuente: str, variante: str, out: Path) -> list[str]:
    o = VARIANTES[variante]
    return [str(cli), "export", str(video_de(fuente)), "--format", "mmaction",
            "--classes", str(o["classes"]), "--label-space", o["label_space"],
            "--persons", o["persons"], "--out", str(out)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variantes", nargs="*", default=list(VARIANTES))
    ap.add_argument("--cli", type=Path, default=CLI_DEFECTO)
    ap.add_argument("--solo-mostrar", action="store_true", help="imprime los comandos y no corre nada")
    args = ap.parse_args(argv)

    fallas = 0
    for variante in args.variantes:
        out = TRABAJO / "exports" / variante
        for fuente in FUENTES:
            pkl = out / f"{video_de(fuente).stem}.mmaction.pkl"
            cmd = comando(args.cli, fuente, variante, out)
            if args.solo_mostrar:
                print(" ".join(cmd))
                continue
            if pkl.exists():
                print(f"{variante}/{fuente}: ya existe")
                continue
            out.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not pkl.exists():
                fallas += 1
                print(f"{variante}/{fuente}: FALLO\n{r.stderr[-800:]}")
            else:
                print(f"{variante}/{fuente}: ok")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
