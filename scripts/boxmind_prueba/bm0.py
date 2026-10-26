"""
BoxTwin - BM-0: el train.py de BoxMind tal como esta publicado, sobre su propio test.

POR QUE EXISTE
  Antes de usar el modelo de BoxMind sobre BoxTwin hay que saber si el codigo publicado da, en
  su dominio y con sus poses, algo cercano a lo que reporta el paper (Tabla S3). Si no se
  acerca, ningun resultado posterior se puede atribuir "al modelo de BoxMind". BM-0 no toca
  nada: corre su train.py con los hiperparametros del paper y lee la ultima epoca.

QUE HACE
  Arma <trabajo_bm>/bm0/ con data/boxingweb -> los datos descargados, corre
  `python <clon>/train.py -m PoseOnly+TCN+MoE -e 12 --lr 0.05 -b 64` desde ahi (su train.py
  usa rutas relativas), guarda el log y escribe bm0.json con el F1 macro de la ultima epoca
  (tecnica, objetivo, distancia, efecto). train.py evalua contra el test en cada epoca; aca se
  toma la ultima, no la mejor (pre-registro, seccion 4.2, B6). Una sola corrida: train.py
  fija la semilla 42.

USO (en un entorno con sus dependencias: torch, opencv, scikit-learn, joblib, torchvision,
tensorboard, matplotlib)
  python scripts/boxmind_prueba/bm0.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import BOXINGWEB, BOXMIND_REPO, HIPER, TRABAJO_BM, fuera_del_repo, verificar_clon  # noqa: E402

PATRON = re.compile(r"test_f1_score:\s*([0-9.eE+-]+)\s*(\[.*\])")


def ultima_epoca(log: str) -> dict | None:
    """La ultima linea `test_f1_score: avg [tech, tgt, dis, eff]` del log de train.py."""
    hallados = PATRON.findall(log)
    if not hallados:
        return None
    prom, lista = hallados[-1]
    vals = [float(v) for v in ast.literal_eval(lista.replace("np.float64(", "(").replace("np.float32(", "("))]
    return {"epocas_vistas": len(hallados), "promedio": float(prom),
            **dict(zip(("tecnica", "objetivo", "distancia", "efecto"), vals))}


def main() -> int:
    commit = verificar_clon()
    base = fuera_del_repo(TRABAJO_BM / "bm0")
    (base / "data").mkdir(parents=True, exist_ok=True)
    enlace = base / "data" / "boxingweb"
    if not enlace.exists():
        enlace.symlink_to(BOXINGWEB.resolve(), target_is_directory=True)
    cmd = [sys.executable, str(BOXMIND_REPO / "train.py"), "-m", HIPER["modelo"], "-e", str(HIPER["epocas"]),
           "--lr", str(HIPER["lr"]), "-b", str(HIPER["batch"])]
    with (base / "train.log").open("w") as fh:
        r = subprocess.run(cmd, cwd=base, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ})
    log = (base / "train.log").read_text(errors="replace")
    res = ultima_epoca(log)
    salida = {"comando": cmd, "boxmind_commit": commit, "codigo_salida": r.returncode, "ultima_epoca": res,
              "referencia_paper_tcn_pose_only": {"tecnica": 0.692, "objetivo": 0.747}}
    (base / "bm0.json").write_text(json.dumps(salida, indent=2) + "\n")
    print(json.dumps(salida, indent=2))
    return 0 if r.returncode == 0 and res else 1


if __name__ == "__main__":
    sys.exit(main())
