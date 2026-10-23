"""
BoxTwin - Arma los conjuntos de la prueba BM en la representacion de BoxMind.

POR QUE EXISTE
  Entrenamiento y evaluacion leen un solo archivo por dominio, ya en la forma que espera el
  modelo (T, 28, 5), para que la representacion se calcule una vez y con una sola funcion
  (representacion.ejemplo) para BoxTwin y para BoxingWeb.

QUE HACE
  Por golpe: cuadros de la ventana anotada (paso 2 por encima de 45 fps), 2D del atacante y del
  rival con huecos de hasta 10 cuadros interpolados, 3D de pose3d_<dominio>.npz, y
  `representacion.ejemplo`. Escribe <trabajo_bm>/datos/conjunto_<dominio>.npz con X (N, L, 28, 5),
  largo, frame_dir, fuente, particion, y_tipo, y_altura, y_distancia, y_efecto (-1 donde no hay
  etiqueta: BoxTwin no tiene distancia ni efecto anotados). Chequea que BoxTwin tenga los 1047
  golpes de E4.

USO (en twinboxing_env)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/armar.py --dominio boxtwin
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/armar.py --dominio boxingweb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import TRABAJO_BM, fuera_del_repo, geo  # noqa: E402
from elevar import cuadros_de, golpes_y_series  # noqa: E402
from representacion import ejemplo  # noqa: E402

sys.path.insert(0, str(Path(geo.__file__).resolve().parent))
from elevar3d import rellenar_huecos  # noqa: E402


def leer_3d(ruta: Path) -> dict:
    """{(frame_dir, persona): P3 (largo, 17, 3)}"""
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: correr elevar.py en boxtwin_mmaction")
    with np.load(ruta) as z:
        return {(str(fd), int(pe)): z["P3"][i, :int(n)]
                for i, (fd, pe, n) in enumerate(zip(z["frame_dir"], z["persona"], z["largo"]))}


def armar(golpes: list[dict], serie, P3: dict) -> dict:
    cache, ejemplos = {}, []
    for g in golpes:
        if g["fuente"] not in cache:
            kp, sc, valid, fps, ancho, alto = serie(g["fuente"])
            cache[g["fuente"]] = ([rellenar_huecos(kp[p], valid[p])[0] for p in (0, 1)], sc, fps, ancho, alto)
        kps, sc, fps, ancho, alto = cache[g["fuente"]]
        T = kps[0].shape[0]
        cs = np.clip(cuadros_de(g["start_frame"], g["end_frame"], fps), 0, T - 1)
        a, r = g["p"], 1 - g["p"]
        ejemplos.append(ejemplo(kps[a][cs], sc[a][cs], kps[r][cs], sc[r][cs],
                                P3.get((g["frame_dir"], 0)), P3.get((g["frame_dir"], 1)), ancho, alto))
    L = max(len(e) for e in ejemplos)
    X = np.zeros((len(ejemplos), L, 28, 5), np.float32)
    for i, e in enumerate(ejemplos):
        X[i, :len(e)] = e
    tipos, alturas = list(geo.TIPOS), list(geo.ALTURAS)
    return {"X": X, "largo": np.array([len(e) for e in ejemplos]),
            "frame_dir": np.array([g["frame_dir"] for g in golpes]),
            "fuente": np.array([g["fuente"] for g in golpes]),
            "particion": np.array([g.get("particion", "") for g in golpes]),
            "y_tipo": np.array([tipos.index(g["tipo"]) for g in golpes]),
            "y_altura": np.array([alturas.index(g["altura"]) for g in golpes]),
            "y_distancia": np.array([int(g.get("distancia", -1)) for g in golpes]),
            "y_efecto": np.array([int(g.get("efecto", -1)) for g in golpes])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dominio", choices=("boxtwin", "boxingweb"), required=True)
    args = ap.parse_args(argv)
    destino = fuera_del_repo(TRABAJO_BM / "datos")
    golpes, serie = golpes_y_series(args.dominio)
    if args.dominio == "boxtwin" and len(golpes) != geo.TOTAL_GOLPES:
        raise SystemExit(f"BoxTwin tiene {len(golpes)} golpes, E4 tiene {geo.TOTAL_GOLPES}")
    d = armar(golpes, serie, leer_3d(destino / f"pose3d_{args.dominio}.npz"))
    np.savez_compressed(destino / f"conjunto_{args.dominio}.npz", **d)
    sin2d = int((np.abs(d["X"][..., :2]).sum((1, 2, 3)) == 0).sum())
    sin3d = int((np.abs(d["X"][..., 2:]).sum((1, 2, 3)) == 0).sum())
    print(f"{args.dominio}: {len(golpes)} golpes, largo max {d['X'].shape[1]}, "
          f"sin 2D {sin2d}, sin 3D {sin3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
