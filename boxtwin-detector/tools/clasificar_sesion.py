#!/usr/bin/env python3
"""
BoxTwin - Que golpe es cada segmento de una sesion de produccion.

POR QUE EXISTE Y POR QUE NO ES clasificar_segmentos.py
  Ese mide: necesita la anotacion para saber la familia real y reporta acierto. Este no
  tiene contra que comparar, porque corre sobre una sesion de un usuario que nadie anoto.
  Son dos contratos distintos y meterlos en un script con un flag terminaria con el camino
  de produccion importando el de medicion, que es como los dos se desincronizan.

  Sigue viviendo aparte del resto del sistema por la razon de siempre: el clasificador es
  PoseConv3D y corre en boxtwin_mmaction, con torch 2.1 y el stack de OpenMMLab pinneado
  contra CUDA 11.8. El detector corre en twinboxing_env con torch 2.6. Entre los dos va un
  JSON.

LO QUE HAY QUE DECIR DEL NUMERO QUE PRODUCE
  El clasificador no generaliza a fuentes nuevas: esta medido en o por debajo de su linea
  de base en los folds que dejan una fuente afuera. Entra al producto igual, por decision
  de producto, y por eso cada tipo sale con su confianza y con la version del checkpoint
  que lo produjo. La Fight-Card lo rotula como estimacion y separa lo medido de lo estimado.

QUE HACE
  Lee segmentos.json y poses.npz de una sesion, clasifica cada segmento y escribe
  tipos.json. El merge a la Fight-Card lo hace el otro entorno, con
  'boxtwin-annotator tipos', para que el contrato del documento viva en un solo lugar.

USO
  ~/miniforge3/envs/boxtwin_mmaction/bin/python tools/clasificar_sesion.py sesion/ \
      --config ../modelos/poseC3D_7fuentes.py --checkpoint ../modelos/poseC3D_7fuentes.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# El clasificador habla en lead-rear sobre seis clases; la Fight-Card muestra cuatro tipos,
# porque jab y cross ya son la familia straight con el brazo puesto. Se guardan los dos: el
# crudo para poder comparar checkpoints, y el de pantalla para el usuario.
CLASES = ["jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut"]
TIPO = {
    "jab": "jab",
    "cross": "cross",
    "lead hook": "hook",
    "rear hook": "hook",
    "lead uppercut": "uppercut",
    "rear uppercut": "uppercut",
}
BRAZO = {"left": "izq", "right": "der"}


class Clasificador:
    """
    Mismo camino que tools/clasificar_segmentos.py y que demo_vivo.py.

    Los tres lo repiten a proposito: este entorno no tiene el paquete del anotador
    instalado y mover ese pin por una clase de veinte lineas no se paga.
    """

    def __init__(self, config: Path, checkpoint: Path, device: str = "cuda:0") -> None:
        from mmaction.apis import init_recognizer
        from mmengine.dataset import Compose

        self.modelo = init_recognizer(str(config), str(checkpoint), device=device)
        self.pipeline = Compose(self.modelo.cfg.test_dataloader.dataset.pipeline)

    def __call__(self, kp, score, alto, ancho):
        import torch

        muestra = {
            "frame_dir": "seg", "label": 0, "img_shape": (alto, ancho),
            "original_shape": (alto, ancho), "total_frames": int(kp.shape[0]),
            "start_index": 0, "modality": "Pose",
            "keypoint": kp[None].astype(np.float32),
            "keypoint_score": score[None].astype(np.float32),
        }
        dato = self.pipeline(muestra)
        with torch.no_grad():
            salida = self.modelo.test_step(
                {"inputs": [dato["inputs"]], "data_samples": [dato["data_samples"]]}
            )[0]
        p = salida.pred_score.cpu().numpy()
        i = int(np.argmax(p))
        return CLASES[i], float(p[i])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("sesion", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-cuadros", type=int, default=2, dest="min_cuadros",
                   help="segmentos mas cortos que esto no se clasifican: con un solo cuadro "
                        "el muestreo temporal de PoseConv3D no tiene de donde sacar nada")
    args = p.parse_args()

    d = json.loads((args.sesion / "segmentos.json").read_text())
    poses = np.load(args.sesion / d.get("poses", "poses.npz"))
    kp_all, sc_all = poses["keypoints"], poses["kp_score"]
    alto = int(d["video"].get("alto") or 0)
    ancho = int(d["video"].get("ancho") or 0)
    if not alto or not ancho:
        print("error: segmentos.json no trae el tamano del cuadro")
        return 1

    clf = Clasificador(args.config, args.checkpoint, args.device)
    indices = {"A": 0, "B": 1}
    tipos: dict[str, dict] = {}
    saltados = 0
    for s in d["segmentos"]:
        i = indices.get(s["peleador"])
        if i is None:
            continue
        kp = kp_all[i, s["inicio"] : s["fin"] + 1]
        sc = sc_all[i, s["inicio"] : s["fin"] + 1]
        # Un golpe que quedo en un cuadro no se clasifica y tampoco se le inventa un tipo:
        # sale sin tipo, que es un estado que la Fight-Card sabe mostrar.
        if len(kp) < args.min_cuadros:
            saltados += 1
            continue
        crudo, conf = clf(kp, sc, alto, ancho)
        gid = f"{s['peleador']}-{BRAZO.get(s['brazo'], s['brazo'])}-{s['inicio']}"
        tipos[gid] = {"tipo": TIPO[crudo], "confianza": round(conf, 4), "crudo": crudo}

    salida = {
        "version": "0.1",
        "checkpoint": args.checkpoint.name,
        "clases": CLASES,
        "segmentos_sin_clasificar": saltados,
        "tipos": tipos,
    }
    dst = args.sesion / "tipos.json"
    dst.write_text(json.dumps(salida, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(tipos)} segmentos clasificados, {saltados} demasiado cortos")
    print(f"  -> {dst}")
    print("\n  El tipo es una ESTIMACION: el clasificador no generaliza a fuentes nuevas.")
    print("  Aplicalo a la Fight-Card con:")
    print(f"    boxtwin-annotator tipos {args.sesion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
