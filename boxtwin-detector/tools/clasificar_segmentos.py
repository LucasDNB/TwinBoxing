#!/usr/bin/env python3
"""
BoxTwin - El sistema completo, etapa 2: que golpe es cada segmento detectado.

POR QUE ESTA SEPARADO DE LA ETAPA 1
  El clasificador vive en boxtwin_mmaction, un entorno pinneado con torch 2.1 y el stack de
  OpenMMLab, y el detector vive en twinboxing_env con torch 2.6. Meter los dos en el mismo
  entorno pediria mover uno de los dos pines por una razon que no es tecnica. Entre las dos
  etapas va un JSON, que ademas deja auditar los segmentos sin correr nada.

QUE MIDE, Y CON QUE ADVERTENCIA
  El numero de punta a punta: de los golpes anotados, cuantos el sistema encuentra Y ademas
  clasifica bien. Es el producto de las dos etapas y es el que responde si BoxTwin cuenta
  golpes sobre video.

  ADVERTENCIA QUE NO ES LETRA CHICA: el clasificador se entreno EN DISTRIBUCION sobre
  sparring-3 y esta medido a 62,5% ahi y por debajo de su linea de base en los tres folds que
  dejan una fuente afuera. Sobre sparring-3 este numero esta inflado porque el clasificador
  vio la mitad de entrenamiento de ese mismo video; sobre cualquier otra fuente, su parte es
  ruido. El detector, en cambio, se entreno SIN la fuente que se mide.

USO
  ~/miniforge3/envs/boxtwin_mmaction/bin/python tools/clasificar_segmentos.py \
      pipeline/sparring-3-rounds.segmentos.json \
      --annot ../anotacion-sparring-3/annotations/sparring-3-rounds.annot.json \
      --cache ../anotacion-sparring-3/cache/sparring-3-rounds.pose.npz \
      --config ../modelos/poseC3D_sparring3_indist.py \
      --checkpoint ../modelos/poseC3D_sparring3_indist.pth
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

CLASES = ["jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut"]
# el clasificador habla en lead-rear; la anotacion guarda tipo y lado por separado
FAMILIA = {"jab": "straight", "cross": "straight", "lead hook": "hook",
           "rear hook": "hook", "lead uppercut": "uppercut", "rear uppercut": "uppercut"}


class Clasificador:
    """Mismo camino que tools/demo_vivo.py del anotador, para que los numeros se comparen."""

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
    p.add_argument("segmentos", type=Path)
    p.add_argument("--annot", type=Path, required=True)
    p.add_argument("--secuencia", type=Path, required=True,
                   help="el .sequence.npz, que trae los keypoints ya resueltos por peleador")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = json.loads(args.segmentos.read_text())
    doc = json.loads(args.annot.read_text())
    seq = np.load(args.secuencia)
    kp_all, sc_all = seq["keypoints"], seq["kp_score"]
    alto, ancho = doc["video"]["height"], doc["video"]["width"]
    peleadores = {"fighter_A": 0, "fighter_B": 1}

    # familia real de cada golpe anotado, por su indice en la lista `anotados`
    eventos = sorted(doc["events"], key=lambda e: (e["fighter"], e["side"], e["start_frame"]))
    por_carril: dict[str, list] = {}
    for e in doc["events"]:
        por_carril.setdefault(f"{e['fighter'][-1]}-{e['side']}", []).append(e)
    for v in por_carril.values():
        v.sort(key=lambda e: e["start_frame"])

    familia_real = {}
    for a in d["anotados"]:
        lista = por_carril.get(a["carril"], [])
        cerca = [e for e in lista if not (e["end_frame"] < a["inicio"] or e["start_frame"] > a["fin"])]
        if cerca:
            familia_real[a["indice"]] = cerca[0]["punch_type"]

    clf = Clasificador(args.config, args.checkpoint, args.device)
    aciertos = evaluados = 0
    conf = Counter()
    for s in d["segmentos"]:
        if s["empareja_con"] is None:
            continue
        real = familia_real.get(s["empareja_con"])
        if real is None:
            continue
        p_i = peleadores[s["fighter"]]
        kp = kp_all[p_i, s["inicio"] : s["fin"] + 1]
        sc = sc_all[p_i, s["inicio"] : s["fin"] + 1]
        if len(kp) < 2:
            continue
        clase, _ = clf(kp, sc, alto, ancho)
        pred = FAMILIA[clase]
        evaluados += 1
        aciertos += pred == real
        conf[(real, pred)] += 1
        s["familia_predicha"] = pred
        s["familia_real"] = real

    det = d["deteccion"]
    fam = aciertos / evaluados if evaluados else 0.0
    e2e = det["emparejados"] / det["golpes"] * fam if det["golpes"] else 0.0
    print(f"\nfamilia sobre los {evaluados} golpes que el detector encontro")
    print(f"  acierto de familia   {fam:.3f}")
    print("\n  matriz (real -> predicha)")
    for real in ("straight", "hook", "uppercut"):
        fila = {pr: conf[(real, pr)] for pr in ("straight", "hook", "uppercut")}
        n = sum(fila.values())
        if n:
            print(f"    {real:9s} n={n:3d}  " +
                  "  ".join(f"{k} {v}" for k, v in fila.items()))
    print(f"\nDE PUNTA A PUNTA, sobre los {det['golpes']} golpes anotados")
    print(f"  encontrados          {det['recall']:.3f}")
    print(f"  y bien clasificados  {e2e:.3f}")
    print(f"  precision de deteccion {det['precision']:.3f}")
    print("\n  El clasificador vio la mitad de entrenamiento de sparring-3, asi que su parte")
    print("  esta inflada. El detector, en cambio, se entreno sin esta fuente.")

    d["familia"] = {"evaluados": evaluados, "acierto": round(fam, 4),
                    "punta_a_punta": round(e2e, 4)}
    dst = args.out or args.segmentos.with_name(args.segmentos.stem + ".clasificado.json")
    dst.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
