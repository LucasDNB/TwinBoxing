#!/usr/bin/env python3
"""
BoxTwin - Etapa de clasificacion de familia, aislada para poder cronometrarla.

POR QUE EXISTE
  El clasificador vive en boxtwin_mmaction, con torch 2.1 y OpenMMLab pinneados, y el resto
  de la cadena en twinboxing_env con torch 2.6. Para medir el tiempo de esta etapa hay que
  correrla en su propio interprete, y el arnes -medicion_cadena.py- la invoca como
  subproceso y cuenta su tiempo de pared entero, carga del modelo incluida.

  Es el MISMO camino que tools/clasificar_segmentos.py del detector: init_recognizer, la
  pipeline de test y test_step por segmento. Si se apartara, estaria midiendo otra cosa que
  la que corre en serio.

QUE HACE
  Lee los segmentos que decodifico el detector, los lleva de la base de tiempo del detector
  -30 fps- a la del video, recorta los keypoints de cada uno y los clasifica. Informa
  cuanto tardo la carga del modelo y cuanto la inferencia, porque son costos distintos: la
  carga se paga una vez por video y la inferencia por golpe.

USO
  ~/miniforge3/envs/boxtwin_mmaction/bin/python medicion_clasifica.py entrada.json \
      --config ../modelos/poseC3D_7fuentes.py --checkpoint ../modelos/poseC3D_7fuentes.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CLASES = ["jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut"]
FAMILIA = {"jab": "straight", "cross": "straight", "lead hook": "hook",
           "rear hook": "hook", "lead uppercut": "uppercut", "rear uppercut": "uppercut"}
FPS_DETECTOR = 30.0


class Clasificador:
    """Mismo camino que tools/clasificar_segmentos.py, para que los tiempos se comparen."""

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
    p.add_argument("entrada", type=Path, help="JSON con segmentos y ruta de la secuencia")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    d = json.loads(a.entrada.read_text())
    seq = np.load(d["secuencia"])
    kp_all, sc_all = seq["keypoints"], seq["kp_score"]
    alto, ancho = int(d["alto"]), int(d["ancho"])
    fps_video = float(d["fps_video"])
    peleadores = {"fighter_A": 0, "fighter_B": 1}

    # el detector trabaja a 30 fps; los keypoints estan en la base del video
    escala = fps_video / FPS_DETECTOR

    t0 = time.perf_counter()
    clf = Clasificador(a.config, a.checkpoint, a.device)
    t_carga = time.perf_counter() - t0

    t_inf = 0.0
    n = 0
    T = kp_all.shape[1]
    for s in d["segmentos"]:
        i0 = int(round(s["inicio"] * escala))
        i1 = int(round(s["fin"] * escala))
        i0, i1 = max(0, i0), min(T - 1, i1)
        if i1 - i0 < 1:
            continue
        pi = peleadores[s["fighter"]]
        kp = kp_all[pi, i0 : i1 + 1]
        sc = sc_all[pi, i0 : i1 + 1]
        if len(kp) < 2:
            continue
        t1 = time.perf_counter()
        clase, conf = clf(kp, sc, alto, ancho)
        t_inf += time.perf_counter() - t1
        s["clase_predicha"] = clase
        s["familia_predicha"] = FAMILIA[clase]
        s["confianza"] = round(conf, 4)
        n += 1

    a.out.write_text(json.dumps({
        "n_clasificados": n,
        "t_carga_modelo_s": round(t_carga, 3),
        "t_inferencia_s": round(t_inf, 3),
        "segmentos": d["segmentos"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"clasificados {n} segmentos: carga {t_carga:.2f} s, inferencia {t_inf:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
