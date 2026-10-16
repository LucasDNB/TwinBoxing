"""
BoxTwin - Elevacion a 3D del atacante con MotionBERT, por golpe, para E5.

POR QUE EXISTE
  El hook es un problema de proyeccion monocular: el mismo arco se ve como una recta segun
  donde este la camara. BoxMind (arXiv 2601.11492) clasifica tecnica y altura en boxeo con
  pose 3D estimada desde una camara, y Skarimva (arXiv 2602.23231) mide que el 3D baja el error
  a la mitad. MotionBERT ya viene con MMPose 1.3.2 en boxtwin_mmaction, asi que E5 no instala
  nada (pre-registro, seccion 4, punto 4).

  MotionBERT es secuencia a secuencia: 243 cuadros de 2D adentro, 243 de 3D afuera. En MMPose
  1.3.2, `inference_pose_lifter_model` arma `lifting_target` con un solo cuadro y falla con
  secuencias de mas de uno, y el demo eleva cuadro por cuadro (seq_len 1), que le saca el
  contexto temporal. Aca se usa el mismo pipeline del config y el mismo `test_step` que llama
  esa funcion, con el data_info armado para 243 cuadros.

QUE HACE
  Por golpe, solo el atacante: contexto de 243 cuadros centrado en la ventana, tomado de kp[p]
  (borde repetido donde se acaba el video; en pacquiao_margarito, a 59,94 fps, un cuadro de
  cada dos). Huecos de pose de hasta 10 cuadros se interpolan antes de elevar; los mas largos se
  rellenan con el vecino y quedan con mascara 0. COCO -> H36M con `convert_keypoint_definition`,
  normalizacion por camara con image_size = (ancho, alto). La salida se remuestrea a los 32
  pasos de la ventana del golpe. Escribe <trabajo>/datos/pose3d.npz (frame_dir, P3 (N, 32, 17, 3),
  M3 (N, 32)) y pose3d.qc.json: coeficiente de variacion del largo de antebrazo y de brazo del
  brazo que golpea a lo largo de cada ventana, mediana y p90 por fuente. Sin umbral.

USO (en boxtwin_mmaction)
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador_geo/elevar3d.py
  ~/miniforge3/envs/boxtwin_mmaction/bin/python scripts/clasificador_geo/elevar3d.py --fuentes 03-sparring --dispositivo cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import FUENTES, PASOS, TRABAJO  # noqa: E402

LARGO = 243
MAX_HUECO = 10
CONFIG = "configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"
PESOS_URL = ("https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/"
             "motionbert_ft_h36m-d80af323_20230531.pth")
LOTE = 8           # 64 no entra en los 8 GB de la 2080 (atencion temporal de 243 x 243)
# H36M: hombro, codo, muneca por lado
BRAZO_H36M = {"left": (11, 12, 13), "right": (14, 15, 16)}


def rellenar_huecos(kp: np.ndarray, presente: np.ndarray, max_hueco: int = MAX_HUECO) -> tuple[np.ndarray, np.ndarray]:
    """
    kp (T, K, 2), presente (T,). Huecos internos de hasta max_hueco cuadros: interpolacion
    lineal y mascara 1. Huecos mas largos o en los bordes: el vecino mas cercano y mascara 0.
    """
    kp = np.array(kp, np.float32, copy=True)
    presente = np.asarray(presente, bool)
    mascara = presente.copy()
    T = len(kp)
    idx = np.flatnonzero(presente)
    if len(idx) == 0:
        return np.zeros_like(kp), mascara
    i = 0
    while i < T:
        if presente[i]:
            i += 1
            continue
        j = i
        while j < T and not presente[j]:
            j += 1
        izq, der = i - 1, j            # vecinos con pose (pueden no existir)
        if izq >= 0 and der < T and (j - i) <= max_hueco:
            a = (np.arange(i, j) - izq) / (der - izq)
            kp[i:j] = kp[izq][None] * (1 - a)[:, None, None] + kp[der][None] * a[:, None, None]
            mascara[i:j] = True
        else:
            # vecino mas cercano; con un solo lado disponible, ese
            for f in range(i, j):
                if izq < 0:
                    kp[f] = kp[der]
                elif der >= T:
                    kp[f] = kp[izq]
                else:
                    kp[f] = kp[izq] if (f - izq) <= (der - f) else kp[der]
        i = j
    return kp, mascara


def paso_de(fps: float) -> int:
    """Un cuadro de cada dos por encima de 45 fps, para que 243 cuadros cubran un tiempo parecido."""
    return 2 if fps > 45 else 1


def indices_contexto(ini: int, fin: int, T: int, paso: int, largo: int = LARGO) -> tuple[np.ndarray, np.ndarray]:
    """Cuadros del contexto centrado en la ventana: sin recortar (para interpolar) y recortados al video."""
    centro = (ini + fin) // 2
    crudo = centro + (np.arange(largo) - largo // 2) * paso
    if crudo[0] > ini or crudo[-1] < fin:
        raise ValueError(f"la ventana [{ini}, {fin}] no entra en el contexto de {largo} cuadros con paso {paso}")
    return crudo, np.clip(crudo, 0, T - 1)


def a_ventana(X: np.ndarray, crudo: np.ndarray, ini: int, fin: int, pasos: int = PASOS) -> np.ndarray:
    """(largo, K, 3) en los cuadros `crudo` -> (pasos, K, 3) en la grilla uniforme de [ini, fin]."""
    pos = np.linspace(ini, fin, pasos)
    flat = X.reshape(len(X), -1)
    out = np.stack([np.interp(pos, crudo, flat[:, j]) for j in range(flat.shape[1])], -1)
    return out.reshape(pasos, *X.shape[1:]).astype(np.float32)


def mascara_ventana(mascara: np.ndarray, ini: int, fin: int, pasos: int = PASOS) -> np.ndarray:
    pos = np.rint(np.linspace(ini, fin, pasos)).astype(int)
    return mascara[pos]


def cv_huesos(P: np.ndarray, M: np.ndarray, side: str) -> tuple[float | None, float | None]:
    """Coeficiente de variacion del largo de antebrazo y de brazo sobre los pasos con mascara."""
    h, c, m = BRAZO_H36M[side]
    sel = M.astype(bool)
    if sel.sum() < 2:
        return None, None
    salida = []
    for a, b in ((c, m), (h, c)):
        L = np.linalg.norm(P[sel, a] - P[sel, b], axis=-1)
        salida.append(float(L.std() / L.mean()) if L.mean() > 0 else None)
    return tuple(salida)


def _gpu_libre() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and not r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False


def elegir_dispositivo(pedido: str) -> str:
    """D5: la elevacion espera de a 10 minutos hasta 2 horas si la GPU esta ocupada; despues, CPU."""
    if pedido != "auto":
        return pedido
    for _ in range(13):
        if _gpu_libre():
            return "cuda:0"
        print("GPU ocupada, espero 10 minutos (D5)", flush=True)
        time.sleep(600)
    return "cpu"


class Elevador:
    """MotionBERT de MMPose con el pipeline del config. Solo se construye en boxtwin_mmaction."""

    def __init__(self, pesos: Path, dispositivo: str):
        import os

        import mmpose
        from mmengine.dataset import Compose
        from mmengine.registry import init_default_scope
        from mmpose.apis import init_model
        cfg = os.path.join(os.path.dirname(mmpose.__file__), ".mim", CONFIG)
        self.modelo = init_model(cfg, str(pesos), device=dispositivo)
        init_default_scope(self.modelo.cfg.get("default_scope", "mmpose"))
        self.pipeline = Compose(self.modelo.cfg.test_dataloader.dataset.pipeline)

    def preparar(self, kp_coco: np.ndarray, ancho: int, alto: int) -> dict:
        from mmpose.apis import convert_keypoint_definition
        T = len(kp_coco)
        k2 = convert_keypoint_definition(kp_coco.astype(np.float32), "coco", "h36m")
        di = dict(keypoints=k2, keypoints_visible=np.ones((T, 17), np.float32),
                  lifting_target=np.zeros((T, 17, 3), np.float32),
                  lifting_target_visible=np.ones((T, 17, 1), np.float32),
                  factor=np.zeros((T,), np.float32), camera_param=dict(w=ancho, h=alto))
        di.update(self.modelo.dataset_meta)
        return self.pipeline(di)

    def elevar(self, datos: list[dict]) -> list[np.ndarray]:
        import torch
        from mmengine.dataset import pseudo_collate
        with torch.no_grad():
            res = self.modelo.test_step(pseudo_collate(datos))
        return [np.asarray(r.pred_instances.keypoints).reshape(-1, 17, 3) for r in res]


def main(argv=None) -> int:
    from datos import leer_csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=FUENTES)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--pesos", type=Path, default=TRABAJO / "pesos" / Path(PESOS_URL).name)
    args = ap.parse_args(argv)
    d = TRABAJO / "datos"
    golpes = [g for g in leer_csv(d / "golpes.csv") if g["fuente"] in args.fuentes]
    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"dispositivo: {dispositivo}; golpes: {len(golpes)}", flush=True)
    elev = Elevador(args.pesos, dispositivo)
    P3, M3, fds, qc = [], [], [], {}
    t0 = time.time()
    for fuente in args.fuentes:
        gs = [g for g in golpes if g["fuente"] == fuente]
        with np.load(d / "series" / f"{fuente}.npz") as z:
            kp, valid, fps, ancho, alto = z["kp"], z["valid"], float(z["fps"]), int(z["ancho"]), int(z["alto"])
        T, paso = kp.shape[1], paso_de(fps)
        rellenos = {p: rellenar_huecos(kp[p], valid[p]) for p in {g["p"] for g in gs}}
        for i in range(0, len(gs), LOTE):
            lote = gs[i:i + LOTE]
            entradas, ctx = [], []
            for g in lote:
                kpf, _ = rellenos[g["p"]]
                crudo, idx = indices_contexto(g["ini"], g["fin"], T, paso)
                entradas.append(elev.preparar(kpf[idx], ancho, alto))
                ctx.append(crudo)
            for g, crudo, X in zip(lote, ctx, elev.elevar(entradas)):
                _, msk = rellenos[g["p"]]
                P3.append(a_ventana(X, crudo, g["ini"], g["fin"]))
                M3.append(mascara_ventana(msk, g["ini"], g["fin"]))
                fds.append(g["frame_dir"])
                qc.setdefault(fuente, []).append(cv_huesos(P3[-1], M3[-1], g["side"]))
        print(f"{fuente}: {len(gs)} golpes, paso {paso}, {time.time() - t0:.0f} s acumulados", flush=True)
    np.savez_compressed(d / "pose3d.npz", frame_dir=np.array(fds), P3=np.stack(P3), M3=np.stack(M3))
    resumen = {}
    for f, vs in qc.items():
        ant = np.array([v[0] for v in vs if v[0] is not None])
        bra = np.array([v[1] for v in vs if v[1] is not None])
        resumen[f] = {"golpes": len(vs), "con_dato": int(len(ant)),
                      "cv_antebrazo": {"mediana": round(float(np.median(ant)), 4), "p90": round(float(np.percentile(ant, 90)), 4)} if len(ant) else None,
                      "cv_brazo": {"mediana": round(float(np.median(bra)), 4), "p90": round(float(np.percentile(bra, 90)), 4)} if len(bra) else None}
    (d / "pose3d.qc.json").write_text(json.dumps({
        "modelo": CONFIG, "pesos": PESOS_URL, "dispositivo": dispositivo, "largo_contexto": LARGO,
        "max_hueco_interpolado": MAX_HUECO, "segundos": round(time.time() - t0, 1), "por_fuente": resumen,
        "pasos_con_mascara": round(float(np.stack(M3).mean()), 4)}, indent=2) + "\n")
    print(json.dumps(resumen, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
