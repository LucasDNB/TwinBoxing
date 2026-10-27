"""
BoxTwin - BoxingWeb re-estimado con el pipeline de pose de BoxTwin, para la prueba BM.

POR QUE EXISTE
  Si el modelo se entrena con las poses de 4D-Humans que trae BoxingWeb y se prueba con las de
  YOLOv8l-pose + MotionBERT de BoxTwin, cualquier diferencia mezcla dos cosas: el dominio del
  video y el estimador de pose. Para que BM-1 y BM-2 midan solo la primera, BoxingWeb pasa por
  el mismo preproceso que las fuentes propias (boxtwin.preprocess, misma configuracion que el
  producto) y la pose de BoxingWeb se usa solo para saber quien es la esquina roja y quien la
  azul en cada cuadro (pre-registro, seccion 4.1, B3).

QUE HACE
  Por ronda de data_train y data_test:
  1. preproceso de BoxTwin sobre el mp4 -> cache de pose (reanudable).
  2. emparejado por cuadro: cada deteccion contra la pose de referencia roja y azul (hombros,
     codos y caderas), asignacion de costo minimo, aceptada si la distancia media queda debajo
     de TAU_ID anchos de hombro de la referencia.
  3. series en el mismo formato que scripts/clasificador_geo/datos.py: kp (2, T, 17, 2),
     sc (2, T, 17), valid (2, T), con 0 = rojo y 1 = azul.
  4. golpes con el filtro de dataset_youtube.py (4 a 30 cuadros) y las etiquetas mapeadas a las
     de BoxTwin: tipo, altura (pecho y abdomen = cuerpo), distancia, efecto, brazo fisico.
     Un golpe se usa si el atacante quedo emparejado en COBERTURA_MIN de sus cuadros; los que
     no, se cuentan y se informan.
  Escribe <trabajo_bm>/boxingweb/{proyectos/, series/<ronda>.npz, golpes.csv, golpes.meta.json}.

USO (en twinboxing_env, con la GPU libre)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/boxingweb.py
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/boxingweb.py --solo-golpes
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import BOXINGWEB, COBERTURA_MIN, LARGO_MAX, LARGO_MIN, TAU_ID, TRABAJO_BM, fuera_del_repo  # noqa: E402

# Reordenamiento de pose_gt.pkl a la indexacion SMPL, como dataset_youtube.py con use_gt_pose.
JOINT_INDEX = (24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7)
JOINT_INDEX_GT = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14)
# Pares (COCO, SMPL) para el emparejado: hombros, codos, caderas.
PARES = ((5, 16), (6, 17), (7, 18), (8, 19), (11, 1), (12, 2))
HOMBROS_SMPL = (16, 17)
MIN_SCORE = 0.3
MIN_PARES = 3

CAMPOS = ("frame_dir", "fuente", "particion", "p", "side", "tipo", "altura", "distancia", "efecto",
          "rol", "fps", "start_frame", "end_frame", "ini", "fin", "cobertura_atacante")


# ---------------------------------------------------------------------------
# Etiquetas
# ---------------------------------------------------------------------------


def etiquetas(ev: dict) -> dict:
    """Un evento de video_event.json -> etiquetas en el vocabulario de BoxTwin (get_label)."""
    tec = ev["technique"]
    resto = tec[1:]
    tipo = "uppercut" if resto == "uppercut" else "hook" if resto == "hook" else "straight"
    tgt = ev["target"]
    altura = "body" if tgt in ("chest", "abdomen") else "head"
    dist = {"close": 2, "medium": 1}.get(ev["distance"], 0)
    return {"tipo": tipo, "altura": altura, "distancia": dist,
            "efecto": 1 if ev["effect"] == "effective" else 0,
            "side": "left" if tec[0] == "l" else "right",
            "p": 0 if ev["side"] == "red" else 1}


def eventos_validos(eventos: list[dict]) -> list[dict]:
    """El filtro de make_dataset: inicio y fin cargados, largo (fin - inicio) entre 4 y 30."""
    salida = []
    for ev in eventos:
        if ev.get("frame_begin", "") == "" or ev.get("frame_end", "") == "":
            continue
        d = int(ev["frame_end"]) - int(ev["frame_begin"])
        if d < LARGO_MIN or d > LARGO_MAX:
            continue
        salida.append(ev)
    return salida


# ---------------------------------------------------------------------------
# Referencia de identidad
# ---------------------------------------------------------------------------


def a_pixeles(p: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Pose normalizada al cuadrado con relleno -> pixeles, como prepare_video_data."""
    lado = max(ancho, alto)
    pad = (lado - np.array([ancho, alto])) // 2
    return np.asarray(p, np.float32)[..., :2] * lado - pad


def referencia(pose: dict, es_gt: bool, ancho: int, alto: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, J, 2) en pixeles para rojo y azul, con la indexacion SMPL que usa el resto."""
    salida = []
    for clave in ("pose_red_2d", "pose_blue_2d"):
        arr = np.array(pose[clave], np.float32, copy=True)
        if es_gt:
            if arr.shape[1] < 25:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 25 - arr.shape[1], arr.shape[2]), np.float32)], 1)
            arr[:, list(JOINT_INDEX)] = arr[:, list(JOINT_INDEX_GT)]
        salida.append(a_pixeles(arr, ancho, alto))
    return salida[0], salida[1]


def distancia(det_kp: np.ndarray, det_sc: np.ndarray, ref: np.ndarray) -> float:
    """Distancia media en pares validos dividida por el ancho de hombros de la referencia."""
    hombros = float(np.linalg.norm(ref[HOMBROS_SMPL[0]] - ref[HOMBROS_SMPL[1]]))
    if not np.isfinite(hombros) or hombros < 1.0:
        return np.inf
    ds = [np.linalg.norm(det_kp[c] - ref[s]) for c, s in PARES
          if det_sc[c] >= MIN_SCORE and np.all(np.isfinite(ref[s]))]
    if len(ds) < MIN_PARES:
        return np.inf
    return float(np.mean(ds)) / hombros


def asignar(costos: np.ndarray, tau: float = TAU_ID) -> tuple[int | None, int | None]:
    """costos (2, n). Indice de deteccion para rojo y azul (distintos), o None si no pasa tau."""
    mejor, elegido = np.inf, (None, None)
    # solo candidatos debajo de tau (suelen ser uno o dos): el producto completo es lento
    cand_a = [i for i in range(costos.shape[1]) if costos[0, i] < tau] + [None]
    cand_b = [i for i in range(costos.shape[1]) if costos[1, i] < tau] + [None]
    for a, b in itertools.product(cand_a, cand_b):
        if a is not None and a == b:
            continue
        ca = costos[0, a] if a is not None else 0.0
        cb = costos[1, b] if b is not None else 0.0
        # se prefiere emparejar a los dos; a igual cantidad, el menor costo total
        total = (2 - (a is not None) - (b is not None)) * 10.0 + ca + cb
        if total < mejor:
            mejor, elegido = total, (a, b)
    return elegido


def emparejar(cache, ref_rojo: np.ndarray, ref_azul: np.ndarray, tau: float = TAU_ID):
    """Cache de pose de BoxTwin + referencias -> kp (2, T, 17, 2), sc (2, T, 17), valid (2, T)."""
    T = min(len(cache), len(ref_rojo), len(ref_azul))
    kp = np.zeros((2, T, 17, 2), np.float32)
    sc = np.zeros((2, T, 17), np.float32)
    valid = np.zeros((2, T), bool)
    for t in range(T):
        d = cache.detections(t)
        if len(d) == 0:
            continue
        costos = np.array([[distancia(d.keypoints[i], d.kp_score[i], ref[t]) for i in range(len(d))]
                           for ref in (ref_rojo, ref_azul)])
        for p, i in enumerate(asignar(costos, tau)):
            if i is not None:
                kp[p, t], sc[p, t], valid[p, t] = d.keypoints[i], d.kp_score[i], True
    return kp, sc, valid


def golpes_de_ronda(eventos: list[dict], ronda: str, particion: str, valid: np.ndarray, fps: float) -> list[dict]:
    T = valid.shape[1]
    salida = []
    for ev in eventos_validos(eventos):
        e = etiquetas(ev)
        ini, fin = int(ev["frame_begin"]), int(ev["frame_end"])
        ini, fin = max(0, ini), min(T - 1, fin)
        if fin < ini:
            continue
        cob = float(valid[e["p"], ini:fin + 1].mean())
        salida.append({"frame_dir": f"bw_{ronda}_{ini}_{fin}_{'rb'[e['p']]}", "fuente": ronda,
                       "particion": particion, "p": e["p"], "side": e["side"], "tipo": e["tipo"],
                       "altura": e["altura"], "distancia": e["distancia"], "efecto": e["efecto"],
                       "rol": "", "fps": float(fps), "start_frame": ini, "end_frame": fin,
                       "ini": ini, "fin": fin, "cobertura_atacante": round(cob, 4)})
    return salida


# ---------------------------------------------------------------------------
# Lectura y escritura
# ---------------------------------------------------------------------------


def rondas(base: Path = None) -> list[tuple[str, str, Path]]:
    """(ronda, particion, carpeta) de data_train y data_test, como make_dataset."""
    base = Path(base or BOXINGWEB)
    salida = []
    for particion, sub in (("train", "data_train"), ("test", "data_test")):
        carpeta = base / sub
        if not carpeta.is_dir():
            raise SystemExit(f"no encuentro {carpeta}: descomprimir BoxingWeb ahi o fijar BOXTWIN_BOXINGWEB")
        for d in sorted(carpeta.iterdir()):
            if d.is_dir() and not d.name.startswith("images"):
                salida.append((d.name, particion, d))
    return salida


def archivos(carpeta: Path) -> dict:
    mp4 = sorted(carpeta.glob("*.mp4"))
    gt = sorted(carpeta.glob("*pose_gt.pkl"))
    uve = sorted(carpeta.glob("*uve.pkl"))
    ev = carpeta / "video_event.json"
    if not mp4 or not ev.is_file() or not (gt or uve):
        raise SystemExit(f"{carpeta}: falta el mp4, video_event.json o el pkl de pose")
    return {"mp4": mp4[0], "pkl": (gt or uve)[0], "es_gt": bool(gt), "eventos": ev}


def leer_csv(origen: Path) -> list[dict]:
    enteros = ("p", "distancia", "efecto", "start_frame", "end_frame", "ini", "fin")
    with Path(origen).open() as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k in enteros:
            f[k] = int(f[k])
        f["fps"] = float(f["fps"])
        f["cobertura_atacante"] = float(f["cobertura_atacante"])
    return filas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solo-golpes", action="store_true", help="no corre el preproceso, usa las series ya escritas")
    args = ap.parse_args(argv)
    destino = fuera_del_repo(TRABAJO_BM / "boxingweb")
    (destino / "series").mkdir(parents=True, exist_ok=True)
    golpes, meta = [], {"rondas": {}, "tau_id": TAU_ID, "cobertura_min": COBERTURA_MIN}
    for ronda, particion, carpeta in rondas():
        a = archivos(carpeta)
        serie = destino / "series" / f"{ronda}.npz"
        if not serie.exists():
            if args.solo_golpes:
                raise SystemExit(f"falta {serie}")
            import cv2
            from boxtwin.core.posecache import PoseCache
            from boxtwin.preprocess.runner import preprocess
            cap = cv2.VideoCapture(str(a["mp4"]))
            ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            r = preprocess(a["mp4"], destino / "proyectos" / ronda)
            with a["pkl"].open("rb") as fh:
                pose = pickle.load(fh)
            rojo, azul = referencia(pose, a["es_gt"], ancho, alto)
            kp, sc, valid = emparejar(PoseCache.open(r.npz_path), rojo, azul)
            np.savez_compressed(serie, kp=kp, sc=sc, valid=valid, fps=fps, ancho=ancho, alto=alto)
        with np.load(serie) as z:
            valid, fps = z["valid"], float(z["fps"])
        gs = golpes_de_ronda(json.loads(a["eventos"].read_text()), ronda, particion, valid, fps)
        usables = [g for g in gs if g["cobertura_atacante"] >= COBERTURA_MIN]
        meta["rondas"][ronda] = {"particion": particion, "pkl_referencia": a["pkl"].name,
                                 "golpes_filtro_largo": len(gs), "usables": len(usables),
                                 "cobertura_roja": round(float(valid[0].mean()), 4),
                                 "cobertura_azul": round(float(valid[1].mean()), 4)}
        golpes.extend(usables)
        print(f"{ronda} ({particion}): {len(usables)}/{len(gs)} golpes usables, "
              f"cobertura roja {valid[0].mean():.3f} azul {valid[1].mean():.3f}", flush=True)
    with (destino / "golpes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(golpes)
    meta["golpes_usables"] = len(golpes)
    meta["descartados_por_identidad"] = sum(v["golpes_filtro_largo"] - v["usables"] for v in meta["rondas"].values())
    (destino / "golpes.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"total: {len(golpes)} golpes usables; descartados por identidad: {meta['descartados_por_identidad']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
