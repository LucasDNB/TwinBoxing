"""
BoxTwin - Analisis de la estructura y la consistencia de BoxingWeb (24-10-2026).

POR QUE EXISTE
  BoxingVI declaraba 30 fps y anotaciones que resultaron falsas. Antes de sacar conclusiones de
  BoxingWeb (o de citar sus numeros) hay que medir lo mismo que alli: con que base de tiempo
  estan los indices, cuanto dura la ventana anotada, si la mano de la etiqueta es consistente
  con la pose y si la particion separa peleadores. La consistencia de la mano no tiene
  referencia absoluta: se calibra corriendo la misma heuristica sobre BoxTwin, donde el brazo
  lo anoto el autor.

QUE HACE
  1. Estructura: rondas por particion, resolucion y fps (ffprobe), duracion, tipos de evento,
     distribucion de cada etiqueta, largo de las ventanas y anomalias.
  2. Base de tiempo: por ronda, el ultimo fin de evento sobre el largo de la pose; por fps, la
     duracion mediana de la ventana en cuadros y en ms, y la fraccion fuera del filtro de BoxMind.
  3. Consistencia de la mano: en cada golpe, la muneca del atacante que mas se aleja de su
     hombro en la ventana (pose_gt 3D, orden JOINT_INDEX_GT) contra la mano de la etiqueta, por
     tecnica y por fps, con los indices tal cual y duplicados (discriminante de base de tiempo).
     La misma heuristica sobre BoxTwin (pose3d_boxtwin.npz, H36M) como calibracion.
  4. Particion: peleadores de test que tambien estan en train y combates partidos.
  Escribe <trabajo_bm>/analisis_boxingweb.json. No toca los datos.

USO (en twinboxing_env)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/boxmind_prueba/analisis_boxingweb.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm import BOXINGWEB, LARGO_MAX, LARGO_MIN, TRABAJO_BM, fuera_del_repo, geo  # noqa: E402
from boxingweb import rondas  # noqa: E402

# hombro izquierdo, muneca izquierda, hombro derecho, muneca derecha
BRAZOS_GT = (5, 7, 2, 4)        # pose_gt: orden de JOINT_INDEX_GT en dataset_youtube.py
BRAZOS_H36M = (11, 13, 14, 16)  # MotionBERT


def mano_que_extiende(P: np.ndarray, brazos: tuple[int, int, int, int]) -> str | None:
    """'l' o 'r': la muneca que mas gana distancia a su hombro respecto del primer cuadro."""
    if len(P) < 2 or not np.isfinite(P).all():
        return None
    hi, mi, hd, md = brazos
    ei = np.linalg.norm(P[:, mi] - P[:, hi], axis=1)
    ed = np.linalg.norm(P[:, md] - P[:, hd], axis=1)
    return "l" if ei.max() - ei[0] > ed.max() - ed[0] else "r"


def fps_de(mp4: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate,width,height", "-of", "json", str(mp4)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    n, d = map(int, s["r_frame_rate"].split("/"))
    return n / d, int(s["width"]), int(s["height"])


def duracion_s(mp4: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "csv=p=0", str(mp4)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def peleadores(ronda: str) -> tuple[str, str]:
    """'2023-02-10_Imam-Khataev_Arlen-Lopez_1' -> ('Imam-Khataev', 'Arlen-Lopez')"""
    partes = ronda.split("_")
    return partes[1], partes[2]


def combate(ronda: str) -> str:
    return "_".join(ronda.split("_")[:3])


def particion(nombres: dict[str, str]) -> dict:
    """nombres: ronda -> 'train' | 'test'."""
    pel = {"train": set(), "test": set()}
    com = {"train": set(), "test": set()}
    for r, p in nombres.items():
        pel[p].update(peleadores(r))
        com[p].add(combate(r))
    rep = sorted(pel["test"] & pel["train"])
    return {"peleadores": len(pel["train"] | pel["test"]), "peleadores_test": len(pel["test"]),
            "peleadores_test_en_train": len(rep), "lista_test_en_train": rep,
            "combates_partidos": len(com["train"] & com["test"])}


def frac(a: int, n: int) -> dict:
    return {"aciertos": a, "n": n, "fraccion": round(a / n, 4) if n else None}


def analizar(base: Path, info_video=fps_de, duracion=duracion_s) -> dict:
    estructura = collections.defaultdict(collections.Counter)
    largos = collections.defaultdict(list)
    ultimo = collections.defaultdict(list)
    mano = collections.defaultdict(lambda: [0, 0])
    nombres, durs, anomalias = {}, [], collections.Counter()
    for nombre, part, carpeta in rondas(base):
        nombres[nombre] = part
        mp4 = next(carpeta.glob("*.mp4"))
        fps, ancho, alto = info_video(mp4)
        durs.append(duracion(mp4))
        clave_fps = str(round(fps))
        estructura["particion"][part] += 1
        estructura["resolucion"][f"{ancho}x{alto}"] += 1
        estructura["fps"][str(round(fps, 2))] += 1
        eventos = json.loads((carpeta / "video_event.json").read_text())
        with next(carpeta.glob("*_pose_gt.pkl")).open("rb") as fh:
            pose = pickle.load(fh)
        T = len(pose["pose_red_3d"])
        golpes = [e for e in eventos if e.get("name") == "punching"]
        for e in eventos:
            estructura["tipo_evento"][e.get("name", "")] += 1
        fines = []
        for e in golpes:
            for k in ("technique", "target", "distance", "effect", "side"):
                estructura[k][e[k]] += 1
            anomalias["falta"] += bool(e.get("foul"))
            b, f = int(e["frame_begin"]), int(e["frame_end"])
            d = f - b
            if d <= 0:
                anomalias["largo<=0"] += 1
            if f >= T:
                anomalias["fin_despues_del_video"] += 1
                continue
            fines.append(f)
            largos[clave_fps].append((d + 1, LARGO_MIN <= d <= LARGO_MAX, fps))
            if not (LARGO_MIN <= d <= LARGO_MAX) or e["side"] not in ("red", "blue"):
                continue
            P = np.asarray(pose[f"pose_{e['side']}_3d"])
            fam, lab = e["technique"][1:], e["technique"][0]
            for esc, sufijo in ((1, ""), (2, "_x2")):
                pred = mano_que_extiende(P[b * esc:(f + 1) * esc], BRAZOS_GT)
                if pred is None:
                    continue
                for k in (f"{fam}{sufijo}", f"{fam}@{clave_fps}{sufijo}"):
                    mano[k][0] += pred == lab
                    mano[k][1] += 1
        if fines:
            ultimo[clave_fps].append(round(max(fines) / T, 4))
    duracion_ventana = {}
    for k, v in sorted(largos.items(), key=lambda x: float(x[0])):
        cuadros = np.array([x[0] for x in v])
        ms = np.array([x[0] / x[2] * 1000 for x in v])
        duracion_ventana[k] = {"golpes": len(v), "mediana_cuadros": float(np.median(cuadros)),
                               "mediana_ms": round(float(np.median(ms)), 1),
                               "fuera_del_filtro": round(1 - float(np.mean([x[1] for x in v])), 4)}
    d = np.array(durs)
    return {"raiz": str(base),
            "estructura": {k: dict(v.most_common()) for k, v in estructura.items()},
            "duracion_rondas_s": {"min": float(d.min()), "mediana": float(np.median(d)),
                                  "max": float(d.max()), "total_min": round(float(d.sum()) / 60, 1)},
            "anomalias": dict(anomalias),
            "ultimo_evento_sobre_largo": dict(ultimo),
            "duracion_ventana_por_fps": duracion_ventana,
            "mano_boxingweb": {k: frac(*v) for k, v in sorted(mano.items())},
            "particion": particion(nombres)}


def calibrar_boxtwin(pose3d: Path, golpes_csv: Path) -> dict:
    """La misma heuristica sobre BoxTwin: atacante (persona 0), brazo anotado en `side`."""
    if not pose3d.exists() or not golpes_csv.exists():
        return {}
    with golpes_csv.open(encoding="utf-8") as fh:
        g = {r["frame_dir"]: r for r in csv.DictReader(fh)}
    cuenta = collections.defaultdict(lambda: [0, 0])
    with np.load(pose3d, allow_pickle=True) as z:
        fd, pe, largo, P3 = z["frame_dir"], z["persona"], z["largo"], z["P3"]
    for i in np.where(pe == 0)[0]:
        r = g.get(str(fd[i]))
        pred = mano_que_extiende(P3[i, :int(largo[i])], BRAZOS_H36M) if r else None
        if pred is None:
            continue
        cuenta[r["tipo"]][0] += pred == r["side"][0]
        cuenta[r["tipo"]][1] += 1
    return {k: frac(*v) for k, v in sorted(cuenta.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--salida", type=Path, default=TRABAJO_BM / "analisis_boxingweb.json")
    args = ap.parse_args(argv)
    salida = fuera_del_repo(args.salida)
    r = analizar(BOXINGWEB)
    r["mano_boxtwin_calibracion"] = calibrar_boxtwin(TRABAJO_BM / "datos" / "pose3d_boxtwin.npz",
                                                     geo.TRABAJO / "datos" / "golpes.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps({k: r[k] for k in ("duracion_ventana_por_fps", "particion")}, indent=1, ensure_ascii=False))
    for k in ("straight", "hook", "uppercut"):
        print(k, "BoxingWeb", r["mano_boxingweb"].get(k), "BoxTwin", r["mano_boxtwin_calibracion"].get(k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
