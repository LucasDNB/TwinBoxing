"""
BoxTwin - Humo H1 a H3 de YOLO26l-pose antes de cachear nada (E8, S6, 29-10-2026).

POR QUE EXISTE
  E8 cambia una sola cosa, el checkpoint de pose, y da por hecho tres cosas que nadie midio
  con YOLO26: que devuelve los 17 keypoints de COCO en el mismo orden que v8 (si no, cambiar
  COCO17_NAMES seria un segundo cambio), que el tracker con ReID `model: auto` funciona sobre
  sus salidas y se reinicia en los cortes de plano, y cuanto tarda y cuanta memoria usa en la
  2080 (P12e). Si alguna falla, los catorce caches de S7 serian una hora y tres cuartos de GPU
  tirada (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §3). T11, decision 17.

QUE HACE
  H1, carga y orden: kpt_shape [17, 3] y 17 keypoints por persona en GPU y en CPU; sobre 50
     cuadros equiespaciados de 01-sparring, las personas que detectan las dos poses con IoU
     >= 0,5 y la mediana de la distancia keypoint a keypoint con el indice propio contra las
     255 permutaciones izquierda-derecha. Pasa si la propia es menor que todas.
  H2, tracking: el camino de inferencia del preproceso (_inferir y _reiniciar_tracker del
     runner, con el yaml dado) sobre los primeros 600 cuadros de 01-sparring y sobre el tramo
     de pacquiao_margarito con sus dos primeros cortes de plano, reiniciando en cada uno.
     Pasa si termina sin error y hubo dos reinicios.
  H3, tiempo y memoria: 1000 cuadros de sparring-3-rounds a 1080p despues de 100 de
     calentamiento, con la GPU sin otros procesos: ms por cuadro de pose y tracking (mediana y
     p95, solo la inferencia, sin decodificar) y pico de memoria reservada por torch, para las
     dos poses. Sin umbral: siempre pasa si corre.
  Lee los videos originales sin copiarlos y escribe solo ~/boxtwin-gemelo/humo/humo.json.
  Sale con 1 si H1 o H2 fallan: E8 se frena (H1) o entra la rama D-H2 (H2), y eso lo decide
  Lucas, no este script.

USO
  python scripts/gemelo/humo.py
  python scripts/gemelo/humo.py --tracker boxtwin-annotator/src/boxtwin/configs/botsort-sin-reid.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "clasificador"))
import guardas  # noqa: E402

from boxtwin.core.constants import COCO17_NAMES  # noqa: E402

UMBRAL_IOU = 0.5
CUADROS_ORDEN = 50
CUADROS_H2 = 600
MARGEN_CORTES = 60
CALENTAMIENTO = 100
MEDIDOS = 1000

# Las ocho parejas izquierda-derecha de COCO-17, por nombre: (1, 2) ojos ... (15, 16) tobillos.
PAREJAS_LR: list[tuple[int, int]] = [
    (i, COCO17_NAMES.index("right_" + n[len("left_"):]))
    for i, n in enumerate(COCO17_NAMES) if n.startswith("left_")
]


def permutaciones_lr() -> list[list[int]]:
    """Todas las formas de cambiar izquierda por derecha en un subconjunto de las parejas."""
    salida = []
    for elegidas in itertools.product((False, True), repeat=len(PAREJAS_LR)):
        if not any(elegidas):
            continue
        p = list(range(len(COCO17_NAMES)))
        for (i, j), cambia in zip(PAREJAS_LR, elegidas):
            if cambia:
                p[i], p[j] = j, i
        salida.append(p)
    return salida


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def emparejar_personas(cajas_a, cajas_b, umbral: float = UMBRAL_IOU) -> list[tuple[int, int]]:
    """Pares (i, j) de mayor IoU primero, cada caja una sola vez, con IoU >= umbral."""
    candidatos = sorted(((_iou(a, b), i, j) for i, a in enumerate(cajas_a) for j, b in enumerate(cajas_b)),
                        reverse=True)
    usados_a, usados_b, pares = set(), set(), []
    for v, i, j in candidatos:
        if v < umbral:
            break
        if i not in usados_a and j not in usados_b:
            usados_a.add(i)
            usados_b.add(j)
            pares.append((i, j))
    return pares


def veredicto_orden(pares: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    pares = [(keypoints de v8, keypoints de YOLO26)] de la misma persona, (17, 2) cada uno.
    Mediana sobre todos los keypoints de todos los pares, con el indice propio y con cada
    permutacion aplicada a YOLO26.
    """
    if not pares:
        return {"pares": 0, "pasa": False, "mediana_propia": None,
                "mediana_mejor_permutacion": None, "mejor_permutacion": None}
    a = np.stack([p[0] for p in pares]).astype(np.float64)
    b = np.stack([p[1] for p in pares]).astype(np.float64)
    propia = float(np.median(np.linalg.norm(a - b, axis=-1)))
    otras = [(float(np.median(np.linalg.norm(a - b[:, p], axis=-1))), p) for p in permutaciones_lr()]
    mejor, perm = min(otras, key=lambda x: x[0])
    return {"pares": len(pares), "mediana_propia": propia, "mediana_mejor_permutacion": mejor,
            "mejor_permutacion": perm, "pasa": propia < mejor}


def estadisticas_tiempo(ms: list[float], calentamiento: int = CALENTAMIENTO) -> dict:
    medidos = np.asarray(ms[calentamiento:], np.float64)
    if len(ms) <= calentamiento or not len(medidos):
        raise ValueError(f"hacen falta mas de {calentamiento} cuadros, hay {len(ms)}")
    return {"n": int(len(medidos)), "calentamiento": calentamiento,
            "mediana_ms": float(np.median(medidos)), "p95_ms": float(np.percentile(medidos, 95))}


def tramo_de_cortes(cortes: list[int], margen: int = MARGEN_CORTES) -> tuple[int, int, list[int]]:
    """De `margen` cuadros antes del primer corte a `margen` despues del segundo."""
    if len(cortes) < 2:
        raise ValueError("hacen falta dos cortes de plano")
    c = sorted(cortes)[:2]
    return max(0, c[0] - margen), c[1] + margen, c


def cuadros_equiespaciados(total: int, n: int = CUADROS_ORDEN) -> list[int]:
    return sorted({int(x) for x in np.linspace(0, total - 1, n)})


def codigo_de_salida(res: dict) -> int:
    return 0 if res["H1"]["pasa"] and res["H2"]["pasa"] else 1


# ---------------------------------------------------------------------------
# Lo que corre en la estacion
# ---------------------------------------------------------------------------


def _video(raiz: Path, dir_: str, fuente: str) -> tuple[Path, dict]:
    meta = json.loads((raiz / dir_ / "cache" / f"{fuente}.meta.json").read_text())
    v = Path(meta["video"]["path"])
    return (v if v.is_file() else raiz / dir_ / "videos" / v.name), meta


def _leer(cap, f: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"no se pudo leer el cuadro {f}")
    return frame


def h1(modelo_v8: Path, modelo_y26: Path, video: Path, device: str) -> dict:
    import cv2
    from ultralytics import YOLO

    y26 = YOLO(str(modelo_y26))
    forma = list(y26.model.yaml.get("kpt_shape") or [])
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame0 = _leer(cap, 0)
    por_disp = {}
    for disp in (device, "cpu"):
        r = y26.predict(frame0, imgsz=640, conf=0.25, iou=0.7, half=False, device=disp, verbose=False)[0]
        por_disp[disp] = {"personas": int(len(r.boxes)),
                          "keypoints": list(r.keypoints.data.shape[1:]) if r.keypoints is not None else None}
    v8 = YOLO(str(modelo_v8))
    pares = []
    for f in cuadros_equiespaciados(total, CUADROS_ORDEN):
        frame = _leer(cap, f)
        ra, rb = (m.predict(frame, imgsz=640, conf=0.25, iou=0.7, half=False, device=device, verbose=False)[0]
                  for m in (v8, y26))
        if ra.keypoints is None or rb.keypoints is None:
            continue
        for i, j in emparejar_personas(ra.boxes.xyxy.cpu().numpy(), rb.boxes.xyxy.cpu().numpy()):
            pares.append((ra.keypoints.xy[i].cpu().numpy(), rb.keypoints.xy[j].cpu().numpy()))
    cap.release()
    orden = veredicto_orden(pares)
    carga = forma == [17, 3] and all(d["personas"] > 0 and d["keypoints"] == [17, 3] for d in por_disp.values())
    return {"kpt_shape": forma, "prueba": por_disp, "orden": orden, "pasa": bool(carga and orden["pasa"])}


def _recorrer(modelo_path: Path, tracker: Path, device: str, video: Path, inicio: int, fin: int,
              cortes: list[int], cronometrar: bool = False) -> dict:
    """El bucle del preproceso sobre [inicio, fin), con sus mismas funciones."""
    import cv2
    import torch
    from boxtwin.preprocess.runner import PreprocessConfig, _cargar_modelo, _inferir, _reiniciar_tracker

    cfg = PreprocessConfig(model=str(modelo_path), tracker=Path(tracker), device=device)
    modelo = _cargar_modelo(modelo_path, cfg)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    pendientes, reinicios, ms = sorted(cortes), 0, []
    if cronometrar and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for f in range(inicio, fin):
        ok, frame = cap.read()
        if not ok:
            break
        if pendientes and f >= pendientes[0]:
            _reiniciar_tracker(modelo)
            reinicios += 1
            pendientes.pop(0)
        if cronometrar and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _inferir(modelo, frame, cfg)
        if cronometrar and torch.cuda.is_available():
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    cap.release()
    pico = torch.cuda.max_memory_reserved() / 2 ** 20 if cronometrar and torch.cuda.is_available() else None
    del modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"cuadros": len(ms), "reinicios": reinicios, "ms": ms, "pico_mib": pico}


def h2(modelo_y26: Path, tracker: Path, device: str, sparring: Path, pacquiao: Path, cortes: list[int]) -> dict:
    salida = {"tracker": str(tracker)}
    try:
        r = _recorrer(modelo_y26, tracker, device, sparring, 0, CUADROS_H2, [])
        salida["sparring"] = {"cuadros": r["cuadros"]}
        a, b, c = tramo_de_cortes(cortes)
        r = _recorrer(modelo_y26, tracker, device, pacquiao, a, b, c)
        salida["pacquiao"] = {"tramo": [a, b], "cortes": c, "cuadros": r["cuadros"], "reinicios": r["reinicios"]}
        salida["pasa"] = salida["sparring"]["cuadros"] == CUADROS_H2 and r["reinicios"] == 2
    except Exception as e:                      # el error es el resultado: va al json
        salida.update({"pasa": False, "error": f"{type(e).__name__}: {e}"})
    return salida


def h3(modelos: dict[str, Path], tracker: Path, device: str, video: Path) -> dict:
    salida = {}
    for nombre, ruta in modelos.items():
        otros = guardas.pids_en_gpu(guardas.salida_nvidia_smi())
        if otros:
            return {"pasa": False, "error": f"la GPU tiene otros procesos: {otros}"}
        r = _recorrer(ruta, tracker, device, video, 0, CALENTAMIENTO + MEDIDOS, [], cronometrar=True)
        salida[nombre] = {**estadisticas_tiempo(r["ms"]), "pico_mib": r["pico_mib"]}
    salida["pasa"] = True
    return salida


def main(argv=None) -> int:
    import comun

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modelo", type=Path, default=guardas.GEMELO / "modelos" / "yolo26l-pose.pt")
    ap.add_argument("--modelo-v8", type=Path, default=guardas.REPO_A / "yolov8l-pose.pt")
    ap.add_argument("--tracker", type=Path,
                    default=guardas.REPO / "boxtwin-annotator" / "src" / "boxtwin" / "configs" / "botsort.yaml")
    ap.add_argument("--raiz-original", type=Path, default=guardas.REPO_A)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=guardas.GEMELO / "humo" / "humo.json")
    args = ap.parse_args(argv)
    out = guardas.exigir_out(args.out)

    spar, _ = _video(args.raiz_original, comun.FUENTES["01-sparring"], "01-sparring")
    pac, meta_pac = _video(args.raiz_original, comun.FUENTES["pacquiao_margarito"], "pacquiao_margarito")
    s3, _ = _video(args.raiz_original, comun.FUENTES["sparring-3-rounds"], "sparring-3-rounds")
    res = {"commit": __import__("subprocess").run(["git", "-C", str(guardas.REPO), "rev-parse", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip()}
    res["H1"] = h1(args.modelo_v8, args.modelo, spar, args.device)
    print(f"H1 {'pasa' if res['H1']['pasa'] else 'FALLA'}: {res['H1']['orden']}", flush=True)
    res["H2"] = h2(args.modelo, args.tracker, args.device, spar, pac, meta_pac["scene_cuts"]["frames"])
    print(f"H2 {'pasa' if res['H2']['pasa'] else 'FALLA'}: {res['H2']}", flush=True)
    res["H3"] = h3({"v8": args.modelo_v8, "y26": args.modelo}, args.tracker, args.device, s3)
    print(f"H3: {res['H3']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    return codigo_de_salida(res)


if __name__ == "__main__":
    sys.exit(main())
