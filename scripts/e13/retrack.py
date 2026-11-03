"""
BoxTwin - E13: re-trackea las detecciones guardadas, offline (3-11-2026).

POR QUE EXISTE
  E13 cambia una sola variable, el algoritmo de asociacion, sobre las mismas detecciones
  (docs/specs/2026-10-30-mejoras-post-e8/06-e13-tracker.md §2). H13-2 prueba que el camino offline
  es el mismo tracker antes de comparar nada.

QUE HACE
  Lee <e13>/det/<fuente>.npz (detectar.py) y escribe un cache con el formato de PoseCache en
  <e13>/<tracker>/<fuente>.pose.npz. Reinicia el tracker en los cortes del meta de raiz-v8 y numera
  los ids como writer.cut_seam.
  - base: BOTSORT de Ultralytics con botsort-sin-reid.yaml, frame_rate=30 como lo arma track.py.
    Usa los cuadros del video porque el GMC (sparseOptFlow) los necesita.
  - ocsort, bytetrack: implementaciones oficiales, importadas desde --terceros, con sus defaults
    salvo el buffer equivalente a 4 s (round(4 * fps) cuadros).
  Las oficiales no devuelven el indice de la deteccion: los keypoints se recuperan emparejando la
  caja de salida con las detecciones del cuadro por IoU, la mayor primero (emparejar_salida).

  H13-2, fijado antes de correr: para cada fuente, comparar_ids contra el cache de raiz-v8, misma
  lista de track_id en el mismo orden en cada cuadro. Vale 100% en las siete o no vale.

USO
  python scripts/e13/retrack.py base --fuente 01-sparring
  python scripts/e13/retrack.py h13-2
  python scripts/e13/retrack.py ocsort --terceros ~/boxtwin-mejoras/e13/terceros
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
import comun_e13 as c  # noqa: E402


def emparejar_salida(cajas: np.ndarray, dets: np.ndarray, minimo: float = 0.3) -> list[int]:
    """Por cada caja de salida, el indice de su deteccion, o -1. Sin repetir, la mayor IoU primero."""
    if not len(cajas) or not len(dets):
        return [-1] * len(cajas)
    x0 = np.maximum(cajas[:, None, 0], dets[None, :, 0])
    y0 = np.maximum(cajas[:, None, 1], dets[None, :, 1])
    x1 = np.minimum(cajas[:, None, 2], dets[None, :, 2])
    y1 = np.minimum(cajas[:, None, 3], dets[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    iou = inter / np.maximum(area(cajas)[:, None] + area(dets)[None, :] - inter, 1e-9)
    salida = [-1] * len(cajas)
    usados: set[int] = set()
    for k in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(k, iou.shape)
        if iou[i, j] < minimo:
            break
        if salida[i] == -1 and j not in usados:
            salida[i] = int(j)
            usados.add(int(j))
    return salida


class Base:
    necesita_cuadros = True

    def __init__(self, fuente: str, _terceros):
        import yaml
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace

        ruta = REPO / "boxtwin-annotator/src/boxtwin/configs/botsort-sin-reid.yaml"
        texto = ruta.read_bytes()
        import hashlib

        if hashlib.sha256(texto).hexdigest() != c.meta_de(fuente)["inference"]["tracker_config_sha256"]:
            raise SystemExit("el yaml del tracker no es el del cache")
        self.t = BOTSORT(args=IterableSimpleNamespace(**yaml.safe_load(texto)), frame_rate=30)

    def reiniciar(self):
        self.t.reset()

    def paso(self, xyxy, conf, cls, cuadro, forma):
        from ultralytics.engine.results import Boxes

        d = Boxes(np.c_[xyxy, conf, cls].astype(np.float32), forma)
        t = self.t.update(d, cuadro, None)
        if len(t) == 0:
            return np.zeros(0, np.int64), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        return t[:, 4].astype(np.int64), t[:, :4].astype(np.float32), t[:, 5].astype(np.float32), \
            t[:, -1].astype(int).tolist()


class OCSort:
    necesita_cuadros = False

    def __init__(self, fuente: str, terceros: Path):
        sys.path.insert(0, str(terceros / "OC_SORT"))
        from trackers.ocsort_tracker.ocsort import KalmanBoxTracker, OCSort as T

        self.K, self.T = KalmanBoxTracker, T
        self.fps = c.meta_de(fuente)["video"]["fps"]
        self.reiniciar()

    def reiniciar(self):
        self.K.count = 0
        # defaults de tools/demo_track.py: track_thresh 0.6, iou 0.3, min_hits 3, inertia 0.2, deltat 3
        self.t = self.T(det_thresh=0.6, max_age=round(c.BUFFER_S * self.fps), min_hits=3, iou_threshold=0.3,
                        delta_t=3, asso_func="iou", inertia=0.2, use_byte=False)

    def paso(self, xyxy, conf, cls, _cuadro, forma):
        h, w = forma
        o = self.t.update(np.c_[xyxy, conf].astype(np.float64), (h, w), (h, w))
        if len(o) == 0:
            return np.zeros(0, np.int64), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        cajas = o[:, :4].astype(np.float32)
        idx = emparejar_salida(cajas, xyxy)
        return o[:, 4].astype(np.int64), cajas, np.array([conf[i] if i >= 0 else 0 for i in idx], np.float32), idx


class ByteTrack:
    necesita_cuadros = False

    def __init__(self, fuente: str, terceros: Path):
        # La oficial es de 2021: usa np.float y cython_bbox. Los dos se reemplazan sin tocar su codigo.
        import types

        if "cython_bbox" not in sys.modules:
            m = types.ModuleType("cython_bbox")

            def bbox_overlaps(a, b):
                a, b = np.asarray(a, float), np.asarray(b, float)
                x0 = np.maximum(a[:, None, 0], b[None, :, 0])
                y0 = np.maximum(a[:, None, 1], b[None, :, 1])
                x1 = np.minimum(a[:, None, 2], b[None, :, 2])
                y1 = np.minimum(a[:, None, 3], b[None, :, 3])
                # cython_bbox cuenta el +1 de VOC; se conserva
                inter = np.clip(x1 - x0 + 1, 0, None) * np.clip(y1 - y0 + 1, 0, None)
                area = lambda z: (z[:, 2] - z[:, 0] + 1) * (z[:, 3] - z[:, 1] + 1)  # noqa: E731
                return inter / (area(a)[:, None] + area(b)[None, :] - inter)

            m.bbox_overlaps = bbox_overlaps
            sys.modules["cython_bbox"] = m
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        # solo el subpaquete del tracker: yolox/__init__ arrastra dependencias que no hacen falta
        raiz = terceros / "ByteTrack"
        pkg = types.ModuleType("yolox")
        pkg.__path__ = [str(raiz / "yolox")]
        sys.modules.setdefault("yolox", pkg)
        from yolox.tracker.basetrack import BaseTrack
        from yolox.tracker.byte_tracker import BYTETracker

        self.B, self.T = BaseTrack, BYTETracker
        self.fps = c.meta_de(fuente)["video"]["fps"]
        self.reiniciar()

    def reiniciar(self):
        self.B._count = 0
        # defaults de tools/demo_track.py: track_thresh 0.5, match_thresh 0.8, mot20 False.
        # buffer_size = frame_rate / 30 * track_buffer: con frame_rate = 30, track_buffer son cuadros.
        a = argparse.Namespace(track_thresh=0.5, track_buffer=round(c.BUFFER_S * self.fps),
                               match_thresh=0.8, mot20=False)
        self.t = self.T(a, frame_rate=30)

    def paso(self, xyxy, conf, cls, _cuadro, forma):
        h, w = forma
        o = self.t.update(np.c_[xyxy, conf].astype(np.float32), (h, w), (h, w))
        if not o:
            return np.zeros(0, np.int64), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        cajas = np.array([t.tlbr for t in o], np.float32)
        idx = emparejar_salida(cajas, xyxy)
        return np.array([t.track_id for t in o], np.int64), cajas, np.array([t.score for t in o], np.float32), idx


TRACKERS = {"base": Base, "ocsort": OCSort, "bytetrack": ByteTrack}


def retrackear(nombre: str, fuente: str, terceros: Path | None) -> dict:
    destino = c.E13 / nombre / f"{fuente}.pose.npz"
    if destino.is_file():
        return json.loads(destino.with_name(f"{fuente}.info.json").read_text())
    import cv2

    meta = c.meta_de(fuente)
    z = np.load(c.DET / f"{fuente}.npz")
    off, xyxy, conf, cls, kp, sc = (z[k] for k in ("frame_index", "xyxy", "conf", "cls", "keypoints", "kp_score"))
    n = len(off) - 1
    forma = (meta["video"]["height"], meta["video"]["width"])
    tr = TRACKERS[nombre](fuente, terceros)
    cap = cv2.VideoCapture(str(c.video_de(fuente))) if tr.necesita_cuadros else None
    cortes = set(meta["scene_cuts"]["frames"])
    ids = c.Ids()
    o_off, o_id, o_box, o_conf, o_kp, o_sc = [0], [], [], [], [], []
    sin_det, t0 = 0, time.time()
    for f in range(n):
        cuadro = None
        if cap is not None:
            ok, cuadro = cap.read()
            if not ok:
                raise SystemExit(f"{fuente}: no decodifica {f}")
        if f in cortes:
            tr.reiniciar()
            ids.corte()
        a, b = off[f], off[f + 1]
        local, cajas, puntaje, idx = tr.paso(xyxy[a:b], conf[a:b], cls[a:b], cuadro, forma)
        idx = np.array(idx, int)
        if len(idx) and (idx < 0).any():
            sin_det += int((idx < 0).sum())
            bien = idx >= 0
            local, cajas, puntaje, idx = local[bien], cajas[bien], puntaje[bien], idx[bien]
        o_off.append(o_off[-1] + len(local))
        if len(local):
            o_id.append(ids.global_(local).astype(np.int32))
            o_box.append(cajas)
            o_conf.append(puntaje)
            o_kp.append(kp[a:b][idx])
            o_sc.append(sc[a:b][idx])
    if cap is not None:
        cap.release()
    base = np.load(c.cache_de(fuente))
    nuevo_meta = json.loads(json.dumps(meta))
    nuevo_meta["e13"] = {"tracker": nombre, "desde": "detecciones guardadas"}
    nuevo_meta["counts"]["n_detections"] = int(o_off[-1])
    nuevo_meta["max_track_id"] = ids.maximo
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(f"{fuente}.tmp.npz")
    np.savez_compressed(
        tmp, format_version=base["format_version"], meta_json=np.array(json.dumps(nuevo_meta)),
        frame_index=np.array(o_off, np.int64), frame_status=base["frame_status"],
        track_id=np.concatenate(o_id), bbox=np.concatenate(o_box), det_conf=np.concatenate(o_conf),
        keypoints=np.concatenate(o_kp), kp_score=np.concatenate(o_sc))
    destino.with_name(f"{fuente}.meta.json").write_text(json.dumps(nuevo_meta))
    info = {"tracker": nombre, "fuente": fuente, "cuadros": n, "salidas": int(o_off[-1]),
            "salidas_sin_deteccion": sin_det, "tracks": len(set(np.concatenate(o_id).tolist())),
            "segundos": round(time.time() - t0, 1)}
    destino.with_name(f"{fuente}.info.json").write_text(json.dumps(info))
    tmp.rename(destino)
    c.log(f"retrack {info}")
    return info


def h13_2(fuentes: list[str]) -> dict:
    salida = {}
    for f in fuentes:
        a = np.load(c.cache_de(f))
        b = np.load(c.E13 / "base" / f"{f}.pose.npz")
        salida[f] = c.comparar_ids(a["frame_index"], a["track_id"], b["frame_index"], b["track_id"])
    salida_ok = all(v["distintos"] == 0 and v["mismo_largo"] for v in salida.values()) and len(salida) == len(c.ORDEN)
    doc = {"fuentes": salida, "pasa": salida_ok}
    (c.E13 / "h13-2.json").write_text(json.dumps(doc, indent=1))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=[*TRACKERS, "h13-2"])
    ap.add_argument("--fuente", nargs="*", default=None)
    ap.add_argument("--terceros", type=Path, default=c.E13 / "terceros")
    a = ap.parse_args(argv)
    fuentes = a.fuente or c.ORDEN
    if a.accion == "h13-2":
        d = h13_2(fuentes)
        print(json.dumps(d, indent=1))
        return 0 if d["pasa"] else 1
    for f in fuentes:
        if (c.DET / f"{f}.npz").is_file():
            print(retrackear(a.accion, f, a.terceros.expanduser()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
