"""
BoxTwin - Re-cache de las siete fuentes con YOLO26l-pose (E8, S7, 27-10-2026).

POR QUE EXISTE
  E8 cambia una sola cosa, el checkpoint de pose, y necesita los catorce caches: los siete
  originales de v8 y siete nuevos con YOLO26 (docs/specs/2026-10-27-gemelo-yolo26,
  01-percepcion.md §4). El preproceso ya piso una anotacion dos veces el 19-08 armando pruebas
  aisladas, porque infiere el proyecto del video y un enlace lo llevaba al original. Este
  script es la unica forma en que la fase 1 llama al preproceso.

QUE HACE
  Por fuente: arma ~/boxtwin-gemelo/raiz-y26/<dir>/ con videos/<video> como enlace simbolico
  al original (solo lectura), se niega si el proyecto no esta bajo el gemelo o si ya tiene
  una anotacion que no escribio el puente, y corre `boxtwin-annotator preprocess` con
  --project explicito y los parametros de A (imgsz 640, conf 0,25, iou 0,7, sin half, cortes
  0,3, el botsort.yaml que se pase). Una fuente con cache completo se saltea; el preproceso
  reanuda por shards. Deja tiempo, cuadros y fps de cada fuente en recache.json.

USO
  python scripts/gemelo/recache.py --modelo ~/boxtwin-gemelo/modelos/yolo26l-pose.pt
  python scripts/gemelo/recache.py --fuentes 01-sparring --tracker boxtwin-annotator/src/boxtwin/configs/botsort-sin-reid.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "clasificador"))
import comun  # noqa: E402
import guardas  # noqa: E402

TRACKER = guardas.REPO / "boxtwin-annotator" / "src" / "boxtwin" / "configs" / "botsort.yaml"
CMD = os.environ.get("BOXTWIN_CMD", "boxtwin-annotator")


def comando(video: Path, proyecto: Path, modelo: Path, tracker: Path = TRACKER, device: str = "0") -> list[str]:
    """El preproceso de A con un solo cambio, el checkpoint. Siempre con --project."""
    return [CMD, "preprocess", str(video), "--project", str(proyecto), "--model", str(modelo),
            "--imgsz", "640", "--conf", "0.25", "--iou", "0.7", "--device", str(device),
            "--tracker", str(tracker), "--cut-threshold", "0.3", "--no-proxy"]


def _video_original(fuente: str, raiz_original: Path) -> Path:
    dir_ = raiz_original / comun.FUENTES[fuente]
    ann = dir_ / "annotations" / f"{fuente}.annot.json"
    v = Path(json.loads(ann.read_text())["video"]["path"])
    if not v.is_file():                         # el proyecto se pudo haber movido
        v = dir_ / "videos" / v.name
    return v


def preparar(fuente: str, raiz_y26: Path, raiz_original: Path) -> tuple[Path, Path]:
    """El proyecto de la fuente bajo el gemelo y el enlace al video. No escribe en A."""
    proyecto = guardas.exigir_fuera_del_repo(guardas.exigir_bajo_gemelo(raiz_y26 / comun.FUENTES[fuente]))
    guardas.exigir_annot_del_puente(proyecto / "annotations" / f"{fuente}.annot.json")
    original = _video_original(fuente, raiz_original)
    enlace = proyecto / "videos" / original.name
    enlace.parent.mkdir(parents=True, exist_ok=True)
    if not enlace.is_symlink():
        enlace.symlink_to(original)
    elif Path(os.readlink(enlace)) != original:
        raise guardas.GuardaError(f"{enlace} ya apunta a otro video: {os.readlink(enlace)}")
    return proyecto, enlace


def _hecho(proyecto: Path, enlace: Path) -> dict | None:
    stem = enlace.stem
    meta, npz = proyecto / "cache" / f"{stem}.meta.json", proyecto / "cache" / f"{stem}.pose.npz"
    return json.loads(meta.read_text()) if meta.is_file() and npz.is_file() else None


def correr(fuentes, raiz_y26: Path, raiz_original: Path, modelo: Path, registro: Path,
           tracker: Path = TRACKER, device: str = "0", ejecutar=None, reloj=time.monotonic) -> dict:
    ejecutar = ejecutar or (lambda cmd: subprocess.run(cmd, check=True))
    registro = guardas.exigir_bajo_gemelo(registro)
    hecho = json.loads(registro.read_text()) if registro.is_file() else {}
    for fuente in fuentes:
        proyecto, enlace = preparar(fuente, raiz_y26, raiz_original)
        if fuente in hecho and _hecho(proyecto, enlace):
            print(f"{fuente}: ya cacheada, se saltea", flush=True)
            continue
        t0 = reloj()
        ejecutar(comando(enlace, proyecto, modelo, tracker, device))
        seg = reloj() - t0
        meta = _hecho(proyecto, enlace) or {}
        cuadros = int(meta.get("video", {}).get("total_frames", 0))
        hecho[fuente] = {"segundos": round(seg, 1), "cuadros": cuadros,
                         "fps": round(cuadros / seg, 2) if seg > 0 else None,
                         "model_sha256": meta.get("inference", {}).get("model_sha256"),
                         "tracker_config_sha256": meta.get("inference", {}).get("tracker_config_sha256")}
        registro.parent.mkdir(parents=True, exist_ok=True)
        registro.write_text(json.dumps(hecho, indent=2) + "\n")
        print(f"{fuente}: {cuadros} cuadros en {seg:.0f} s", flush=True)
    return hecho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=list(comun.FUENTES), choices=list(comun.FUENTES))
    ap.add_argument("--modelo", type=Path, default=guardas.GEMELO / "modelos" / "yolo26l-pose.pt")
    ap.add_argument("--raiz-y26", type=Path, default=guardas.GEMELO / "raiz-y26")
    ap.add_argument("--raiz-original", type=Path, default=guardas.REPO_A)
    ap.add_argument("--tracker", type=Path, default=TRACKER)
    ap.add_argument("--device", default="0")
    ap.add_argument("--registro", type=Path, default=guardas.GEMELO / "e8" / "recache.json")
    args = ap.parse_args(argv)
    correr(args.fuentes, args.raiz_y26, args.raiz_original, args.modelo, args.registro,
           args.tracker, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
