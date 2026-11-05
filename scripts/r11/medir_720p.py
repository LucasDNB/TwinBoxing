"""
BoxTwin - Pose a 720p contra el cache de 1080p (cierre del 4-11-2026).

POR QUE EXISTE
  CLAUDE.md §6.4 pide saber cuanto se pierde si el dispositivo transcodifica antes de subir. La
  regla esta en docs/specs/2026-11-04-cierre/00-preregistro.md §1, escrita antes de medir.

QUE HACE
  Por video de R-11: transcodifica hasta el ultimo cuadro de la lista, corre medir_celda.py con la
  misma lista y yolov8l-pose en PyTorch a 640, lleva cajas y keypoints a escala 1080p y compara con
  el cache de raiz-v8 con regla.concordancia. Toma el candado de GPU y espera las colas. Escribe
  <r11>/720p/resultado.json. Un video ya medido no se repite.

USO
  ~/boxtwin-mejoras/r11/venv/bin/python scripts/r11/medir_720p.py
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "e13")]
import comun_e13 as c  # noqa: E402
import regla  # noqa: E402

R11 = Path("~/boxtwin-mejoras/r11").expanduser()
BASE = R11 / "720p"
RAIZ = Path("~/boxtwin-gemelo/raiz-v8").expanduser()
VIDEOS = {"sparring-3": ("anotacion-sparring-3", "sparring-3-rounds"), "pacquiao": ("anotacion-pacquiao", "pacquiao_margarito")}
ESCALA = 1080 / 720


def transcodificar(lista: dict, destino: Path) -> None:
    if destino.is_file():
        return
    ultimo = max(fin for _, fin in lista["tramos"])
    tmp = destino.with_name("tmp-" + destino.name)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", lista["video"], "-frames:v", str(ultimo + 1), "-vf", "scale=-2:720",
                    "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-an", "-fps_mode", "passthrough", str(tmp)], check=True)
    tmp.rename(destino)


def concordancia(v: str, salida: Path) -> dict:
    dir_, base = VIDEOS[v]
    z = np.load(RAIZ / dir_ / "cache" / f"{base}.pose.npz")
    off, bbox, kp, sc = z["frame_index"], z["bbox"], z["keypoints"], z["kp_score"]
    d = np.load(f"{salida}.npz")
    cajas, puntos, puntajes = d["cajas"] * ESCALA, d["kp"] * ESCALA, d["sc"]
    ini = np.r_[0, np.cumsum(d["n"])]

    def cuadros():
        for i in range(regla.WARMUP, len(d["cuadro"])):
            f = int(d["cuadro"][i])
            a, b, x, y = off[f], off[f + 1], ini[i], ini[i + 1]
            yield bbox[a:b], kp[a:b], sc[a:b], cajas[x:y], puntos[x:y], puntajes[x:y]

    return regla.concordancia(cuadros())


def main() -> int:
    BASE.mkdir(exist_ok=True)
    candado = open(c.CANDADO, "a+")
    res = {}
    for v in VIDEOS:
        lista = json.loads((R11 / "cuadros" / f"{v}.json").read_text())
        video = BASE / f"{v}.mp4"
        transcodificar(lista, video)
        salida = BASE / v
        if not Path(f"{salida}.npz").is_file():
            nueva = BASE / f"{v}.lista.json"
            nueva.write_text(json.dumps({**lista, "video": str(video)}))
            c.esperar_colas()
            fcntl.flock(candado, fcntl.LOCK_EX)
            try:
                subprocess.run([sys.executable, str(AQUI / "medir_celda.py"), "--modelo", str(R11 / "pesos/yolov8l-pose.pt"),
                                "--imgsz", "640", "--formato", "pytorch", "--cuadros", str(nueva), "-o", str(salida)], check=True)
            finally:
                fcntl.flock(candado, fcntl.LOCK_UN)
        res[v] = concordancia(v, salida)
        res[v]["mib"] = video.stat().st_size / 2**20
    r = {"videos": res, "veredicto": regla.veredicto_720p(res)}
    (BASE / "resultado.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
