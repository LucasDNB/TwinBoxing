"""
BoxTwin - Bajar la resolucion antes de procesar los videos 4K (5-11-2026).

POR QUE EXISTE
  sparr y sparr01, los dos videos nuevos de gimnasio, dieron 8,88 y 4,24 veces su duracion en A. Las
  reglas estan en docs/specs/2026-11-04-cierre/01-transcodificar-4k.md, escritas antes de medir.

QUE HACE
  Por condicion: transcodifica, corre el `procesar` del entorno de A con los modelos de A y escribe
  solo en <cierre>/4k/<video>-<condicion>/. Espera las colas, toma el candado y repite la condicion si
  A o BoxTwinZ tomaron un trabajo mientras corria. Compara la pose con el cache 4K de la sesion de A.
  Escribe <cierre>/4k/resultado.json. Una condicion terminada no se repite.

USO
  python3 scripts/r11/medir_4k.py          # con el python del venv de R-11
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "e13")]
import comun_e13 as c  # noqa: E402
import regla  # noqa: E402

BASE = Path("~/boxtwin-mejoras/cierre/4k").expanduser()
DATOS = Path("~/boxtwin/datos/sesiones").expanduser()
A = Path("~/Proyectos/TwinBoxing").expanduser()
CMD = Path("~/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator").expanduser()
SESIONES = {"sparr01": ("fad18e9f490c4619a6ee2f56ff3575a8", "sparr01.mp4"),
            "sparr": ("dbe27831b1ec4483a1ce73480ef3d242", "sparr.mp4")}
# (video, condicion, alto, crf, fps)
CONDICIONES = [("sparr01", "4k", None, None, None), ("sparr01", "1080", 1080, 20, None), ("sparr01", "720", 720, 23, None),
               ("sparr", "1080", 1080, 20, None), ("sparr", "720", 720, 23, None), ("sparr", "1080-30", 1080, 20, 30)]


def colas() -> tuple:
    import sqlite3

    res = []
    for p in ("~/boxtwin/datos/boxtwin.db", "~/boxtwinz-datos/boxtwinz.db"):
        con = sqlite3.connect(f"file:{Path(p).expanduser()}?mode=ro", uri=True)
        res.append(con.execute("select count(*) from trabajos").fetchone()[0])
        con.close()
    return tuple(res)


def transcodificar(origen: Path, destino: Path, alto: int, crf: int, fps) -> float:
    t0 = time.time()
    filtro = f"scale=-2:{alto}" + (f",fps={fps}" if fps else "")
    tmp = destino.with_name("tmp-" + destino.name)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(origen), "-vf", filtro, "-c:v", "libx264", "-crf", str(crf),
                    "-preset", "veryfast", "-an", str(tmp)], check=True)
    tmp.rename(destino)
    return time.time() - t0


def procesar(video: Path, salida: Path) -> dict:
    # el entorno de A sin PYTHONPATH: el codigo que corre es el de la copia de A, que solo se lee
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([str(CMD), "procesar", str(video), "--out", str(salida), "--modelo-guantes",
                    str(A / "boxtwin-guantes/modelos/guantes-v2.pt"), "--modelo-pose", str(A / "yolov8l-pose.pt"), "--device", "0"],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    ses = json.loads((salida / "sesion.json").read_text())
    return {e["etapa"]: e["segundos"] for e in ses["etapas"]}


def cache_de(d: Path):
    return np.load(next(d.glob("cache/*.pose.npz")))


def concordancia(sesion: Path, salida: Path, escala: float, paso: float) -> dict:
    z, n = cache_de(sesion), cache_de(salida)
    o1, b1, k1, s1 = z["frame_index"], z["bbox"], z["keypoints"], z["kp_score"]
    o2, b2, k2, s2 = n["frame_index"], n["bbox"] * escala, n["keypoints"] * escala, n["kp_score"]

    def cuadros():
        for i in range(len(o2) - 1):
            f = int(round(i * paso))
            if f + 1 >= len(o1):
                break
            a, b, x, y = o1[f], o1[f + 1], o2[i], o2[i + 1]
            yield b1[a:b], k1[a:b], s1[a:b], b2[x:y], k2[x:y], s2[x:y]

    return regla.concordancia(cuadros())


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    candado = open(c.CANDADO, "a+")
    res = {}
    for v, cond, alto, crf, fps in CONDICIONES:
        sid, nombre = SESIONES[v]
        sesion = DATOS / sid
        origen = sesion / "videos" / nombre
        ses = json.loads((sesion / "sesion.json").read_text())
        dur = ses["video"]["duracion_s"]
        clave = f"{v}-{cond}"
        salida = BASE / clave
        hecho = salida / "medida.json"
        if not hecho.is_file():
            salida.mkdir(exist_ok=True)
            video = origen
            t_trans = 0.0
            if alto:
                video = BASE / f"{clave}.mp4"
                if not video.is_file():
                    t_trans = transcodificar(origen, video, alto, crf, fps)
                    (BASE / f"{clave}.trans").write_text(str(t_trans))
                else:
                    t_trans = float((BASE / f"{clave}.trans").read_text())
            while True:
                c.esperar_colas()
                fcntl.flock(candado, fcntl.LOCK_EX)
                antes = colas()
                t0 = time.time()
                try:
                    etapas = procesar(video, salida)
                finally:
                    fcntl.flock(candado, fcntl.LOCK_UN)
                muro = time.time() - t0
                if colas() == antes:
                    break
                c.log(f"4k {clave}: entro un trabajo, se repite")
                for p in salida.iterdir():      # rutas de este directorio, por nombre exacto
                    if p.is_file():
                        p.unlink()
            hecho.write_text(json.dumps({"transcodificar_s": t_trans, "procesar_s": muro, "etapas": etapas}))
            c.log(f"4k {clave}: {t_trans:.0f} s + {muro:.0f} s")
        m = json.loads(hecho.read_text())
        escala = (ses["video"]["alto"] / alto) if alto else 1.0
        paso = (ses["video"]["fps"] / fps) if fps else 1.0
        resto = sum(_resto(sid))
        f = regla.factor_4k(m["transcodificar_s"], m["procesar_s"], resto, dur)
        conc = concordancia(sesion, salida, escala, paso)
        res[clave] = {**m, "duracion_s": dur, "resto_s": resto, "factor": f, "concordancia": conc,
                      "veredicto": regla.veredicto_4k(f["con_transcodificado"], conc)}
    e = res["sparr01-4k"]["etapas"]
    res["k_4k"] = {"offline": e["preproceso"] + e["evidencia"], "a": 742.4,
                   "pasa": abs(e["preproceso"] + e["evidencia"] - 742.4) / 742.4 < 0.10}
    (BASE / "resultado.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("factor"), v.get("veredicto")) if "factor" in v else v for k, v in res.items()}, indent=1))
    return 0


def _resto(sid: str) -> list[float]:
    import sqlite3
    from datetime import datetime

    con = sqlite3.connect(f"file:{Path('~/boxtwin/datos/boxtwin.db').expanduser()}?mode=ro", uri=True)
    filas = con.execute("select tomado_en, terminado_en from trabajos where sesion_id=? and intento=1 and etapa in "
                        "('completar','clasificar')", (sid,)).fetchall()
    return [(datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() for a, b in filas]


if __name__ == "__main__":
    sys.exit(main())
