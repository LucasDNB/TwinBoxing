"""
BoxTwin - M6 del modo academico E7: cuanto agrega E7 a la etapa clasificar, contra E4.

POR QUE EXISTE
  La enmienda del 25-10 (docs/specs/2026-10-25-e7-modo-academico/00-enmienda.md, condicion 4)
  no deja encender el modo sin medir su tiempo contra C5 (reloj hasta la Fight-Card, umbral
  2x). E7 suma una elevacion con HMR2 por golpe, que en E7-3 costo unos 2,2 s por golpe. Hay
  que saber cuanto es eso en una sesion real, en veces la duracion del video.

QUE HACE
  Por sesion: copia segmentos.json, poses.npz, sesion.json y fightcard.json a un directorio de
  trabajo (la sesion original no se toca; el video se lee de su ruta), y cronometra con los
  comandos que arma el worker:
    E4: boxtwin clasificar --modelo clasificador_geo_v1 y boxtwin tipos.
    E7: tools/elevar_hmr2_sesion.py en boxtwin_4dhumans, boxtwin clasificar --modelo
        clasificador_e7_v1 --pose3d y boxtwin tipos.
  Escribe <salida>/tiempo_e7.json: segundos por comando, total por modo, duracion del video y
  el agregado de E7 sobre E4 en veces la duracion del video. No decide nada: el numero va a la
  enmienda y a la Fight-Card si corresponde.

USO (en twinboxing_env, con la GPU libre)
  ~/miniforge3/envs/twinboxing_env/bin/python scripts/clasificador_geo/medir_tiempo_e7.py <sesion> [<sesion> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = Path("~/miniforge3/envs/twinboxing_env/bin/python").expanduser()
PY_4DH = Path("~/miniforge3/envs/boxtwin_4dhumans/bin/python").expanduser()
MODELO_E4 = Path("~/boxtwin/modelos/clasificador_geo_v1").expanduser()
MODELO_E7 = Path("~/boxtwin/modelos/clasificador_e7_v1").expanduser()
ARCHIVOS = ("segmentos.json", "poses.npz", "sesion.json", "fightcard.json")


def boxtwin(*args: str) -> list[str]:
    """El CLI del anotador de este checkout, no el instalado (puede ser de otra rama)."""
    return [str(PY), "-c", "import sys; from boxtwin.cli import main; sys.exit(main(sys.argv[1:]))", *args]


def comandos(d: Path) -> dict[str, list[list[str]]]:
    return {
        "E4": [boxtwin("clasificar", str(d), "--modelo", str(MODELO_E4)), boxtwin("tipos", str(d))],
        "E7": [[str(PY_4DH), str(REPO / "boxtwin-annotator" / "tools" / "elevar_hmr2_sesion.py"), str(d)],
               boxtwin("clasificar", str(d), "--modelo", str(MODELO_E7), "--pose3d", str(d / "pose3d_hmr2.npz")),
               boxtwin("tipos", str(d))],
    }


def copiar(sesion: Path, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    for n in ARCHIVOS:
        if (sesion / n).is_file():
            shutil.copy2(sesion / n, destino / n)
    return destino


def cronometrar(cmds: list[list[str]], entorno: dict) -> list[dict]:
    salida = []
    for c in cmds:
        t0 = time.perf_counter()
        r = subprocess.run(c, capture_output=True, text=True, env=entorno)
        salida.append({"comando": " ".join(Path(x).name if "/" in x else x for x in c[:2] + c[-1:]),
                       "segundos": round(time.perf_counter() - t0, 2), "rc": r.returncode,
                       "error": r.stderr[-400:] if r.returncode else None})
        if r.returncode:
            break
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sesiones", nargs="+", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("~/boxtwin/e7-4dhumans/tiempo").expanduser())
    args = ap.parse_args(argv)
    entorno = {**os.environ, "PYTHONPATH": f"{REPO / 'boxtwin-annotator' / 'src'}"}
    resultado = []
    for s in args.sesiones:
        s = s.expanduser()
        video = json.loads((s / "segmentos.json").read_text())["video"]
        fila = {"sesion": s.name, "duracion_s": video.get("duracion_s"), "fps": video.get("fps"),
                "segmentos": len(json.loads((s / "segmentos.json").read_text())["segmentos"])}
        for modo in ("E4", "E7"):
            d = copiar(s, args.salida / s.name / modo)       # cada modo sobre su propia copia
            pasos = cronometrar(comandos(d)[modo], entorno)
            fila[modo] = {"pasos": pasos, "total_s": round(sum(p["segundos"] for p in pasos), 2),
                          "ok": all(p["rc"] == 0 for p in pasos)}
        if fila["E4"]["ok"] and fila["E7"]["ok"] and fila["duracion_s"]:
            fila["agregado_e7_s"] = round(fila["E7"]["total_s"] - fila["E4"]["total_s"], 2)
            fila["agregado_e7_x"] = round(fila["agregado_e7_s"] / fila["duracion_s"], 3)
        resultado.append(fila)
        print(json.dumps({k: v for k, v in fila.items() if k not in ("E4", "E7")} |
                         {m: fila[m]["total_s"] for m in ("E4", "E7")}), flush=True)
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "tiempo_e7.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
