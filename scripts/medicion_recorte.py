#!/usr/bin/env python3
"""
BoxTwin - Arma el recorte de un video anotado, video y anotacion juntos.

POR QUE EXISTE
  pacquiao_margarito dura 2275 s pero su anotacion cubre los cuadros 938 a 11225, o sea
  los primeros tres minutos. Medir la cadena sobre el video entero gastaria 1,75 h por
  corrida para que las etapas de identidad, deteccion y clasificacion trabajen sobre el 8%
  de los cuadros.

  Recortar solo la decodificacion no alcanza: la deteccion de cortes de plano corre por
  ffmpeg sobre el archivo entero y su costo quedaria atribuido a un tramo corto, inflando
  el ratio. Hay que recortar el archivo de verdad.

  Y hay que recortar la anotacion con el: el export dimensiona sus arreglos con
  doc.video.total_frames y despues le pide cada cuadro al cache, asi que una anotacion de
  136364 cuadros contra un cache de 11451 se cae con IndexError.

  NO TOCA NADA DEL PROYECTO ORIGINAL. Escribe un proyecto nuevo, con el video recortado,
  la anotacion recortada y un enlace al cache de referencia para poder verificar.

QUE HACE
  Corta el video con -c copy -que no reencodea, asi los cuadros son los mismos bit a bit-,
  cuenta los cuadros que salieron de verdad, y reescribe la anotacion con ese total,
  filtrando eventos y recortando asignaciones de identidad. Informa cuanto se perdio.

USO
  python ../scripts/medicion_recorte.py \
      --video anotacion-pacquiao/videos/pacquiao_margarito.mp4 --segundos 191 \
      --destino /tmp/pacq
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cuadros_de(video: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


def duracion_de(video: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--segundos", type=float, required=True)
    p.add_argument("--destino", type=Path, required=True)
    a = p.parse_args()

    src = a.video.resolve()
    proyecto = src.parent.parent
    base = src.stem
    annot_src = proyecto / "annotations" / f"{base}.annot.json"
    if not annot_src.is_file():
        print(f"error: no hay anotacion en {annot_src}", file=sys.stderr)
        return 1

    dst = a.destino.resolve()
    (dst / "videos").mkdir(parents=True, exist_ok=True)
    (dst / "annotations").mkdir(parents=True, exist_ok=True)
    (dst / "cache").mkdir(parents=True, exist_ok=True)

    video_dst = dst / "videos" / src.name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-t", str(a.segundos), "-c", "copy",
         "-y", str(video_dst)],
        check=True,
    )
    n = cuadros_de(video_dst)
    dur = duracion_de(video_dst)
    print(f"video recortado: {n} cuadros, {dur:.3f} s")

    d = json.loads(annot_src.read_text(encoding="utf-8"))
    n_ev0 = len(d.get("events", []))
    n_as0 = len(d.get("identity", {}).get("assignments", []))

    d["video"]["total_frames"] = n
    d["video"]["duration_s"] = dur
    d["events"] = [e for e in d.get("events", []) if e["end_frame"] < n]

    asg = []
    for x in d.get("identity", {}).get("assignments", []):
        if x["start_frame"] >= n:
            continue
        x = dict(x)
        x["end_frame_excl"] = min(x["end_frame_excl"], n)
        asg.append(x)
    d.setdefault("identity", {})["assignments"] = asg
    for k in ("manual_tracks", "interpolations"):
        v = d["identity"].get(k)
        if isinstance(v, list):
            d["identity"][k] = [x for x in v if x.get("start_frame", 0) < n]
    d["unreliable_segments"] = [
        u for u in d.get("unreliable_segments", []) if u.get("start_frame", 0) < n
    ]

    (dst / "annotations" / annot_src.name).write_text(
        json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # el cache de referencia se enlaza para que medicion_cadena.py pueda comparar y para
    # que adopte la misma configuracion de inferencia
    for suf in (".pose.npz", ".meta.json"):
        ref = proyecto / "cache" / f"{base}{suf}"
        if ref.is_file():
            enlace = dst / "cache" / ref.name
            if not enlace.exists():
                enlace.symlink_to(ref)

    print(f"eventos      {len(d['events'])} de {n_ev0}")
    print(f"asignaciones {len(asg)} de {n_as0}")
    if len(d["events"]) == n_ev0 and len(asg) == n_as0:
        print("el recorte no pierde nada de la anotacion")
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
