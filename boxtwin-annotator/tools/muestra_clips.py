#!/usr/bin/env python3
"""
BoxTwin - Arma una muestra estratificada de clips y la pega en un solo video para revisar.

POR QUE EXISTE
  Un golpe dura unos 10 cuadros, o sea un tercio de segundo. Abrir 20 clips de a uno y
  mirarlos a velocidad real no permite juzgar nada: cuando el ojo llega, el clip termino.
  Y revisar "unos cuantos" sin criterio ya se demostro que no informa. En BoxingVI el video
  V4 figuraba como confiable con 5 aciertos de 5, y la muestra formal de 18 lo dejo en 12;
  los dos resultados no se contradicen, el piso de confianza de 5/5 es 56,6%. La muestra
  chica no estaba mal, no decia nada.

  Por eso esto sortea con semilla fija, estratifica por clase con un minimo por clase, y
  escribe la muestra a CSV antes de generar nada. Resortear despues de ver resultados
  parciales convierte el numero en lo que uno quiera que sea, asi que se niega a pisar una
  muestra existente salvo --force.

QUE HACE
  Sortea la muestra desde el manifest de un export de clips, la congela en CSV, y produce un
  solo mp4 con los clips en camara lenta, cada uno con su clase y su id estampados y repetido
  para poder juzgarlo.

USO
  python tools/muestra_clips.py proyecto/exports/Sparring.manifest.csv
  python tools/muestra_clips.py .../Sparring.manifest.csv -n 30 --fps 4 --repeticiones 3
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def sortear(filas: list[dict], n: int, min_por_clase: int, seed: int) -> list[dict]:
    """
    Estratificada: primero el minimo de cada clase, despues se reparte el resto.

    El minimo por clase va primero a proposito. Con un desbalance de 20 a 1 un sorteo
    proporcional deja las clases raras sin un solo ejemplo, y esas son justo las que hay que
    mirar: son las que menos veces anoto uno y donde el criterio esta menos asentado.
    """
    rng = random.Random(seed)
    por_clase: dict[str, list[dict]] = defaultdict(list)
    for f in filas:
        por_clase[f["clase"]].append(f)

    elegidos: list[dict] = []
    resto: list[dict] = []
    for clase in sorted(por_clase):
        disponibles = sorted(por_clase[clase], key=lambda r: r["event_id"])
        rng.shuffle(disponibles)
        elegidos.extend(disponibles[:min_por_clase])
        resto.extend(disponibles[min_por_clase:])

    if len(elegidos) < n:
        rng.shuffle(resto)
        elegidos.extend(resto[: n - len(elegidos)])
    return sorted(elegidos, key=lambda r: int(r["start_frame"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Muestra estratificada de clips, en un solo video.")
    p.add_argument("manifest", type=Path)
    p.add_argument("-n", type=int, default=20, help="tamano de la muestra")
    p.add_argument("--min-por-clase", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=5.0, help="fps de salida; 5 sobre 30 es ~0,17x")
    p.add_argument("--repeticiones", type=int, default=2, help="veces que se repite cada clip")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="resortea pisando la muestra existente")
    args = p.parse_args()

    filas = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not filas:
        raise SystemExit(f"{args.manifest} no tiene filas")
    base = args.manifest.parent

    csv_muestra = args.manifest.with_name(args.manifest.stem.replace(".manifest", "") + ".muestra.csv")
    if csv_muestra.exists() and not args.force:
        muestra = list(csv.DictReader(csv_muestra.open(encoding="utf-8")))
        print(f"muestra ya congelada: {csv_muestra} ({len(muestra)} clips)")
        print("  para resortear: --force")
    else:
        muestra = sortear(filas, args.n, args.min_por_clase, args.seed)
        with csv_muestra.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(muestra)
        print(f"muestra sorteada con seed {args.seed}: {csv_muestra} ({len(muestra)} clips)")

    conteo: dict[str, int] = defaultdict(int)
    for r in muestra:
        conteo[r["clase"]] += 1
    for clase in sorted(conteo):
        print(f"    {clase:26s} {conteo[clase]}")

    salida = args.out or csv_muestra.with_suffix(".mp4")
    lista = base / "_muestra_concat.txt"
    trozos: list[Path] = []
    tmp = base / "_muestra_tmp"
    tmp.mkdir(exist_ok=True)

    for i, r in enumerate(muestra):
        origen = base / r["clip"]
        if not origen.exists():
            print(f"  falta {origen}", file=sys.stderr)
            continue
        # El texto se quema DESPUES de bajar los fps, sobre el resultado, para que se lea en
        # todos los cuadros del clip lento y no solo en los originales.
        etiqueta = (
            f"{i + 1}/{len(muestra)}  {r['clase']}   {r['event_id']}  "
            f"{r['fighter']}  {r['landed']}  {r['quality']}"
        ).replace(":", r"\:")
        destino = tmp / f"{i:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(origen),
                "-vf",
                f"setpts=PTS*{29.643 / args.fps:.4f},scale=960:-2,"
                f"drawbox=y=0:h=54:w=iw:t=fill:color=black@0.75,"
                f"drawtext=text='{etiqueta}':x=12:y=16:fontsize=22:fontcolor=white",
                "-r", str(args.fps), "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-an", str(destino),
            ],
            check=True,
        )
        trozos.extend([destino] * args.repeticiones)

    lista.write_text("".join(f"file '{t.resolve()}'\n" for t in trozos), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lista), "-c", "copy", str(salida)],
        check=True,
    )
    for t in set(trozos):
        t.unlink()
    tmp.rmdir()
    lista.unlink()

    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(salida)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\nlisto -> {salida}  ({float(dur):.0f} s, {len(muestra)} clips x{args.repeticiones})")
    print(f"muestra congelada en {csv_muestra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
