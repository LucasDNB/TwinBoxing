#!/usr/bin/env python3
"""
BoxingVI - Deteccion de placas de titulo entre los clips.

POR QUE EXISTE
  V1 intercala placas de titulo ("ROUND 2 / 2-2 / 1 minute") entre el metraje, y
  el anotador original les puso etiqueta de golpe igual. Son clips rotulados
  Cross o Jab que no contienen una persona. Entran a la extraccion de pose, YOLO
  no detecta nada y se propagan como muestras degeneradas al entrenamiento.

  Un primer intento uso brillo medio del frame con umbral global calibrado sobre
  V1, y fallo feo: marco 528 de 810 clips de V3 como placas, todos falsos
  positivos. V3 es metraje real de estudio con fondo oscuro, brillo uniforme
  entre 20 y 50. El umbral partia una distribucion continua por el medio. V1
  funcionaba solo porque es bimodal: placas en 6-11, gimnasio en 93-96.
  La leccion es que el brillo medio confunde "video oscuro" con "pantalla negra",
  y que un umbral calibrado en un video no transfiere a otro con otra exposicion.

QUE HACE
  Mide dos cosas que no dependen de la exposicion del video:
    frac_negro  fraccion de pixeles con luma < 16 en el frame central. Una placa
                es negro puro con texto encima y da ~0.85-0.95. Un video oscuro
                pero real tiene fondo gris o azul y un sujeto iluminado, no negro.
    mov         diferencia absoluta media entre el primer y el ultimo frame,
                submuestreados. Una placa es estatica y da ~0. Cualquier metraje
                con un boxeador moviendose da mucho mas.
  Marca placa si frac_negro > --min-negro y mov < --max-mov.

  Ademas diagnostica cada video por separado: si la distribucion de frac_negro no
  es bimodal, no hay dos poblaciones que separar y el conteo se reporta como "sin
  placas detectables" en vez de devolver un numero. Ese chequeo es justamente el
  que le faltaba a la version por brillo.

  Reanudable: escribe el CSV incrementalmente y saltea lo ya medido.

USO
  # desde test-BoxingVI/
  python ../scripts/boxingvi_placas.py --csv ./clips/manifest_filtrado.csv \
      --out ./clips/placas.csv

  # calibrar contra casos conocidos antes de confiar en el conteo
  python ../scripts/boxingvi_placas.py --calibrar
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Placas de titulo de V1 confirmadas a ojo, y controles de gimnasio del mismo video.
CALIB_PLACAS = ["V1_27396_27405", "V1_28996_29008", "V1_32504_32512", "V1_8469_8475"]
CALIB_REALES = ["V1_22649_22658", "V1_4349_4358", "V1_33838_33851"]

CAMPOS = ["clip", "video_key", "cls", "frac_negro", "mov", "es_placa", "placa_parcial"]


def medir(path: str, luma_negro: int = 16):
    """Devuelve (frac_negro, mov) del clip, o (nan, nan) si no se puede leer."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok_a, primero = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok_m, medio = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    ok_z, ultimo = cap.read()
    cap.release()

    if not (ok_a and ok_m and ok_z):
        return float("nan"), float("nan")

    g = cv2.cvtColor(medio, cv2.COLOR_BGR2GRAY)
    frac_negro = float((g < luma_negro).mean())

    # Submuestreo antes de comparar: alcanza para ver si hay movimiento y es barato.
    chico = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)
    mov = float(np.abs(chico(ultimo) - chico(primero)).mean())

    return frac_negro, mov


def diagnostico_bimodal(vals: np.ndarray, corte: float, ancho: float = 0.15):
    """Busca dos poblaciones separadas en frac_negro, con prueba de valle.

    La pregunta que importa no es si hay un hueco grande entre valores
    consecutivos, sino si el umbral cae en una zona vacia entre dos modas
    pobladas. Medir el hueco maximo falla apenas hay un puñado de clips
    intermedios que lo tapan: en V1 hay 37 clips de transicion entre las dos
    modas (1617 abajo de 0.1 y 209 arriba de 0.9) y el hueco maximo baja a 0.29,
    suficiente para descartar 181 placas que existen.

    Asi que se mide la densidad alrededor del corte. Si hay poblacion arriba y la
    banda que rodea al umbral esta casi vacia comparada con esa poblacion, son
    dos grupos. Si no hay nada arriba, no hay segunda poblacion y no hay placas:
    ese es el caso de V3, donde el metodo anterior invento 528.
    """
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return False, 0, 0
    arriba = int((v > corte).sum())
    valle = int(((v > corte - ancho) & (v < corte + ancho)).sum())
    # Se exige poblacion arriba y un valle chico contra esa poblacion.
    bimodal = arriba >= 3 and valle < 0.2 * arriba
    return bimodal, arriba, valle


def calibrar(df, luma_negro):
    """Imprime las dos metricas sobre los casos conocidos. No escribe nada."""
    print("\ncalibracion sobre casos ya confirmados a ojo\n")
    print(f"  {'clip':24}{'frac_negro':>12}{'mov':>9}")
    for grupo, nombres in [("PLACAS", CALIB_PLACAS), ("REALES", CALIB_REALES)]:
        print(f"  --- {grupo}")
        for k in nombres:
            fila = df[df["clip"].str.contains(k + ".mp4", regex=False)]
            if fila.empty:
                print(f"  {k:24}{'no esta en el manifest':>21}")
                continue
            fn, mv = medir(fila.iloc[0]["clip"], luma_negro)
            print(f"  {k:24}{fn:>12.3f}{mv:>9.2f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Detecta placas de titulo entre los clips.")
    ap.add_argument("--csv", default="./clips/manifest_filtrado.csv")
    ap.add_argument("--out", default="./clips/placas.csv")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="limitar a estos video_key (default: todos)")
    ap.add_argument("--luma-negro", type=int, default=16,
                    help="un pixel cuenta como negro por debajo de este valor")
    ap.add_argument("--min-negro", type=float, default=0.70,
                    help="fraccion de negro minima para considerar placa")
    ap.add_argument("--max-mov", type=float, default=2.0,
                    help="movimiento maximo para considerar placa")
    ap.add_argument("--calibrar", action="store_true",
                    help="solo medir los casos conocidos y salir")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.videos:
        df = df[df["video_key"].isin(args.videos)]

    if args.calibrar:
        calibrar(df, args.luma_negro)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hechos = set()
    if out.exists():
        hechos = set(pd.read_csv(out)["clip"])
        print(f"reanudando: {len(hechos)} clips ya medidos")

    pendientes = df[~df["clip"].isin(hechos)]
    print(f"midiendo {len(pendientes)} clips de {df['video_key'].nunique()} videos\n")

    nuevo = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        for i, (_, r) in enumerate(pendientes.iterrows(), 1):
            fn, mv = medir(r["clip"], args.luma_negro)
            w.writerow({
                "clip": r["clip"], "video_key": r["video_key"], "cls": r["cls"],
                "frac_negro": round(fn, 4) if fn == fn else "",
                "mov": round(mv, 3) if mv == mv else "",
                "es_placa": "", "placa_parcial": "",  # se completan al reclasificar
            })
            if i % 250 == 0:
                fh.flush()
                print(f"  {i}/{len(pendientes)}")

    # Clasificacion a partir de las metricas guardadas. Se recalcula entera en
    # cada corrida: permite mover los umbrales sin volver a leer 4751 videos.
    res = pd.read_csv(out)
    negro = res["frac_negro"] > args.min_negro
    quieto = res["mov"] < args.max_mov
    res["es_placa"] = (negro & quieto).astype("Int64")
    # Frac_negro alto con movimiento alto: el clip cruza un corte de escena, es
    # mitad placa y mitad metraje. Tambien es basura, pero de otra clase.
    res["placa_parcial"] = (negro & ~quieto).astype("Int64")
    res.loc[res["frac_negro"].isna(), ["es_placa", "placa_parcial"]] = pd.NA
    res.to_csv(out, index=False)

    print(f"\n{'video':7}{'clips':>7}{'placas':>8}{'parc':>6}{'%':>7}   diagnostico")
    for v in sorted(res["video_key"].unique(), key=lambda s: (len(s), s)):
        b = res[res["video_key"] == v]
        bim, arriba, valle = diagnostico_bimodal(b["frac_negro"].values, args.min_negro)
        n = int(b["es_placa"].fillna(0).sum())
        par = int(b["placa_parcial"].fillna(0).sum())
        if bim:
            diag = f"bimodal: {arriba} arriba del corte, valle de {valle}"
        else:
            diag = (f"sin segunda poblacion ({arriba} arriba, valle {valle}): "
                    f"sin placas detectables")
            n, par = 0, 0
        pct = n / len(b) * 100 if len(b) else 0
        print(f"{v:7}{len(b):>7}{n:>8}{par:>6}{pct:>6.1f}%   {diag}")

    ilegibles = int(res["frac_negro"].isna().sum())
    if ilegibles:
        print(f"\n{ilegibles} clips ilegibles")
    print(f"\nescrito: {out}")


if __name__ == "__main__":
    main()
