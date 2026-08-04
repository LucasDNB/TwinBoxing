#!/usr/bin/env python3
"""
BoxingVI - Fase C: split train/val y filtrado del manifest.

Decisiones que implementa (justificadas en docs/experiments/):
  1. FILTRADO: descarta clips de largo no plausible para un golpe unico.
     Cota inferior: un golpe necesita minimo ~3 frames para tener cinematica
     (guardia -> extension -> retorno). Cota superior: 25 frames es el maximo
     estadistico que reporta el paper de BoxingVI para un golpe a 30 fps;
     se usa un margen de 30 para no descartar anotaciones laxas legitimas.
  2. SPLIT POR VIDEO, no aleatorio. Dos clips del mismo video comparten
     boxeador, ring, iluminacion y angulo de camara: mezclarlos entre train y
     val produce fuga de dominio y metricas infladas. El split por video mide
     generalizacion a boxeadores no vistos.
  3. Videos de validacion elegidos, no sorteados: se exige que el split de val
     contenga las 6 clases (V2 y V8 no tienen ningun Rear Hook, quedan en train)
     y que no tenga distribucion invertida respecto del resto (V4 tiene mas Jab
     que Cross, unico caso; queda en train).
  4. PESOS DE CLASE por frecuencia inversa, calculados solo sobre train.

Uso:
  python boxingvi_split.py --manifest ./clips/manifest.csv --out ./clips
"""

import argparse
import json
from pathlib import Path

import pandas as pd

VAL_VIDEOS = ["V5", "V9", "V10"]

MIN_FRAMES = 3
MAX_FRAMES = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="./clips")
    ap.add_argument("--val-videos", nargs="+", default=VAL_VIDEOS)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    out = Path(args.out)
    df = pd.read_csv(args.manifest)
    n0 = len(df)
    df["n_frames"] = df["end_frame"] - df["start_frame"] + 1

    short = df[df["n_frames"] < args.min_frames]
    long_ = df[df["n_frames"] > args.max_frames]
    print("### FILTRADO POR LARGO ###")
    print(f"  descartados por cortos (<{args.min_frames} frames): {len(short)}")
    for _, r in short.iterrows():
        print(f"    {r['video_key']} {r['start_frame']}-{r['end_frame']} "
              f"({r['n_frames']}f) {r['cls']}")
    print(f"  descartados por largos (>{args.max_frames} frames): {len(long_)}")
    for _, r in long_.iterrows():
        print(f"    {r['video_key']} {r['start_frame']}-{r['end_frame']} "
              f"({r['n_frames']}f) {r['cls']}")

    df = df[(df["n_frames"] >= args.min_frames) &
            (df["n_frames"] <= args.max_frames)].copy()
    print(f"  quedan: {len(df)} de {n0}\n")

    val_set = set(args.val_videos)
    unknown = val_set - set(df["video_key"].unique())
    if unknown:
        raise SystemExit(f"[ERROR] video_key inexistente en el manifest: {unknown}")

    df["split"] = df["video_key"].apply(lambda v: "val" if v in val_set else "train")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]

    print("### SPLIT ###")
    print(f"  train: {len(tr):5d} ({len(tr)/len(df)*100:.1f}%)  videos: "
          f"{sorted(tr['video_key'].unique(), key=lambda x: int(x[1:]))}")
    print(f"  val:   {len(va):5d} ({len(va)/len(df)*100:.1f}%)  videos: "
          f"{sorted(va['video_key'].unique(), key=lambda x: int(x[1:]))}")

    print("\n### CHEQUEOS ###")
    classes = sorted(df["cls"].unique())
    missing_va = [c for c in classes if (va["cls"] == c).sum() == 0]
    missing_tr = [c for c in classes if (tr["cls"] == c).sum() == 0]
    print(f"  clases ausentes en val:   {missing_va or 'ninguna'}")
    print(f"  clases ausentes en train: {missing_tr or 'ninguna'}")
    overlap = set(tr["video_key"]) & set(va["video_key"])
    print(f"  videos compartidos entre splits: {overlap or 'ninguno'}  "
          f"({'OK, sin fuga' if not overlap else 'FUGA DE DOMINIO'})")
    dom = tr["video_key"].value_counts(normalize=True)
    print(f"  video dominante en train: {dom.index[0]} = {dom.iloc[0]*100:.1f}%")
    if dom.iloc[0] > 0.30:
        print("    [ATENCION] un solo video aporta mas del 30% del train. Si el modelo "
              "colapsa hacia ese estilo, submuestrear y volver a medir.")

    print("\n### DISTRIBUCION POR CLASE ###")
    print(f"  {'clase':16s} {'train':>6s} {'val':>6s} {'%val':>6s} {'peso':>7s}")
    n_tr, k = len(tr), len(classes)
    weights = {}
    for c in classes:
        a, b = int((tr["cls"] == c).sum()), int((va["cls"] == c).sum())
        w = n_tr / (k * a) if a else 0.0
        weights[c] = round(w, 4)
        pct = b / (a + b) * 100 if (a + b) else 0
        print(f"  {c:16s} {a:6d} {b:6d} {pct:5.1f}% {w:7.3f}")

    cols = ["clip", "cls", "video_key", "start_frame", "end_frame", "n_frames",
            "src_video", "src_fps"]
    cols = [c for c in cols if c in df.columns]
    tr[cols].to_csv(out / "train.csv", index=False)
    va[cols].to_csv(out / "val.csv", index=False)
    df.to_csv(out / "manifest_filtrado.csv", index=False)
    with open(out / "class_weights.json", "w") as fh:
        json.dump({"weights": weights, "classes": classes,
                   "criterio": "frecuencia inversa balanceada sobre train: n/(k*n_c)"},
                  fh, indent=2, ensure_ascii=False)

    print(f"\nescritos: {out/'train.csv'}, {out/'val.csv'}, "
          f"{out/'manifest_filtrado.csv'}, {out/'class_weights.json'}")


if __name__ == "__main__":
    main()
