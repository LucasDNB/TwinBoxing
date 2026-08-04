#!/usr/bin/env python3
"""
BoxingVI - Fase A: inspeccion de anotaciones + descarga de videos.

Que hace:
  1. Inspecciona los .xlsx de RGB_videos/ y Annotation_files/ (columnas + muestra),
     para poder escribir el recorte de clips (Fase B) calzado al formato real.
  2. Baja los videos de YouTube en lote leyendo la planilla de links,
     manejando link rot y registrando la cobertura efectiva.
  3. Reporta el fps real de cada video bajado (ffprobe) y avisa si difiere de 30,
     porque las anotaciones de BoxingVI estan en frames a 30 fps.

Requisitos:
  pip install pandas openpyxl yt-dlp
  ffmpeg/ffprobe instalados en el sistema (para el reporte de fps)

Uso tipico:
  # 1) primero bajar la carpeta del Drive con gdown (ver plan)
  # 2) inspeccionar sin bajar nada:
  python boxingvi_fetch.py --dataset-dir ./Dataset --inspect-only
  # 3) inspeccionar + bajar:
  python boxingvi_fetch.py --dataset-dir ./Dataset --out ./videos
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

YT_RE = re.compile(r"(youtube\.com/|youtu\.be/|https?://)", re.IGNORECASE)


def find_xlsx(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xls")) + sorted(folder.glob("*.ods"))


def inspect_excel(path: Path):
    """Imprime hojas, forma, columnas y las primeras filas de cada hoja."""
    print(f"\n{'='*70}\nARCHIVO: {path}")
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"  [ERROR] no se pudo abrir: {e}")
        return
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        print(f"\n  Hoja: '{sheet}'  |  filas x cols: {df.shape}")
        print(f"  Columnas: {list(df.columns)}")
        print(f"  Muestra (5 filas):")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(5).to_string(index=False))


def detect_link_column(df: pd.DataFrame):
    """Elige la columna con mas celdas que parecen URLs. Devuelve nombre o None."""
    best_col, best_hits = None, 0
    for col in df.columns:
        hits = df[col].astype(str).str.contains(YT_RE).sum()
        if hits > best_hits:
            best_col, best_hits = col, hits
    return best_col if best_hits > 0 else None


def load_links(rgb_dir: Path, link_col: str | None):
    """Lee el/los xlsx de RGB_videos/ y devuelve lista de (fila_id, url)."""
    files = find_xlsx(rgb_dir)
    if not files:
        sys.exit(f"[ERROR] no encontre .xlsx en {rgb_dir}")
    links = []
    for f in files:
        df = pd.read_excel(f)
        col = link_col or detect_link_column(df)
        if col is None:
            print(f"[AVISO] no detecte columna de links en {f.name}. "
                  f"Columnas: {list(df.columns)}. Usa --link-col para especificar.")
            continue
        for i, val in df[col].dropna().astype(str).items():
            if YT_RE.search(val):
                links.append((f"{f.stem}_{i}", val.strip()))
    # dedup por url conservando orden
    seen, uniq = set(), []
    for rid, url in links:
        if url not in seen:
            seen.add(url)
            uniq.append((rid, url))
    return uniq


def download(links, out_dir: Path, fmt: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "download_archive.txt"   # permite reanudar
    ok, failed = [], []
    for rid, url in links:
        print(f"\n--- bajando {rid}: {url}")
        cmd = [
            "yt-dlp", "-f", fmt, "--no-playlist",
            "--download-archive", str(archive),
            "-o", str(out_dir / "%(id)s.%(ext)s"), url,
        ]
        rc = subprocess.run(cmd).returncode
        (ok if rc == 0 else failed).append(url)
    return ok, failed


def report_fps(out_dir: Path):
    """Corre ffprobe sobre cada video y avisa si el fps difiere de 30."""
    vids = [p for p in out_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    rows = []
    for v in vids:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", str(v)],
                capture_output=True, text=True)
            raw = r.stdout.strip()          # viene como "30000/1001" o "30/1"
            num, den = (raw.split("/") + ["1"])[:2]
            fps = round(float(num) / float(den), 3) if float(den) else None
        except Exception:
            fps = None
        flag = "" if fps == 30 else "  <-- OJO, no es 30 fps"
        rows.append((v.name, fps))
        print(f"  {v.name}: {fps} fps{flag}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="Raiz del Dataset descargado del Drive (con RGB_videos/ y Annotation_files/)")
    ap.add_argument("--out", default="./videos", help="Carpeta destino de los videos")
    ap.add_argument("--link-col", default=None, help="Nombre de la columna de links (si la autodeteccion falla)")
    ap.add_argument("--format", default="bestvideo[ext=mp4]/bestvideo/best",
                    help="Formato yt-dlp (video sin audio por defecto; audio esta fuera de scope)")
    ap.add_argument("--inspect-only", action="store_true", help="Solo inspecciona los Excel, no baja nada")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    rgb_dir = root / "RGB_videos"
    ann_dir = root / "Annotation_files"

    # 1) Inspeccion de ambos Excel
    print("### INSPECCION DE PLANILLAS ###")
    for folder in (rgb_dir, ann_dir):
        for f in find_xlsx(folder):
            inspect_excel(f)

    if args.inspect_only:
        print("\n[inspect-only] listo. Pegame esta salida y escribo el recorte (Fase B).")
        return

    # 2) Descarga
    links = load_links(rgb_dir, args.link_col)
    print(f"\n### DESCARGA ###  links unicos detectados: {len(links)}")
    ok, failed = download(links, Path(args.out), args.format)

    # 3) Cobertura + fps
    coverage = {"total_links": len(links), "descargados": len(ok), "fallidos": len(failed)}
    Path(args.out, "coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False))
    if failed:
        Path(args.out, "failed_links.txt").write_text("\n".join(failed))

    print(f"\n### COBERTURA ###  {coverage}")
    if failed:
        print(f"  {len(failed)} links caidos -> ver failed_links.txt (link rot esperable)")

    print("\n### FPS DE LOS VIDEOS (deben ser 30 para calzar con las anotaciones) ###")
    report_fps(Path(args.out))
    print("\nSi hay videos que no son 30 fps, hay que reencodear a 30 antes de recortar,")
    print("o mapear los frames de anotacion segun el fps real. No recortes hasta resolver esto.")


if __name__ == "__main__":
    main()