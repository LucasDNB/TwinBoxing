#!/usr/bin/env python3
"""
BoxingVI - Fase B: recorte de clips por (Start Frame, End Frame, Class).

CAMBIO v0.2 (2026-07-30) -- FIX DE FPS
  La v0.1 reencodeaba cada video a 30 fps constante asumiendo que las anotaciones
  estaban expresadas en frames a 30 fps (como afirma el paper). Verificacion empirica:
  las anotaciones estan en el fps NATIVO de cada video. El reencoding introducia un
  corrimiento proporcional (factor fps_nativo/30) y acumulativo a lo largo del video.
  -> Se elimina la etapa de normalizacion. El corte se hace sobre el video original.

Que hace:
  1. Lee las anotaciones de Annotation_files/ (limpia columnas Unnamed y espacios).
  2. Sondea cada video con ffprobe: fps nativo (racional exacto), cantidad de frames,
     y deteccion de VFR (r_frame_rate != avg_frame_rate).
  3. Valida cada anotacion contra la duracion real del video (descarta rangos fuera).
  4. Recorta por seek exacto: -ss (start-0.5)/fps + -frames:v (end-start+1).
  5. Verifica el clip resultante contando frames con ffprobe y lo registra en manifest.

Requisitos: pip install pandas openpyxl odfpy ; ffmpeg/ffprobe en el sistema.

Uso:
  # 1ro: auditoria sin cortar nada (fps de cada video, anotaciones fuera de rango)
  python boxingvi_clip.py --dataset-dir ./Dataset --videos ./videos --out ./clips --dry-run

  # 2do: smoke test de 20 clips
  python boxingvi_clip.py --dataset-dir ./Dataset --videos ./videos --out ./clips --limit 20

  # 3ro: corrida completa
  python boxingvi_clip.py --dataset-dir ./Dataset --videos ./videos --out ./clips
"""

import argparse
import csv
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd

VIDEO_EXTS = {".mp4", ".mkv", ".webm"}


# ----------------------------------------------------------------------
# 1. Anotaciones
# ----------------------------------------------------------------------
# Clases canonicas de BoxingVI (6 golpes ofensivos, sin altura ni defensas)
CANON = {
    "jab": "Jab",
    "cross": "Cross",
    "lead hook": "Lead Hook",
    "rear hook": "Rear Hook",
    "lead uppercut": "Lead Uppercut",
    "rear uppercut": "Rear Uppercut",
}


def _canon_class(raw):
    """Normaliza mayusculas/espacios y valida contra las 6 clases. None si desconocida."""
    key = " ".join(str(raw).strip().lower().split())
    return CANON.get(key)


def _pick_by_name(cols):
    low = {c: str(c).strip().lower() for c in cols}
    cs = next((c for c in cols if low[c].startswith("start")), None)
    ce = next((c for c in cols if low[c].startswith(("end", "ending"))), None)
    cc = next((c for c in cols if low[c].startswith("class") or low[c] == "type"), None)
    return cs, ce, cc


def load_annotations(ann_dir: Path):
    """Lee los .xlsx (uno por video). Cada planilla tiene un formato distinto:
    algunas con headers usables (con variantes sucias), otras sin header (columnas
    Unnamed) o con el header pegoteado. Estrategia: detectar Start/End/Class por
    nombre; si no, caer a las 3 primeras columnas por posicion. Se valida cada fila
    (frames enteros, end>start, clase canonica) y se reporta lo descartado por planilla."""
    files = sorted(ann_dir.glob("*.xlsx")) + sorted(ann_dir.glob("*.xls"))
    if not files:
        sys.exit(f"[ERROR] no hay .xlsx en {ann_dir}")
    rows, report = [], []
    for f in files:
        df = pd.read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        cs, ce, cc = _pick_by_name(df.columns)
        if cs and ce and cc:
            sub, modo = df[[cs, ce, cc]].copy(), "nombre"
        else:
            sub, modo = df.iloc[:, :3].copy(), "posicional"  # start, end, class
        sub.columns = ["start", "end", "cls"]
        kept = dropped = 0
        unknown = set()
        for _, r in sub.iterrows():
            try:
                s, e = int(float(r["start"])), int(float(r["end"]))
            except (ValueError, TypeError):
                dropped += 1
                continue
            cls = _canon_class(r["cls"])
            if cls is None:
                if str(r["cls"]).strip().lower() not in ("nan", ""):
                    unknown.add(str(r["cls"]).strip())
                dropped += 1
                continue
            if e <= s or s < 0:
                dropped += 1
                continue
            rows.append({"video_key": f.stem, "start": s, "end": e, "cls": cls})
            kept += 1
        report.append((f.stem, modo, kept, dropped, unknown))

    print("\n### ANOTACIONES CARGADAS ###")
    for stem, modo, kept, dropped, unknown in report:
        extra = f"  clases raras: {unknown}" if unknown else ""
        print(f"  {stem}: {kept} validos, {dropped} descartados  [{modo}]{extra}")
    print(f"  TOTAL: {len(rows)} anotaciones validas")
    return rows


# ----------------------------------------------------------------------
# 2. Resolucion anotacion -> archivo de video
# ----------------------------------------------------------------------
def extract_youtube_id(url: str):
    """Saca el id del parametro v= (o del path de youtu.be)."""
    u = urlparse(url)
    q = parse_qs(u.query)
    if "v" in q:
        return q["v"][0]
    if "youtu.be" in u.netloc:
        return u.path.lstrip("/").split("/")[0] or None
    return None


def build_rgb_map(rgb_dir: Path):
    """Lee Meta_data.ods (columnas Video, Link) y arma {video_key: youtube_id}."""
    files = (sorted(rgb_dir.glob("*.ods")) + sorted(rgb_dir.glob("*.xlsx")))
    if not files:
        print(f"[AVISO] no encontre Meta_data en {rgb_dir}; resolucion caera al match directo/difuso")
        return {}
    f = files[0]
    df = pd.read_excel(f, engine="odf" if f.suffix.lower() == ".ods" else None)
    df.columns = [str(c).strip() for c in df.columns]
    col_v = next((c for c in df.columns if c.lower().startswith("video")), None)
    col_l = next((c for c in df.columns if c.lower().startswith("link")), None)
    if not (col_v and col_l):
        print(f"[AVISO] {f.name}: no veo columnas Video/Link. Columnas: {list(df.columns)}")
        return {}
    mapping = {}
    for _, r in df.dropna(subset=[col_v, col_l]).iterrows():
        yid = extract_youtube_id(str(r[col_l]))
        if yid:
            mapping[str(r[col_v]).strip()] = yid
    return mapping


def resolve_video_file(video_key: str, videos_dir: Path, rgb_map: dict):
    """Estrategias en orden: match directo por nombre; via rgb_map (subject->id);
    match difuso (el archivo contiene la key)."""
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{video_key}{ext}"
        if p.exists():
            return p
    if video_key in rgb_map:
        for ext in VIDEO_EXTS:
            p = videos_dir / f"{rgb_map[video_key]}{ext}"
            if p.exists():
                return p
    for p in videos_dir.iterdir():
        if p.suffix.lower() in VIDEO_EXTS and video_key in p.stem:
            return p
    return None


# ----------------------------------------------------------------------
# 3. Sondeo del video (reemplaza a la vieja normalizacion a 30 fps)
# ----------------------------------------------------------------------
def probe_video(src: Path):
    """Devuelve dict con fps nativo exacto (Fraction), nro de frames y flag de VFR.
    El fps se toma de r_frame_rate como racional (ej. 30000/1001) para no perder
    precision al convertir indice de frame -> timestamp de seek."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
           "-of", "json", str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        st = json.loads(r.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None

    def _frac(v):
        try:
            f = Fraction(v)
            return f if f > 0 else None
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    r_fps = _frac(st.get("r_frame_rate"))
    a_fps = _frac(st.get("avg_frame_rate"))
    if r_fps is None:
        return None

    # nb_frames suele faltar en webm/mkv -> derivar de duracion
    n_frames = None
    if st.get("nb_frames", "N/A") not in ("N/A", None):
        try:
            n_frames = int(st["nb_frames"])
        except ValueError:
            pass
    if n_frames is None and st.get("duration"):
        try:
            n_frames = int(float(st["duration"]) * float(r_fps))
        except ValueError:
            pass

    # VFR: si r_frame_rate y avg_frame_rate difieren >1%, el indice de frame
    # no mapea linealmente a timestamp y el corte por frame es poco confiable.
    vfr = bool(a_fps and abs(float(r_fps) - float(a_fps)) / float(r_fps) > 0.01)
    return {"fps": r_fps, "avg_fps": a_fps, "n_frames": n_frames, "vfr": vfr}


def count_frames(path: Path):
    """Cuenta frames reales decodificando (exacto, lento pero solo sobre el clip)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


# ----------------------------------------------------------------------
# 4. Recorte por frame sobre el video NATIVO
# ----------------------------------------------------------------------
def cut_clip(src: Path, start: int, end: int, fps: Fraction, out_path: Path):
    """Seek de entrada a medio frame antes del target: ffmpeg emite el primer frame
    con pts >= t, asi que t=(start-0.5)/fps garantiza arrancar exactamente en `start`
    sin depender de redondeo de punto flotante. -frames:v fija el largo (end inclusive)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = max(Fraction(0), (Fraction(start) - Fraction(1, 2)) / fps)
    n = end - start + 1
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{float(t):.6f}", "-i", str(src),
         "-frames:v", str(n), "-vf", "setpts=PTS-STARTPTS",
         "-an", "-sn", str(out_path)],
        capture_output=True)
    return out_path.exists() and out_path.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--videos", required=True, help="Carpeta con los .mp4 bajados (Fase A)")
    ap.add_argument("--out", default="./clips")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo audita: fps por video, VFR, anotaciones fuera de rango. No corta.")
    ap.add_argument("--limit", type=int, default=0, help="Cortar solo los primeros N clips")
    ap.add_argument("--verify", action="store_true",
                    help="Contar frames de cada clip generado (lento, pero da evidencia)")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    videos_dir = Path(args.videos)
    out_dir = Path(args.out)

    ann = load_annotations(root / "Annotation_files")
    rgb_map = build_rgb_map(root / "RGB_videos")

    # --- Sondeo: una vez por video fuente ---
    print("\n### SONDEO DE VIDEOS (fps nativo) ###")
    src_cache, meta_cache = {}, {}
    for key in sorted({a["video_key"] for a in ann}):
        src = resolve_video_file(key, videos_dir, rgb_map)
        src_cache[key] = src
        if src is None:
            print(f"  {key}: [NO RESUELTO] no encontre archivo de video")
            continue
        meta = probe_video(src)
        meta_cache[key] = meta
        if meta is None:
            print(f"  {key}: [ERROR] ffprobe fallo sobre {src.name}")
            continue
        flag = "  <-- VFR, corte por frame NO confiable" if meta["vfr"] else ""
        print(f"  {key}: {src.name} | fps={float(meta['fps']):.4f} "
              f"({meta['fps']}) | frames={meta['n_frames']}{flag}")

    # --- Validacion de rangos contra la duracion real ---
    ok_ann, out_of_range = [], 0
    for a in ann:
        meta = meta_cache.get(a["video_key"])
        if src_cache.get(a["video_key"]) is None or meta is None:
            continue
        if meta["n_frames"] and a["end"] >= meta["n_frames"]:
            out_of_range += 1
            continue
        ok_ann.append(a)

    print(f"\nAnotaciones utilizables: {len(ok_ann)} | fuera de rango del video: {out_of_range} "
          f"| sin video: {len(ann) - len(ok_ann) - out_of_range}")
    if out_of_range:
        print("  [ATENCION] anotaciones que exceden el largo del video sugieren que ese video "
              "se bajo en otra version/corte que la usada por los autores.")

    if args.dry_run:
        print("\n--dry-run: no se corto nada. Revisa el fps de cada video antes de la corrida real.")
        return

    # --- Corte ---
    target = ok_ann[:args.limit] if args.limit else ok_ann
    manifest, n_ok, n_fail, n_mismatch = [], 0, 0, 0

    for i, a in enumerate(target, 1):
        src = src_cache[a["video_key"]]
        meta = meta_cache[a["video_key"]]
        cls_dir = out_dir / a["cls"].replace(" ", "_")
        clip_path = cls_dir / f"{a['video_key']}_{a['start']}_{a['end']}.mp4"
        expected = a["end"] - a["start"] + 1

        if not cut_clip(src, a["start"], a["end"], meta["fps"], clip_path):
            n_fail += 1
            continue

        got = count_frames(clip_path) if args.verify else None
        if got is not None and got != expected:
            n_mismatch += 1

        n_ok += 1
        manifest.append({
            "clip": str(clip_path), "cls": a["cls"], "video_key": a["video_key"],
            "start_frame": a["start"], "end_frame": a["end"],
            "expected_frames": expected, "actual_frames": got if got is not None else "",
            "src_video": src.name, "src_fps": str(meta["fps"]), "src_vfr": meta["vfr"],
        })
        if i % 100 == 0:
            print(f"  ... {i}/{len(target)} clips")

    if manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    print(f"\n### RESULTADO ###  clips OK: {n_ok} | fallidos: {n_fail}")
    if args.verify:
        print(f"clips con largo distinto al esperado: {n_mismatch}")
    print(f"manifest: {out_dir/'manifest.csv'}")


if __name__ == "__main__":
    main()