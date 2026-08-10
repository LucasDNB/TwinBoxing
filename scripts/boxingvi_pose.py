#!/usr/bin/env python3
"""
BoxingVI - Fase D: extraccion de poses y seleccion del boxeador atacante.

PROBLEMA QUE RESUELVE
  Las anotaciones de BoxingVI dicen que en un clip hay un Cross, pero no dicen
  QUIEN lo tiro. En sparring el estimador detecta 2+ personas por frame. Si se
  alimenta al clasificador con el esqueleto del defensor, la etiqueta queda
  sistematicamente desalineada: el modelo aprende a asociar "Cross" con la
  cinematica de quien lo recibe. Es un error silencioso, del mismo tipo que el
  del fps: el entrenamiento corre, la loss baja, y el modelo aprende mal.

HEURISTICA DE SELECCION (atacante)
  Se puntua cada persona por el desplazamiento maximo de muneca dentro de la
  ventana del clip, normalizado por el largo del torso para ser invariante a la
  escala (una persona cerca de la camara desplaza mas pixeles que una lejana
  haciendo el mismo movimiento). El que mas desplaza la muneca es el atacante.

  Supuesto explicito: en una ventana de ~9 frames centrada en un golpe, el
  atacante extiende el brazo y el defensor no, o lo hace menos. NO es cierto en
  intercambios simultaneos ni en contragolpes. Por eso el modo --validate: la
  heuristica hay que medirla, no asumirla.

ASOCIACION ENTRE FRAMES
  Ventanas de ~9 frames, sin necesidad de tracking robusto: asociacion greedy
  por distancia de centroide con umbral. No se usa BoT-SORT a proposito, para no
  arrastrar a la construccion del dataset las decisiones sin calibrar del
  selector de LIVE (ver fighter_selector). Los dos caminos se comparan despues.

SALIDA
  Pickle en formato mmaction2/PoseConv3D: lista de dicts con keypoint (M,T,V,C),
  keypoint_score (M,T,V), label, img_shape, total_frames, frame_dir.

Uso:
  # validacion visual sobre 30 clips (hacer ESTO primero)
  python boxingvi_pose.py --csv ./clips/train.csv --out ./pose --validate 30

  # extraccion completa
  python boxingvi_pose.py --csv ./clips/train.csv --out ./pose/train.pkl
  python boxingvi_pose.py --csv ./clips/val.csv   --out ./pose/val.pkl
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# COCO-17
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12

CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ----------------------------------------------------------------------
# 1. Deteccion cuadro a cuadro
# ----------------------------------------------------------------------
def detect_clip(model, path: Path, conf: float, imgsz: int, device="cpu"):
    """Corre el estimador sobre todos los frames del clip.
    Devuelve (dets, (h, w)) donde dets[t] es lista de dicts por persona."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None, None
    h, w = frames[0].shape[:2]

    res = model.predict(frames, conf=conf, imgsz=imgsz, verbose=False, device=device)
    dets = []
    for r in res:
        per_frame = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            xy = r.keypoints.xy.cpu().numpy()          # (n, 17, 2)
            sc = (r.keypoints.conf.cpu().numpy()
                  if r.keypoints.conf is not None else np.ones(xy.shape[:2]))
            box_c = (r.boxes.conf.cpu().numpy()
                     if r.boxes is not None else np.ones(len(xy)))
            for i in range(len(xy)):
                per_frame.append({"kp": xy[i], "sc": sc[i], "det": float(box_c[i])})
        dets.append(per_frame)
    return dets, (h, w)


# ----------------------------------------------------------------------
# 2. Asociacion greedy entre frames (ventanas cortas, sin tracker)
# ----------------------------------------------------------------------
def _centroid(kp, sc, thr=0.3):
    v = sc > thr
    return kp[v].mean(axis=0) if v.sum() >= 3 else None


def build_tracks(dets, max_dist_ratio=0.15, img_w=1920):
    """Encadena detecciones por proximidad de centroide. Umbral proporcional al
    ancho de imagen: dos personas distintas raramente estan a <15% del ancho, y
    una misma persona raramente se mueve mas que eso entre frames consecutivos."""
    max_dist = max_dist_ratio * img_w
    tracks = []          # cada track: {"frames": {t: det}}
    for t, per_frame in enumerate(dets):
        used = set()
        for tr in tracks:
            last_t = max(tr["frames"])
            if t - last_t > 2:                    # perdido hace mas de 2 frames
                continue
            c_prev = _centroid(tr["frames"][last_t]["kp"], tr["frames"][last_t]["sc"])
            if c_prev is None:
                continue
            best, best_d = None, max_dist
            for i, d in enumerate(per_frame):
                if i in used:
                    continue
                c = _centroid(d["kp"], d["sc"])
                if c is None:
                    continue
                dist = float(np.linalg.norm(c - c_prev))
                if dist < best_d:
                    best, best_d = i, dist
            if best is not None:
                tr["frames"][t] = per_frame[best]
                used.add(best)
        for i, d in enumerate(per_frame):
            if i not in used:
                tracks.append({"frames": {t: d}})
    return tracks


# ----------------------------------------------------------------------
# 3. Heuristica del atacante
# ----------------------------------------------------------------------
def torso_len(kp, sc, thr=0.3):
    """Distancia media hombro-cadera. Escala corporal para normalizar."""
    ds = []
    for s, hp in ((L_SHO, L_HIP), (R_SHO, R_HIP)):
        if sc[s] > thr and sc[hp] > thr:
            ds.append(float(np.linalg.norm(kp[s] - kp[hp])))
    return float(np.mean(ds)) if ds else None


def score_track(tr, min_frames=3, thr=0.3):
    """Puntaje = desplazamiento maximo de muneca / largo de torso.
    Devuelve (score, detalle) o (None, motivo) si el track no es puntuable."""
    ts = sorted(tr["frames"])
    if len(ts) < min_frames:
        return None, "track demasiado corto"

    torsos = [x for x in (torso_len(tr["frames"][t]["kp"], tr["frames"][t]["sc"])
                          for t in ts) if x]
    if not torsos:
        return None, "sin torso visible"
    L = float(np.median(torsos))
    if L < 1e-6:
        return None, "torso degenerado"

    best, best_side = 0.0, None
    for wri in (L_WRI, R_WRI):
        pts = [tr["frames"][t]["kp"][wri] for t in ts
               if tr["frames"][t]["sc"][wri] > thr]
        if len(pts) < min_frames:
            continue
        pts = np.array(pts)
        # rango total recorrido, no solo extremos: captura extension + retorno
        d = float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)))
        if d > best:
            best, best_side = d, "L" if wri == L_WRI else "R"
    if best_side is None:
        return None, "munecas no visibles"

    vis = float(np.mean([tr["frames"][t]["sc"].mean() for t in ts]))
    return best / L, {"disp_norm": best / L, "side": best_side,
                      "torso_px": L, "n_frames": len(ts), "vis": vis}


def pick_attacker(tracks):
    scored = []
    for tr in tracks:
        s, info = score_track(tr)
        if s is not None:
            scored.append((s, tr, info))
    if not scored:
        return None, None, None
    scored.sort(key=lambda x: -x[0])
    margin = (scored[0][0] - scored[1][0]) / scored[0][0] if len(scored) > 1 else 1.0
    return scored[0][1], scored[0][2], {"n_cand": len(scored), "margin": margin}


# ----------------------------------------------------------------------
# 4. Empaquetado formato mmaction2
# ----------------------------------------------------------------------
def track_to_array(tr, total_frames):
    """(1, T, 17, 2) + scores (1, T, 17). Frames sin deteccion quedan en cero,
    que es lo que PoseConv3D espera para ausencia."""
    kp = np.zeros((1, total_frames, 17, 2), dtype=np.float32)
    sc = np.zeros((1, total_frames, 17), dtype=np.float32)
    for t, d in tr["frames"].items():
        if t < total_frames:
            kp[0, t] = d["kp"]
            sc[0, t] = d["sc"]
    return kp, sc


# ----------------------------------------------------------------------
# 5. Validacion visual
# ----------------------------------------------------------------------
def draw_overlay(frame, tracks, chosen, t):
    for tr in tracks:
        if t not in tr["frames"]:
            continue
        d = tr["frames"][t]
        is_pick = tr is chosen
        col = (0, 255, 0) if is_pick else (0, 0, 255)
        kp, sc = d["kp"], d["sc"]
        for a, b in SKELETON:
            if sc[a] > 0.3 and sc[b] > 0.3:
                cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                         col, 2)
        for j in range(17):
            if sc[j] > 0.3:
                cv2.circle(frame, tuple(kp[j].astype(int)), 3, col, -1)
        c = _centroid(kp, sc)
        if c is not None:
            cv2.putText(frame, "ATACANTE" if is_pick else "otro",
                        (int(c[0]) - 40, int(c[1]) - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return frame


def validate(model, rows, out_dir: Path, conf, imgsz, device="cpu"):
    """Genera una tira de contactos por clip con el atacante en verde."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for _, r in rows.iterrows():
        p = Path(r["clip"])
        dets, shape = detect_clip(model, p, conf, imgsz, device)
        if dets is None:
            continue
        tracks = build_tracks(dets, img_w=shape[1])
        chosen, info, meta = pick_attacker(tracks)

        cap = cv2.VideoCapture(str(p))
        frames = []
        t = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(draw_overlay(fr, tracks, chosen, t))
            t += 1
        cap.release()
        if not frames:
            continue

        n = len(frames)
        cols = min(5, n)
        rows_g = int(np.ceil(n / cols))
        hh = 200
        ww = int(frames[0].shape[1] * hh / frames[0].shape[0])
        canvas = np.zeros((rows_g * hh, cols * ww, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            small = cv2.resize(f, (ww, hh))
            y, x = divmod(i, cols)
            canvas[y * hh:(y + 1) * hh, x * ww:(x + 1) * ww] = small

        name = f"{r['video_key']}_{r['start_frame']}_{r['end_frame']}_{r['cls'].replace(' ','_')}.jpg"
        cv2.imwrite(str(out_dir / name), canvas)
        log.append({
            "clip": name, "cls": r["cls"], "n_personas": len(tracks),
            "n_candidatos": meta["n_cand"] if meta else 0,
            "margen": round(meta["margin"], 3) if meta else None,
            "disp_norm": round(info["disp_norm"], 3) if info else None,
            "lado": info["side"] if info else None,
        })

    df = pd.DataFrame(log)
    df.to_csv(out_dir / "validacion.csv", index=False)
    print(f"\n### VALIDACION: {len(df)} clips -> {out_dir} ###")
    if len(df):
        print(df.to_string(index=False))
        amb = df[df["margen"] < 0.15] if "margen" in df else []
        print(f"\ncasos ambiguos (margen <15%): {len(amb)} de {len(df)}")
        print("Abri las imagenes. El esqueleto VERDE tiene que ser quien tira el golpe.\n"
              "Contá cuantos estan mal: esa es la tasa de error de la heuristica y va a Cap 4.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv o val.csv de la Fase C")
    ap.add_argument("--out", required=True, help="ruta .pkl, o carpeta si --validate")
    ap.add_argument("--model", default="yolov8l-pose.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--validate", type=int, default=0,
                    help="N clips de muestra con overlay visual, no escribe pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    if args.device == "auto":
        dev = 0 if torch.cuda.is_available() else "cpu"
    else:
        dev = int(args.device) if args.device.isdigit() else args.device
    print(f"device: {dev}  (cuda disponible: {torch.cuda.is_available()})")
    if dev == "cpu":
        print("[ATENCION] corriendo en CPU. Para 3858 clips va a tardar horas. "
              "Revisa driver/NVML antes de la corrida completa.")
    model = YOLO(args.model)

    df = pd.read_csv(args.csv)
    print(f"clips en {args.csv}: {len(df)}")

    if args.validate:
        # muestra estratificada por clase: se valida sobre todas, no solo las frecuentes
        n_per = max(1, args.validate // df["cls"].nunique())
        parts = [g.sample(min(len(g), n_per), random_state=args.seed)
                 for _, g in df.groupby("cls")]
        sample = pd.concat(parts).reset_index(drop=True)
        validate(model, sample, Path(args.out), args.conf, args.imgsz, dev)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anns, skipped = [], {"sin_frames": 0, "sin_persona": 0, "sin_score": 0}

    for i, r in enumerate(df.itertuples(), 1):
        p = Path(r.clip)
        dets, shape = detect_clip(model, p, args.conf, args.imgsz, dev)
        if dets is None:
            skipped["sin_frames"] += 1
            continue
        if not any(dets):
            skipped["sin_persona"] += 1
            continue
        tracks = build_tracks(dets, img_w=shape[1])
        chosen, info, meta = pick_attacker(tracks)
        if chosen is None:
            skipped["sin_score"] += 1
            continue

        T = len(dets)
        kp, sc = track_to_array(chosen, T)
        anns.append({
            "frame_dir": p.stem,
            "label": CLS2IDX[r.cls],
            "img_shape": shape,
            "original_shape": shape,
            "total_frames": T,
            "keypoint": kp,
            "keypoint_score": sc,
            # trazabilidad para Cap 4
            "video_key": r.video_key,
            "n_personas": len(tracks),
            "sel_margin": round(meta["margin"], 4),
            "sel_disp_norm": round(info["disp_norm"], 4),
        })
        if i % 200 == 0:
            print(f"  ... {i}/{len(df)}")

    with open(out, "wb") as fh:
        pickle.dump({"split": {out.stem: [a["frame_dir"] for a in anns]},
                     "annotations": anns}, fh)

    print(f"\n### RESULTADO ###  poses: {len(anns)} | descartados: {skipped}")
    if anns:
        m = np.array([a["sel_margin"] for a in anns])
        npers = np.array([a["n_personas"] for a in anns])
        print(f"personas por clip: media {npers.mean():.2f}, "
              f"1 persona en {int((npers==1).sum())} clips ({(npers==1).mean()*100:.1f}%)")
        print(f"margen de seleccion: mediana {np.median(m):.3f}, "
              f"ambiguos (<0.15): {int((m<0.15).sum())} ({(m<0.15).mean()*100:.1f}%)")
    print(f"pkl: {out}")


if __name__ == "__main__":
    main()
    