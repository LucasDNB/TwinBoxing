#!/usr/bin/env python3
"""
BoxTwin - Parches ajustados de cada deteccion dudosa, para revisar de a uno.

POR QUE EXISTE
  El intento anterior uso el recorte de persona entero como negativo, con etiqueta vacia, y
  fue un desastre medido: la fraccion de guante sobre tracks de PELEADOR cayo de 0,868 a
  0,344, los tracks que superan el umbral pasaron de 119 de 125 a 45, y cuatro de las seis
  fuentes dejaron de decidir nada. El mAP de validacion apenas se movio -0,895 a 0,876-
  porque se mide sobre fotografia de producto, asi que mirando solo ese numero el
  experimento parecia neutro.

  La causa es el tamano de la unidad. Un recorte de persona de un entrenador puede tener un
  guante DE VERDAD adentro, y la etiqueta vacia afirma que ahi no hay ninguno. El modelo
  aprendio a ignorar guantes legitimos.

  Un parche ajustado a la deteccion afirma solo sobre esa deteccion. Y revisarlo es rapido:
  se ve de una si es un almohadon de esquina o un guante.

QUE HACE
  Por cada deteccion sobre los tracks que reciben rol de peleador sin serlo, guarda un parche
  con algo de contexto y lo indexa. Arma hojas de contactos numeradas para revisar en bloque.

  El veredicto se devuelve como una lista de los que SI son guante, que se espera que sean
  los menos; el resto quedan confirmados como negativos.

USO
  python tools/parches.py --evidencia <dir> --out data/parches
  # revisar las hojas, anotar los indices que SI son guante
  python tools/parches.py --evidencia <dir> --out data/parches --son-guante 3,17,40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROYECTOS = [
    ("anotacion", "Sparring"),
    ("anotacion-spar-01", "01-sparring"),
    ("anotacion-spar-02", "02-sparring"),
    ("anotacion-spar-03", "03-sparring"),
    ("anotacion-spar-04", "04-sparring"),
    ("anotacion-sparring-3", "sparring-3-rounds"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("--raiz", type=Path, default=Path.home() / "Proyectos/TwinBoxing")
    p.add_argument("--evidencia", type=Path, required=True)
    p.add_argument("--modelo", default="modelos/guantes.pt")
    p.add_argument("--out", type=Path, default=Path("data/parches"))
    p.add_argument("--por-track", type=int, default=20, dest="por_track")
    p.add_argument("--contexto", type=float, default=1.6,
                   help="cuanto se agranda la caja de la deteccion al recortar el parche")
    p.add_argument("--son-guante", default="", dest="son_guante",
                   help="indices, separados por coma, de los parches que SI son un guante")
    args = p.parse_args()

    from boxtwin.core.annotations import load
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, cargar_evidencia, proponer
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import TrackRole

    cfg = ConfigIdentidadAuto()
    det = DetectorGuantes(args.modelo)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parches").mkdir(exist_ok=True)

    registro = []
    for proy, vid in PROYECTOS:
        ev_path = args.evidencia / f"{proy}.json"
        if not ev_path.is_file():
            continue
        ev = cargar_evidencia(ev_path)
        doc, _ = load(args.raiz / proy / f"annotations/{vid}.annot.json")
        prop = proponer(ev, cfg, doc.video.total_frames, doc.video.fps)
        verdad = {a.track_id for a in doc.identity.assignments
                  if a.role in (TrackRole.A, TrackRole.B)}
        falsos = sorted(t for t, r in prop.roles.items()
                        if r in (TrackRole.A, TrackRole.B) and t not in verdad)
        if not falsos:
            continue
        cache = PoseCache.open(args.raiz / proy / f"cache/{vid}.pose.npz")
        cap = cv2.VideoCapture(str(args.raiz / proy / f"videos/{vid}.mp4"))
        for t in falsos:
            e = ev[t]
            cuadros = [f for f in range(e.primer_frame, e.ultimo_frame + 1, cfg.paso)
                       if t in cache.detections(f).track_id.tolist()]
            if not cuadros:
                continue
            paso = max(1, len(cuadros) // args.por_track)
            for f in cuadros[::paso][:args.por_track]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                d = cache.detections(f)
                i = d.track_id.tolist().index(t)
                alto, ancho = frame.shape[:2]
                for k, c in enumerate(det.detectar(frame, d.bbox[i])):
                    x0, y0, x1, y1 = c.xyxy
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    w = (x1 - x0) * args.contexto / 2
                    h = (y1 - y0) * args.contexto / 2
                    px0, py0 = max(0, int(cx - w)), max(0, int(cy - h))
                    px1, py1 = min(ancho, int(cx + w)), min(alto, int(cy + h))
                    parche = frame[py0:py1, px0:px1]
                    if parche.size == 0 or parche.shape[0] < 12:
                        continue
                    registro.append({
                        "proy": proy, "vid": vid, "track": t, "frame": f, "det": k,
                        "xyxy": [float(v) for v in c.xyxy], "conf": c.conf,
                        "parche": f"{vid}_t{t}_f{f}_d{k}.jpg",
                    })
                    cv2.imwrite(str(args.out / "parches" / registro[-1]["parche"]), parche)
        cap.release()

    for n, r in enumerate(registro):
        r["idx"] = n
    (args.out / "indice.json").write_text(json.dumps(registro, indent=1) + "\n")

    # Hojas numeradas, para poder nombrar los indices sin abrir archivo por archivo.
    POR_FILA, LADO = 12, 110
    hojas = 0
    for inicio in range(0, len(registro), POR_FILA * 8):
        bloque = registro[inicio:inicio + POR_FILA * 8]
        celdas = []
        for r in bloque:
            img = cv2.imread(str(args.out / "parches" / r["parche"]))
            if img is None:
                img = np.zeros((LADO, LADO, 3), np.uint8)
            img = cv2.resize(img, (LADO, LADO))
            cv2.rectangle(img, (0, 0), (38, 16), (0, 0, 0), -1)
            cv2.putText(img, str(r["idx"]), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            celdas.append(img)
        filas = [np.hstack(celdas[i:i + POR_FILA])
                 for i in range(0, len(celdas) - len(celdas) % POR_FILA, POR_FILA)]
        if not filas:
            continue
        cv2.imwrite(str(args.out / f"hoja_{hojas:02d}.jpg"), np.vstack(filas))
        hojas += 1

    print(f"\n{len(registro)} parches de {len({(r['vid'], r['track']) for r in registro})} "
          f"tracks, en {hojas} hoja/s")

    son_guante = {int(x) for x in args.son_guante.split(",") if x.strip().isdigit()}
    if son_guante:
        negativos = [r for r in registro if r["idx"] not in son_guante]
        (args.out / "veredicto.json").write_text(json.dumps({
            "son_guante": sorted(son_guante),
            "negativos": [r["parche"] for r in negativos],
        }, indent=1) + "\n")
        print(f"veredicto: {len(son_guante)} son guante, {len(negativos)} confirmados negativos")
    else:
        print("revisar las hojas y volver a correr con --son-guante con los indices que SI "
              "sean un guante")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
