#!/usr/bin/env python3
"""
BoxTwin - Recortes negativos: lo que el detector confunde con un guante.

POR QUE EXISTE
  El dataset publico tiene UNA sola clase y ningun ejemplo de que NO es un guante. Entrenado
  asi, el detector dispara sobre cualquier bulto redondo, liso y saturado, y sobre material
  de ring eso tiene un nombre concreto: los almohadones rojos de las esquinas y las cuerdas.
  Verificado mirando que recorta cuando se equivoca.

  La consecuencia se midio: 20 tracks reciben rol de peleador sin serlo, y revisados uno por
  uno son todos gente AFUERA del ring -espectadores y entrenadores apoyados en las cuerdas-
  que por estar cerca de la camara pasa el filtro de altura y tiene un almohadon rojo justo
  donde estan sus manos. Exigir que el guante caiga donde la pose dice que esta la mano baja
  de 20 a 17 y nada mas: el almohadon esta ahi, asi que ninguna restriccion geometrica lo
  separa. El arreglo no es una heuristica mejor, son negativos.

  La idea de que los negativos salen solos ES FALSA, y se probo. El razonamiento era: los
  tracks que reciben rol de peleador sin serlo ya estan identificados, asi que sus
  detecciones son por construccion falsas. Mirando la hoja de contactos se cae: un entrenador
  puede estar SOSTENIENDO guantes de verdad, y el detector los encuadra bien. Marcar eso como
  negativo le ensena al modelo a ignorar un guante legitimo, que es el error opuesto al que
  se queria arreglar.

  Se intento entonces separar por geometria, quedandose con las detecciones lejos de las
  manos -escenografia, no puede ser de esa persona-. Medido sobre 70 detecciones de tracks
  equivocados: 69 caen CERCA de las manos y una sola lejos. No hay nada que separar
  automaticamente.

  Asi que el set sale completo y la poda es manual. Queda una copia de cada recorte con las
  cajas dibujadas en revisar/, y sacar uno del entrenamiento es borrar su .jpg de images/ y
  su .txt de labels/. Con --confirmados se puede pasar en cambio una lista blanca.

QUE HACE
  Encuentra los tracks a los que la asignacion automatica les da rol de peleador y la
  anotacion manual no, y guarda sus recortes de persona con un archivo de etiquetas VACIO,
  que es como YOLO representa un negativo.

USO
  python tools/negativos.py --evidencia <dir> --out data/negativos
  # revisar data/negativos/revisar/ y armar la lista
  python tools/negativos.py --evidencia <dir> --out data/negativos --confirmados lista.txt
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--evidencia", type=Path, required=True,
                   help="directorio con la evidencia ya medida, uno por proyecto")
    p.add_argument("--modelo", default="modelos/guantes.pt")
    p.add_argument("--out", type=Path, default=Path("data/negativos"))
    p.add_argument("--por-track", type=int, default=25, dest="por_track",
                   help="cuantos recortes tomar de cada track equivocado")
    p.add_argument("--margen", type=float, default=0.08)
    p.add_argument("--confirmados", type=Path, default=None,
                   help="archivo con un nombre de recorte por linea. Si se pasa, solo esos "
                        "reciben etiqueta; sin el, la reciben todos")
    args = p.parse_args()

    confirmados = set()
    if args.confirmados and args.confirmados.is_file():
        confirmados = {l.strip() for l in args.confirmados.read_text().splitlines() if l.strip()}

    from boxtwin.core.annotations import load
    from boxtwin.core.deteccion_guantes import DetectorGuantes
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, cargar_evidencia, proponer
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import TrackRole

    cfg = ConfigIdentidadAuto()
    det = DetectorGuantes(args.modelo)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    # Una copia con las cajas dibujadas, una por recorte, para poder podar despues sin
    # volver a generar nada: se borra el .jpg de images/ y su .txt y listo.
    (args.out / "revisar").mkdir(parents=True, exist_ok=True)
    indice: list[str] = []
    (args.out / "revisar").mkdir(parents=True, exist_ok=True)
    indice: list[str] = []

    total = 0
    con_deteccion = 0
    hoja: list[np.ndarray] = []
    resumen = []

    for proy, vid in PROYECTOS:
        ev_path = args.evidencia / f"{proy}.json"
        if not ev_path.is_file():
            print(f"  {proy}: sin evidencia, se saltea")
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
        n_proy = 0
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
                rec, ox, oy = det.recorte(frame, d.bbox[i])
                if rec.size == 0 or rec.shape[0] < 32:
                    continue
                cajas = det.detectar(frame, d.bbox[i])
                nombre = f"{vid}_t{t}_f{f}"
                cv2.imwrite(str(args.out / "images" / f"{nombre}.jpg"), rec)
                # La etiqueta vacia -que en YOLO es un negativo- SOLO si alguien lo confirmo.
                # Escribirla sin revisar es afirmar que ahi no hay ningun guante, y en varios
                # de estos recortes hay guantes de verdad en manos de un entrenador.
                # Etiqueta vacia: en YOLO, una imagen sin cajas es un negativo.
                if not confirmados or nombre in confirmados:
                    (args.out / "labels" / f"{nombre}.txt").write_text("")
                indice.append(nombre)
                total += 1
                n_proy += 1
                if cajas:
                    con_deteccion += 1
                    vis = rec.copy()
                    for c in cajas:
                        x0, y0, x1, y1 = (int(v) for v in c.xyxy)
                        cv2.rectangle(vis, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy),
                                      (0, 0, 255), 2)
                    cv2.imwrite(str(args.out / "revisar" / f"{nombre}.jpg"), vis)
                    if len(hoja) < 48:
                        h, w = vis.shape[:2]
                        hoja.append(cv2.resize(vis, (max(20, int(w * 150 / h)), 150)))
        cap.release()
        resumen.append((vid, len(falsos), n_proy))

    (args.out / "indice.txt").write_text("\n".join(sorted(indice)) + "\n")
    print(f"\n{total} recortes candidatos, {con_deteccion} con al menos una deteccion "
          f"({con_deteccion / max(1, total):.0%})")
    escritas = len(confirmados & set(indice)) if confirmados else total
    print(f"etiquetas vacias escritas: {escritas}")
    for vid, nf, nr in resumen:
        print(f"  {vid:<20} {nf:>3} tracks equivocados -> {nr:>4} recortes")

    if hoja:
        anchos = [c.shape[1] for c in hoja]
        filas = []
        i = 0
        while i < len(hoja):
            fila, ancho = [], 0
            while i < len(hoja) and ancho + anchos[i] < 1400:
                fila.append(hoja[i]); ancho += anchos[i]; i += 1
            if fila:
                filas.append(np.hstack(fila))
        ancho = max(f.shape[1] for f in filas)
        lienzo = np.zeros((150 * len(filas), ancho, 3), np.uint8)
        for j, f in enumerate(filas):
            lienzo[j * 150:(j + 1) * 150, :f.shape[1]] = f
        destino = args.out / "revisar.jpg"
        cv2.imwrite(str(destino), lienzo)
        print(f"\nhoja de contactos en {destino}, y uno por archivo en {args.out}/revisar/")
        print("Hay que revisarlos: varios de estos recortes tienen guantes DE VERDAD en la")
        print("mano de un entrenador, y el detector los encuadra bien. Medido, 69 de 70")
        print("detecciones caen en posicion de mano, asi que no hay regla que los separe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
