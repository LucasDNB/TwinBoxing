#!/usr/bin/env python3
"""
BoxTwin - ¿El color del guante distingue al peleador A del B?

POR QUE EXISTE
  El detector de guantes ya separa peleador de no peleador: medido sobre seis fuentes, los
  tracks de peleador dan fraccion de guante entre 0,59 y 0,96 contra 0,22 a 0,33 del resto.
  Lo que NO resuelve es A contra B, que es donde esta el trabajo manual: de los 390 relevos
  de track medidos, el 65% tiene al otro peleador como candidato, y los dos tienen guantes.

  El color es la unica senal de apariencia que podria separarlos. El ReID generico ya se
  midio y no aporta a 640x360 (18-08-2026), pero ese encoder promedia el cuerpo entero, que
  a esa resolucion no deja senal. El guante es lo contrario: el objeto mas saturado de la
  escena, y el color es senal de baja frecuencia que sobrevive al downsampling mucho mejor
  que la forma.

  El riesgo, y por eso esto se mide antes de disenar nada: que los dos peleadores lleven
  guantes del mismo color. En sparring de gimnasio es perfectamente posible, y si pasa la
  senal es cero y no hay modelo que la invente.

QUE HACE
  Sobre los cuadros donde la anotacion dice quien es quien, detecta los guantes, les muestrea
  el color, y arma un perfil por peleador. Reporta el color de cada uno y que tan separables
  son, clasificando cada guante contra los dos perfiles.

  Solo usa cuadros donde los dos peleadores estan AISLADOS -cajas que no se superponen- para
  que el parche no tenga cuerpo del rival adentro. En un clinch las cajas se solapan casi por
  completo y ahi el color medido no es de quien dice ser.

USO
  python tools/color_guante.py ~/Proyectos/TwinBoxing/anotacion-spar-01 --modelo modelos/guantes.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Cortes sobre el tono de OpenCV, que va de 0 a 179 y no de 0 a 359. La primera version
# usaba los cortes de la escala de 360 y llamaba "violeta" a un azul y "cian" a un verde
# amarillento: los nombres del reporte quedaban al lado del tono correcto y contradiciendolo.
NOMBRES = [
    (8, "rojo"), (22, "naranja"), (33, "amarillo"), (78, "verde"), (99, "cian"),
    (129, "azul"), (155, "violeta"), (170, "rosa"), (180, "rojo"),
]


def _nombre_de_tono(h: float, s: float, v: float) -> str:
    if v < 50:
        return "negro"
    if s < 45:
        return "blanco/gris"
    for lim, n in NOMBRES:
        if h <= lim:
            return n
    return "rojo"


def color_de_parche(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """
    Tono, saturacion y valor del parche, sobre los pixeles que tienen color.

    El tono se promedia circularmente: es un angulo, y el rojo vive en los dos extremos de
    la escala, asi que una media aritmetica de 5 y 175 daria verde.
    """
    if bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    # El centro del 60%: los bordes de la caja traen fondo y guante del rival.
    bgr = bgr[int(h * 0.2):int(h * 0.8) or 1, int(w * 0.2):int(w * 0.8) or 1]
    if bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(float), hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    mascara = (S > 40) & (V > 40)
    if mascara.sum() < 12:
        # Sin pixeles con color: es negro o blanco, y el tono no significa nada.
        return (-1.0, float(np.median(S)), float(np.median(V)))
    ang = H[mascara] * 2 * np.pi / 180.0
    medio = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    tono = (np.degrees(medio) % 360) / 2.0
    return (float(tono), float(np.median(S[mascara])), float(np.median(V[mascara])))


def perfil_de(colores: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Color representativo de un conjunto. El tono se promedia circularmente."""
    arr = np.array(colores)
    con = arr[arr[:, 0] >= 0]
    if len(con):
        ang = con[:, 0] * 2 * np.pi / 180
        t = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2
    else:
        t = -1.0
    return (float(t), float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Distancia entre dos colores. El tono pesa mas, pero solo si los dos lo tienen."""
    ha, sa, va = a
    hb, sb, vb = b
    d_sv = (abs(sa - sb) + abs(va - vb)) / 255.0
    if ha < 0 or hb < 0:
        # Alguno es acromatico: solo comparan saturacion y valor, y se penaliza si uno
        # tiene color y el otro no.
        return d_sv + (0.5 if (ha < 0) != (hb < 0) else 0.0)
    dh = abs(ha - hb)
    dh = min(dh, 180 - dh) / 90.0
    return 2.0 * dh + d_sv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("USO")[0])
    p.add_argument("proyecto", type=Path)
    p.add_argument("--modelo", default="modelos/guantes.pt")
    p.add_argument("--paso", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--guardar", default=None,
                   help="volcar los colores medidos a un .json, para re-analizar sin volver "
                        "a detectar")
    p.add_argument("--min-guantes", type=int, default=10, dest="min_guantes",
                   help="tracks con menos guantes que esto no entran al voto")
    p.add_argument("--iou-max", type=float, default=0.02, dest="iou_max",
                   help="solapamiento maximo entre las cajas de los dos peleadores para que "
                        "el cuadro cuente. En clinch el parche tendria cuerpo del rival")
    args = p.parse_args()

    from boxtwin.core.annotations import load
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.interpolation import iou
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.types import FighterId
    from ultralytics import YOLO

    P = args.proyecto
    video = next((P / "videos").glob("*.mp4"))
    doc, _ = load(P / f"annotations/{video.stem}.annot.json")
    cache = PoseCache.open(P / f"cache/{video.stem}.pose.npz")
    res = IdentityResolver(doc, cache)
    modelo = YOLO(args.modelo)

    cap = cv2.VideoCapture(str(video))
    alto_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ancho_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    colores = {FighterId.A: [], FighterId.B: []}
    # Mismo dato agrupado por track: la identidad se decide por track, no por guante suelto.
    por_track: dict[tuple, list] = {}
    aislados = solapados = 0

    for f in range(0, doc.video.total_frames, args.paso):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        porpel = res.by_fighter(f)
        a, b = porpel[FighterId.A], porpel[FighterId.B]
        if a is None or b is None or a.interpolated or b.interpolated:
            continue
        ca = np.asarray(a.bbox, dtype=np.float64)
        cb = np.asarray(b.bbox, dtype=np.float64)
        if iou(ca, cb) > args.iou_max:
            solapados += 1
            continue
        aislados += 1

        for pel, pose in ((FighterId.A, a), (FighterId.B, b)):
            x0, y0, x1, y1 = (int(v) for v in pose.bbox)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(ancho_img, x1), min(alto_img, y1)
            recorte = frame[y0:y1, x0:x1]
            if recorte.size == 0:
                continue
            r = modelo.predict(recorte, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if r.boxes is None:
                continue
            for gx0, gy0, gx1, gy1 in r.boxes.xyxy.cpu().numpy():
                parche = recorte[int(gy0):int(gy1), int(gx0):int(gx1)]
                c = color_de_parche(parche)
                if c is not None:
                    colores[pel].append(c)
                    por_track.setdefault((pel, pose.track_id), []).append(c)
    cap.release()

    if args.guardar:
        import json
        Path(args.guardar).write_text(json.dumps({
            "fuente": video.stem,
            "aislados": aislados, "solapados": solapados,
            "por_track": [{"rol": pel.value, "track": tid, "colores": cs}
                          for (pel, tid), cs in por_track.items()],
        }) + "\n")

    print(f"\n=== {video.stem}")
    print(f"cuadros con los dos peleadores aislados: {aislados}  (descartados por "
          f"solapamiento: {solapados})")
    for pel in (FighterId.A, FighterId.B):
        cs = colores[pel]
        if not cs:
            print(f"  {pel.value}: sin guantes detectados")
            continue
        arr = np.array(cs)
        con_tono = arr[arr[:, 0] >= 0]
        if len(con_tono):
            ang = con_tono[:, 0] * 2 * np.pi / 180
            tono = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2
        else:
            tono = -1.0
        s, v = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
        print(f"  {pel.value}: {len(cs):>4} guantes | tono {tono:6.1f} sat {s:5.1f} "
              f"val {v:5.1f} | {_nombre_de_tono(tono, s, v)}"
              f"  ({len(con_tono)}/{len(cs)} con color)")

    # Separabilidad: cada guante contra el perfil mediano del otro peleador, dejandolo afuera.
    if colores[FighterId.A] and colores[FighterId.B]:
        perfiles = {}
        for pel in (FighterId.A, FighterId.B):
            arr = np.array(colores[pel])
            con = arr[arr[:, 0] >= 0]
            if len(con):
                ang = con[:, 0] * 2 * np.pi / 180
                t = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2
            else:
                t = -1.0
            perfiles[pel] = (t, float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))
        ok_tot = n_tot = 0
        for pel in (FighterId.A, FighterId.B):
            otro = FighterId.B if pel is FighterId.A else FighterId.A
            ok = sum(1 for c in colores[pel]
                     if _dist(c, perfiles[pel]) < _dist(c, perfiles[otro]))
            print(f"  guantes de {pel.value} mas cerca de SU perfil: {ok}/{len(colores[pel])}"
                  f" = {ok/len(colores[pel]):.1%}")
            ok_tot += ok; n_tot += len(colores[pel])
        print(f"\n  acierto POR GUANTE: {ok_tot}/{n_tot} = {ok_tot/n_tot:.1%}   (azar = 50%)")
        print(f"  distancia entre los dos perfiles: "
              f"{_dist(perfiles[FighterId.A], perfiles[FighterId.B]):.3f}")

        # Por track: cada track vota con todos sus guantes. Es la decision real, y se mide
        # en vez de deducirla de una binomial: los errores estan correlacionados entre
        # cuadros vecinos, asi que el N efectivo es mucho menor que el crudo y una cuenta
        # de independencia daria una certeza que no existe.
        ok_t = n_t = empates = 0
        malos = []
        for (pel, tid), cs in sorted(por_track.items(), key=lambda x: -len(x[1])):
            if len(cs) < args.min_guantes:
                continue
            otro = FighterId.B if pel is FighterId.A else FighterId.A
            # Perfil SIN este track: armarlo con los guantes que despues se clasifican es
            # fuga, y el numero sale inflado. Asi mide lo que de verdad va a pasar con un
            # track nuevo contra perfiles ya establecidos.
            resto = [c for (pp, tt), lst in por_track.items() if pp is pel and tt != tid
                     for c in lst]
            if len(resto) < args.min_guantes:
                continue
            perfil_propio = perfil_de(resto)
            votos = sum(1 for c in cs
                        if _dist(c, perfil_propio) < _dist(c, perfiles[otro]))
            n_t += 1
            if votos * 2 > len(cs):
                ok_t += 1
            else:
                if votos * 2 == len(cs):
                    empates += 1
                malos.append((pel.value, tid, votos, len(cs)))
        if n_t:
            print(f"  acierto POR TRACK, perfil sin ese track (minimo {args.min_guantes} "
                  f"guantes): {ok_t}/{n_t} = {ok_t/n_t:.1%}"
                  + (f"   [{empates} empate/s]" if empates else ""))
            for rol, tid, v, n in malos:
                print(f"      falla: {rol} track {tid} voto {v}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
