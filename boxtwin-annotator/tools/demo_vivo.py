#!/usr/bin/env python3
"""
BoxTwin - Demo del sistema completo sobre video, con ventana OpenCV.

POR QUE EXISTE
  Los numeros de una matriz de confusion no muestran COMO falla un sistema. Ver el pipeline
  corriendo sobre video, con los golpes apareciendo a medida que pasan, hace visibles cosas
  que una tabla esconde: que un hook lejano se clasifica como straight, que en el clinch la
  identidad se mezcla, que hay golpes que el detector simplemente no ve.

QUE CAMBIO RESPECTO DE LA PRIMERA VERSION
  La version del 01-09 disparaba con una heuristica de extension de muneca, porque el
  detector no existia. Despues se midio que esa heuristica NO SUPERA AL AZAR: con 42 golpes
  por minuto, casi la mitad de la linea de tiempo esta a menos de medio segundo de un golpe
  por construccion, y la heuristica acertaba 0,52 contra 0,46 de tirar dardos.

  Ahora dispara el DETECTOR entrenado, un ensamble de cinco semillas sobre una TCN de
  convoluciones dilatadas. Medido sobre sparring-3 entero, con el detector entrenado SIN esa
  fuente: precision 0,885 contra 0,21 de la heuristica, y error de fronteras de 1,14 cuadros
  al inicio y 0,97 al final, contra 1,12 y 1,55 del acuerdo intra-anotador.

  DOS ADVERTENCIAS QUE NO SON LETRA CHICA:

  1. El detector se entrena SIN la fuente que se mira. El checkpoint por defecto es el del
     fold que deja sparring-3 afuera, asi que sobre ese video lo que se ve es honesto. Sobre
     otro video hay que pasar el ensamble de SU fold, o se estaria mirando el entrenamiento.

  2. El clasificador de familia no generaliza. Reentrenado sobre las siete fuentes acierta
     0,746 por familia EN DISTRIBUCION, y confunde 38 de 76 hooks con straight. Ese eje esta
     medido cinco veces por caminos independientes y no mejora con mas datos: reentrenar con
     3,7 veces mas movio el numero 0,001. Lo que se ve en la etiqueta del golpe es, sobre una
     fuente nueva, poco mas que el prior.

QUE HACE
  Reproduce el video con el overlay de pose coloreado por peleador, corre el detector sobre
  la secuencia entera, y lleva UNA CONSOLA POR PELEADOR a cada lado del cuadro: su cuenta
  acumulada, el desglose por familia y los ultimos golpes a medida que salen.

  Las consolas van al costado y no encima del video, porque superpuestas taparian justo lo
  que se esta mirando. Y son dos y no una compartida: con las dos columnas juntas hay que
  leer el rol de cada renglon para saber de quien es el golpe, y mirando el video al mismo
  tiempo eso no se hace.

  Con --out escribe el video procesado en vez de -o ademas de- mostrarlo.

  La pose sale del cache del preproceso y las detecciones se calculan una vez al arrancar:
  el objetivo es mirar el sistema, no medir su velocidad.

USO
  ~/miniforge3/envs/boxtwin_mmaction/bin/python tools/demo_vivo.py <proyecto>
  ... --desde 3000 --velocidad 0.5 --umbral 0.85
  ... --sin-ventana --out /tmp/procesado.mp4          # escribir sin abrir ventana

  Teclas: espacio pausa, flechas mueven de a un cuadro en pausa, q sale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# El detector vive en el paquete hermano, que no esta instalado en boxtwin_mmaction: ese
# entorno esta pinneado y no se toca. La TCN es PyTorch puro y corre igual con torch 2.1,
# verificado, asi que alcanza con ponerlo en el path.
_RAIZ = Path(__file__).resolve().parents[2]
for _p in (_RAIZ / "boxtwin-detector" / "src", _RAIZ / "boxtwin-annotator" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CLASES = ["jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut"]

# COCO-17. Se repiten aca en vez de importarlas de boxtwin.core.constants porque este script
# corre en boxtwin_mmaction, donde el paquete del anotador no esta instalado y donde no se
# tocan las dependencias.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10

COLOR = {"fighter_A": (76, 88, 236), "fighter_B": (235, 158, 74)}  # BGR, igual que el anotador
GRIS = (128, 128, 128)


# ---------------------------------------------------------------------------
# Lectura del cache y de la anotacion, sin depender del paquete boxtwin
# ---------------------------------------------------------------------------


class Cache:
    """Cache de pose en formato CSR: frame_index[i] es donde arrancan las detecciones de i."""

    def __init__(self, path: Path) -> None:
        z = np.load(path)
        self.idx = z["frame_index"]
        self.track_id = z["track_id"]
        self.bbox = z["bbox"]
        self.kp = z["keypoints"]
        self.score = z["kp_score"]
        self.n_frames = len(self.idx) - 1

    def en(self, f: int):
        """(track_id, bbox, keypoints, score) de cada deteccion del cuadro f."""
        if not 0 <= f < self.n_frames:
            return []
        a, b = int(self.idx[f]), int(self.idx[f + 1])
        return [
            (int(self.track_id[i]), self.bbox[i], self.kp[i], self.score[i])
            for i in range(a, b)
        ]


class Identidad:
    """Resuelve track_id -> peleador por intervalos, como hace el anotador al exportar."""

    def __init__(self, annot: Path) -> None:
        d = json.loads(annot.read_text(encoding="utf-8"))
        self.video = d["video"]
        self.por_track: dict[int, list[tuple[int, int, str]]] = {}
        for a in d["identity"]["assignments"]:
            self.por_track.setdefault(a["track_id"], []).append(
                (a["start_frame"], a["end_frame_excl"], a["role"])
            )

    def rol(self, track_id: int, frame: int) -> str | None:
        for ini, fin, rol in self.por_track.get(track_id, ()):
            if ini <= frame < fin:
                return rol
        return None


# ---------------------------------------------------------------------------
# Detector: el ensamble entrenado que reemplazo a la heuristica
# ---------------------------------------------------------------------------


class Detector:
    """
    Decide CUANDO hay un golpe, con el ensamble entrenado.

    Reemplaza a la heuristica de extension de muneca de la primera version, que quedo medida
    sin superar al azar. Corre una sola vez sobre la secuencia entera al arrancar: son unos
    segundos y evita reproducir el video a merced de la GPU.

    El ensamble tiene que ser el del fold que deja afuera la fuente que se mira. Si se le pasa
    uno que la vio, lo que se ve en pantalla es el entrenamiento y no una prediccion.
    """

    def __init__(self, ensamble_path: Path, umbral: float, device: str = "cuda:0") -> None:
        import torch

        from boxtwin_detector.ensamble import cargar

        self.ens = cargar(Path(ensamble_path), torch.device(device))
        self.umbral = umbral
        self.device = torch.device(device)

    def segmentos(self, kp_por_rol: dict, sc_por_rol: dict, T: int) -> list[dict]:
        """
        Todos los golpes del video, por peleador y brazo.

        kp_por_rol[rol] es (T, 17, 2) con la identidad ya resuelta; los cuadros sin pose van
        en cero y quedan fuera de la mascara, que es lo que el detector espera.
        """
        import numpy as np
        import torch

        from boxtwin_detector.decodificacion import decodificar_score, probabilidad_de_golpe
        from boxtwin_detector.entrenamiento import predecir_secuencia
        from boxtwin_detector.features import features_de

        out = []
        for rol in ("A", "B"):
            kp, sc = kp_por_rol[rol], sc_por_rol[rol]
            for brazo in ("left", "right"):
                f, usable = features_de(kp, sc, brazo)
                x = self.ens.estandarizador.aplicar(f)
                acum = np.zeros(T, np.float64)
                for m in self.ens.modelos:
                    acum += probabilidad_de_golpe(
                        predecir_secuencia(m, x, self.ens.config, self.device))
                prob = (acum / self.ens.n).astype(np.float32)
                for s in decodificar_score(prob, self.umbral, valido=usable):
                    out.append({
                        "rol": rol, "brazo": brazo,
                        "inicio": s.inicio, "fin": s.fin,
                        "prob": float(prob[s.inicio : s.fin + 1].max()),
                    })
        out.sort(key=lambda d: d["inicio"])
        return out


# ---------------------------------------------------------------------------
# Clasificador
# ---------------------------------------------------------------------------


class Clasificador:
    def __init__(self, config: Path, checkpoint: Path, device: str = "cuda:0") -> None:
        from mmaction.apis import init_recognizer
        from mmengine.dataset import Compose

        self.modelo = init_recognizer(str(config), str(checkpoint), device=device)
        cfg = self.modelo.cfg
        pipe = cfg.test_dataloader.dataset.pipeline
        self.pipeline = Compose(pipe)
        self.device = device

    def __call__(self, kp: np.ndarray, score: np.ndarray, alto: int, ancho: int):
        """kp (T,17,2) y score (T,17) de UN peleador. Devuelve (clase, probabilidad)."""
        import torch

        muestra = {
            "frame_dir": "vivo",
            "label": 0,
            "img_shape": (alto, ancho),
            "original_shape": (alto, ancho),
            "total_frames": int(kp.shape[0]),
            "start_index": 0,
            "modality": "Pose",
            "keypoint": kp[None].astype(np.float32),
            "keypoint_score": score[None].astype(np.float32),
        }
        dato = self.pipeline(muestra)
        with torch.no_grad():
            salida = self.modelo.test_step(
                {"inputs": [dato["inputs"]], "data_samples": [dato["data_samples"]]}
            )[0]
        p = salida.pred_score.cpu().numpy()
        i = int(np.argmax(p))
        return CLASES[i], float(p[i])


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------


def dibujar_pose(img, kp, score, color, umbral=0.3):
    for a, b in EDGES:
        if score[a] >= umbral and score[b] >= umbral:
            cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), color, 2)
    for i in range(len(kp)):
        if score[i] >= umbral:
            cv2.circle(img, tuple(kp[i].astype(int)), 3, color, -1)


ANCHO_CONSOLA = 300


def dibujar_consola(lienzo, x0, ancho, rol, conteo, eventos, fps):
    """
    La consola de un peleador: su cuenta acumulada y los golpes a medida que salen.

    Una consola por peleador y no una compartida, que es como estaba. Con las dos columnas
    juntas hay que leer el rol de cada renglon para saber de quien es el golpe, y mirando el
    video al mismo tiempo eso no se hace: la vista va al peleador, no a la etiqueta. Separadas
    y del color de cada uno, la cuenta se lee de reojo.
    """
    alto = lienzo.shape[0]
    f = cv2.FONT_HERSHEY_SIMPLEX
    color = COLOR[rol]
    cv2.rectangle(lienzo, (x0, 0), (x0 + ancho, alto), (18, 18, 20), -1)
    cv2.line(lienzo, (x0, 0), (x0, alto), color, 3)
    x = x0 + 16

    etiqueta = "PELEADOR A" if rol.endswith("A") else "PELEADOR B"
    cv2.putText(lienzo, etiqueta, (x, 34), f, 0.62, color, 2)

    total = sum(conteo.values())
    cv2.putText(lienzo, str(total), (x, 92), f, 1.6, color, 3)
    cv2.putText(lienzo, "golpes", (x + 12 + 34 * len(str(total)), 92), f, 0.5, (150, 150, 150), 1)

    y = 124
    cv2.line(lienzo, (x - 6, y), (x0 + ancho - 14, y), (60, 60, 60), 1)
    y += 24
    for cl in CLASES:
        n = conteo.get(cl, 0)
        tono = (190, 190, 190) if n else (85, 85, 85)
        cv2.putText(lienzo, cl, (x, y), f, 0.44, tono, 1)
        cv2.putText(lienzo, f"{n:3d}", (x0 + ancho - 52, y), f, 0.48, color if n else (85, 85, 85),
                    2 if n else 1)
        y += 21

    y += 8
    cv2.line(lienzo, (x - 6, y), (x0 + ancho - 14, y), (60, 60, 60), 1)
    y += 22
    cv2.putText(lienzo, "ultimos", (x, y), f, 0.42, (130, 130, 130), 1)
    y += 20
    # Del mas nuevo al mas viejo: lo que acaba de pasar es lo que se esta mirando.
    for fr, cl, pr in list(eventos)[::-1]:
        if y > alto - 14:
            break
        cv2.putText(lienzo, f"{fr/fps:6.1f}s", (x, y), f, 0.4, (120, 120, 120), 1)
        cv2.putText(lienzo, cl[:13], (x + 56, y), f, 0.42, color, 1)
        cv2.putText(lienzo, f"{pr:.2f}", (x0 + ancho - 46, y), f, 0.38, (120, 120, 120), 1)
        y += 18


def componer(img, conteo, eventos, frame, fps, pausado, ancho_consola=ANCHO_CONSOLA):
    """
    Arma el cuadro final: consola de A, el video, consola de B.

    Las consolas van al costado y no encima del video. Superpuestas taparian justo lo que se
    esta mirando, y el punto de esto es poder ver el golpe y su etiqueta a la vez.
    """
    alto, ancho = img.shape[:2]
    lienzo = np.zeros((alto, ancho + 2 * ancho_consola, 3), np.uint8)
    lienzo[:, ancho_consola:ancho_consola + ancho] = img
    dibujar_consola(lienzo, 0, ancho_consola, "fighter_A", conteo["fighter_A"],
                    eventos["fighter_A"], fps)
    dibujar_consola(lienzo, ancho_consola + ancho, ancho_consola, "fighter_B",
                    conteo["fighter_B"], eventos["fighter_B"], fps)

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lienzo, f"cuadro {frame}   {frame/fps:6.1f}s", (ancho_consola + 14, alto - 14),
                f, 0.5, (210, 210, 210), 1)
    if pausado:
        cv2.putText(lienzo, "PAUSA", (ancho_consola + 220, alto - 14), f, 0.5, (0, 215, 255), 2)
    return lienzo


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Demo de reconocimiento sobre video.")
    p.add_argument("proyecto", type=Path, help="directorio del proyecto anotado")
    RAIZ = Path("/home/lucasb/Proyectos/TwinBoxing/modelos")
    p.add_argument("--detector", type=Path, default=RAIZ / "detector_sin_sparring-3.ens.pt",
                   help="ensamble del fold que deja AFUERA la fuente que se mira. Pasarle uno "
                        "que la vio es mirar el entrenamiento, no una prediccion.")
    p.add_argument("--modelo", type=Path, default=RAIZ / "poseC3D_7fuentes.pth")
    p.add_argument("--config", type=Path, default=RAIZ / "poseC3D_7fuentes.py")
    p.add_argument("--desde", type=int, default=0)
    p.add_argument("--hasta", type=int, default=None)
    p.add_argument("--umbral", type=float, default=0.80,
                   help="umbral del detector. 0,80 es el punto de mejor F1 medido; bajarlo "
                        "sube recall y hunde precision, y hay un acantilado angosto entre "
                        "0,75 y 0,80 donde se caen las marcas espurias")
    p.add_argument("--velocidad", type=float, default=1.0)
    p.add_argument("--ancho", type=int, default=1280)
    p.add_argument("--sin-ventana", action="store_true", help="no abrir ventana")
    p.add_argument("--ancho-consola", type=int, default=ANCHO_CONSOLA, dest="ancho_consola",
                   # El %% va escapado: argparse formatea el help con %, y un % suelto lo
                   # rompe al imprimir --help.
                   help="ancho de cada consola en pixeles. Sobre el proxy de 960 las de 300 "
                        "se llevan el 38%% del cuadro; sobre el original de 1920, el 24%%")
    p.add_argument("--out", type=Path, default=None,
                   help="escribir el video procesado, con una consola por peleador a cada "
                        "lado. Funciona con --sin-ventana, que es como conviene para un "
                        "video largo")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    proy = args.proyecto
    videos = sorted((proy / "videos").glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no hay video en {proy/'videos'}")
    base = videos[0].stem
    annot = proy / "annotations" / f"{base}.annot.json"
    npz = proy / "cache" / f"{base}.pose.npz"
    proxy = proy / "cache" / f"{base}.proxy.mp4"
    for f in (annot, npz):
        if not f.is_file():
            raise SystemExit(f"falta {f}")

    ident = Identidad(annot)
    cache = Cache(npz)
    fuente = proxy if proxy.is_file() else videos[0]
    cap = cv2.VideoCapture(str(fuente))
    if not cap.isOpened():
        raise SystemExit(f"no se pudo abrir {fuente}")

    ancho_v, alto_v = ident.video["width"], ident.video["height"]
    esc_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / ancho_v
    esc_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / alto_v
    fps = float(ident.video["fps"])

    print(f"video     : {fuente.name}  ({'proxy' if fuente == proxy else 'original'})")
    print(f"pose      : {npz.name}, {cache.n_frames} cuadros")
    print(f"detector  : {args.detector.name}, umbral {args.umbral}")
    print(f"familia   : {args.modelo.name}")
    print("  OJO: el detector tiene que ser el del fold que deja esta fuente afuera.")
    print("       Y la familia del golpe no generaliza: 38 de 76 hooks se llaman straight,")
    print("       y ese eje no mejora con mas datos. La etiqueta es lo mas debil que se ve.")
    print()

    clf = Clasificador(args.config, args.modelo, args.device)
    det = Detector(args.detector, args.umbral, args.device)

    # -- las poses de todo el video, por rol, para correr el detector de una sola vez
    print("resolviendo identidad y corriendo el detector sobre la secuencia...")
    T = cache.n_frames
    kp_rol = {r: np.zeros((T, 17, 2), np.float32) for r in ("A", "B")}
    sc_rol = {r: np.zeros((T, 17), np.float32) for r in ("A", "B")}
    for f2 in range(T):
        for tid2, _, kp2, sc2 in cache.en(f2):
            rol = ident.rol(tid2, f2)
            if rol in ("fighter_A", "fighter_B"):
                kp_rol[rol[-1]][f2] = kp2
                sc_rol[rol[-1]][f2] = sc2
    golpes = det.segmentos(kp_rol, sc_rol, T)
    print(f"  {len(golpes)} golpes detectados\n")

    # -- la familia de cada uno, una vez
    por_inicio: dict[int, list] = {}
    for g in golpes:
        rol = f"fighter_{g['rol']}"
        a, b = g["inicio"], g["fin"] + 1
        kp_seg, sc_seg = kp_rol[g["rol"]][a:b], sc_rol[g["rol"]][a:b]
        if len(kp_seg) >= 3 and sc_seg.sum() > 0:
            cl, prob = clf(kp_seg, sc_seg, alto_v, ancho_v)
        else:
            cl, prob = "?", 0.0
        g["clase"], g["p_clase"] = cl, prob
        por_inicio.setdefault(a, []).append(g)
        print(f"{a:7d}  {a/fps:7.2f}s  {rol:10s} {g['brazo']:5s} {cl:15s} "
              f"p_det={g['prob']:.2f} p_cls={prob:.2f}")
    print()

    # Una cola por peleador: cada consola muestra la suya.
    eventos: dict[str, deque] = {"fighter_A": deque(maxlen=30), "fighter_B": deque(maxlen=30)}
    conteo: dict[str, dict[str, int]] = {"fighter_A": {}, "fighter_B": {}}
    hasta = min(args.hasta or cache.n_frames, cache.n_frames)

    if args.desde:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.desde)
    frame = args.desde
    pausado = False
    # Se dibuja si hay que mostrarlo o si hay que escribirlo. El escritor se crea con el
    # primer cuadro, que es cuando se sabe el tamano del lienzo con las dos consolas.
    dibuja = (not args.sin_ventana) or args.out is not None
    escritor = None
    espera_ms = max(1, int(1000 / (fps * max(args.velocidad, 0.01))))

    while frame < hasta:
        if not pausado:
            ok, img = cap.read()
            if not ok:
                break

        dets = cache.en(frame)
        # Pose por peleador en este cuadro, para el disparador.
        por_rol = {}
        for tid, bbox, kp, sc in dets:
            rol = ident.rol(tid, frame)
            if rol in ("fighter_A", "fighter_B"):
                por_rol[rol] = (kp, sc)

        if not pausado:
            for g in por_inicio.get(frame, []):
                rol = f"fighter_{g['rol']}"
                eventos[rol].append((frame, g["clase"], g["p_clase"]))
                conteo[rol][g["clase"]] = conteo[rol].get(g["clase"], 0) + 1

        if dibuja:
            vis = img.copy()
            for tid, bbox, kp, sc in dets:
                rol = ident.rol(tid, frame)
                color = COLOR.get(rol, GRIS)
                if rol not in ("fighter_A", "fighter_B") and color is GRIS:
                    continue  # el publico no se dibuja: satura la pantalla
                k = kp.copy()
                k[:, 0] *= esc_x
                k[:, 1] *= esc_y
                dibujar_pose(vis, k, sc, color)
            lienzo = componer(vis, conteo, eventos, frame, fps, pausado, args.ancho_consola)

            if args.out is not None and not pausado:
                if escritor is None:
                    alto_l, ancho_l = lienzo.shape[:2]
                    escritor = cv2.VideoWriter(
                        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho_l, alto_l)
                    )
                    if not escritor.isOpened():
                        raise SystemExit(f"no se pudo abrir {args.out} para escribir")
                escritor.write(lienzo)

        if not args.sin_ventana:
            h, w = lienzo.shape[:2]
            if w != args.ancho:
                lienzo = cv2.resize(lienzo, (args.ancho, round(h * args.ancho / w)))
            cv2.imshow("BoxTwin - reconocimiento", lienzo)

            k = cv2.waitKey(1 if pausado else espera_ms) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                pausado = not pausado
            if pausado and k in (81, 83):  # flechas
                paso = 1 if k == 83 else -1
                frame = max(0, frame + paso)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
                if not ok:
                    break
                continue

        if not pausado:
            frame += 1

    cap.release()
    if escritor is not None:
        escritor.release()
        print(f"\nvideo procesado en {args.out}")
    if not args.sin_ventana:
        cv2.destroyAllWindows()

    print()
    print("=== resumen ===")
    for rol in ("fighter_A", "fighter_B"):
        c = conteo[rol]
        print(f"{rol}: {sum(c.values())} golpes  {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
