#!/usr/bin/env python3
"""
BoxTwin - Demo de reconocimiento sobre video, con ventana OpenCV.

POR QUE EXISTE
  Los numeros de una matriz de confusion no muestran COMO falla un sistema. Ver el pipeline
  corriendo sobre video, con los golpes apareciendo en pantalla a medida que pasan, hace
  visibles cosas que una tabla esconde: que el detector dispara con el movimiento de guardia,
  que un hook lejano se clasifica como straight, que en el clinch la identidad se mezcla.

  Tambien es la primera vez que las tres piezas del sistema corren juntas: pose, identidad y
  clasificacion. Hasta ahora se habian medido por separado.

  DOS ADVERTENCIAS QUE NO SON LETRA CHICA:

  1. El clasificador NO TIENE CLASE "no hay golpe". Fue entrenado sobre ventanas que siempre
     contienen uno, asi que a cualquier ventana que se le pase le va a devolver uno de los
     seis. Por eso hace falta un DETECTOR que decida cuando preguntar, y ese detector no
     existe todavia: el modelo de secuencia con carriles BIO esta sin construir. Lo que hay
     aca es un disparador por movimiento de muneca, una heuristica de reemplazo, no un
     detector entrenado. Sus falsos positivos son suyos, no del clasificador.

  2. El clasificador solo tiene senal EN DISTRIBUCION. Medido: 62,5% sobre sparring-3 contra
     37,5% de linea de base, y por debajo de la linea de base en los tres folds que dejan una
     fuente afuera. Sobre un video que no sea sparring-3, lo que se ve en pantalla es ruido
     con formato de prediccion.

QUE HACE
  Reproduce el video con el overlay de pose coloreado por peleador, dispara la clasificacion
  cuando detecta la extension de un brazo, y muestra los golpes reconocidos en un panel
  lateral y por consola.

  La pose sale del cache del preproceso, no se calcula en vivo: el objetivo es mirar la
  clasificacion, no medir el rendimiento del detector de pose.

USO
  python tools/demo_vivo.py <proyecto>
  python tools/demo_vivo.py <proyecto> --desde 3000 --umbral 0.5 --velocidad 0.5
  python tools/demo_vivo.py <proyecto> --sin-ventana          # solo consola

  Corre en el entorno boxtwin_mmaction, que es donde vive el modelo:
  ~/miniforge3/envs/boxtwin_mmaction/bin/python tools/demo_vivo.py ...

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
# Disparador por movimiento: el detector que falta
# ---------------------------------------------------------------------------


class Disparador:
    """
    Decide CUANDO preguntarle al clasificador.

    Mide la extension de cada muneca respecto del centro de hombros, normalizada por el ancho
    de hombros para que no dependa de la distancia a camara, y dispara cuando esa extension
    hace un maximo local por encima de un umbral.

    Es una heuristica, no un detector entrenado, y se nota: dispara con el movimiento de
    guardia y se pierde golpes cortos. El detector de verdad es el modelo de secuencia con
    carriles BIO, que todavia no existe. Mientras tanto esto permite ver el clasificador
    funcionando sin tener que clasificar cada cuadro, que devolveria un golpe por cuadro
    porque el modelo no tiene clase "no hay golpe".
    """

    def __init__(self, umbral_ext: float = 1.6, refractario: int = 20, ventana: int = 5) -> None:
        self.umbral_ext = umbral_ext
        self.refractario = refractario
        self.hist: dict[tuple[str, str], deque] = {}
        self.ultimo: dict[tuple[str, str], int] = {}
        self.ventana = ventana

    def evaluar(self, rol: str, kp, score, frame: int) -> str | None:
        """Devuelve 'left'/'right' si hay que disparar para ese brazo, o None."""
        if min(score[L_SH], score[R_SH]) < 0.3:
            return None
        esc = float(np.linalg.norm(kp[L_SH] - kp[R_SH]))
        if esc < 1e-3:
            return None
        centro = (kp[L_SH] + kp[R_SH]) / 2

        for lado, wr in (("left", L_WR), ("right", R_WR)):
            clave = (rol, lado)
            h = self.hist.setdefault(clave, deque(maxlen=self.ventana))
            h.append(float(np.linalg.norm(kp[wr] - centro)) / esc if score[wr] >= 0.3 else 0.0)
            if len(h) < self.ventana:
                continue
            medio = h[len(h) // 2]
            # Maximo local: el punto medio de la ventana es mayor que sus vecinos.
            if medio < self.umbral_ext or medio != max(h):
                continue
            if frame - self.ultimo.get(clave, -10**9) < self.refractario:
                continue
            self.ultimo[clave] = frame
            return lado
        return None


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


def dibujar_panel(img, recientes, conteo, frame, fps, pausado):
    alto, ancho = img.shape[:2]
    w = 330
    panel = img[:, ancho - w:]
    cv2.rectangle(panel, (0, 0), (w, alto), (20, 20, 20), -1)
    img[:, ancho - w:] = cv2.addWeighted(panel, 0.75, img[:, ancho - w:], 0.25, 0)
    x = ancho - w + 14
    f = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, f"cuadro {frame}  {frame/fps:6.1f}s", (x, 28), f, 0.55, (200, 200, 200), 1)
    if pausado:
        cv2.putText(img, "PAUSA", (x + 215, 28), f, 0.55, (0, 215, 255), 2)

    y = 62
    for rol in ("fighter_A", "fighter_B"):
        c = conteo.get(rol, {})
        cv2.putText(img, rol, (x, y), f, 0.5, COLOR[rol], 2)
        cv2.putText(img, f"{sum(c.values())}", (x + 250, y), f, 0.6, COLOR[rol], 2)
        y += 20
        for cl in CLASES:
            if c.get(cl):
                cv2.putText(img, f"   {cl:15s} {c[cl]:3d}", (x, y), f, 0.42, (185, 185, 185), 1)
                y += 16
        y += 10

    cv2.line(img, (x - 4, y), (ancho - 12, y), (70, 70, 70), 1)
    y += 24
    cv2.putText(img, "ultimos golpes", (x, y), f, 0.45, (150, 150, 150), 1)
    y += 22
    for fr, rol, cl, p in list(recientes)[-12:][::-1]:
        cv2.putText(img, f"{fr:6d} {cl:14s} {p:.2f}", (x, y), f, 0.44, COLOR[rol], 1)
        y += 18
        if y > alto - 30:
            break


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Demo de reconocimiento sobre video.")
    p.add_argument("proyecto", type=Path, help="directorio del proyecto anotado")
    p.add_argument("--modelo", type=Path,
                   default=Path("/home/lucasb/Proyectos/TwinBoxing/modelos/poseC3D_sparring3_indist.pth"))
    p.add_argument("--config", type=Path,
                   default=Path("/home/lucasb/Proyectos/TwinBoxing/modelos/poseC3D_sparring3_indist.py"))
    p.add_argument("--desde", type=int, default=0)
    p.add_argument("--hasta", type=int, default=None)
    p.add_argument("--umbral", type=float, default=0.0,
                   help="probabilidad minima para mostrar el golpe. 0 muestra todo, que es lo "
                        "honesto: el modelo no tiene clase 'no hay golpe' y su confianza no "
                        "esta calibrada")
    p.add_argument("--ventana", type=int, default=12, help="cuadros que se le pasan al modelo")
    p.add_argument("--extension", type=float, default=1.6,
                   help="extension de muneca, en anchos de hombro, para disparar")
    p.add_argument("--refractario", type=int, default=20,
                   help="cuadros minimos entre dos disparos del mismo brazo")
    p.add_argument("--velocidad", type=float, default=1.0)
    p.add_argument("--ancho", type=int, default=1280)
    p.add_argument("--sin-ventana", action="store_true", help="solo consola")
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
    print(f"modelo    : {args.modelo.name}")
    print(f"disparador: extension >= {args.extension}, refractario {args.refractario} cuadros")
    print("  OJO: el clasificador no tiene clase 'no hay golpe' y solo tiene senal medida")
    print("       sobre sparring-3. Fuera de ahi lo que se ve es ruido con formato.")
    print()

    clf = Clasificador(args.config, args.modelo, args.device)
    disp = Disparador(args.extension, args.refractario)

    recientes: deque = deque(maxlen=40)
    conteo: dict[str, dict[str, int]] = {"fighter_A": {}, "fighter_B": {}}
    hasta = min(args.hasta or cache.n_frames, cache.n_frames)

    if args.desde:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.desde)
    frame = args.desde
    pausado = False
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
            for rol, (kp, sc) in por_rol.items():
                lado = disp.evaluar(rol, kp, sc, frame)
                if lado is None:
                    continue
                # Ventana centrada en el disparo, con los keypoints de ESE peleador.
                a = max(0, frame - args.ventana // 2)
                b = min(cache.n_frames, a + args.ventana)
                seq_kp, seq_sc = [], []
                for f2 in range(a, b):
                    hallado = None
                    for tid2, _, kp2, sc2 in cache.en(f2):
                        if ident.rol(tid2, f2) == rol:
                            hallado = (kp2, sc2)
                            break
                    if hallado is None:
                        hallado = (kp, sc)  # se repite el ultimo visto: no se inventa pose
                    seq_kp.append(hallado[0])
                    seq_sc.append(hallado[1])
                if len(seq_kp) < 3:
                    continue
                cl, prob = clf(np.stack(seq_kp), np.stack(seq_sc), alto_v, ancho_v)
                if prob < args.umbral:
                    continue
                recientes.append((frame, rol, cl, prob))
                conteo[rol][cl] = conteo[rol].get(cl, 0) + 1
                print(f"{frame:7d}  {frame/fps:7.2f}s  {rol:10s} {cl:15s} p={prob:.2f}")

        if not args.sin_ventana:
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
            dibujar_panel(vis, recientes, conteo, frame, fps, pausado)
            h, w = vis.shape[:2]
            if w != args.ancho:
                vis = cv2.resize(vis, (args.ancho, round(h * args.ancho / w)))
            cv2.imshow("BoxTwin - reconocimiento", vis)

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
