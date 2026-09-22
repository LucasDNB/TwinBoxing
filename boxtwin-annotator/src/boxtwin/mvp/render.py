"""
BoxTwin - El video procesado: los dos peleadores marcados y una consola para cada uno.

POR QUE EXISTE
  La Fight-Card da los numeros y no muestra de donde salen. Un total de 40 golpes no deja
  ver si el sistema conto bien, si se comio la mitad de un intercambio o si le atribuyo a uno
  los golpes del otro. Con el video al lado de la cuenta, eso se ve en segundos.

  Tres decisiones de disposicion, y ninguna es estetica:

  Las consolas van AL COSTADO y no encima del video. Superpuestas taparian justo lo que se
  esta mirando, y el punto es ver el golpe y su etiqueta a la vez.

  Son DOS y no una compartida. Con las dos columnas juntas hay que leer el rol de cada
  renglon para saber de quien es el golpe, y mirando el video al mismo tiempo eso no se hace:
  la vista va al peleador, no a la etiqueta.

  Y se dibujan SOLO los dos peleadores. En un gimnasio el detector de pose encuentra al
  publico, a la otra pareja y al entrenador -medido: en 04-sparring son 375 tracks de los
  cuales dos son los que importan- y marcarlos a todos convierte la pantalla en ruido. Peor
  todavia, sugiere que el sistema los esta contando, que es justo lo que no hace.

QUE HACE
  Lee el video, el cache de pose, la identidad ya resuelta y la Fight-Card, y escribe un mp4
  con los dos peleadores marcados y una consola por peleador acumulando su cuenta.

  No detecta nada: los golpes ya estan en fightcard.json, con las correcciones del entrenador
  aplicadas encima si las hubo. Volver a correr el detector aca daria numeros que no son los
  que la Fight-Card muestra.

USO
  from boxtwin.mvp.render import renderizar
  renderizar(directorio_de_la_sesion)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from boxtwin.core.types import FighterId, TrackRole

__all__ = ["ANCHO_CONSOLA", "PROCESADO", "renderizar", "dibujar_consola", "componer"]

PROCESADO = "procesado.mp4"
ANCHO_CONSOLA = 300

# El codec del video procesado, y no es un detalle de implementacion. mp4v es MPEG-4 Parte 2
# y NINGUN navegador lo reproduce en un <video> de HTML5: el archivo se escribe sin error y
# la pagina muestra un reproductor vacio, que es el peor modo de falla porque no se parece a
# un fallo. Ademas pesa: los mismos tres minutos son 230 MB en mp4v contra 55 en h264.
CODEC = "avc1"

# BGR, los mismos del anotador para que el color signifique lo mismo en toda la herramienta.
COLOR = {"A": (76, 88, 236), "B": (235, 158, 74)}

# COCO-17.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def _dibujar_pose(cv2, img, kp, score, color, umbral=0.3):
    """El esqueleto de un peleador. cv2 entra por parametro: se importa una sola vez arriba."""
    for a, b in EDGES:
        if score[a] >= umbral and score[b] >= umbral:
            cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                     color, 2)
    for i in range(len(kp)):
        if score[i] >= umbral:
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 3, color, -1)


def dibujar_consola(lienzo, x0, ancho, peleador, conteo, eventos, total_golpes):
    """
    La consola de un peleador: su cuenta acumulada, el desglose y los ultimos golpes.

    El desglose por tipo va en gris cuando el tipo es None, que es el caso sin clasificador
    corrido, y eso NO es lo mismo que cero: es "no se estimo". Mostrarlo como cero seria
    afirmar que no hubo golpes de esa familia.
    """
    import cv2

    alto = lienzo.shape[0]
    f = cv2.FONT_HERSHEY_SIMPLEX
    color = COLOR[peleador]
    # x2 inclusive en cv2.rectangle: con x0+ancho el panel pisaba la primera columna del
    # video.
    cv2.rectangle(lienzo, (x0, 0), (x0 + ancho - 1, alto), (18, 18, 20), -1)
    # La linea de acento va en el borde EXTERIOR del lienzo, no contra el video: dibujada
    # del lado de adentro se comia la ultima columna de imagen, porque una linea de 3 px se
    # centra en su coordenada. Simetrica para los dos, ademas.
    borde = x0 if x0 == 0 else x0 + ancho - 3
    cv2.rectangle(lienzo, (borde, 0), (borde + 3, alto), color, -1)
    x = x0 + 16

    cv2.putText(lienzo, f"PELEADOR {peleador}", (x, 34), f, 0.62, color, 2)
    cv2.putText(lienzo, str(total_golpes), (x, 94), f, 1.6, color, 3)
    cv2.putText(lienzo, "golpes", (x + 20 + 34 * len(str(total_golpes)), 94), f, 0.5,
                (150, 150, 150), 1)

    y = 126
    cv2.line(lienzo, (x - 6, y), (x0 + ancho - 14, y), (60, 60, 60), 1)
    y += 24
    if conteo:
        for tipo, n in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0])):
            if y > alto - 160:
                break
            etiqueta = tipo if tipo else "sin estimar"
            tono = (190, 190, 190) if tipo else (120, 120, 120)
            cv2.putText(lienzo, etiqueta[:16], (x, y), f, 0.44, tono, 1)
            cv2.putText(lienzo, f"{n:3d}", (x0 + ancho - 52, y), f, 0.48, color, 2)
            y += 21
    else:
        cv2.putText(lienzo, "sin golpes todavia", (x, y), f, 0.42, (100, 100, 100), 1)
        y += 21

    y += 8
    cv2.line(lienzo, (x - 6, y), (x0 + ancho - 14, y), (60, 60, 60), 1)
    y += 22
    cv2.putText(lienzo, "ultimos", (x, y), f, 0.42, (130, 130, 130), 1)
    y += 20
    # Del mas nuevo al mas viejo: lo que acaba de pasar es lo que se esta mirando.
    for t_s, brazo, tipo in list(eventos)[::-1]:
        if y > alto - 14:
            break
        cv2.putText(lienzo, f"{t_s:6.1f}s", (x, y), f, 0.4, (120, 120, 120), 1)
        cv2.putText(lienzo, (tipo or brazo or "?")[:14], (x + 56, y), f, 0.42, color, 1)
        y += 18


def componer(img, estado, ancho_consola=ANCHO_CONSOLA):
    """El cuadro final: consola de A, el video, consola de B."""
    import cv2

    alto, ancho = img.shape[:2]
    lienzo = np.zeros((alto, ancho + 2 * ancho_consola, 3), np.uint8)
    lienzo[:, ancho_consola:ancho_consola + ancho] = img
    dibujar_consola(lienzo, 0, ancho_consola, "A", estado["A"]["conteo"],
                    estado["A"]["eventos"], estado["A"]["total"])
    dibujar_consola(lienzo, ancho_consola + ancho, ancho_consola, "B", estado["B"]["conteo"],
                    estado["B"]["eventos"], estado["B"]["total"])
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lienzo, estado["pie"], (ancho_consola + 14, alto - 14), f, 0.5,
                (210, 210, 210), 1)
    return lienzo


def _mover_indice_al_principio(ruta: Path) -> bool:
    """
    Remuxea el mp4 con el indice adelante, para que empiece a reproducir sin esperar.

    opencv deja el atomo `moov` al final, asi que el navegador tiene que pedir primero el
    tramo final del archivo y recien despues puede arrancar. Con rangos funciona igual, pero
    sobre 55 MB la espera se nota justo al abrir la Fight-Card, que es cuando el usuario
    decide si la herramienta sirve.

    Es copia sin recodificar: segundos, no minutos. Si no hay ffmpeg se deja como esta, que
    reproduce lo mismo pero arranca mas lento.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    tmp = ruta.with_suffix(".faststart.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta),
         "-c", "copy", "-movflags", "+faststart", str(tmp)],
        capture_output=True,
    )
    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(ruta)
    return True


def renderizar(
    directorio: Path,
    salida: Path | None = None,
    ancho_consola: int = ANCHO_CONSOLA,
    desde: int = 0,
    hasta: int | None = None,
    progreso=None,
) -> Path:
    """
    Escribe el video procesado de una sesion ya completada.

    Se apoya en `fightcard.json` y no vuelve a correr el detector: esa es la fuente que la
    interfaz muestra, con las correcciones del entrenador ya aplicadas, y rehacer la
    deteccion aca daria un video que no coincide con los numeros de al lado.
    """
    import cv2

    from boxtwin.core.annotations import load as load_doc
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths

    directorio = Path(directorio)
    videos = sorted((directorio / "videos").glob("*.*"))
    if not videos:
        raise FileNotFoundError(f"no hay video en {directorio/'videos'}")
    paths = project_paths(videos[0])
    if not paths.npz.is_file():
        raise FileNotFoundError(f"falta el cache de pose {paths.npz}")
    if not paths.annot.is_file():
        raise FileNotFoundError(f"falta la anotacion {paths.annot}")

    fc_path = directorio / "fightcard.json"
    if not fc_path.is_file():
        raise FileNotFoundError(
            f"falta {fc_path.name}: el video procesado se arma sobre la Fight-Card, asi que "
            "la sesion tiene que estar completa"
        )
    fc = json.loads(fc_path.read_text())

    doc, _ = load_doc(paths.annot)
    cache = PoseCache.open(paths.npz)
    res = IdentityResolver(doc, cache)
    fps = float(doc.video.fps)

    # Los golpes, indexados por el cuadro en que arrancan.
    por_cuadro: dict[int, list[tuple[str, dict]]] = {}
    for pel in ("A", "B"):
        for g in (fc.get("peleadores", {}).get(pel, {}) or {}).get("golpes", []):
            por_cuadro.setdefault(int(g["cuadro_inicio"]), []).append((pel, g))

    cap = cv2.VideoCapture(str(paths.proxy if paths.proxy.is_file() else videos[0]))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video de {directorio}")
    ancho_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # La pose esta en coordenadas del original; el video puede ser el proxy, mas chico.
    esc_x = ancho_v / doc.video.width
    esc_y = alto_v / doc.video.height

    fin = min(hasta or doc.video.total_frames, doc.video.total_frames, len(cache))
    salida = Path(salida) if salida else directorio / PROCESADO
    escritor = cv2.VideoWriter(
        str(salida), cv2.VideoWriter_fourcc(*CODEC), fps,
        (ancho_v + 2 * ancho_consola, alto_v),
    )
    if not escritor.isOpened():
        raise RuntimeError(
            f"no se pudo abrir {salida} para escribir. Si el codec avc1 no esta en este "
            "opencv, el video saldria en un formato que el navegador no reproduce"
        )

    from collections import deque

    estado = {
        p: {"conteo": {}, "eventos": deque(maxlen=30), "total": 0} for p in ("A", "B")
    }
    if desde:
        cap.set(cv2.CAP_PROP_POS_FRAMES, desde)

    try:
        for f in range(desde, fin):
            ok, img = cap.read()
            if not ok:
                break

            for pel, g in por_cuadro.get(f, []):
                e = estado[pel]
                tipo = g.get("tipo")
                e["conteo"][tipo] = e["conteo"].get(tipo, 0) + 1
                e["eventos"].append((f / fps, g.get("brazo"), tipo))
                e["total"] += 1

            # Solo los dos peleadores. El publico, el arbitro y la otra pareja del gimnasio
            # no se marcan: marcarlos sugiere que el sistema los esta contando.
            for pose in res.resolve_frame(f):
                if pose.role not in (TrackRole.A, TrackRole.B) or pose.shadowed:
                    continue
                pel = pose.role.value[-1]
                kp = np.asarray(pose.keypoints, dtype=np.float64).copy()
                kp[:, 0] *= esc_x
                kp[:, 1] *= esc_y
                _dibujar_pose(cv2, img, kp, np.asarray(pose.kp_score), COLOR[pel])
                x0, y0 = int(pose.bbox[0] * esc_x), int(pose.bbox[1] * esc_y)
                cv2.putText(img, pel, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, COLOR[pel], 2)

            estado["pie"] = f"{f/fps:6.1f}s"
            escritor.write(componer(img, estado, ancho_consola))
            if progreso is not None and (f - desde) % 60 == 0:
                progreso(f - desde, fin - desde)
    finally:
        cap.release()
        escritor.release()
    _mover_indice_al_principio(salida)
    if progreso is not None:
        progreso(fin - desde, fin - desde)
    return salida
