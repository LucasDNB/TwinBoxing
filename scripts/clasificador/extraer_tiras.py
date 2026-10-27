"""
BoxTwin - Tiras de cuadros por golpe para etiquetado a ciegas.

POR QUE EXISTE
  Claude Code puede mirar imagenes pero no video. Para que proponga tipo y altura de un
  golpe hay que darle el golpe como una sola imagen: ocho cuadros del tramo, recortados
  alrededor de los dos peleadores, con el brazo que golpea marcado. El brazo lo pone el
  detector (un carril por brazo), asi que darlo no regala la etiqueta.

  El cegado es la otra mitad. `export --format clips` guarda cada clip en una carpeta con el
  nombre de su clase, asi que un etiquetador que lee la ruta lee la respuesta. Aca las tiras
  salen con un id opaco y en orden aleatorio, y la clave (id -> evento y etiqueta real) va a
  un directorio aparte que el subagente etiquetador no puede abrir (hook en .claude/hooks/).

  Las dos fuentes de golpes pasan por el mismo dibujo. Si el piloto midiera sobre tiras
  dibujadas de una forma y el preetiquetado de material nuevo usara otra, el acuerdo medido
  no diria nada sobre el uso real.

QUE HACE
  anotado: muestra estratificada por (tipo, altura) de los golpes completos de las fuentes
           pedidas, con un minimo por celda para que entren uppercut al cuerpo y compania.
  sesion:  todos los golpes detectados de una sesion procesada por el producto
           (fightcard.json + poses.npz + videos/), sin etiqueta real.
  Escribe <lote>/tiras/<id>.png, <lote>/lote.jsonl (id y brazo, nada mas) y
  <lote>/clave/clave.csv. Es reanudable: una tira que ya existe no se vuelve a dibujar.

USO
  # piloto (corre en twinboxing_env, necesita boxtwin instalado y los videos de la estacion)
  python scripts/clasificador/extraer_tiras.py anotado --lote piloto --n 150 --seed 20261014
  # material nuevo
  python scripts/clasificador/extraer_tiras.py sesion --lote amateur --sesion ~/boxtwin/sesiones/123
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import ALTURAS, BRAZO_COCO, FUENTES, MIN_SCORE, REPO, TIPOS, TRABAJO, ruta_annot  # noqa: E402

N_CUADROS = 8
MARGEN_CUADROS = 2
ALTO_TILE = 260
COLUMNAS = 4

CAMPOS_CLAVE = ["id", "origen", "fuente", "event_id", "fighter", "side", "start_frame", "end_frame",
                "punch_type", "target"]


@dataclass
class Golpe:
    fuente: str
    event_id: str
    fighter: str
    side: str                      # left | right, anatomico, como lo guarda el anotador
    start_frame: int
    end_frame: int
    video: Path
    # (T, 17, 2) y (T, 17) sobre los cuadros [start - margen, end + margen]; NaN si no hay pose
    kp_atac: np.ndarray = field(repr=False, default=None)
    sc_atac: np.ndarray = field(repr=False, default=None)
    kp_rival: np.ndarray = field(repr=False, default=None)
    sc_rival: np.ndarray = field(repr=False, default=None)
    primer_cuadro: int = 0
    punch_type: str = ""
    target: str = ""
    origen: str = "anotado"


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------


def asignar_cupos(conteos: dict, n: int, minimo: int) -> dict:
    """
    Cuantos golpes salen de cada celda. Primero el minimo (o todo lo que haya), despues el
    resto proporcional a lo que queda en cada celda, con restos mayores. Determinista.
    """
    cupo = {c: min(minimo, k) for c, k in conteos.items()}
    resto = n - sum(cupo.values())
    if resto < 0:
        raise ValueError(f"n={n} no alcanza para el minimo de {minimo} por celda")
    disponibles = {c: conteos[c] - cupo[c] for c in conteos}
    total = sum(disponibles.values())
    if total == 0 or resto == 0:
        return cupo
    resto = min(resto, total)
    ideal = {c: resto * disponibles[c] / total for c in conteos}
    for c in conteos:
        cupo[c] += int(ideal[c])
    faltan = resto - sum(int(v) for v in ideal.values())
    for c in sorted(conteos, key=lambda c: (-(ideal[c] - int(ideal[c])), str(c)))[:faltan]:
        cupo[c] += 1
    return cupo


def muestrear(eventos: list[dict], n: int, minimo: int, seed: int) -> list[dict]:
    """eventos: dicts con al menos punch_type y target. Devuelve la muestra en orden aleatorio."""
    rng = random.Random(seed)
    celdas: dict = {}
    for e in sorted(eventos, key=lambda e: (e["fuente"], e["event_id"])):
        celdas.setdefault((e["punch_type"], e["target"]), []).append(e)
    cupo = asignar_cupos({c: len(v) for c, v in celdas.items()}, n, minimo)
    elegidos = []
    for c in sorted(celdas):
        elegidos += rng.sample(celdas[c], cupo[c])
    rng.shuffle(elegidos)
    return elegidos


# ---------------------------------------------------------------------------
# Fuentes de golpes
# ---------------------------------------------------------------------------


def golpes_anotados(fuentes: list[str]) -> list[dict]:
    """Golpes completos de las anotaciones, sin poses todavia (barato, para muestrear)."""
    salida = []
    for f in fuentes:
        doc = json.loads(ruta_annot(f).read_text())
        for ev in doc["events"]:
            if ev["completeness"] != "full":
                continue
            salida.append({
                "fuente": f, "event_id": ev["id"], "fighter": ev["fighter"], "side": ev["side"],
                "start_frame": ev["start_frame"], "end_frame": ev["end_frame"],
                "punch_type": ev["punch_type"], "target": ev["target"],
            })
    return salida


def poses_anotadas(elegidos: list[dict]) -> list[Golpe]:
    """
    Carga las poses de los golpes elegidos con el resolvedor de identidad del anotador, que es
    el mismo que usan los exports. Importa boxtwin aca adentro para que el muestreo y los
    tests no dependan del paquete.
    """
    from boxtwin.core.annotations import load as cargar_doc
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import project_paths
    from boxtwin.core.types import FighterId

    por_fuente: dict = {}
    for e in elegidos:
        por_fuente.setdefault(e["fuente"], []).append(e)

    golpes = []
    for fuente, evs in por_fuente.items():
        crudo = json.loads(ruta_annot(fuente).read_text())
        video = Path(crudo["video"]["path"])
        if not video.is_file():
            # Las rutas absolutas del annot.json son de la estacion; si el repo se movio,
            # el video esta en <proyecto>/videos/.
            video = REPO / FUENTES[fuente] / "videos" / video.name
        paths = project_paths(video)
        doc, migradas = cargar_doc(paths.annot)
        if migradas:
            raise SystemExit(f"{fuente}: el cargador migro la anotacion ({migradas}); frenar y avisar")
        res = IdentityResolver(doc, PoseCache.open(paths.npz))
        for e in evs:
            atac = FighterId(e["fighter"])
            rival = FighterId.B if atac is FighterId.A else FighterId.A
            g = Golpe(fuente=fuente, event_id=e["event_id"], fighter=e["fighter"], side=e["side"],
                      start_frame=e["start_frame"], end_frame=e["end_frame"], video=paths.video,
                      punch_type=e["punch_type"], target=e["target"])
            _cargar_ventana(g, lambda fr, quien: res.by_fighter(fr)[quien], atac, rival)
            golpes.append(g)
    return golpes


def _cargar_ventana(g: Golpe, pose_de, atac, rival) -> None:
    ini = max(0, g.start_frame - MARGEN_CUADROS)
    fin = g.end_frame + MARGEN_CUADROS
    T = fin - ini + 1
    g.primer_cuadro = ini
    g.kp_atac = np.full((T, 17, 2), np.nan, np.float32)
    g.kp_rival = np.full((T, 17, 2), np.nan, np.float32)
    g.sc_atac = np.zeros((T, 17), np.float32)
    g.sc_rival = np.zeros((T, 17), np.float32)
    for i, fr in enumerate(range(ini, fin + 1)):
        for quien, kp, sc in ((atac, g.kp_atac, g.sc_atac), (rival, g.kp_rival, g.sc_rival)):
            p = pose_de(fr, quien)
            if p is not None:
                kp[i], sc[i] = p.keypoints, p.kp_score


def golpes_de_sesion(sesion: Path) -> list[Golpe]:
    """Golpes detectados de una sesion del producto. Sin etiqueta real: es material nuevo."""
    fc = json.loads((sesion / "fightcard.json").read_text())
    z = np.load(sesion / "poses.npz")
    kp, sc, valid = z["keypoints"], z["kp_score"], z["valid"]
    videos = sorted((sesion / "videos").glob("*"))
    if not videos:
        raise SystemExit(f"{sesion}: no hay video en videos/")
    golpes = []
    for lado, datos in fc["peleadores"].items():
        p = 0 if lado == "A" else 1
        r = 1 - p
        for gol in datos["golpes"]:
            if gol.get("descartado"):
                continue
            g = Golpe(fuente=sesion.name, event_id=gol["id"], fighter=lado,
                      side="left" if gol["brazo"] == "izq" else "right",
                      start_frame=int(gol["cuadro_inicio"]), end_frame=int(gol["cuadro_fin"]),
                      video=videos[0], origen="sesion")

            def pose_de(fr, quien, _kp=kp, _sc=sc, _v=valid):
                if fr >= _kp.shape[1] or not _v[quien, fr]:
                    return None
                return _Pose(_kp[quien, fr], _sc[quien, fr])

            if r >= kp.shape[0]:      # sesion de un solo peleador
                r = p
            _cargar_ventana(g, pose_de, p, r)
            golpes.append(g)
    return golpes


@dataclass
class _Pose:
    keypoints: np.ndarray
    kp_score: np.ndarray


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------


def _caja(kp: np.ndarray, sc: np.ndarray) -> tuple[float, float, float, float] | None:
    ok = sc >= MIN_SCORE
    if ok.sum() < 3:
        return None
    pts = kp[ok]
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def region(g: Golpe, ancho: int, alto: int) -> tuple[int, int, int, int]:
    """Union de las cajas de los dos peleadores en la ventana, con margen. Cuadro entero si no hay pose."""
    cajas = [c for t in range(len(g.sc_atac)) for c in (_caja(g.kp_atac[t], g.sc_atac[t]),
                                                         _caja(g.kp_rival[t], g.sc_rival[t])) if c]
    if not cajas:
        return 0, 0, ancho, alto
    x1 = min(c[0] for c in cajas); y1 = min(c[1] for c in cajas)
    x2 = max(c[2] for c in cajas); y2 = max(c[3] for c in cajas)
    mx, my = 0.15 * (x2 - x1) + 10, 0.20 * (y2 - y1) + 10
    return (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(ancho, int(x2 + mx)), min(alto, int(y2 + my)))


def indices_muestra(T: int, n: int = N_CUADROS) -> list[int]:
    return sorted({int(round(i)) for i in np.linspace(0, T - 1, num=min(n, T))})


def leer_cuadros(video: Path, primero: int, indices: list[int]) -> dict[int, np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"no se pudo abrir {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, primero)
    salida, i, ultimo = {}, 0, max(indices)
    while i <= ultimo:
        ok, img = cap.read()
        if not ok:
            break
        if i in indices:
            salida[i] = img
        i += 1
    cap.release()
    return salida


def dibujar(g: Golpe, id_tira: str, cuadros: dict[int, np.ndarray]) -> np.ndarray:
    import cv2

    if not cuadros:
        raise ValueError(f"{id_tira}: no se leyo ningun cuadro")
    alto, ancho = next(iter(cuadros.values())).shape[:2]
    x1, y1, x2, y2 = region(g, ancho, alto)
    escala = ALTO_TILE / max(1, (y2 - y1))
    tiles = []
    hombro, codo, muneca = BRAZO_COCO[g.side]
    for i in sorted(cuadros):
        img = cuadros[i].copy()
        c_r = _caja(g.kp_rival[i], g.sc_rival[i])
        c_a = _caja(g.kp_atac[i], g.sc_atac[i])
        if c_r:
            cv2.rectangle(img, (int(c_r[0]), int(c_r[1])), (int(c_r[2]), int(c_r[3])), (255, 200, 0), 2)
        if c_a:
            cv2.rectangle(img, (int(c_a[0]), int(c_a[1])), (int(c_a[2]), int(c_a[3])), (0, 0, 230), 2)
        pts = [g.kp_atac[i, k] for k in (hombro, codo, muneca) if g.sc_atac[i, k] >= MIN_SCORE]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 230, 255), 4)
        if g.sc_atac[i, muneca] >= MIN_SCORE:
            m = g.kp_atac[i, muneca]
            cv2.circle(img, (int(m[0]), int(m[1])), 6, (0, 230, 255), -1)
        tile = cv2.resize(img[y1:y2, x1:x2], None, fx=escala, fy=escala)
        cv2.putText(tile, f"{i - (g.start_frame - g.primer_cuadro):+d}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % COLUMNAS:
        tiles.append(np.zeros_like(tiles[0]))
    filas = [np.hstack(tiles[k:k + COLUMNAS]) for k in range(0, len(tiles), COLUMNAS)]
    cuerpo = np.vstack(filas)
    banda = np.zeros((40, cuerpo.shape[1], 3), np.uint8)
    brazo = "IZQUIERDO" if g.side == "left" else "DERECHO"
    cv2.putText(banda, f"{id_tira}  brazo que golpea: {brazo} del atacante (amarillo)  "
                       f"atacante: caja roja  rival: caja celeste", (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([banda, cuerpo])


# ---------------------------------------------------------------------------
# Lote
# ---------------------------------------------------------------------------


def escribir_lote(golpes: list[Golpe], lote: Path, seed: int) -> int:
    """Ids opacos en orden aleatorio. Reanudable: si la clave ya existe se reusan sus ids."""
    import cv2

    (lote / "tiras").mkdir(parents=True, exist_ok=True)
    (lote / "clave").mkdir(parents=True, exist_ok=True)
    ruta_clave = lote / "clave" / "clave.csv"
    if ruta_clave.exists():
        previos = {(r["fuente"], r["event_id"]): r["id"] for r in csv.DictReader(ruta_clave.open())}
    else:
        previos = {}
    rng = random.Random(seed + 1)
    numeros = list(range(1, len(golpes) + 1))
    rng.shuffle(numeros)
    filas, hechas = [], 0
    for g, num in zip(golpes, numeros):
        id_tira = previos.get((g.fuente, g.event_id), f"T{num:04d}")
        destino = lote / "tiras" / f"{id_tira}.png"
        if not destino.exists():
            T = len(g.sc_atac)
            cuadros = leer_cuadros(g.video, g.primer_cuadro, indices_muestra(T))
            cv2.imwrite(str(destino), dibujar(g, id_tira, cuadros))
            hechas += 1
        filas.append({"id": id_tira, "origen": g.origen, "fuente": g.fuente, "event_id": g.event_id,
                      "fighter": g.fighter, "side": g.side, "start_frame": g.start_frame,
                      "end_frame": g.end_frame, "punch_type": g.punch_type, "target": g.target})
    filas.sort(key=lambda f: f["id"])
    with ruta_clave.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_CLAVE)
        w.writeheader()
        w.writerows(filas)
    with (lote / "lote.jsonl").open("w") as fh:
        for f in filas:
            fh.write(json.dumps({"id": f["id"], "brazo": f["side"]}) + "\n")
    return hechas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="modo", required=True)
    an = sub.add_parser("anotado")
    an.add_argument("--lote", required=True)
    an.add_argument("--n", type=int, default=150)
    an.add_argument("--minimo-celda", type=int, default=10)
    an.add_argument("--seed", type=int, default=20261014)
    an.add_argument("--fuentes", nargs="*", default=list(FUENTES))
    se = sub.add_parser("sesion")
    se.add_argument("--lote", required=True)
    se.add_argument("--sesion", type=Path, required=True)
    se.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args(argv)

    lote = TRABAJO / "lotes" / args.lote
    if args.modo == "anotado":
        todos = golpes_anotados(args.fuentes)
        elegidos = muestrear(todos, args.n, args.minimo_celda, args.seed)
        golpes = poses_anotadas(elegidos)
        orden = {(e["fuente"], e["event_id"]): k for k, e in enumerate(elegidos)}
        golpes.sort(key=lambda g: orden[(g.fuente, g.event_id)])
    else:
        golpes = golpes_de_sesion(args.sesion.expanduser())
    hechas = escribir_lote(golpes, lote, args.seed)
    celdas = {}
    for g in golpes:
        celdas[(g.punch_type, g.target)] = celdas.get((g.punch_type, g.target), 0) + 1
    print(f"{len(golpes)} golpes en {lote} ({hechas} tiras nuevas)")
    if args.modo == "anotado":
        for t in TIPOS:
            print("  " + t + ": " + ", ".join(f"{a} {celdas.get((t, a), 0)}" for a in ALTURAS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
