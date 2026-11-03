"""
BoxTwin - Puente de identidad de los caches de YOLOv8 a los de YOLO26 (E8, 27-10-2026).

POR QUE EXISTE
  Las anotaciones guardan la identidad como tramos de track_id del cache de v8. Un cache nuevo
  renumera los tracks y cambia sus cortes, asi que esas asignaciones apuntarian a otra
  persona. Sin pasarlas al cache de YOLO26, E8 no tiene identidad para la condicion nueva y
  no se puede comparar nada (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §5).

QUE HACE
  Por fuente, leyendo la anotacion y el cache originales sin escribirlos:
  1. verdad por cuadro: la caja observada de cada peleador (IdentityResolver.by_fighter, sin
     interpoladas);
  2. emparejamiento: en el mismo cuadro del cache nuevo, la deteccion de mayor IoU, si es
     >= 0,5; si A y B quieren la misma, se la queda el de mayor IoU y el otro queda sin
     emparejar (conflicto);
  3. asignaciones nuevas: corridas de cuadros consecutivos con el mismo track nuevo y rol,
     con origin.op puente y anotador puente-e8;
  4. interpolaciones: se transporta una aceptada si el rol queda sin emparejar en todo el
     hueco y emparejado en el cuadro anterior y en el siguiente. No se inventan otras;
  5. el resto del documento se copia igual; pose apunta al cache nuevo;
  6. escribe <raiz-y26>/<dir>/annotations/<fuente>.annot.json, validado con load, y
     <raiz-y26>/<dir>/puente.json. Se niega a escribir dentro del repo o fuera del gemelo.
  Con --k1 corre el puente del cache original contra si mismo (control K1).

USO
  python scripts/gemelo/puente_identidad.py                    # las siete fuentes
  python scripts/gemelo/puente_identidad.py --fuentes 01-sparring --k1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "clasificador"))
import guardas  # noqa: E402

UMBRAL_IOU = 0.5        # el umbral MOT que ya usa tools/comparar_tracker.py
UMBRAL_K4 = 0.70
PELEADORES = ("A", "B")


class PuenteError(RuntimeError):
    """Un caso que el puente no sabe resolver sin adivinar."""


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    from boxtwin.core.interpolation import iou
    return float(iou(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def verdad(doc, cache) -> dict[str, dict[int, tuple[np.ndarray, int]]]:
    """Por peleador y cuadro, la caja observada y su track en el cache original."""
    from boxtwin.core.identity import IdentityResolver
    from boxtwin.core.types import FighterId

    res = IdentityResolver(doc, cache)
    salida: dict[str, dict[int, tuple[np.ndarray, int]]] = {"A": {}, "B": {}}
    for f in range(min(doc.video.total_frames, len(cache))):
        for pel, pose in res.by_fighter(f).items():
            if pose is not None and not pose.interpolated:
                p = "A" if pel is FighterId.A else "B"
                salida[p][f] = (np.asarray(pose.bbox, np.float64), int(pose.track_id))
    return salida


def emparejar(gt, cache, umbral: float = UMBRAL_IOU):
    """
    Track nuevo de cada peleador por cuadro, con su IoU, y cuantos conflictos hubo.
    Tambien cuenta, por peleador, los cuadros sin verdad donde el cache nuevo tiene una
    deteccion que no se emparejo con nadie: lo que YOLO26 ve y la verdad de v8 no.
    """
    asign: dict[str, dict[int, int]] = {"A": {}, "B": {}}
    ious: dict[str, dict[int, float]] = {"A": {}, "B": {}}
    conflictos = 0
    sin_verdad = {"A": 0, "B": 0}
    cuadros = sorted(set(gt["A"]) | set(gt["B"]) | set(range(len(cache))))
    for f in cuadros:
        if f >= len(cache):
            continue
        dets = cache.detections(f)
        mejor: dict[str, tuple[int, float]] = {}
        for p in PELEADORES:
            if f not in gt[p]:
                continue
            caja = gt[p][f][0]
            m, v = None, umbral
            for i in range(len(dets)):
                x = _iou(caja, dets.bbox[i])
                if x >= v and (m is None or x > v):
                    m, v = i, x
            if m is not None:
                mejor[p] = (m, v)
        if "A" in mejor and "B" in mejor and mejor["A"][0] == mejor["B"][0]:
            conflictos += 1
            pierde = "B" if mejor["A"][1] >= mejor["B"][1] else "A"
            del mejor[pierde]
        for p, (i, v) in mejor.items():
            asign[p][f] = int(dets.track_id[i])
            ious[p][f] = v
        usados = {i for i, _ in mejor.values()}
        if len(dets) > len(usados):
            for p in PELEADORES:
                if f not in gt[p]:
                    sin_verdad[p] += 1
    return asign, ious, conflictos, sin_verdad


def corridas(asign: dict[str, dict[int, int]]) -> list[tuple[int, str, int, int]]:
    """(track, rol, inicio, fin_excl) de cuadros consecutivos con el mismo track y rol."""
    salida = []
    for p in PELEADORES:
        fs = sorted(asign[p])
        i = 0
        while i < len(fs):
            j = i
            while j + 1 < len(fs) and fs[j + 1] == fs[j] + 1 and asign[p][fs[j + 1]] == asign[p][fs[i]]:
                j += 1
            salida.append((asign[p][fs[i]], p, fs[i], fs[j] + 1))
            i = j + 1
    return sorted(salida, key=lambda c: (c[2], c[1], c[0]))


def puente(doc, cache_orig, cache_nuevo, umbral: float = UMBRAL_IOU, ahora: datetime | None = None):
    """
    Devuelve (documento nuevo, informe, asignacion por cuadro). El documento lleva todavia la
    pose del original: la cambia `escribir`, que es quien sabe donde quedo el cache nuevo.
    """
    from boxtwin.core.schema import Assignment, Interpolation, Origin
    from boxtwin.core.types import AssignmentOp, TrackRole

    if doc.identity.manual_tracks:
        raise PuenteError(f"la anotacion tiene {len(doc.identity.manual_tracks)} manual_tracks; "
                          "el puente no sabe pasarlos sin adivinar")
    ahora = ahora or datetime.now().astimezone()
    gt = verdad(doc, cache_orig)
    asign, ious, conflictos, sin_verdad = emparejar(gt, cache_nuevo, umbral)
    rol = {"A": TrackRole.A, "B": TrackRole.B}
    op_id = f"op_puente_e8_{ahora:%Y%m%d}"

    asignaciones = [
        Assignment(id=f"as_pe8_{k:05d}", track_id=tid, role=rol[p], start_frame=a, end_frame_excl=b,
                   origin=Origin(op=AssignmentOp.PUENTE, op_id=op_id, at_frame=a,
                                 annotator=guardas.ANOTADOR_PUENTE, created_at=ahora))
        for k, (tid, p, a, b) in enumerate(corridas(asign))
    ]

    interps, no_transportadas = [], 0
    for it in doc.identity.interpolations:
        p = {TrackRole.A: "A", TrackRole.B: "B"}.get(it.role)
        ant, sig = it.gap_start_frame - 1, it.gap_end_frame_excl
        hueco_libre = p is not None and not any(
            f in asign[p] for f in range(it.gap_start_frame, it.gap_end_frame_excl))
        if not (hueco_libre and ant in asign[p] and sig in asign[p]):
            no_transportadas += 1
            continue
        de, a = asign[p][ant], asign[p][sig]
        da, db = cache_nuevo.detections(ant), cache_nuevo.detections(sig)
        union = _iou(da.bbox[da.index_of_track(de)], db.bbox[db.index_of_track(a)])
        interps.append(Interpolation(
            id=f"in_pe8_{len(interps):05d}", role=it.role, from_track_id=de, to_track_id=a,
            gap_start_frame=it.gap_start_frame, gap_end_frame_excl=it.gap_end_frame_excl,
            gap_len=it.gap_len, method=it.method, iou_at_join=min(1.0, max(0.0, union)),
            accepted_by=f"{guardas.ANOTADOR_PUENTE} (de {it.id})", created_at=ahora))

    nuevo = doc.model_copy(deep=True)
    nuevo.identity.assignments = asignaciones
    nuevo.identity.interpolations = interps

    todos = [v for p in PELEADORES for v in ious[p].values()]
    por_pel = {p: (float(np.median(list(ious[p].values()))) if ious[p] else None) for p in PELEADORES}
    mediana = float(np.median(todos)) if todos else None
    informe = {
        "cuadros_con_verdad": {p: len(gt[p]) for p in PELEADORES},
        "emparejados": {p: len(asign[p]) for p in PELEADORES},
        "conflictos": conflictos,
        "iou_mediana": mediana,
        "iou_p10": float(np.percentile(todos, 10)) if todos else None,
        "iou_mediana_por_peleador": por_pel,
        "puente_debil": mediana is None or mediana < UMBRAL_K4,
        "interpolaciones": {"transportadas": len(interps), "no_transportadas": no_transportadas},
        "tracks_nuevos_usados": len({a.track_id for a in asignaciones}),
        "asignaciones": len(asignaciones),
        "cuadros_con_deteccion_sin_verdad": sin_verdad,
        "umbral_iou": umbral,
    }
    return nuevo, informe, asign


def k1(doc, cache) -> dict:
    """El puente del cache contra si mismo tiene que dar el track original en cada cuadro."""
    gt = verdad(doc, cache)
    asign, _, conflictos, _ = emparejar(gt, cache)
    total = sum(len(gt[p]) for p in PELEADORES)
    iguales = sum(1 for p in PELEADORES for f, (_, tid) in gt[p].items() if asign[p].get(f) == tid)
    return {"cuadros": total, "iguales": iguales, "conflictos": conflictos,
            "pasa": iguales == total and conflictos == 0}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def escribir(doc, informe: dict, destino: Path, npz_nuevo: Path) -> Path:
    """Escribe la anotacion del puente y puente.json. Valida la salida con load."""
    from boxtwin.core.annotations import load, save
    from boxtwin.core.posecache import PoseCache
    from boxtwin.core.project import _modelo_de_pose
    from boxtwin.core.schema import PoseRef
    from boxtwin.core.types import KeypointFormat

    destino = guardas.exigir_fuera_del_repo(guardas.exigir_bajo_gemelo(destino))
    guardas.exigir_annot_del_puente(destino)
    npz_nuevo = Path(npz_nuevo)
    meta_path = npz_nuevo.with_name(npz_nuevo.name.replace(".pose.npz", ".meta.json"))
    meta = PoseCache.open(npz_nuevo).meta
    doc = doc.model_copy(deep=True)
    doc.pose = PoseRef(
        npz_path=str(npz_nuevo), meta_path=str(meta_path),
        meta_sha256=_sha(meta_path) if meta_path.is_file() else "0" * 64,
        keypoint_format=KeypointFormat(meta.get("keypoint_format", "coco17")),
        keypoint_sources={"0-16": _modelo_de_pose(meta)},
    )
    save(doc, destino)
    load(destino)
    (destino.parents[1] / "puente.json").write_text(
        json.dumps({**informe, "annot": str(destino), "annot_sha256": _sha(destino),
                    "cache": str(npz_nuevo)}, indent=2, ensure_ascii=False) + "\n")
    return destino


def _original(raiz: Path, fuente: str):
    """Anotacion y cache originales de una fuente, solo lectura."""
    from boxtwin.core.annotations import load
    from boxtwin.core.posecache import PoseCache
    import comun

    ruta = raiz / comun.FUENTES[fuente] / "annotations" / f"{fuente}.annot.json"
    doc, _ = load(ruta)
    npz = Path(doc.pose.npz_path)
    if not npz.is_file():
        npz = ruta.parents[1] / "cache" / npz.name
    return doc, PoseCache.open(npz)


def main(argv=None) -> int:
    import comun
    from boxtwin.core.posecache import PoseCache

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuentes", nargs="*", default=list(comun.FUENTES), choices=list(comun.FUENTES))
    ap.add_argument("--raiz-original", type=Path, default=guardas.REPO_A,
                    help="copia principal de A, solo lectura")
    ap.add_argument("--raiz-y26", type=Path, default=guardas.GEMELO / "raiz-y26")
    ap.add_argument("--k1", action="store_true", help="solo el control K1, sin escribir anotaciones")
    args = ap.parse_args(argv)
    raiz = guardas.exigir_bajo_gemelo(args.raiz_y26)
    falla = False
    for fuente in args.fuentes:
        doc, cache = _original(args.raiz_original, fuente)
        dir_y26 = raiz / comun.FUENTES[fuente]
        if args.k1:
            k = k1(doc, cache)
            (guardas.exigir_bajo_gemelo(dir_y26) / "k1.json").write_text(json.dumps(k, indent=2) + "\n")
            print(f"{fuente}: K1 {k['iguales']}/{k['cuadros']}, conflictos {k['conflictos']}", flush=True)
            falla |= not k["pasa"]
            continue
        npz_n = dir_y26 / "cache" / f"{fuente}.pose.npz"
        nuevo, inf, _ = puente(doc, cache, PoseCache.open(npz_n))
        escribir(nuevo, inf, dir_y26 / "annotations" / f"{fuente}.annot.json", npz_n)
        print(f"{fuente}: emparejados {inf['emparejados']}, conflictos {inf['conflictos']}, "
              f"IoU mediana {inf['iou_mediana']:.3f}{' PUENTE DEBIL' if inf['puente_debil'] else ''}",
              flush=True)
    return 1 if falla else 0


if __name__ == "__main__":
    sys.exit(main())
