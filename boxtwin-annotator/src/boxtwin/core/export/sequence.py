"""
BoxTwin - Export de series temporales continuas en esquema BIO.

POR QUE EXISTE
  Es el formato del clasificador orientado a eventos, el que tiene que decidir por cuadro si
  hay un golpe empezando, en curso, o nada. A diferencia del export a MMAction2, que recorta
  ventanas ya centradas en el golpe, este conserva el video entero y por lo tanto conserva
  las transiciones, que es lo que hay que aprender.

  Un carril por BRAZO y no por peleador. En una combinacion el jab izquierdo todavia esta
  retrayendo cuando arranca el cross derecho, y BIO no puede representar dos segmentos
  solapados en un mismo carril: habria que descartar uno. Un 1-2 es el golpe mas frecuente
  del boxeo, asi que el formato perderia datos exactamente donde mas hay. Con un carril por
  brazo el solapamiento es imposible salvo en el doble jab, que se resuelve truncando y se
  reporta.

  La mascara de validez es lo que distingue "no hay golpe" de "no sabemos": un cuadro sin
  identidad resuelta o dentro de un tramo no confiable no es fondo, es ausencia de dato. Sin
  esa distincion el modelo aprende que las oclusiones son fondo.

QUE HACE
  Escribe un npz con las etiquetas por cuadro y brazo, los keypoints, la mascara de validez
  y la marca de que cuadros son interpolados.

USO
  export sequence --classes 12   (label-space side por defecto)
"""

from __future__ import annotations

import numpy as np

from boxtwin.core.export.base import (
    ExportContext,
    ExportResult,
    base_metadata,
    escribir_json,
    registrar,
)
from boxtwin.core.export.labels import LabelSpace, class_index, class_list
from boxtwin.core.gloves import derive_gloves
from boxtwin.core.types import FighterId, Side

__all__ = ["exportar", "bio_names"]

FUERA = 0  # etiqueta O


def bio_names(clases: list[str]) -> list[str]:
    """Nombres del espacio BIO en orden de indice: O, B-c0, I-c0, B-c1, I-c1, ..."""
    nombres = ["O"]
    for c in clases:
        nombres += [f"B-{c}", f"I-{c}"]
    return nombres


def _bio(indice_clase: int) -> tuple[int, int]:
    """(etiqueta B, etiqueta I) de una clase."""
    return 1 + 2 * indice_clase, 2 + 2 * indice_clase


@registrar("sequence")
def exportar(ctx: ExportContext) -> ExportResult:
    space = LabelSpace(ctx.opcion("label_space", LabelSpace.SIDE.value))
    classes = int(ctx.opcion("classes", 12))
    n_kp = 19 if ctx.opcion("keypoints", "coco17") == "coco17+gloves" else 17
    por_brazo = ctx.opcion("channels", "per-arm") != "per-fighter"

    clases = class_list(space, classes)
    nombres_bio = bio_names(clases)
    glove_k = ctx.doc.settings_snapshot.glove_extrapolation_k

    peleadores = [FighterId.A, FighterId.B]
    T = ctx.doc.video.total_frames
    n_carriles = 2 if por_brazo else 1

    labels = np.zeros((len(peleadores), n_carriles, T), np.int16)
    valid = np.zeros((len(peleadores), T), bool)
    interp = np.zeros((len(peleadores), T), bool)
    kp = np.zeros((len(peleadores), T, n_kp, 2), np.float32)
    sc = np.zeros((len(peleadores), T, n_kp), np.float32)

    avisos: list[str] = []
    truncados = 0
    descartados = 0

    # -- keypoints y validez
    for p, fighter in enumerate(peleadores):
        for f in range(T):
            pose = ctx.resolver.by_fighter(f)[fighter]
            if pose is None:
                continue
            valid[p, f] = pose.reliable
            interp[p, f] = pose.interpolated
            xy, s = pose.keypoints, pose.kp_score
            if n_kp == 19:
                g_xy, g_sc = derive_gloves(xy[None, ...], s[None, ...], glove_k)
                xy = np.concatenate([xy, g_xy[0]], axis=0)
                s = np.concatenate([s, g_sc[0]], axis=0)
            kp[p, f] = xy
            sc[p, f] = s

    # -- etiquetas BIO
    for p, fighter in enumerate(peleadores):
        eventos = sorted(
            (e for e in ctx.doc.events if e.fighter is fighter),
            key=lambda e: (e.start_frame, e.id),
        )
        ocupado_hasta = {0: -1, 1: -1}  # por carril, ultimo cuadro ya escrito
        for ev in eventos:
            idx = class_index(ev, space, classes)
            if idx is None:
                descartados += 1
                continue
            carril = 0 if not por_brazo else (0 if ev.side is Side.LEFT else 1)
            b, i = _bio(idx)

            inicio = ev.start_frame
            if inicio <= ocupado_hasta[carril]:
                # Solapamiento en el mismo carril: solo puede pasar en un doble jab. Se
                # trunca el anterior y se registra; el annot.json no se toca.
                truncados += 1
                labels[p, carril, inicio : ocupado_hasta[carril] + 1] = FUERA
            fin = min(ev.end_frame, T - 1)
            labels[p, carril, inicio] = b
            if fin > inicio:
                labels[p, carril, inicio + 1 : fin + 1] = i
            ocupado_hasta[carril] = fin

    if truncados:
        avisos.append(
            f"{truncados} solapamientos del mismo brazo se truncaron en el export; "
            "el annot.json no se modifico"
        )
    if descartados:
        avisos.append(f"{descartados} eventos quedaron fuera del espacio de {classes} clases")
    if not por_brazo:
        avisos.append(
            "con --channels per-fighter, un 1-2 pierde uno de los dos golpes: en un solo "
            "carril BIO no entran dos segmentos solapados"
        )

    base = ctx.video_path.stem
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    npz = ctx.out_dir / f"{base}.sequence.npz"
    np.savez_compressed(
        npz,
        labels=labels,
        valid=valid,
        interpolated=interp,
        keypoints=kp,
        kp_score=sc,
        fighters=np.array([f.value for f in peleadores]),
        lanes=np.array(["left", "right"] if por_brazo else ["both"]),
        bio_names=np.array(nombres_bio),
        classes=np.array(clases),
    )

    meta = base_metadata(ctx, "sequence")
    meta["classes"] = clases
    meta["bio_names"] = nombres_bio
    meta["label_space"] = space.value
    meta["channels"] = "per-arm" if por_brazo else "per-fighter"
    meta["shapes"] = {
        "labels": list(labels.shape),
        "valid": list(valid.shape),
        "keypoints": list(kp.shape),
        "kp_score": list(sc.shape),
    }
    meta["counts"] = {
        "frames": T,
        "valid_frames": int(valid.sum()),
        "interpolated_frames": int(interp.sum()),
        "labeled_frames": int((labels != FUERA).sum()),
        "events_used": len(ctx.doc.events) - descartados,
        "events_skipped": descartados,
        "same_arm_truncations": truncados,
    }
    meta_path = ctx.out_dir / f"{base}.sequence.meta.json"
    escribir_json(meta_path, meta)

    return ExportResult(
        formato="sequence", archivos=[npz, meta_path], resumen=meta["counts"], avisos=avisos
    )
