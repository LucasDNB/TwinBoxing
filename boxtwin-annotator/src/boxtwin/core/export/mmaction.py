"""
BoxTwin - Export a MMAction2 para PoseConv3D.

POR QUE EXISTE
  Es el formato que consume el pipeline de entrenamiento que ya existe en el proyecto. Tres
  decisiones de este export no son de forma sino de que se le ensena al modelo.

  Por defecto va una sola persona por muestra, el que pega. Incluir tambien al rival da
  contexto sobre si el golpe llega o lo bloquean, pero cambia el problema: se pasa de
  clasificar un movimiento a clasificar un intercambio, y deja de ser comparable con lo que
  ya se entreno. Por eso `both` existe como opcion y no como default.

  Por defecto van 17 keypoints y no 19. PoseConv3D arma volumenes de heatmaps con V
  canales; pasar a 19 cambia la primera convolucion y los pesos preentrenados en NTU dejan
  de cargar directo. La variante con guantes es una rama comparativa, no un cambio de base.

  Los ejemplos de fondo se muestrean con margen a los bordes de todo evento. Un fondo que
  contiene la cola de un golpe le ensena al modelo que ese movimiento es fondo, y rompe
  justamente la frontera que tiene que aprender.

QUE HACE
  Escribe el pickle con `keypoint` de forma (M, T, V, C) y `keypoint_score` de forma
  (M, T, V), mas la lista de clases y la metadata de trazabilidad.

USO
  export mmaction --label-space lead-rear --classes 12
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from boxtwin.core.export.base import (
    ExportContext,
    ExportResult,
    base_metadata,
    escribir_json,
    registrar,
)
from boxtwin.core.export.labels import BACKGROUND, LabelSpace, class_index, class_list
from boxtwin.core.export.windows import Ventana, ventana_de_evento, ventanas_de_fondo
from boxtwin.core.gloves import derive_gloves
from boxtwin.core.types import FighterId

__all__ = ["exportar"]


def _pose_de(ctx: ExportContext, frame: int, fighter: FighterId):
    return ctx.resolver.by_fighter(frame)[fighter]


def _rival(f: FighterId) -> FighterId:
    return FighterId.B if f is FighterId.A else FighterId.A


def _muestra(
    ctx: ExportContext,
    ventana: Ventana,
    label: int,
    nombre: str,
    *,
    personas: int,
    n_kp: int,
    glove_k: float,
) -> dict[str, Any]:
    """
    Arma una muestra en el formato que espera PoseConv3D.

    Los cuadros sin pose entran en cero con score cero. No se descartan ni se rellenan con
    el cuadro vecino: un hueco es informacion, y taparlo copiando el anterior inventaria
    quietud donde hubo oclusion.
    """
    T = ventana.n_frames
    peleadores = [ventana.fighter]
    if personas == 2:
        peleadores.append(_rival(ventana.fighter))

    kp = np.zeros((personas, T, n_kp, 2), np.float32)
    sc = np.zeros((personas, T, n_kp), np.float32)

    for m, quien in enumerate(peleadores):
        for i, f in enumerate(range(ventana.start_frame, ventana.end_frame + 1)):
            pose = _pose_de(ctx, f, quien)
            if pose is None:
                continue
            xy, s = pose.keypoints, pose.kp_score
            if n_kp == 19:
                g_xy, g_sc = derive_gloves(xy[None, ...], s[None, ...], glove_k)
                xy = np.concatenate([xy, g_xy[0]], axis=0)
                s = np.concatenate([s, g_sc[0]], axis=0)
            kp[m, i] = xy
            sc[m, i] = s

    alto, ancho = ctx.doc.video.height, ctx.doc.video.width
    return {
        "frame_dir": nombre,
        "label": label,
        "img_shape": (alto, ancho),
        "original_shape": (alto, ancho),
        "total_frames": T,
        "keypoint": kp,
        "keypoint_score": sc,
    }


@registrar("mmaction")
def exportar(ctx: ExportContext) -> ExportResult:
    space = LabelSpace(ctx.opcion("label_space", LabelSpace.LEAD_REAR.value))
    classes = int(ctx.opcion("classes", 12))
    personas = 2 if ctx.opcion("persons", "attacker") == "both" else 1
    n_kp = 19 if ctx.opcion("keypoints", "coco17") == "coco17+gloves" else 17
    pad = int(ctx.opcion("pad", 0))
    n_fondo = int(ctx.opcion("background", 0))
    seed = int(ctx.opcion("seed", 42))

    nombres = class_list(space, classes)
    idx_fondo = nombres.index(BACKGROUND) if BACKGROUND in nombres else None
    glove_k = ctx.doc.settings_snapshot.glove_extrapolation_k
    base = ctx.video_path.stem

    muestras: list[dict[str, Any]] = []
    avisos: list[str] = []
    descartados = 0

    # Orden por evento, que ya viene canonico: el pickle sale igual en cada corrida.
    for ev in ctx.doc.events:
        label = class_index(ev, space, classes)
        if label is None:
            descartados += 1
            continue
        muestras.append(
            _muestra(
                ctx, ventana_de_evento(ev, ctx.doc, pad=pad), label, f"{base}_{ev.id}",
                personas=personas, n_kp=n_kp, glove_k=glove_k,
            )
        )

    if n_fondo and idx_fondo is None:
        avisos.append(
            f"se pidieron {n_fondo} ejemplos de fondo pero el espacio de {classes} clases "
            "no tiene clase de fondo; se ignoran"
        )
    elif n_fondo:
        largo = int(ctx.opcion("background_len", 20))
        ventanas = ventanas_de_fondo(
            ctx.doc, ctx.resolver, cantidad=n_fondo, largo=largo, seed=seed
        )
        if len(ventanas) < n_fondo:
            avisos.append(
                f"solo se encontraron {len(ventanas)} ventanas de fondo de las {n_fondo} "
                "pedidas: el video no tiene mas tramos limpios sin eventos"
            )
        for i, v in enumerate(ventanas):
            muestras.append(
                _muestra(
                    ctx, v, idx_fondo, f"{base}_bg_{v.fighter.value}_{v.start_frame:07d}",
                    personas=personas, n_kp=n_kp, glove_k=glove_k,
                )
            )

    if descartados:
        avisos.append(
            f"{descartados} eventos quedaron fuera del espacio de {classes} clases "
            "(amagues o abortados)"
        )

    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    pkl = ctx.out_dir / f"{base}.mmaction.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(
            {"split": {base: [m["frame_dir"] for m in muestras]}, "annotations": muestras},
            fh,
            protocol=4,
        )

    meta = base_metadata(ctx, "mmaction")
    meta["classes"] = nombres
    meta["label_space"] = space.value
    meta["shapes"] = {"keypoint": "(M, T, V, C)", "keypoint_score": "(M, T, V)", "M": personas, "V": n_kp}
    meta["counts"] = {
        "samples": len(muestras),
        "events_used": len(muestras) - (0 if idx_fondo is None else sum(
            1 for m in muestras if m["label"] == idx_fondo
        )),
        "background": 0 if idx_fondo is None else sum(1 for m in muestras if m["label"] == idx_fondo),
        "events_skipped": descartados,
    }
    meta["per_class"] = {
        nombre: sum(1 for m in muestras if m["label"] == i) for i, nombre in enumerate(nombres)
    }
    # Advertencia que va en el archivo y no solo en la consola: el que arme el split seis
    # meses despues no va a leer esta salida.
    meta["warning_split"] = (
        "Partir train/val por evento filtra datos: en una combinacion las ventanas de dos "
        "golpes comparten cuadros. Partir por video o por round."
    )
    meta_path = ctx.out_dir / f"{base}.mmaction.meta.json"
    escribir_json(meta_path, meta)

    return ExportResult(
        formato="mmaction",
        archivos=[pkl, meta_path],
        resumen=meta["counts"] | {"per_class": meta["per_class"]},
        avisos=avisos,
    )
