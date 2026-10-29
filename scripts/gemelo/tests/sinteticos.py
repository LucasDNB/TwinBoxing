"""Caches y anotaciones chicos, para probar el puente y el trackeo sin material real."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from boxtwin.core.annotations import save
from boxtwin.core.posecache import FrameStatus, PoseArrays, write_pose_cache
from boxtwin.core.schema import (Assignment, Event, Interpolation, Origin, PoseRef, VideoInfo,
                                 new_document)
from boxtwin.core.types import AssignmentOp, FpsSource, Guard, TrackRole

AHORA = datetime(2026, 10, 28, tzinfo=timezone.utc)
FPS = 30.0
ROL = {"A": TrackRole.A, "B": TrackRole.B, "ignore": TrackRole.IGNORE}


def caja(cx: float, ancho: float = 100.0) -> tuple:
    return (cx - ancho / 2, 100.0, cx + ancho / 2, 400.0)


def cache(ruta: Path, cuadros: list[list[tuple]], modelo: str = "yolov8l-pose.pt") -> Path:
    """cuadros[f] = [(track_id, (x1, y1, x2, y2)), ...]."""
    indice, ids, cajas = [0], [], []
    for dets in cuadros:
        for tid, b in dets:
            ids.append(tid)
            cajas.append(b)
        indice.append(len(ids))
    n = len(ids)
    arrays = PoseArrays(
        frame_index=np.asarray(indice, np.int64),
        frame_status=np.full(len(cuadros), FrameStatus.OK, np.uint8),
        track_id=np.asarray(ids, np.int32).reshape(n),
        bbox=np.asarray(cajas, np.float32).reshape(n, 4),
        det_conf=np.full(n, 0.9, np.float32),
        keypoints=np.zeros((n, 17, 2), np.float32),
        kp_score=np.full((n, 17), 0.9, np.float32),
    )
    meta = {"keypoint_format": "coco17",
            "inference": {"model_path": f"/m/{modelo}", "model_sha256": "cd" * 32}}
    ruta = Path(ruta)
    write_pose_cache(ruta, arrays, meta)
    ruta.with_name(ruta.name.replace(".pose.npz", ".meta.json")).write_text(json.dumps(meta))
    return ruta


def documento(npz: Path, T: int, asignaciones: list[tuple], interps: list[tuple] = (),
              eventos: int = 1):
    """asignaciones = [(track_id, 'A'|'B'|'ignore', inicio, fin_excl)];
    interps = [(id, 'A'|'B', de_track, a_track, hueco_inicio, hueco_fin_excl)]."""
    doc = new_document(
        app_version="test", now=AHORA,
        video=VideoInfo(path="/v/spar.mp4", sha256="0" * 64, size_bytes=1, mtime=AHORA, fps=FPS,
                        fps_source=FpsSource.MEASURED, width=1280, height=720, total_frames=T,
                        duration_s=T / FPS, codec="h264"),
        pose=PoseRef(npz_path=str(npz), meta_path=str(npz).replace(".pose.npz", ".meta.json"),
                     meta_sha256="0" * 64, keypoint_sources={"0-16": "yolov8l-pose"}),
        guard_a=Guard.ORTHODOX, guard_b=Guard.ORTHODOX,
    )
    for i, (tid, rol, a, b) in enumerate(asignaciones):
        doc.identity.assignments.append(Assignment(
            id=f"as_{i:04d}", track_id=tid, role=ROL[rol], start_frame=a, end_frame_excl=b,
            origin=Origin(op=AssignmentOp.MANUAL, op_id=f"op_{i:04d}", at_frame=a,
                          annotator="lucasb", created_at=AHORA)))
    for iid, rol, de, a, gs, ge in interps:
        doc.identity.interpolations.append(Interpolation(
            id=iid, role=ROL[rol], from_track_id=de, to_track_id=a, gap_start_frame=gs,
            gap_end_frame_excl=ge, gap_len=ge - gs, iou_at_join=0.8, accepted_by="lucasb",
            created_at=AHORA))
    for i in range(eventos):
        doc.events.append(Event.model_validate({
            "id": f"ev_{i:04d}", "fighter": "fighter_A", "start_frame": 1, "end_frame": 3,
            "side": "left", "punch_type": "straight", "target": "head", "completeness": "full",
            "guard": "orthodox"}))
    return doc


def guardar(doc, ruta: Path) -> Path:
    save(doc, ruta)
    return ruta
