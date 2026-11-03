"""
T1 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6).

`keypoint_sources` del documento nuevo sale del modelo que escribio el cache, no fijo en
yolov8l-pose. Sin `model_path` en el meta, queda lo de hoy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from boxtwin.core.project import cargar_o_crear, project_paths


def _meta(inference: dict | None) -> dict:
    meta = {
        "video": {
            "sha256": "0" * 64, "size_bytes": 10,
            "mtime": datetime.now(timezone.utc).isoformat(), "fps": 30.0,
            "fps_source": "measured", "width": 1280, "height": 720,
            "total_frames": 300, "duration_s": 10.0, "codec": "h264",
        },
        "keypoint_format": "coco17",
    }
    if inference is not None:
        meta["inference"] = inference
    return meta


def _doc(tmp_path, meta):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "spar.mp4").write_bytes(b"x")
    doc, _ = cargar_o_crear(project_paths(tmp_path / "videos" / "spar.mp4"), meta)
    return doc


def test_meta_de_yolo26_da_yolo26(tmp_path):
    meta = _meta({"model_path": "/home/x/boxtwin-gemelo/modelos/yolo26l-pose.pt"})
    assert _doc(tmp_path, meta).pose.keypoint_sources == {"0-16": "yolo26l-pose"}


def test_meta_sin_model_path_da_yolov8_como_hoy(tmp_path):
    assert _doc(tmp_path, _meta(None)).pose.keypoint_sources == {"0-16": "yolov8l-pose"}


def test_meta_con_inference_sin_model_path_da_yolov8(tmp_path):
    assert _doc(tmp_path, _meta({"imgsz": 640})).pose.keypoint_sources == {"0-16": "yolov8l-pose"}
