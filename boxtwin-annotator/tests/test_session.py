"""
Apertura de una sesion de anotacion.

El caso que sostiene el flujo de trabajo real: se preprocesa en la maquina con GPU y se
anota en otra, por lo que el video original puede no estar. Sobre 4K el original pesa
490 MB cada 10 minutos contra 97 del proxy, asi que exigirlo obligaria a mover cinco veces
mas datos para ganar solo el zoom en resolucion completa.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest




pytest.importorskip("cv2")

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, write_pose_cache  # noqa: E402
from boxtwin.server.session import Session, SessionError, project_paths  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")

W, H, N = 64, 48, 25
SHA = "b" * 64


def _meta() -> dict:
    return {
        "format_version": 1,
        "kind": "boxtwin.pose_meta",
        "keypoint_format": "coco17",
        "video": {
            "path": "videos/mini.mp4", "sha256": SHA, "size_bytes": 1234,
            "mtime": "2026-08-11T10:00:00-03:00", "width": W, "height": H,
            "codec": "h264", "fps": 25.0, "fps_declared": 25.0,
            "fps_source": "container_verified", "duration_s": 1.0,
            "total_frames": N, "total_frames_declared": N,
        },
        "counts": {"n_detections": N},
        "resume_seams": [{"frame": 10, "id_offset": 3}],
    }


@pytest.fixture
def proyecto(tmp_path: Path) -> Path:
    """Proyecto completo: video, proxy, npz y meta."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "cache").mkdir()

    for destino, ancho in ((tmp_path / "videos" / "mini.mp4", W),
                           (tmp_path / "cache" / "mini.proxy.mp4", W // 2)):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc=duration=1:size={ancho}x{H if ancho == W else H // 2}:rate=25",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12", str(destino)],
            check=True,
        )

    frame_index = np.arange(N + 1, dtype=np.int64)  # una deteccion por cuadro
    arrays = PoseArrays(
        frame_index=frame_index,
        frame_status=np.full(N, FrameStatus.OK, np.uint8),
        track_id=np.ones(N, np.int32),
        bbox=np.tile(np.array([1.0, 1.0, 20.0, 30.0], np.float32), (N, 1)),
        det_conf=np.full(N, 0.9, np.float32),
        keypoints=np.zeros((N, N_KEYPOINTS, 2), np.float32),
        kp_score=np.full((N, N_KEYPOINTS), 0.8, np.float32),
    )
    meta = _meta()
    write_pose_cache(tmp_path / "cache" / "mini.pose.npz", arrays, meta)
    (tmp_path / "cache" / "mini.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


# -- layout ----------------------------------------------------------------


def test_layout_del_proyecto(tmp_path: Path) -> None:
    p = project_paths(tmp_path / "proyecto" / "videos" / "spar.mp4")
    assert p.project == (tmp_path / "proyecto").resolve()
    assert p.npz.name == "spar.pose.npz"
    assert p.annot.parent.name == "annotations"


# -- apertura --------------------------------------------------------------


def test_abre_con_todo(proyecto: Path) -> None:
    s = Session.open(proyecto / "videos" / "mini.mp4")
    try:
        assert s.total_frames == N
        assert s.using_proxy is True
        assert s.has_video is True
        assert s.source.scale == pytest.approx(0.5)
        assert s.seams == [10]
        assert s.source.frame(5) is not None
    finally:
        s.close()


def test_crea_el_annot_si_no_existe(proyecto: Path) -> None:
    annot = proyecto / "annotations" / "mini.annot.json"
    assert not annot.exists()
    s = Session.open(proyecto / "videos" / "mini.mp4")
    try:
        assert annot.is_file()
        assert s.doc.video.sha256 == SHA
        assert s.doc.video.total_frames == N
    finally:
        s.close()


def test_abre_sin_el_video_original(proyecto: Path) -> None:
    """El caso del flujo de dos maquinas: llega el bundle sin el original."""
    (proyecto / "videos" / "mini.mp4").unlink()
    s = Session.open(proyecto / "videos" / "mini.mp4")
    try:
        assert s.has_video is False
        assert s.using_proxy is True
        assert s.source.frame(3) is not None
        # Sin original no hay resolucion completa, y eso no puede romper nada.
        assert s.hires() is None
    finally:
        s.close()


def test_abre_sin_proxy_usando_el_original(proyecto: Path) -> None:
    (proyecto / "cache" / "mini.proxy.mp4").unlink()
    s = Session.open(proyecto / "videos" / "mini.mp4")
    try:
        assert s.using_proxy is False
        assert s.source.scale == 1.0
        assert s.hires() is s.source
    finally:
        s.close()


# -- fallas ----------------------------------------------------------------


def test_sin_cache_de_pose(proyecto: Path) -> None:
    (proyecto / "cache" / "mini.pose.npz").unlink()
    with pytest.raises(SessionError, match="preprocess"):
        Session.open(proyecto / "videos" / "mini.mp4")


def test_sin_imagen_de_ningun_tipo(proyecto: Path) -> None:
    (proyecto / "videos" / "mini.mp4").unlink()
    (proyecto / "cache" / "mini.proxy.mp4").unlink()
    with pytest.raises(SessionError, match="al menos uno"):
        Session.open(proyecto / "videos" / "mini.mp4")


def test_anotacion_de_otro_video(proyecto: Path) -> None:
    """
    Reemplazar el video dejando el mismo nombre haria que los keypoints caigan sobre otro
    material. Se anotaria igual y el dataset saldria mal sin que nada lo delate.
    """
    s = Session.open(proyecto / "videos" / "mini.mp4")
    s.close()
    annot = proyecto / "annotations" / "mini.annot.json"
    data = json.loads(annot.read_text(encoding="utf-8"))
    data["video"]["sha256"] = "c" * 64
    annot.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SessionError, match="otro video"):
        Session.open(proyecto / "videos" / "mini.mp4")
