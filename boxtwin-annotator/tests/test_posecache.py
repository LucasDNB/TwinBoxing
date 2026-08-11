"""Formato del cache de pose: indice CSR, round-trip y las invariantes que lo sostienen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxtwin.core.posecache import (
    CACHE_FORMAT_VERSION,
    N_KEYPOINTS,
    FrameStatus,
    PoseArrays,
    PoseCache,
    write_pose_cache,
)


def arrays_sinteticos(dets_por_frame: list[int]) -> PoseArrays:
    """Cache con la cantidad de detecciones pedida por frame. Los valores son el indice global."""
    n_frames = len(dets_por_frame)
    total = sum(dets_por_frame)
    frame_index = np.zeros(n_frames + 1, np.int64)
    frame_index[1:] = np.cumsum(dets_por_frame)

    return PoseArrays(
        frame_index=frame_index,
        frame_status=np.full(n_frames, FrameStatus.OK, np.uint8),
        track_id=np.arange(total, dtype=np.int32),
        bbox=np.tile(np.array([0.0, 0.0, 10.0, 20.0], np.float32), (total, 1)),
        det_conf=np.full(total, 0.9, np.float32),
        keypoints=np.arange(total * N_KEYPOINTS * 2, dtype=np.float32).reshape(
            total, N_KEYPOINTS, 2
        ),
        kp_score=np.full((total, N_KEYPOINTS), 0.8, np.float32),
    )


# -- indice CSR ------------------------------------------------------------


def test_rebanadas_por_frame() -> None:
    cache = PoseCache(arrays_sinteticos([2, 0, 3, 1]), meta={})
    assert len(cache) == 4
    assert cache.n_detections == 6
    assert [len(cache.detections(f)) for f in range(4)] == [2, 0, 3, 1]
    # Los track_id son el indice global, asi que la rebanada del frame 2 es 2,3,4.
    assert cache.detections(2).track_id.tolist() == [2, 3, 4]


def test_frame_sin_detecciones_no_es_frame_sin_procesar() -> None:
    """
    Con CSR los dos casos dan rebanada vacia. Distinguirlos es lo que hace posible reanudar.
    """
    a = arrays_sinteticos([1, 0, 1])
    a.frame_status[2] = FrameStatus.NO_PROCESADO
    cache = PoseCache(a, meta={})

    vacio_procesado = cache.detections(1)
    sin_procesar = cache.detections(2)
    assert len(vacio_procesado) == 0 and vacio_procesado.processed
    assert not sin_procesar.processed
    assert cache.unprocessed_frames().tolist() == [2]


def test_detections_es_una_vista_sin_copia() -> None:
    cache = PoseCache(arrays_sinteticos([3]), meta={})
    dets = cache.detections(0)
    assert dets.keypoints.base is not None


def test_frame_fuera_de_rango() -> None:
    cache = PoseCache(arrays_sinteticos([1, 1]), meta={})
    with pytest.raises(IndexError):
        cache.detections(2)
    with pytest.raises(IndexError):
        cache.detections(-1)


def test_frames_of_track() -> None:
    a = arrays_sinteticos([2, 1, 2])
    # Se fuerza el track 7 en los frames 0 y 2.
    a.track_id[:] = np.array([7, 1, 2, 7, 3], np.int32)
    cache = PoseCache(a, meta={})
    assert cache.frames_of_track(7).tolist() == [0, 2]
    assert cache.frames_of_track(999).tolist() == []
    assert cache.track_ids().tolist() == [1, 2, 3, 7]


def test_index_of_track() -> None:
    a = arrays_sinteticos([3])
    a.track_id[:] = np.array([5, 9, 2], np.int32)
    dets = PoseCache(a, meta={}).detections(0)
    assert dets.index_of_track(9) == 1
    assert dets.index_of_track(404) is None


# -- invariantes -----------------------------------------------------------


def test_valida_offset_inicial() -> None:
    a = arrays_sinteticos([1, 1])
    a.frame_index[0] = 1
    with pytest.raises(ValueError, match=r"frame_index\[0\]"):
        a.validate()


def test_valida_offset_final() -> None:
    a = arrays_sinteticos([1, 1])
    a.frame_index[-1] = 5
    with pytest.raises(ValueError, match="track_id"):
        a.validate()


def test_valida_monotonia() -> None:
    a = arrays_sinteticos([2, 2, 2])  # frame_index = 0, 2, 4, 6
    a.frame_index[1] = 5  # queda 0, 5, 4, 6: el total sigue bien pero baja en el medio
    with pytest.raises(ValueError, match="monotono"):
        a.validate()


@pytest.mark.parametrize("campo", ["track_id", "bbox", "det_conf", "keypoints", "kp_score"])
def test_valida_formas(campo: str) -> None:
    a = arrays_sinteticos([2, 2])
    setattr(a, campo, getattr(a, campo)[:-1])
    with pytest.raises(ValueError, match=campo):
        a.validate()


# -- round-trip ------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    a = arrays_sinteticos([2, 0, 3])
    meta = {"kind": "boxtwin.pose_meta", "video": {"sha256": "abc"}, "counts": {}}
    p = tmp_path / "spar.pose.npz"
    write_pose_cache(p, a, meta)

    cache = PoseCache.open(p)
    assert cache.meta == meta
    assert len(cache) == 3
    np.testing.assert_array_equal(cache.detections(2).track_id, a.track_id[2:5])
    np.testing.assert_allclose(cache.detections(0).keypoints, a.keypoints[:2])


def test_open_rechaza_otra_version(tmp_path: Path) -> None:
    """Un cache de otro formato tiene que hacer reprocesar, no leerse a medias."""
    p = tmp_path / "viejo.pose.npz"
    write_pose_cache(p, arrays_sinteticos([1]), {})
    datos = dict(np.load(p, allow_pickle=False))
    datos["format_version"] = np.int32(CACHE_FORMAT_VERSION + 1)
    np.savez(p, **datos)

    with pytest.raises(ValueError, match="reprocesar"):
        PoseCache.open(p)


def test_escritura_no_deja_temporales(tmp_path: Path) -> None:
    write_pose_cache(tmp_path / "spar.pose.npz", arrays_sinteticos([1, 1]), {})
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_escritura_rechaza_arrays_invalidos(tmp_path: Path) -> None:
    a = arrays_sinteticos([1, 1])
    a.frame_index[-1] = 99
    with pytest.raises(ValueError):
        write_pose_cache(tmp_path / "roto.npz", a, {})
    assert not (tmp_path / "roto.npz").exists()


def test_cache_vacio(tmp_path: Path) -> None:
    """Un video sin ninguna deteccion tiene que producir un cache valido, no un error."""
    p = tmp_path / "vacio.pose.npz"
    write_pose_cache(p, arrays_sinteticos([0, 0, 0]), {})
    cache = PoseCache.open(p)
    assert cache.n_detections == 0
    assert len(cache.detections(1)) == 0
