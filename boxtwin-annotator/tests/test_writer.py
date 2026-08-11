"""
Shards y reanudacion.

El test que mas importa es el del offset de track_id. BoT-SORT tiene estado: si el proceso
muere y arranca una instancia nueva, los ids vuelven a empezar en 1. Sin offset, el track 1
del segundo tramo se confundiria con el track 1 del primero y una asignacion de identidad
se aplicaria en silencio a otra persona. Es el peor tipo de bug posible en este sistema
porque no rompe nada, solo produce datos mal etiquetados.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseCache
from boxtwin.preprocess.writer import (
    FrameDetections,
    ResumeMismatch,
    ResumeState,
    ShardWriter,
)

SHA = "a" * 64
CFG = "b" * 64


def dets(ids: list[int]) -> FrameDetections:
    n = len(ids)
    if n == 0:
        return FrameDetections.empty()
    return FrameDetections(
        track_id=np.asarray(ids, np.int32),
        bbox=np.tile(np.array([0.0, 0.0, 10.0, 20.0], np.float32), (n, 1)),
        det_conf=np.full(n, 0.9, np.float32),
        keypoints=np.zeros((n, N_KEYPOINTS, 2), np.float32),
        kp_score=np.full((n, N_KEYPOINTS), 0.8, np.float32),
    )


def escribir(writer: ShardWriter, desde: int, patron: list[list[int]]) -> None:
    for i, ids in enumerate(patron):
        writer.add_frame(desde + i, FrameStatus.OK, dets(ids))


def nuevo(work: Path, **kw) -> ShardWriter:
    opciones = dict(video_sha256=SHA, config_hash=CFG, n_frames_expected=100, shard_frames=4)
    opciones.update(kw)
    return ShardWriter(work, **opciones)


# -- acumulacion basica ----------------------------------------------------


def test_flush_automatico_por_shard(tmp_path: Path) -> None:
    w = nuevo(tmp_path / "work")
    assert w.resume_from() == 0
    escribir(w, 0, [[1, 2]] * 4)
    assert len(list((tmp_path / "work").glob("shard_*.npz"))) == 1
    assert w.state.next_frame == 4


def test_frames_fuera_de_orden_fallan(tmp_path: Path) -> None:
    w = nuevo(tmp_path / "work")
    w.resume_from()
    w.add_frame(0, FrameStatus.OK, dets([1]))
    with pytest.raises(ValueError, match="se esperaba el frame 1"):
        w.add_frame(5, FrameStatus.OK, dets([1]))


def test_finalize_produce_cache_leible(tmp_path: Path) -> None:
    w = nuevo(tmp_path / "work")
    w.resume_from()
    escribir(w, 0, [[1, 2], [1], [], [1, 2, 3], [2]])
    npz = tmp_path / "spar.pose.npz"
    w.finalize(npz, {"counts": {}}, total_frames=5)

    cache = PoseCache.open(npz)
    assert len(cache) == 5
    assert [len(cache.detections(f)) for f in range(5)] == [2, 1, 0, 3, 1]
    assert cache.detections(3).track_id.tolist() == [1, 2, 3]
    assert not (tmp_path / "work").exists()  # el directorio de trabajo se limpia


# -- reanudacion -----------------------------------------------------------


def test_reanuda_desde_el_ultimo_shard(tmp_path: Path) -> None:
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 8)  # dos shards completos

    w2 = nuevo(work)
    assert w2.resume_from() == 8


def test_offset_de_track_id_al_reanudar(tmp_path: Path) -> None:
    """
    El tracker nuevo vuelve a emitir ids desde 1. Tienen que quedar desplazados por encima
    del maximo del tramo anterior para que no colisionen.
    """
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1, 2], [1, 2], [1, 3], [3, 7]])  # max_track_id = 7

    w2 = nuevo(work)
    assert w2.resume_from() == 4
    assert w2.id_offset == 7
    escribir(w2, 4, [[1, 2], [1, 2], [1, 2], [1, 2]])

    npz = tmp_path / "spar.pose.npz"
    w2.finalize(npz, {"counts": {}}, total_frames=8)
    cache = PoseCache.open(npz)

    # Antes de la costura, ids crudos. Despues, desplazados.
    assert cache.detections(0).track_id.tolist() == [1, 2]
    assert cache.detections(4).track_id.tolist() == [8, 9]
    # Ningun id del segundo tramo pisa uno del primero.
    previos = set(np.concatenate([cache.detections(f).track_id for f in range(4)]).tolist())
    posteriores = set(np.concatenate([cache.detections(f).track_id for f in range(4, 8)]).tolist())
    assert previos.isdisjoint(posteriores)


def test_costura_queda_registrada_en_el_meta(tmp_path: Path) -> None:
    """El anotador tiene que poder saber donde hay un corte de identidad garantizado."""
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1, 5]] * 4)

    w2 = nuevo(work)
    w2.resume_from()
    escribir(w2, 4, [[1]] * 4)
    npz = tmp_path / "spar.pose.npz"
    w2.finalize(npz, {"counts": {}}, total_frames=8)

    meta = PoseCache.open(npz).meta
    assert meta["resume_seams"] == [{"frame": 4, "id_offset": 5}]
    assert meta["max_track_id"] == 6


def test_dos_reanudaciones_acumulan_offsets(tmp_path: Path) -> None:
    work = tmp_path / "work"
    for tramo in range(3):
        w = nuevo(work)
        inicio = w.resume_from()
        escribir(w, inicio, [[1, 2]] * 4)
    npz = tmp_path / "spar.pose.npz"
    w.finalize(npz, {"counts": {}}, total_frames=12)

    cache = PoseCache.open(npz)
    assert cache.detections(0).track_id.tolist() == [1, 2]
    assert cache.detections(4).track_id.tolist() == [3, 4]
    assert cache.detections(8).track_id.tolist() == [5, 6]
    assert len(cache.meta["resume_seams"]) == 2


def test_sin_trabajo_previo_no_hay_costura(tmp_path: Path) -> None:
    w = nuevo(tmp_path / "work")
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    npz = tmp_path / "spar.pose.npz"
    w.finalize(npz, {"counts": {}}, total_frames=4)
    assert PoseCache.open(npz).meta["resume_seams"] == []


# -- proteccion del estado -------------------------------------------------


def test_rechaza_reanudar_otro_video(tmp_path: Path) -> None:
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    with pytest.raises(ResumeMismatch, match="otro video"):
        nuevo(work, video_sha256="c" * 64)


def test_rechaza_reanudar_con_otra_configuracion(tmp_path: Path) -> None:
    """
    Reanudar con otro modelo o imgsz mezclaria dos configuraciones en un mismo cache y el
    resultado seria incomparable consigo mismo.
    """
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    with pytest.raises(ResumeMismatch, match="configuracion"):
        nuevo(work, config_hash="d" * 64)


def test_estado_se_escribe_despues_del_shard(tmp_path: Path) -> None:
    """
    Si el proceso muere entre el shard y el estado, el shard huerfano se reescribe en la
    proxima corrida. Al reves se perderia trabajo.
    """
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    estado = ResumeState.from_json((work / "state.json").read_text(encoding="utf-8"))
    assert estado.next_frame == 4
    assert estado.shards == ["shard_000000000.npz"]


def test_state_json_es_legible(tmp_path: Path) -> None:
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    data = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert set(data) == {
        "video_sha256", "config_hash", "n_frames_expected", "next_frame",
        "max_track_id", "id_offset", "shards", "seams",
    }


# -- finalize contra el conteo real ----------------------------------------


def test_frames_de_mas_quedan_como_no_procesados(tmp_path: Path) -> None:
    """
    El total real se sabe recien al terminar de decodificar. Si el video tiene mas frames
    de los procesados, los que faltan tienen que quedar marcados, no desaparecer.
    """
    w = nuevo(tmp_path / "work")
    w.resume_from()
    escribir(w, 0, [[1]] * 4)
    npz = tmp_path / "spar.pose.npz"
    w.finalize(npz, {"counts": {}}, total_frames=10)

    cache = PoseCache.open(npz)
    assert len(cache) == 10
    assert cache.unprocessed_frames().tolist() == [4, 5, 6, 7, 8, 9]
    assert len(cache.detections(7)) == 0


def test_finalize_recorta_si_el_contenedor_prometio_de_mas(tmp_path: Path) -> None:
    w = nuevo(tmp_path / "work")
    w.resume_from()
    escribir(w, 0, [[1, 2]] * 8)
    npz = tmp_path / "spar.pose.npz"
    w.finalize(npz, {"counts": {}}, total_frames=6)

    cache = PoseCache.open(npz)
    assert len(cache) == 6
    assert cache.n_detections == 12


def test_shards_no_contiguos_fallan(tmp_path: Path) -> None:
    """Un hueco desalinearia el indice CSR y saldrian los keypoints corridos, en silencio."""
    work = tmp_path / "work"
    w = nuevo(work)
    w.resume_from()
    escribir(w, 0, [[1]] * 8)
    (work / "shard_000000000.npz").unlink()
    w.state.shards = ["shard_000000004.npz"]

    with pytest.raises(ValueError, match="no contiguos"):
        w.finalize(tmp_path / "roto.npz", {"counts": {}}, total_frames=8)
