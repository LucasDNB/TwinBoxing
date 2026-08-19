"""
Resolucion de identidad por rango.

El caso que mas importa es el ID switch: antes del cuadro 4120 el track 1 es fighter_A y
despues es fighter_B. Si el resolver se equivoca ahi, los keypoints se exportan bajo el
peleador equivocado y no hay nada en el archivo que lo delate.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.core.identity import IdentityResolver
from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, PoseCache
from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import FighterId, TrackRole


def cache_con(por_frame: dict[int, list[tuple[int, float]]], n_frames: int) -> PoseCache:
    """Cache sintetico: por cuadro, una lista de (track_id, confianza)."""
    conteos = [len(por_frame.get(f, [])) for f in range(n_frames)]
    total = sum(conteos)
    frame_index = np.zeros(n_frames + 1, np.int64)
    frame_index[1:] = np.cumsum(conteos)

    ids, confs = [], []
    for f in range(n_frames):
        for tid, conf in por_frame.get(f, []):
            ids.append(tid)
            confs.append(conf)

    arrays = PoseArrays(
        frame_index=frame_index,
        frame_status=np.full(n_frames, FrameStatus.OK, np.uint8),
        track_id=np.asarray(ids or [], np.int32).reshape(total),
        bbox=np.tile(np.array([0.0, 0.0, 10.0, 20.0], np.float32), (total, 1)),
        det_conf=np.asarray(confs or [], np.float32).reshape(total),
        keypoints=np.zeros((total, N_KEYPOINTS, 2), np.float32),
        kp_score=np.full((total, N_KEYPOINTS), 0.9, np.float32),
    )
    return PoseCache(arrays, meta={})


@pytest.fixture
def resolver(doc_rich: AnnotationDoc) -> IdentityResolver:
    # Los tracks 1 y 2 conviven todo el video, el 11 aparece tras el re-seed y el 5 es el
    # arbitro. Coincide con las asignaciones de la fixture.
    por_frame = {}
    for f in range(0, 9004):
        por_frame[f] = [(1, 0.9), (2, 0.85), (5, 0.4)]
    for f in range(9008, 9100):
        por_frame[f] = [(1, 0.9), (11, 0.8)]
    return IdentityResolver(doc_rich, cache_con(por_frame, 9100))


# -- resolucion por rango --------------------------------------------------


def test_id_switch_en_4120(resolver: IdentityResolver) -> None:
    """Antes del switch el track 1 es A; desde el cuadro 4120 pasa a ser B."""
    assert resolver.role_of(4119, 1) is TrackRole.A
    assert resolver.role_of(4120, 1) is TrackRole.B
    assert resolver.role_of(4119, 2) is TrackRole.B
    assert resolver.role_of(4120, 2) is TrackRole.A


def test_bordes_del_intervalo_son_semiabiertos(resolver: IdentityResolver) -> None:
    assert resolver.role_of(0, 1) is TrackRole.A
    assert resolver.role_of(4119, 1) is TrackRole.A
    assert resolver.role_of(4120, 1) is TrackRole.B


def test_track_sin_assignment_es_none(resolver: IdentityResolver) -> None:
    assert resolver.role_of(100, 999) is None


def test_fuera_de_todo_intervalo_es_none(resolver: IdentityResolver) -> None:
    """El track 2 deja de estar asignado en 9004: el hueco del re-seed."""
    assert resolver.role_of(9003, 2) is TrackRole.A
    assert resolver.role_of(9004, 2) is None


def test_ignore_no_es_peleador(resolver: IdentityResolver) -> None:
    assert resolver.role_of(100, 5) is TrackRole.IGNORE
    pose = next(p for p in resolver.resolve_frame(100) if p.track_id == 5)
    assert pose.fighter is None


# -- resolucion de un cuadro -----------------------------------------------


def test_resolve_frame_devuelve_todas_las_detecciones(resolver: IdentityResolver) -> None:
    poses = resolver.resolve_frame(100)
    assert {p.track_id for p in poses} == {1, 2, 5}
    assert {p.role for p in poses} == {TrackRole.A, TrackRole.B, TrackRole.IGNORE}


def test_by_fighter_despues_del_switch(resolver: IdentityResolver) -> None:
    antes = resolver.by_fighter(4000)
    despues = resolver.by_fighter(4200)
    assert antes[FighterId.A].track_id == 1
    assert despues[FighterId.A].track_id == 2
    assert despues[FighterId.B].track_id == 1


def test_peleador_ausente_es_none(resolver: IdentityResolver) -> None:
    """En 9008 el track 2 ya no esta y B lo lleva el track 1: A queda en el 11."""
    porf = resolver.by_fighter(9050)
    assert porf[FighterId.A].track_id == 11
    assert porf[FighterId.B].track_id == 1


# -- desempate -------------------------------------------------------------


def test_colision_de_rol_gana_la_mayor_confianza(doc_rich: AnnotationDoc) -> None:
    """
    Dos tracks resolviendo al mismo peleador pasa en cada clinch. Sin regla explicita el
    ganador dependeria del orden de las detecciones y el export dejaria de ser reproducible.
    """
    from boxtwin.core.schema import Assignment, Origin
    from boxtwin.core.types import AssignmentOp
    from tests.conftest import T0

    doc_rich.identity.assignments.append(
        Assignment(
            id="as_9001", track_id=77, role=TrackRole.A, start_frame=0, end_frame_excl=1000,
            origin=Origin(op=AssignmentOp.MANUAL, op_id="op_x", at_frame=0,
                          annotator="lucas", created_at=T0),
        )
    )
    res = IdentityResolver(doc_rich, cache_con({10: [(1, 0.6), (77, 0.95)]}, 20))
    poses = {p.track_id: p for p in res.resolve_frame(10)}
    assert poses[77].shadowed is False
    assert poses[1].shadowed is True
    assert res.by_fighter(10)[FighterId.A].track_id == 77


def test_desempate_es_determinista_con_confianzas_iguales(doc_rich: AnnotationDoc) -> None:
    from boxtwin.core.schema import Assignment, Origin
    from boxtwin.core.types import AssignmentOp
    from tests.conftest import T0

    doc_rich.identity.assignments.append(
        Assignment(
            id="as_9001", track_id=77, role=TrackRole.A, start_frame=0, end_frame_excl=1000,
            origin=Origin(op=AssignmentOp.MANUAL, op_id="op_x", at_frame=0,
                          annotator="lucas", created_at=T0),
        )
    )
    cache = cache_con({10: [(1, 0.8), (77, 0.8)]}, 20)
    ganadores = {IdentityResolver(doc_rich, cache).by_fighter(10)[FighterId.A].track_id
                 for _ in range(5)}
    assert len(ganadores) == 1


# -- tramos no confiables --------------------------------------------------


def test_tramo_no_confiable_marca_pero_no_borra(doc_rich: AnnotationDoc) -> None:
    """
    Excluir del export es decision del export, no de la lectura. La deteccion tiene que
    seguir estando para que el anotador la vea y pueda juzgarla.
    """
    res = IdentityResolver(doc_rich, cache_con({21500: [(1, 0.9)]}, 21600))
    # El tramo un_0001 es de fighter_B entre 21400 y 21620, y en 21500 el track 1 es B.
    pose = res.resolve_frame(21500)[0]
    assert pose.role is TrackRole.B
    assert pose.reliable is False
    assert pose.keypoints is not None


def test_fuera_del_tramo_es_confiable(doc_rich: AnnotationDoc) -> None:
    res = IdentityResolver(doc_rich, cache_con({21700: [(1, 0.9)]}, 21800))
    assert res.resolve_frame(21700)[0].reliable is True


def test_is_reliable_respeta_el_peleador(doc_rich: AnnotationDoc) -> None:
    """El tramo es de B: A no tiene por que quedar marcado."""
    res = IdentityResolver(doc_rich, cache_con({}, 21800))
    assert res.is_reliable(21500, FighterId.B) is False
    assert res.is_reliable(21500, FighterId.A) is True


# -- tracks manuales -------------------------------------------------------


def test_track_manual_resuelve_y_se_marca(doc_rich: AnnotationDoc) -> None:
    res = IdentityResolver(doc_rich, cache_con({21445: [(-1, 1.0)]}, 21500))
    pose = res.resolve_frame(21445)[0]
    assert pose.role is TrackRole.B
    assert pose.manual is True


# -- utilidades ------------------------------------------------------------


def test_assigned_tracks(resolver: IdentityResolver) -> None:
    assert resolver.assigned_tracks(100) == {1: TrackRole.A, 2: TrackRole.B, 5: TrackRole.IGNORE}


def test_unassigned_tracks_encuentra_lo_pendiente(doc_rich: AnnotationDoc) -> None:
    res = IdentityResolver(doc_rich, cache_con({5: [(1, 0.9), (404, 0.7)]}, 10))
    assert res.unassigned_tracks(range(10)) == {404}


def test_refresh_toma_los_cambios(doc_rich: AnnotationDoc) -> None:
    from boxtwin.core.schema import Assignment, Origin
    from boxtwin.core.types import AssignmentOp
    from tests.conftest import T0

    res = IdentityResolver(doc_rich, cache_con({5: [(404, 0.9)]}, 10))
    assert res.role_of(5, 404) is None
    doc_rich.identity.assignments.append(
        Assignment(
            id="as_9002", track_id=404, role=TrackRole.A, start_frame=0, end_frame_excl=10,
            origin=Origin(op=AssignmentOp.MANUAL, op_id="op_y", at_frame=0,
                          annotator="lucas", created_at=T0),
        )
    )
    res.refresh()
    assert res.role_of(5, 404) is TrackRole.A
