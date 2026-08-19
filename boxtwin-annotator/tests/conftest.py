"""Fixtures compartidas: un documento minimo y uno que ejercita todo el esquema."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from boxtwin.core.schema import (
    SCHEMA_VERSION,
    AnnotationDoc,
    Assignment,
    AnnotatorInfo,
    ComboOverride,
    Counters,
    Event,
    EventMetrics,
    FighterDef,
    Generator,
    GuardOverride,
    Identity,
    Interpolation,
    ManualBox,
    ManualTrack,
    Origin,
    PoseRef,
    Process,
    SessionMetrics,
    Totals,
    UnreliableSegment,
    VideoInfo,
    new_document,
)
from boxtwin.core.types import (
    AssignmentOp,
    ComboOp,
    Completeness,
    FighterId,
    FpsSource,
    Guard,
    KeypointFormat,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
    UnreliableReason,
    UnreliableSource,
)

AR = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 11, 14, 3, 11, tzinfo=AR)
SHA_VIDEO = "9f2b8e41c07a5d3f6b1e2a94cc8017d5aa3f61b0e7d2c948f15a6b3c0d7e4128"
SHA_META = "1a7743ce90de5b28f4c6017a9b3e5d2077bb31ca44e0918f6d2a5c37be04198d"


@pytest.fixture
def video_info() -> VideoInfo:
    # Numeros tomados de tSNybMUpMKk.mp4, 1080p a 30000/1001. El contenedor no declara
    # nb_frames, de ahi que total_frames_declared sea None.
    return VideoInfo(
        path="videos/spar_2026-03-11_cam1.mp4",
        sha256=SHA_VIDEO,
        size_bytes=734920184,
        mtime=datetime(2026, 3, 11, 21, 10, 3, tzinfo=AR),
        fps=30000 / 1001,
        fps_declared=30.0,
        fps_source=FpsSource.MEASURED,
        width=1920,
        height=1080,
        total_frames=53412,
        total_frames_declared=None,
        duration_s=1782.111,
        codec="h264",
    )


@pytest.fixture
def pose_ref() -> PoseRef:
    return PoseRef(
        npz_path="cache/spar_2026-03-11_cam1.pose.npz",
        meta_path="cache/spar_2026-03-11_cam1.meta.json",
        meta_sha256=SHA_META,
        keypoint_format=KeypointFormat.COCO17,
        keypoint_sources={"0-16": "yolov8l-pose"},
    )


@pytest.fixture
def doc_min(video_info: VideoInfo, pose_ref: PoseRef) -> AnnotationDoc:
    """Documento vacio y valido, tal como sale del preproceso."""
    return new_document(
        app_version="0.1.0",
        now=T0,
        video=video_info,
        pose=pose_ref,
        guard_a=Guard.ORTHODOX,
        guard_b=Guard.SOUTHPAW,
        label_a="Rojo",
        label_b="Azul",
    )


def _origin(op: AssignmentOp, op_id: str, at_frame: int, **kw) -> Origin:
    return Origin(op=op, op_id=op_id, at_frame=at_frame, annotator="lucas", created_at=T0, **kw)


@pytest.fixture
def doc_rich(video_info: VideoInfo, pose_ref: PoseRef) -> AnnotationDoc:
    """
    Documento que ejercita todo el esquema y que no produce ningun Issue.

    Incluye un ID switch en 4120, un re-seed en 9012 con su interpolacion, una caja
    manual, un tramo ocluido, un cambio de guardia y una combinacion 1-2, que es el caso
    que obliga a los carriles por brazo.
    """
    assignments = [
        # Tramo inicial.
        Assignment(
            id="as_0001",
            track_id=1,
            role=TrackRole.A,
            start_frame=0,
            end_frame_excl=4120,
            origin=_origin(AssignmentOp.MANUAL, "op_0001", 0),
        ),
        Assignment(
            id="as_0002",
            track_id=2,
            role=TrackRole.B,
            start_frame=0,
            end_frame_excl=4120,
            origin=_origin(AssignmentOp.MANUAL, "op_0001", 0),
        ),
        # ID switch en 4120: los dos nacen del mismo op_id con los roles cruzados.
        Assignment(
            id="as_0003",
            track_id=1,
            role=TrackRole.B,
            start_frame=4120,
            end_frame_excl=53412,
            origin=_origin(AssignmentOp.SWAP, "op_0021", 4120),
        ),
        Assignment(
            id="as_0004",
            track_id=2,
            role=TrackRole.A,
            start_frame=4120,
            end_frame_excl=9004,
            origin=_origin(AssignmentOp.SWAP, "op_0021", 4120),
        ),
        # Re-seed: el tracker perdio a A en 9004 y lo recupero como track 11 en 9008.
        Assignment(
            id="as_0005",
            track_id=11,
            role=TrackRole.A,
            start_frame=9008,
            end_frame_excl=53412,
            origin=_origin(
                AssignmentOp.RESEED,
                "op_0034",
                9008,
                seed_box_xyxy=[812.0, 340.5, 1044.0, 902.0],
                seed_iou=0.83,
            ),
        ),
        # El arbitro.
        Assignment(
            id="as_0006",
            track_id=5,
            role=TrackRole.IGNORE,
            start_frame=0,
            end_frame_excl=53412,
            origin=_origin(AssignmentOp.MANUAL, "op_0009", 300),
        ),
    ]

    identity = Identity(
        assignments=assignments,
        manual_tracks=[
            ManualTrack(
                track_id=-1,
                role=TrackRole.B,
                boxes=[
                    ManualBox(frame=21440, xyxy=[640.0, 288.0, 902.0, 860.0]),
                    ManualBox(frame=21455, xyxy=[655.0, 291.0, 918.0, 866.0]),
                ],
                notes="clinch contra las cuerdas, el tracker fusiona los dos cuerpos",
                annotator="lucas",
                created_at=T0,
            )
        ],
        interpolations=[
            Interpolation(
                id="in_0001",
                role=TrackRole.A,
                from_track_id=2,
                to_track_id=11,
                gap_start_frame=9004,
                gap_end_frame_excl=9008,
                gap_len=4,
                iou_at_join=0.71,
                accepted_by="lucas",
                created_at=T0,
            )
        ],
    )

    eventos = [
        # Combinacion 1-2 de fighter_A: el jab izquierdo todavia retrae cuando sale el
        # cross derecho. Lados distintos, no tiene que avisar nada.
        Event(
            id="ev_0041",
            fighter=FighterId.A,
            start_frame=5108,
            peak_frame=5114,
            end_frame=5126,
            side=Side.LEFT,
            punch_type=PunchType.STRAIGHT,
            target=Target.HEAD,
            completeness=Completeness.FULL,
            landed=Landed.LANDED,
            guard=Guard.ORTHODOX,
            quality=Quality.CLEAN,
        ),
        Event(
            id="ev_0042",
            fighter=FighterId.A,
            start_frame=5120,
            peak_frame=5127,
            end_frame=5139,
            side=Side.RIGHT,
            punch_type=PunchType.HOOK,
            target=Target.HEAD,
            completeness=Completeness.FULL,
            landed=Landed.BLOCKED,
            guard=Guard.ORTHODOX,
            quality=Quality.CLEAN,
        ),
        Event(
            id="ev_0043",
            fighter=FighterId.B,
            start_frame=5131,
            end_frame=5142,
            side=Side.LEFT,
            punch_type=PunchType.STRAIGHT,
            target=Target.HEAD,
            completeness=Completeness.FEINT,
            landed=Landed.UNKNOWN,
            guard=Guard.SOUTHPAW,
            quality=Quality.AMBIGUOUS,
            notes="amaga y sale con el paso lateral",
        ),
        # Dentro del tramo en que B esta en guardia ortodoxa.
        Event(
            id="ev_0044",
            fighter=FighterId.B,
            start_frame=13010,
            peak_frame=13018,
            end_frame=13029,
            side=Side.RIGHT,
            punch_type=PunchType.UPPERCUT,
            target=Target.BODY,
            completeness=Completeness.FULL,
            landed=Landed.MISSED,
            guard=Guard.ORTHODOX,
            quality=Quality.PARTIAL_OCCLUSION,
        ),
    ]

    metrics = {
        "ev_0041": EventMetrics(
            annotator="lucas", session_id="se_0003", created_at=T0, confirmed_at=T0,
            active_ms=6120, replays=2, edits=0,
        ),
        "ev_0042": EventMetrics(
            annotator="lucas", session_id="se_0003", created_at=T0, confirmed_at=T0,
            active_ms=8420, replays=3, edits=1, last_edited_at=T0,
        ),
        "ev_0043": EventMetrics(
            annotator="lucas", session_id="se_0003", created_at=T0, confirmed_at=T0,
            active_ms=11030, replays=5, edits=0,
        ),
        "ev_0044": EventMetrics(
            annotator="lucas", session_id="se_0003", created_at=T0, confirmed_at=T0,
            active_ms=7240, replays=2, edits=0,
        ),
    }

    return AnnotationDoc(
        schema_version=SCHEMA_VERSION,
        generator=Generator(app_version="0.1.0", created_at=T0, updated_at=T0),
        counters=Counters(event=44, assignment=6, unreliable=1, interpolation=1, combo=1, session=3, op=34),
        video=video_info,
        pose=pose_ref,
        fighters={
            FighterId.A: FighterDef(label="Rojo", guard=Guard.ORTHODOX),
            FighterId.B: FighterDef(
                label="Azul",
                guard=Guard.SOUTHPAW,
                guard_overrides=[
                    GuardOverride(start_frame=12800, end_frame_excl=13950, guard=Guard.ORTHODOX, notes="cambia de guardia")
                ],
            ),
        },
        identity=identity,
        unreliable_segments=[
            UnreliableSegment(
                id="un_0001",
                fighter=FighterId.B,
                start_frame=21400,
                end_frame_excl=21620,
                reason=UnreliableReason.OCCLUDED,
                notes="clinch, el arbitro tapa el torso",
                source=UnreliableSource.MANUAL,
                annotator="lucas",
                created_at=T0,
            )
        ],
        events=eventos,
        combo_overrides=[
            ComboOverride(
                id="co_0001",
                op=ComboOp.SPLIT,
                event_ids=["ev_0042", "ev_0043"],
                reason="son de peleadores distintos, no es una combinacion",
                annotator="lucas",
                created_at=T0,
            )
        ],
        process=Process(
            annotators=[AnnotatorInfo(id="lucas", name="Lucas Benitez")],
            sessions=[
                SessionMetrics(
                    id="se_0003",
                    annotator="lucas",
                    started_at=T0,
                    ended_at=T0 + timedelta(hours=4),
                    active_ms=8412300,
                    events_created=4,
                    events_edited=1,
                    app_version="0.1.0",
                )
            ],
            totals=Totals(active_ms=8412300, events=4, median_ms_per_event=7830),
            event_metrics=metrics,
        ),
    )
