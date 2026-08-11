"""Invariantes duras del esquema: lo que tiene que hacer fallar el parseo."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from boxtwin.core.schema import (
    SCHEMA_VERSION,
    AnnotationDoc,
    Assignment,
    Event,
    Interpolation,
    ManualBox,
    ManualTrack,
    Origin,
)
from boxtwin.core.types import (
    AssignmentOp,
    Completeness,
    FighterId,
    Guard,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
)

from tests.conftest import T0


def _origin() -> Origin:
    return Origin(op=AssignmentOp.MANUAL, op_id="op_0001", at_frame=0, annotator="lucas", created_at=T0)


def _event(**kw):
    base = dict(
        id="ev_0001",
        fighter=FighterId.A,
        start_frame=100,
        end_frame=120,
        side=Side.LEFT,
        punch_type=PunchType.STRAIGHT,
        target=Target.HEAD,
        completeness=Completeness.FULL,
        guard=Guard.ORTHODOX,
    )
    base.update(kw)
    return Event(**base)


# -- eventos ---------------------------------------------------------------


def test_evento_valido_minimo() -> None:
    ev = _event()
    assert ev.duration_frames == 21  # limites inclusivos
    assert ev.peak_frame is None
    assert ev.quality is Quality.CLEAN


@pytest.mark.parametrize("end", [100, 99, 0])
def test_evento_rechaza_end_no_posterior_a_start(end: int) -> None:
    with pytest.raises(ValidationError, match="start_frame < end_frame"):
        _event(end_frame=end)


@pytest.mark.parametrize("peak", [99, 121])
def test_evento_rechaza_peak_fuera_de_rango(peak: int) -> None:
    with pytest.raises(ValidationError, match="peak_frame"):
        _event(peak_frame=peak)


@pytest.mark.parametrize("peak", [100, 110, 120])
def test_evento_acepta_peak_en_los_bordes(peak: int) -> None:
    assert _event(peak_frame=peak).peak_frame == peak


def test_evento_rechaza_campo_desconocido() -> None:
    """extra=forbid: un campo raro es archivo de otra version, no un dato extra."""
    with pytest.raises(ValidationError):
        _event(punch_subtype="overhand")


def test_evento_rechaza_valor_fuera_del_vocabulario() -> None:
    with pytest.raises(ValidationError):
        _event(punch_type="overhand")


def test_solapamiento_de_eventos_se_mide_inclusivo() -> None:
    a = _event(id="ev_0001", start_frame=100, end_frame=120)
    b = _event(id="ev_0002", start_frame=120, end_frame=140)
    assert a.overlaps(b)
    assert a.overlap_frames(b) == 1

    c = _event(id="ev_0003", start_frame=121, end_frame=140)
    assert not a.overlaps(c)
    assert a.overlap_frames(c) == 0


# -- rangos semiabiertos ---------------------------------------------------


@pytest.mark.parametrize("end_excl", [100, 99])
def test_assignment_rechaza_rango_vacio_o_invertido(end_excl: int) -> None:
    with pytest.raises(ValidationError, match="start_frame < end_frame_excl"):
        Assignment(
            id="as_0001",
            track_id=1,
            role=TrackRole.A,
            start_frame=100,
            end_frame_excl=end_excl,
            origin=_origin(),
        )


def test_assignment_covers_es_semiabierto() -> None:
    a = Assignment(
        id="as_0001", track_id=1, role=TrackRole.A,
        start_frame=100, end_frame_excl=120, origin=_origin(),
    )
    assert a.covers(100)
    assert a.covers(119)
    assert not a.covers(120)


def test_interpolacion_rechaza_gap_len_inconsistente() -> None:
    with pytest.raises(ValidationError, match="gap_len"):
        Interpolation(
            id="in_0001",
            role=TrackRole.A,
            from_track_id=2,
            to_track_id=11,
            gap_start_frame=9004,
            gap_end_frame_excl=9008,
            gap_len=5,
            iou_at_join=0.7,
            accepted_by="lucas",
            created_at=T0,
        )


# -- cajas y tracks manuales -----------------------------------------------


@pytest.mark.parametrize("xyxy", [[10.0, 10.0, 10.0, 20.0], [10.0, 10.0, 20.0, 10.0], [30.0, 10.0, 20.0, 20.0]])
def test_caja_rechaza_coordenadas_invertidas(xyxy: list[float]) -> None:
    with pytest.raises(ValidationError, match="caja invalida"):
        ManualBox(frame=10, xyxy=xyxy)


def test_caja_rechaza_largo_distinto_de_cuatro() -> None:
    with pytest.raises(ValidationError):
        ManualBox(frame=10, xyxy=[10.0, 10.0, 20.0])


def test_track_manual_exige_id_negativo() -> None:
    """Los positivos son de BoT-SORT. Reusarlos haria colisionar los dos espacios."""
    with pytest.raises(ValidationError):
        ManualTrack(
            track_id=3,
            role=TrackRole.A,
            boxes=[ManualBox(frame=10, xyxy=[1.0, 1.0, 2.0, 2.0])],
            annotator="lucas",
            created_at=T0,
        )


def test_track_manual_exige_al_menos_una_caja() -> None:
    with pytest.raises(ValidationError):
        ManualTrack(track_id=-1, role=TrackRole.A, boxes=[], annotator="lucas", created_at=T0)


def test_track_manual_expone_su_rango(doc_rich: AnnotationDoc) -> None:
    mt = doc_rich.identity.manual_tracks[0]
    assert mt.start_frame == 21440
    assert mt.end_frame_excl == 21456


# -- redondeo estable ------------------------------------------------------


def test_coordenadas_se_redondean_al_validar() -> None:
    """
    El redondeo se aplica al validar y no al serializar, asi el objeto en memoria y el
    archivo coinciden y un round-trip no mueve ningun digito.
    """
    box = ManualBox(frame=10, xyxy=[10.123456, 10.987654, 20.5, 20.5])
    assert box.xyxy == [10.12, 10.99, 20.5, 20.5]


def test_iou_se_redondea_a_cuatro_decimales() -> None:
    o = Origin(
        op=AssignmentOp.RESEED, op_id="op_1", at_frame=0, annotator="lucas", created_at=T0,
        seed_box_xyxy=[1.0, 1.0, 2.0, 2.0], seed_iou=0.8333333333,
    )
    assert o.seed_iou == 0.8333


def test_los_timestamps_se_truncan_a_segundos() -> None:
    """
    Mismo criterio que el redondeo de coordenadas: si el objeto en memoria tuviera mas
    precision que el archivo, cargar lo que se acaba de guardar daria un documento distinto
    del que se tenia, por una diferencia que nunca llego al disco.
    """
    from datetime import datetime, timedelta, timezone

    ar = timezone(timedelta(hours=-3))
    con_micros = datetime(2026, 8, 11, 14, 3, 11, 123456, tzinfo=ar)
    o = Origin(
        op=AssignmentOp.MANUAL, op_id="op_1", at_frame=0, annotator="lucas",
        created_at=con_micros,
    )
    assert o.created_at.microsecond == 0
    assert o.created_at == con_micros.replace(microsecond=0)


@pytest.mark.parametrize("iou", [-0.1, 1.5])
def test_iou_rechaza_fuera_de_cero_uno(iou: float) -> None:
    with pytest.raises(ValidationError):
        Origin(
            op=AssignmentOp.RESEED, op_id="op_1", at_frame=0, annotator="lucas", created_at=T0,
            seed_box_xyxy=[1.0, 1.0, 2.0, 2.0], seed_iou=iou,
        )


# -- documento -------------------------------------------------------------


def test_documento_rechaza_version_distinta(doc_min: AnnotationDoc) -> None:
    """Parsear sin migrar tiene que fallar, no adivinar."""
    raw = doc_min.model_dump(mode="json")
    raw["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValidationError, match="migrations.migrate"):
        AnnotationDoc.model_validate(raw)


def test_documento_exige_los_dos_peleadores(doc_min: AnnotationDoc) -> None:
    raw = doc_min.model_dump(mode="json")
    del raw["fighters"]["fighter_B"]
    with pytest.raises(ValidationError, match="faltan peleadores"):
        AnnotationDoc.model_validate(raw)


def test_documento_rechaza_sha_que_no_es_sha(doc_min: AnnotationDoc) -> None:
    raw = doc_min.model_dump(mode="json")
    raw["video"]["sha256"] = "no-soy-un-hash"
    with pytest.raises(ValidationError):
        AnnotationDoc.model_validate(raw)


def test_guard_at_resuelve_overrides(doc_rich: AnnotationDoc) -> None:
    # Fuera del tramo manda el default del peleador.
    assert doc_rich.guard_at(FighterId.B, 100) is Guard.SOUTHPAW
    # El borde inferior entra y el superior no: el rango es semiabierto.
    assert doc_rich.guard_at(FighterId.B, 12800) is Guard.ORTHODOX
    assert doc_rich.guard_at(FighterId.B, 13949) is Guard.ORTHODOX
    assert doc_rich.guard_at(FighterId.B, 13950) is Guard.SOUTHPAW
    # fighter_A no tiene overrides.
    assert doc_rich.guard_at(FighterId.A, 12800) is Guard.ORTHODOX


def test_events_of_filtra_por_peleador(doc_rich: AnnotationDoc) -> None:
    assert [e.id for e in doc_rich.events_of(FighterId.A)] == ["ev_0041", "ev_0042"]
    assert [e.id for e in doc_rich.events_of(FighterId.B)] == ["ev_0043", "ev_0044"]
    assert doc_rich.event_by_id("no-existe") is None
