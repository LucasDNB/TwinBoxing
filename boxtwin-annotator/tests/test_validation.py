"""
Chequeos a nivel documento.

El caso negativo importa tanto como los positivos: una combinacion 1-2 no tiene que
producir ninguna advertencia. Si avisara, la advertencia saltaria en casi todos los
intercambios y el anotador aprenderia a ignorarlas.
"""

from __future__ import annotations

import pytest

from boxtwin.core.schema import (
    AnnotationDoc,
    Assignment,
    ComboOverride,
    Event,
    GuardOverride,
    Interpolation,
    Origin,
    UnreliableSegment,
)
from boxtwin.core.types import (
    AssignmentOp,
    ComboOp,
    Completeness,
    FighterId,
    Guard,
    IssueLevel,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
    UnreliableReason,
)
from boxtwin.core.validation import ISSUE_CODES, has_errors, validate_document

from tests.conftest import T0


def codes(doc: AnnotationDoc) -> list[str]:
    return [i.code for i in validate_document(doc)]


def _origin(op: AssignmentOp = AssignmentOp.MANUAL) -> Origin:
    return Origin(op=op, op_id="op_9999", at_frame=0, annotator="lucas", created_at=T0)


def _ev(doc: AnnotationDoc, **kw) -> Event:
    base = dict(
        id="ev_9000",
        fighter=FighterId.A,
        start_frame=6000,
        end_frame=6018,
        side=Side.LEFT,
        punch_type=PunchType.STRAIGHT,
        target=Target.HEAD,
        completeness=Completeness.FULL,
        guard=Guard.ORTHODOX,
    )
    base.update(kw)
    return Event(**base)


# -- linea base ------------------------------------------------------------


def test_documento_rico_no_tiene_issues(doc_rich: AnnotationDoc) -> None:
    assert validate_document(doc_rich) == []


def test_documento_minimo_no_tiene_issues(doc_min: AnnotationDoc) -> None:
    assert validate_document(doc_min) == []


def test_combinacion_uno_dos_no_avisa_nada(doc_rich: AnnotationDoc) -> None:
    """
    ev_0041 es un jab izquierdo que termina en 5126 y ev_0042 un cross derecho que arranca
    en 5120: se solapan 7 cuadros. Es la definicion de combinacion y no es un defecto.
    """
    a = doc_rich.event_by_id("ev_0041")
    b = doc_rich.event_by_id("ev_0042")
    assert a.overlaps(b) and a.side is not b.side
    assert validate_document(doc_rich) == []


# -- eventos ---------------------------------------------------------------


def test_ev_out_of_bounds(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, start_frame=53400, end_frame=53420))
    assert "EV_OUT_OF_BOUNDS" in codes(doc_rich)
    assert has_errors(validate_document(doc_rich))


def test_ev_too_long(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, start_frame=6000, end_frame=6060))
    assert "EV_TOO_LONG" in codes(doc_rich)


def test_ev_overlap_same_side_corto(doc_rich: AnnotationDoc) -> None:
    """Doble jab: se solapan 2 cuadros, por debajo del umbral. Avisa suave."""
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=5125, end_frame=5140))
    resultado = codes(doc_rich)
    assert "EV_OVERLAP_SAME_SIDE" in resultado
    assert "EV_OVERLAP_SAME_SIDE_LONG" not in resultado


def test_ev_overlap_same_side_largo(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=5115, end_frame=5140))
    resultado = codes(doc_rich)
    assert "EV_OVERLAP_SAME_SIDE_LONG" in resultado
    assert "EV_OVERLAP_SAME_SIDE" not in resultado


def test_ev_refire_too_fast(doc_rich: AnnotationDoc) -> None:
    """Dos golpes del mismo brazo arrancando a 2 cuadros: es el mismo golpe marcado dos veces."""
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=5110, end_frame=5128))
    resultado = codes(doc_rich)
    assert "EV_REFIRE_TOO_FAST" in resultado
    # El aviso de refire ya dice todo; repetirlo como solapamiento seria ruido.
    assert "EV_OVERLAP_SAME_SIDE" not in resultado
    assert "EV_OVERLAP_SAME_SIDE_LONG" not in resultado


def test_ev_guard_mismatch(doc_rich: AnnotationDoc) -> None:
    ev = doc_rich.event_by_id("ev_0044")
    ev.guard = Guard.SOUTHPAW  # la vigente en 13010 es orthodox
    issues = [i for i in validate_document(doc_rich) if i.code == "EV_GUARD_MISMATCH"]
    assert len(issues) == 1
    assert issues[0].level is IssueLevel.INFO
    assert issues[0].ref == "ev_0044"


def test_ev_unreliable_but_clean(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(
        _ev(doc_rich, id="ev_9001", fighter=FighterId.B, start_frame=21450, end_frame=21470,
            guard=Guard.SOUTHPAW, quality=Quality.CLEAN)
    )
    assert "EV_UNRELIABLE_BUT_CLEAN" in codes(doc_rich)


def test_ev_unreliable_no_avisa_si_ya_esta_marcado(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(
        _ev(doc_rich, id="ev_9001", fighter=FighterId.B, start_frame=21450, end_frame=21470,
            guard=Guard.SOUTHPAW, quality=Quality.PARTIAL_OCCLUSION)
    )
    assert "EV_UNRELIABLE_BUT_CLEAN" not in codes(doc_rich)


def test_ev_no_identity(doc_rich: AnnotationDoc) -> None:
    """El hueco [9004, 9008) es justo donde el tracker perdio a A."""
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=9000, end_frame=9020))
    assert "EV_NO_IDENTITY" in codes(doc_rich)


def test_ev_no_identity_no_avisa_con_track_manual(doc_rich: AnnotationDoc) -> None:
    """El track manual de B cubre 21440..21455, asi que un evento ahi tiene identidad."""
    doc_rich.events.append(
        _ev(doc_rich, id="ev_9001", fighter=FighterId.B, start_frame=21441, end_frame=21454,
            guard=Guard.SOUTHPAW, quality=Quality.AMBIGUOUS)
    )
    assert "EV_NO_IDENTITY" not in codes(doc_rich)


# -- guardia ---------------------------------------------------------------


def test_fg_guard_override_overlap(doc_rich: AnnotationDoc) -> None:
    fd = doc_rich.fighters[FighterId.B]
    fd.guard_overrides = fd.guard_overrides + [
        GuardOverride(start_frame=13900, end_frame_excl=14500, guard=Guard.SOUTHPAW)
    ]
    assert "FG_GUARD_OVERRIDE_OVERLAP" in codes(doc_rich)


# -- identidad -------------------------------------------------------------


def test_id_assignment_overlap(doc_rich: AnnotationDoc) -> None:
    doc_rich.identity.assignments.append(
        Assignment(id="as_9001", track_id=1, role=TrackRole.A, start_frame=3000,
                   end_frame_excl=5000, origin=_origin())
    )
    resultado = codes(doc_rich)
    assert "ID_ASSIGNMENT_OVERLAP" in resultado
    assert has_errors(validate_document(doc_rich))


def test_id_role_collision(doc_rich: AnnotationDoc) -> None:
    """Dos tracks distintos siendo fighter_A al mismo tiempo."""
    doc_rich.identity.assignments.append(
        Assignment(id="as_9001", track_id=77, role=TrackRole.A, start_frame=1000,
                   end_frame_excl=2000, origin=_origin())
    )
    assert "ID_ROLE_COLLISION" in codes(doc_rich)


def test_ignore_no_produce_colision(doc_rich: AnnotationDoc) -> None:
    """Puede haber varios ignorados a la vez: el arbitro y los que miran."""
    doc_rich.identity.assignments.append(
        Assignment(id="as_9001", track_id=78, role=TrackRole.IGNORE, start_frame=0,
                   end_frame_excl=53412, origin=_origin())
    )
    assert "ID_ROLE_COLLISION" not in codes(doc_rich)


def test_id_interp_too_long(doc_rich: AnnotationDoc) -> None:
    # El umbral se fija en el test y no se hereda del default: la validacion tiene que probar
    # la regla, no el valor que hoy trae el esquema. Antes dependia del default y el test se
    # rompio al subirlo de 5 a 60.
    doc_rich.settings_snapshot.interp_max_gap_frames = 10
    doc_rich.identity.interpolations.append(
        Interpolation(id="in_9001", role=TrackRole.B, from_track_id=1, to_track_id=2,
                      gap_start_frame=30000, gap_end_frame_excl=30012, gap_len=12,
                      iou_at_join=0.6, accepted_by="lucas", created_at=T0)
    )
    assert "ID_INTERP_TOO_LONG" in codes(doc_rich)


def test_id_orphan_interp(doc_rich: AnnotationDoc) -> None:
    doc_rich.identity.interpolations.append(
        Interpolation(id="in_9001", role=TrackRole.A, from_track_id=404, to_track_id=405,
                      gap_start_frame=30000, gap_end_frame_excl=30003, gap_len=3,
                      iou_at_join=0.6, accepted_by="lucas", created_at=T0)
    )
    issues = [i for i in validate_document(doc_rich) if i.code == "ID_ORPHAN_INTERP"]
    assert len(issues) == 1
    assert "404" in issues[0].message


# -- combinaciones y metricas ----------------------------------------------


def test_cb_unknown_event(doc_rich: AnnotationDoc) -> None:
    doc_rich.combo_overrides.append(
        ComboOverride(id="co_9001", op=ComboOp.JOIN, event_ids=["ev_0041", "ev_9999"],
                      annotator="lucas", created_at=T0)
    )
    assert "CB_UNKNOWN_EVENT" in codes(doc_rich)


def test_pm_missing_metrics(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=30000, end_frame=30018))
    issues = [i for i in validate_document(doc_rich) if i.code == "PM_MISSING_METRICS"]
    assert [i.ref for i in issues] == ["ev_9001"]


# -- forma de la salida ----------------------------------------------------


def test_issues_ordenados_por_nivel_y_frame(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=53400, end_frame=53411))
    doc_rich.events.append(_ev(doc_rich, id="ev_9002", start_frame=6000, end_frame=6060))
    issues = validate_document(doc_rich)
    rangos = [(i.level, i.frames[0] if i.frames else -1) for i in issues]
    orden = {IssueLevel.ERROR: 0, IssueLevel.WARNING: 1, IssueLevel.INFO: 2}
    assert rangos == sorted(rangos, key=lambda r: (orden[r[0]], r[1]))


def test_validacion_es_determinista(doc_rich: AnnotationDoc) -> None:
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=5110, end_frame=5128))
    assert validate_document(doc_rich) == validate_document(doc_rich)


def test_todos_los_codigos_declarados_tienen_el_nivel_que_emiten() -> None:
    """ISSUE_CODES es la tabla de referencia; si se desincroniza deja de servir."""
    assert set(ISSUE_CODES) == {
        "EV_OUT_OF_BOUNDS", "EV_TOO_LONG", "EV_OVERLAP_SAME_SIDE", "EV_OVERLAP_SAME_SIDE_LONG",
        "EV_REFIRE_TOO_FAST", "EV_GUARD_MISMATCH", "EV_UNRELIABLE_BUT_CLEAN", "EV_NO_IDENTITY",
        "FG_GUARD_OVERRIDE_OVERLAP", "ID_ASSIGNMENT_OVERLAP", "ID_ROLE_COLLISION",
        "ID_INTERP_TOO_LONG", "ID_ORPHAN_INTERP", "CB_UNKNOWN_EVENT", "PM_MISSING_METRICS",
    }


@pytest.mark.parametrize("code", sorted(ISSUE_CODES))
def test_nivel_emitido_coincide_con_la_tabla(code: str, doc_rich: AnnotationDoc) -> None:
    # Se rompe el documento a lo bruto para forzar la mayor cantidad de codigos posible.
    doc_rich.events.append(_ev(doc_rich, id="ev_9001", start_frame=5110, end_frame=5180))
    doc_rich.identity.assignments.append(
        Assignment(id="as_9001", track_id=1, role=TrackRole.A, start_frame=3000,
                   end_frame_excl=5000, origin=_origin())
    )
    for issue in validate_document(doc_rich):
        assert issue.level is ISSUE_CODES[issue.code]


# -- colisiones de rol: solo las reales -------------------------------------


class CacheFalso:
    """Lo minimo que mira el filtro: en que cuadros existe cada track."""

    def __init__(self, por_track: dict[int, list[int]]) -> None:
        self._t = por_track

    def frames_of_track(self, track_id: int):
        import numpy as np

        return np.asarray(self._t.get(track_id, []), dtype=np.int64)


def _dos_tracks_al_mismo_rol(doc_rich: AnnotationDoc) -> AnnotationDoc:
    from boxtwin.core.schema import Assignment, Origin

    o = doc_rich.identity.assignments[0].origin
    doc_rich.identity.assignments = [
        Assignment(id="as_9001", track_id=101, role=TrackRole.A,
                   start_frame=0, end_frame_excl=1000, origin=o),
        Assignment(id="as_9002", track_id=102, role=TrackRole.A,
                   start_frame=500, end_frame_excl=1000, origin=o),
    ]
    return doc_rich


def test_sin_cache_se_reportan_todas(doc_rich: AnnotationDoc) -> None:
    d = _dos_tracks_al_mismo_rol(doc_rich)
    assert "ID_ROLE_COLLISION" in codes(d)


def test_con_cache_no_se_reporta_si_los_tracks_no_coexisten(doc_rich: AnnotationDoc) -> None:
    """
    Solaparse en rango es normal y hasta util: hace de respaldo cuando el tracker fragmenta a
    un peleador. Lo que importa es si los dos tracks estan detectados en el mismo cuadro.
    """
    d = _dos_tracks_al_mismo_rol(doc_rich)
    cache = CacheFalso({101: list(range(0, 400)), 102: list(range(500, 900))})
    assert "ID_ROLE_COLLISION" not in {i.code for i in validate_document(d, cache)}


def test_con_cache_si_se_reporta_cuando_coexisten(doc_rich: AnnotationDoc) -> None:
    d = _dos_tracks_al_mismo_rol(doc_rich)
    cache = CacheFalso({101: list(range(0, 900)), 102: list(range(500, 900))})
    assert "ID_ROLE_COLLISION" in {i.code for i in validate_document(d, cache)}


def test_el_cache_no_toca_los_otros_codigos(doc_rich: AnnotationDoc) -> None:
    d = _dos_tracks_al_mismo_rol(doc_rich)
    cache = CacheFalso({101: list(range(0, 400)), 102: list(range(500, 900))})
    sin = {i.code for i in validate_document(d)} - {"ID_ROLE_COLLISION"}
    con = {i.code for i in validate_document(d, cache)} - {"ID_ROLE_COLLISION"}
    assert sin == con
