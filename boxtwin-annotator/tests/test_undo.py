"""
Historial de deshacer.

Lo que se fija aca es que deshacer devuelva el documento a un estado que realmente existio.
Un deshacer parcial es peor que no tener deshacer: deja datos que parecen buenos y no lo
son, y nadie vuelve a revisarlos.
"""

from __future__ import annotations

import pytest

from boxtwin.core.schema import AnnotationDoc, Event, EventMetrics
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Guard,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)
from boxtwin.core.undo import (
    AddEvent,
    CompositeCommand,
    DeleteEvent,
    EditEvent,
    UndoStack,
)

from tests.conftest import T0


def ev(eid: str = "ev_9001", **kw) -> Event:
    base = dict(
        id=eid, fighter=FighterId.A, start_frame=100, end_frame=120,
        side=Side.LEFT, punch_type=PunchType.STRAIGHT, target=Target.HEAD,
        completeness=Completeness.FULL, landed=Landed.UNKNOWN,
        guard=Guard.ORTHODOX, quality=Quality.CLEAN,
    )
    base.update(kw)
    return Event(**base)


def metricas() -> EventMetrics:
    return EventMetrics(
        annotator="lucas", session_id="se_0001", created_at=T0, confirmed_at=T0,
        active_ms=4200, replays=2,
    )


# -- alta ------------------------------------------------------------------


def test_alta_y_deshacer(doc_min: AnnotationDoc) -> None:
    pila = UndoStack(doc_min)
    pila.do(AddEvent(ev(), metricas()))
    assert [e.id for e in doc_min.events] == ["ev_9001"]
    assert "ev_9001" in doc_min.process.event_metrics

    pila.undo()
    assert doc_min.events == []
    # Las metricas se van con el evento: si quedaran, el reporte de tiempo contaria un
    # evento que ya no existe.
    assert "ev_9001" not in doc_min.process.event_metrics


def test_rehacer_restaura_todo(doc_min: AnnotationDoc) -> None:
    pila = UndoStack(doc_min)
    pila.do(AddEvent(ev(), metricas()))
    pila.undo()
    pila.redo()
    assert [e.id for e in doc_min.events] == ["ev_9001"]
    assert doc_min.process.event_metrics["ev_9001"].replays == 2


# -- baja ------------------------------------------------------------------


def test_baja_y_deshacer(doc_rich: AnnotationDoc) -> None:
    pila = UndoStack(doc_rich)
    antes = [e.id for e in doc_rich.events]
    m_antes = doc_rich.process.event_metrics["ev_0042"].model_dump()

    pila.do(DeleteEvent("ev_0042"))
    assert "ev_0042" not in [e.id for e in doc_rich.events]

    pila.undo()
    assert sorted(e.id for e in doc_rich.events) == sorted(antes)
    assert doc_rich.process.event_metrics["ev_0042"].model_dump() == m_antes


def test_baja_de_evento_inexistente(doc_min: AnnotationDoc) -> None:
    with pytest.raises(KeyError):
        UndoStack(doc_min).do(DeleteEvent("ev_9999"))


def test_no_se_puede_deshacer_una_baja_no_ejecutada(doc_min: AnnotationDoc) -> None:
    with pytest.raises(RuntimeError):
        DeleteEvent("ev_0001").undo(doc_min)


# -- edicion ---------------------------------------------------------------


def test_edicion_y_deshacer(doc_rich: AnnotationDoc) -> None:
    pila = UndoStack(doc_rich)
    pila.do(EditEvent("ev_0042", "target", Target.BODY))
    assert doc_rich.event_by_id("ev_0042").target is Target.BODY
    pila.undo()
    assert doc_rich.event_by_id("ev_0042").target is Target.HEAD


def test_la_edicion_lee_el_valor_anterior_al_ejecutar(doc_rich: AnnotationDoc) -> None:
    """
    El `antes` no lo pasa el llamador: si pudiera pasar uno equivocado, deshacer dejaria el
    documento en un estado que nunca existio.
    """
    pila = UndoStack(doc_rich)
    doc_rich.event_by_id("ev_0042").target = Target.BODY  # cambio por fuera de la pila
    pila.do(EditEvent("ev_0042", "target", Target.HEAD))
    pila.undo()
    assert doc_rich.event_by_id("ev_0042").target is Target.BODY


def test_edicion_rechaza_campo_inexistente(doc_rich: AnnotationDoc) -> None:
    with pytest.raises(KeyError):
        UndoStack(doc_rich).do(EditEvent("ev_0042", "no_existe", 1))


def test_edicion_invalida_la_rechaza_el_esquema(doc_rich: AnnotationDoc) -> None:
    """Un fin anterior al inicio no puede entrar ni siquiera pasando por la pila."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UndoStack(doc_rich).do(EditEvent("ev_0042", "end_frame", 10))


# -- compuesto -------------------------------------------------------------


def test_compuesto_es_un_solo_paso(doc_rich: AnnotationDoc) -> None:
    pila = UndoStack(doc_rich)
    pila.do(CompositeCommand([
        EditEvent("ev_0042", "target", Target.BODY),
        EditEvent("ev_0043", "target", Target.BODY),
    ]))
    assert doc_rich.event_by_id("ev_0042").target is Target.BODY
    assert doc_rich.event_by_id("ev_0043").target is Target.BODY

    pila.undo()
    assert doc_rich.event_by_id("ev_0042").target is Target.HEAD
    assert doc_rich.event_by_id("ev_0043").target is Target.HEAD
    assert len(pila) == 0


def test_compuesto_es_todo_o_nada(doc_rich: AnnotationDoc) -> None:
    """
    Si uno falla, se revierte lo ya aplicado. Un intercambio de identidad a medias dejaria
    un peleador corregido y el otro no, que es peor que no haberlo intentado.
    """
    pila = UndoStack(doc_rich)
    with pytest.raises(KeyError):
        pila.do(CompositeCommand([
            EditEvent("ev_0042", "target", Target.BODY),
            EditEvent("ev_9999", "target", Target.BODY),  # no existe
        ]))
    assert doc_rich.event_by_id("ev_0042").target is Target.HEAD
    assert len(pila) == 0


# -- pila ------------------------------------------------------------------


def test_estado_de_la_pila(doc_min: AnnotationDoc) -> None:
    pila = UndoStack(doc_min)
    assert not pila.can_undo and not pila.can_redo
    pila.do(AddEvent(ev()))
    assert pila.can_undo and not pila.can_redo
    assert pila.undo_label == "agregar evento"
    pila.undo()
    assert not pila.can_undo and pila.can_redo
    assert pila.redo_label == "agregar evento"


def test_una_operacion_nueva_invalida_el_rehacer(doc_min: AnnotationDoc) -> None:
    """La rama deshecha ya no aplica sobre este documento."""
    pila = UndoStack(doc_min)
    pila.do(AddEvent(ev("ev_0001")))
    pila.undo()
    assert pila.can_redo
    pila.do(AddEvent(ev("ev_0002")))
    assert not pila.can_redo


def test_tope_descarta_lo_mas_viejo(doc_min: AnnotationDoc) -> None:
    pila = UndoStack(doc_min, depth=3)
    for i in range(6):
        pila.do(AddEvent(ev(f"ev_{i:04d}")))
    assert len(pila) == 3
    for _ in range(3):
        pila.undo()
    # Los tres primeros quedaron fuera del historial y siguen en el documento.
    assert sorted(e.id for e in doc_min.events) == ["ev_0000", "ev_0001", "ev_0002"]


def test_profundidad_minima(doc_min: AnnotationDoc) -> None:
    with pytest.raises(ValueError):
        UndoStack(doc_min, depth=0)


def test_cumple_el_minimo_pedido(doc_min: AnnotationDoc) -> None:
    """El enunciado pide al menos 50 pasos sobre todas las operaciones."""
    from boxtwin.core.undo import DEFAULT_DEPTH

    assert DEFAULT_DEPTH >= 50
    pila = UndoStack(doc_min)
    for i in range(50):
        pila.do(AddEvent(ev(f"ev_{i:04d}")))
    for _ in range(50):
        assert pila.undo() is not None
    assert doc_min.events == []


def test_deshacer_sin_nada_no_explota(doc_min: AnnotationDoc) -> None:
    pila = UndoStack(doc_min)
    assert pila.undo() is None
    assert pila.redo() is None
