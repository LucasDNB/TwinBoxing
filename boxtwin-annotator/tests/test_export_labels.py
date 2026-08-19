"""
Espacios de clases y ventanas del export.

El orden de la lista de clases es contrato: el indice es la etiqueta numerica, asi que si
cambiara, los modelos ya entrenados quedarian con las clases permutadas sin que nada avise.
"""

from __future__ import annotations

import pytest

from boxtwin.core.export.labels import (
    BACKGROUND,
    FEINT,
    LabelSpace,
    class_index,
    class_list,
    class_name,
)
from boxtwin.core.export.windows import (
    frames_ocupados,
    ventana_de_evento,
    ventanas_de_fondo,
)
from boxtwin.core.identity import IdentityResolver
from boxtwin.core.schema import AnnotationDoc, Event
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Guard,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
)

from tests.test_identity_ops import CAJA, cache_con


def ev(**kw) -> Event:
    base = dict(
        id="ev_9001", fighter=FighterId.A, start_frame=100, end_frame=120,
        side=Side.LEFT, punch_type=PunchType.STRAIGHT, target=Target.HEAD,
        completeness=Completeness.FULL, landed=Landed.UNKNOWN,
        guard=Guard.ORTHODOX, quality=Quality.CLEAN,
    )
    base.update(kw)
    return Event(**base)


# -- listas de clases ------------------------------------------------------


@pytest.mark.parametrize("space", list(LabelSpace))
@pytest.mark.parametrize("n", [6, 12, 14])
def test_cantidad_de_clases(space: LabelSpace, n: int) -> None:
    assert len(class_list(space, n)) == n


def test_las_seis_de_lead_rear_son_las_de_boxingvi() -> None:
    """
    Jab, Cross, Lead/Rear Hook, Lead/Rear Uppercut. Que coincidan hace que un dataset propio
    exportado asi sea concatenable con lo que ya esta cortado.
    """
    assert set(class_list(LabelSpace.LEAD_REAR, 6)) == {
        "straight-lead", "straight-rear",
        "hook-lead", "hook-rear",
        "uppercut-lead", "uppercut-rear",
    }


def test_el_espacio_de_14_extiende_al_de_12_sin_renumerar() -> None:
    """Agregar el espacio de 14 no puede cambiar el indice de las 12 anteriores."""
    doce = class_list(LabelSpace.LEAD_REAR, 12)
    catorce = class_list(LabelSpace.LEAD_REAR, 14)
    assert catorce[:12] == doce
    assert catorce[12:] == [FEINT, BACKGROUND]


def test_el_orden_es_estable() -> None:
    assert class_list(LabelSpace.SIDE, 12) == class_list(LabelSpace.SIDE, 12)


def test_conjunto_invalido() -> None:
    with pytest.raises(ValueError):
        class_list(LabelSpace.SIDE, 8)


# -- traduccion de eventos -------------------------------------------------


def test_side_usa_el_lado_observado() -> None:
    assert class_name(ev(side=Side.LEFT), LabelSpace.SIDE, 6) == "straight-left"


def test_lead_rear_deriva_de_la_guardia() -> None:
    """Izquierda en ortodoxo es adelantada; en zurdo, la misma izquierda es atrasada."""
    orto = ev(side=Side.LEFT, guard=Guard.ORTHODOX)
    zurdo = ev(side=Side.LEFT, guard=Guard.SOUTHPAW)
    assert class_name(orto, LabelSpace.LEAD_REAR, 6) == "straight-lead"
    assert class_name(zurdo, LabelSpace.LEAD_REAR, 6) == "straight-rear"
    # En el espacio side son la misma clase: es lo que se observa, no depende de la guardia.
    assert class_name(orto, LabelSpace.SIDE, 6) == class_name(zurdo, LabelSpace.SIDE, 6)


def test_doce_agrega_la_altura() -> None:
    assert class_name(ev(target=Target.BODY), LabelSpace.LEAD_REAR, 12) == "straight-lead-body"


def test_el_amague_queda_fuera_de_6_y_12() -> None:
    """
    Un amague no es un recto mal hecho. Meterlo en la clase recto ensucia justamente los
    ejemplos que definen la frontera.
    """
    a = ev(completeness=Completeness.FEINT)
    assert class_name(a, LabelSpace.LEAD_REAR, 6) is None
    assert class_name(a, LabelSpace.LEAD_REAR, 12) is None
    assert class_name(a, LabelSpace.LEAD_REAR, 14) == FEINT


def test_el_abortado_queda_fuera_de_todos() -> None:
    a = ev(completeness=Completeness.ABORTED)
    for n in (6, 12, 14):
        assert class_name(a, LabelSpace.LEAD_REAR, n) is None


def test_class_index_coincide_con_la_lista() -> None:
    e = ev(punch_type=PunchType.HOOK, target=Target.BODY, guard=Guard.ORTHODOX, side=Side.RIGHT)
    idx = class_index(e, LabelSpace.LEAD_REAR, 12)
    assert class_list(LabelSpace.LEAD_REAR, 12)[idx] == "hook-rear-body"


# -- ventanas --------------------------------------------------------------


def test_ventana_de_evento(doc_min: AnnotationDoc) -> None:
    doc_min.video.total_frames = 1000
    v = ventana_de_evento(ev(start_frame=100, end_frame=120), doc_min)
    assert (v.start_frame, v.end_frame, v.n_frames) == (100, 120, 21)


def test_el_relleno_se_recorta_contra_el_video(doc_min: AnnotationDoc) -> None:
    doc_min.video.total_frames = 125
    v = ventana_de_evento(ev(start_frame=2, end_frame=120), doc_min, pad=10)
    assert (v.start_frame, v.end_frame) == (0, 124)


def test_frames_ocupados_incluye_el_margen(doc_min: AnnotationDoc) -> None:
    doc_min.video.total_frames = 1000
    doc_min.events = [ev(start_frame=100, end_frame=110)]
    ocupados = frames_ocupados(doc_min, FighterId.A, margen=5)
    assert 95 in ocupados and 115 in ocupados
    assert 94 not in ocupados and 116 not in ocupados


# -- fondo -----------------------------------------------------------------


def _doc_con_identidad(doc_min: AnnotationDoc, n: int = 300) -> tuple[AnnotationDoc, IdentityResolver]:
    from boxtwin.core.identity_ops import AssignRole
    from boxtwin.core.undo import UndoStack

    doc_min.video.total_frames = n
    pila = UndoStack(doc_min)
    pila.do(AssignRole(track_id=1, role=TrackRole.A, start_frame=0, end_frame_excl=n))
    cache = cache_con({f: [(1, CAJA)] for f in range(n)}, n)
    return doc_min, IdentityResolver(doc_min, cache)


def test_el_fondo_evita_los_eventos_y_su_margen(doc_min: AnnotationDoc) -> None:
    """
    Un fondo que contiene la cola de un golpe le ensena al modelo que ese movimiento es
    fondo, y rompe la frontera que tiene que aprender.
    """
    doc, res = _doc_con_identidad(doc_min)
    doc.events = [ev(start_frame=100, end_frame=150)]
    ventanas = ventanas_de_fondo(doc, res, cantidad=50, largo=10, seed=1, margen=5)
    for v in ventanas:
        assert v.end_frame < 95 or v.start_frame > 155


def test_el_fondo_excluye_tramos_no_confiables(doc_min: AnnotationDoc) -> None:
    from datetime import datetime

    from boxtwin.core.schema import UnreliableSegment
    from boxtwin.core.types import UnreliableReason

    doc, res = _doc_con_identidad(doc_min)
    doc.unreliable_segments = [
        UnreliableSegment(
            id="un_0001", fighter=FighterId.A, start_frame=50, end_frame_excl=150,
            reason=UnreliableReason.OCCLUDED, annotator="t",
            created_at=datetime.now().astimezone(),
        )
    ]
    res.refresh()
    for v in ventanas_de_fondo(doc, res, cantidad=50, largo=10, seed=1, margen=0):
        if v.fighter is FighterId.A:
            assert v.end_frame < 50 or v.start_frame >= 150


def test_el_fondo_exige_identidad_en_toda_la_ventana(doc_min: AnnotationDoc) -> None:
    """Un fondo sin keypoints entraria como ceros y el modelo puede aprender esa clase espuria."""
    doc, res = _doc_con_identidad(doc_min)
    # fighter_B nunca tiene identidad resuelta
    assert all(v.fighter is FighterId.A for v in ventanas_de_fondo(
        doc, res, cantidad=50, largo=10, seed=1, margen=0
    ))


def test_el_muestreo_es_determinista(doc_min: AnnotationDoc) -> None:
    """Sin esto, dos corridas del mismo comando dan datasets distintos."""
    doc, res = _doc_con_identidad(doc_min)
    a = ventanas_de_fondo(doc, res, cantidad=5, largo=10, seed=7, margen=0)
    b = ventanas_de_fondo(doc, res, cantidad=5, largo=10, seed=7, margen=0)
    assert a == b
    c = ventanas_de_fondo(doc, res, cantidad=5, largo=10, seed=8, margen=0)
    assert a != c or len(a) < 5


def test_las_ventanas_de_fondo_no_se_solapan(doc_min: AnnotationDoc) -> None:
    doc, res = _doc_con_identidad(doc_min)
    ventanas = sorted(
        ventanas_de_fondo(doc, res, cantidad=100, largo=10, seed=3, margen=0),
        key=lambda v: v.start_frame,
    )
    for a, b in zip(ventanas, ventanas[1:]):
        assert b.start_frame > a.end_frame


def test_pedir_mas_fondo_del_disponible(doc_min: AnnotationDoc) -> None:
    doc, res = _doc_con_identidad(doc_min, n=50)
    ventanas = ventanas_de_fondo(doc, res, cantidad=1000, largo=10, seed=1, margen=0)
    assert 0 < len(ventanas) <= 5


def test_largo_invalido(doc_min: AnnotationDoc) -> None:
    doc, res = _doc_con_identidad(doc_min)
    with pytest.raises(ValueError):
        ventanas_de_fondo(doc, res, cantidad=1, largo=0, seed=1)
