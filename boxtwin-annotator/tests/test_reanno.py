"""
Muestra de reanotacion ciega.

Lo que se fija es que la muestra se pueda defender: determinista por semilla, congelada una
vez sorteada, y con ventanas cuyos bordes no revelen las fronteras que se van a medir.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from boxtwin.core.reanno import (
    PAD_MAX,
    PAD_MIN,
    REANNO_KIND,
    ReannoTrial,
    SampleFrozenError,
    TrialLabels,
    annot_sha,
    cargar,
    cargar_o_sortear,
    guardar,
    sortear,
)
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
)

AR = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 11, 14, 0, 0, tzinfo=AR)


def ev(eid: str, ini: int) -> Event:
    return Event(
        id=eid, fighter=FighterId.A, start_frame=ini, end_frame=ini + 20,
        side=Side.LEFT, punch_type=PunchType.STRAIGHT, target=Target.HEAD,
        completeness=Completeness.FULL, landed=Landed.LANDED,
        guard=Guard.ORTHODOX, quality=Quality.CLEAN,
    )


@pytest.fixture
def doc(doc_min: AnnotationDoc) -> AnnotationDoc:
    doc_min.video.total_frames = 5000
    doc_min.events = [ev(f"ev_{i:04d}", 100 + i * 50) for i in range(1, 41)]
    return doc_min


# -- sorteo ----------------------------------------------------------------


def test_sortea_la_fraccion_pedida(doc: AnnotationDoc) -> None:
    r = sortear(doc, fraction=0.10, seed=1)
    assert r.sample.n == 4  # 10% de 40
    assert len(r.sample.event_ids) == 4


def test_siempre_sortea_al_menos_uno(doc: AnnotationDoc) -> None:
    doc.events = doc.events[:3]
    assert sortear(doc, fraction=0.01, seed=1).sample.n == 1


def test_el_sorteo_es_determinista(doc: AnnotationDoc) -> None:
    """Sin esto, la muestra no se puede reproducir ni auditar."""
    a = sortear(doc, fraction=0.25, seed=42)
    b = sortear(doc, fraction=0.25, seed=42)
    assert a.sample.event_ids == b.sample.event_ids
    assert a.sample.windows == b.sample.windows


def test_semillas_distintas_dan_muestras_distintas(doc: AnnotationDoc) -> None:
    a = sortear(doc, fraction=0.25, seed=1)
    b = sortear(doc, fraction=0.25, seed=2)
    assert a.sample.event_ids != b.sample.event_ids


def test_el_sorteo_no_depende_del_orden_en_memoria(doc: AnnotationDoc) -> None:
    """Los ids se ordenan antes de sortear: el orden de la lista puede variar entre corridas."""
    a = sortear(doc, fraction=0.25, seed=5)
    doc.events = list(reversed(doc.events))
    b = sortear(doc, fraction=0.25, seed=5)
    assert a.sample.event_ids == b.sample.event_ids


def test_guarda_el_hash_de_la_anotacion(doc: AnnotationDoc) -> None:
    r = sortear(doc, fraction=0.1, seed=1)
    assert r.source_annot_sha256 == annot_sha(doc)
    assert r.video_sha256 == doc.video.sha256
    assert r.kind == REANNO_KIND


def test_fraccion_invalida(doc: AnnotationDoc) -> None:
    for f in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            sortear(doc, fraction=f, seed=1)


def test_sin_eventos(doc_min: AnnotationDoc) -> None:
    with pytest.raises(ValueError, match="no hay eventos"):
        sortear(doc_min, fraction=0.1, seed=1)


# -- ventanas --------------------------------------------------------------


def test_la_ventana_tiene_relleno_a_los_dos_lados(doc: AnnotationDoc) -> None:
    r = sortear(doc, fraction=0.5, seed=3)
    for eid, (ini, fin) in r.sample.windows.items():
        e = doc.event_by_id(eid)
        assert ini < e.start_frame
        assert fin > e.end_frame
        assert PAD_MIN <= e.start_frame - ini <= PAD_MAX
        assert PAD_MIN <= fin - e.end_frame <= PAD_MAX


def test_el_relleno_varia_entre_intentos(doc: AnnotationDoc) -> None:
    """
    Con relleno fijo, restarlo recuperaria las fronteras originales y el error de fronteras
    mediria cero por construccion.
    """
    r = sortear(doc, fraction=1.0, seed=11)
    previos = {
        doc.event_by_id(eid).start_frame - ini for eid, (ini, _) in r.sample.windows.items()
    }
    assert len(previos) > 1


def test_la_ventana_se_recorta_contra_el_video(doc: AnnotationDoc) -> None:
    doc.events = [ev("ev_0001", 0)]
    doc.video.total_frames = 25
    r = sortear(doc, fraction=1.0, seed=1)
    ini, fin = r.sample.windows["ev_0001"]
    assert ini == 0
    assert fin == 24


# -- persistencia ----------------------------------------------------------


def test_round_trip(doc: AnnotationDoc, tmp_path: Path) -> None:
    r = sortear(doc, fraction=0.25, seed=9, annotator="lucas")
    r.trials = [
        ReannoTrial(
            event_id=r.sample.event_ids[0], annotator="lucas", annotated_at=T0,
            active_ms=4200, replays=2,
            labels=TrialLabels(
                start_frame=100, end_frame=120, side=Side.RIGHT,
                punch_type=PunchType.HOOK, target=Target.BODY,
                completeness=Completeness.FULL,
            ),
        )
    ]
    p = tmp_path / "x.reanno.json"
    guardar(r, p)
    assert cargar(p) == r


def test_guardar_no_deja_temporales(doc: AnnotationDoc, tmp_path: Path) -> None:
    guardar(sortear(doc, fraction=0.1, seed=1), tmp_path / "x.reanno.json")
    assert list(tmp_path.glob("*.tmp")) == []


def test_cargar_rechaza_otro_tipo_de_archivo(doc: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text('{"kind": "boxtwin.annot"}', encoding="utf-8")
    with pytest.raises(ValueError, match="no es un archivo de reanotacion"):
        cargar(p)


def test_pendientes(doc: AnnotationDoc) -> None:
    r = sortear(doc, fraction=0.25, seed=9)
    assert r.pendientes() == r.sample.event_ids
    r.trials = [
        ReannoTrial(
            event_id=r.sample.event_ids[0], annotator="l", annotated_at=T0,
            labels=TrialLabels(
                start_frame=1, end_frame=2, side=Side.LEFT, punch_type=PunchType.HOOK,
                target=Target.HEAD, completeness=Completeness.FULL,
            ),
        )
    ]
    assert r.sample.event_ids[0] not in r.pendientes()


# -- congelada -------------------------------------------------------------


def test_no_resortea_si_ya_existe(doc: AnnotationDoc, tmp_path: Path) -> None:
    """
    Resortear despues de ver resultados parciales convierte el numero en lo que uno quiera
    que sea.
    """
    p = tmp_path / "x.reanno.json"
    primera, nueva = cargar_o_sortear(p, doc, fraction=0.25, seed=1, annotator="l")
    assert nueva
    guardar(primera, p)

    segunda, nueva2 = cargar_o_sortear(p, doc, fraction=0.9, seed=999, annotator="l")
    assert not nueva2
    assert segunda.sample.event_ids == primera.sample.event_ids
    assert segunda.sample.seed == 1


def test_force_resortea(doc: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "x.reanno.json"
    primera, _ = cargar_o_sortear(p, doc, fraction=0.25, seed=1, annotator="l")
    guardar(primera, p)
    segunda, nueva = cargar_o_sortear(
        p, doc, fraction=0.25, seed=77, annotator="l", force=True
    )
    assert nueva
    assert segunda.sample.seed == 77


def test_rechaza_una_muestra_de_otra_version_de_la_anotacion(
    doc: AnnotationDoc, tmp_path: Path
) -> None:
    """Comparar contra un estado distinto del que se sorteo invalidaria el pre-registro."""
    p = tmp_path / "x.reanno.json"
    primera, _ = cargar_o_sortear(p, doc, fraction=0.25, seed=1, annotator="l")
    guardar(primera, p)

    doc.events[0].target = Target.BODY  # la anotacion cambio
    with pytest.raises(SampleFrozenError, match="otra version"):
        cargar_o_sortear(p, doc, fraction=0.25, seed=1, annotator="l")
