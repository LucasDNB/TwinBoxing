"""IO del archivo de anotacion: atomicidad, contadores y orden canonico."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from boxtwin.core.annotations import ID_PREFIXES, canonicalize, dumps, load, new_id, save, touch
from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import FighterId

from tests.conftest import T0


# -- contadores de ID ------------------------------------------------------


def test_new_id_formato_y_avance(doc_min: AnnotationDoc) -> None:
    assert new_id(doc_min, "event") == "ev_0001"
    assert new_id(doc_min, "event") == "ev_0002"
    assert doc_min.counters.event == 2


def test_new_id_arranca_del_contador_persistido(doc_rich: AnnotationDoc) -> None:
    assert doc_rich.counters.event == 44
    assert new_id(doc_rich, "event") == "ev_0045"


def test_new_id_nunca_reusa_tras_borrar(doc_rich: AnnotationDoc) -> None:
    """
    El caso que obliga a persistir los contadores. Derivar el proximo ID del maximo
    existente haria que borrar el ultimo evento y crear otro devuelva el mismo ID, y una
    referencia externa pasaria a apuntar a otra cosa.
    """
    primero = new_id(doc_rich, "event")
    doc_rich.events = [e for e in doc_rich.events if e.id != "ev_0044"]
    segundo = new_id(doc_rich, "event")
    assert primero != segundo
    assert segundo == "ev_0046"


def test_new_id_sobrevive_al_guardado(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    emitido = new_id(doc_rich, "assignment")
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    recargado, _ = load(p)
    assert new_id(recargado, "assignment") != emitido


@pytest.mark.parametrize("kind, prefijo", sorted(ID_PREFIXES.items()))
def test_new_id_cubre_todos_los_tipos(doc_min: AnnotationDoc, kind: str, prefijo: str) -> None:
    assert new_id(doc_min, kind) == f"{prefijo}_0001"


def test_new_id_rechaza_tipo_desconocido(doc_min: AnnotationDoc) -> None:
    with pytest.raises(KeyError):
        new_id(doc_min, "golpe")


# -- escritura atomica -----------------------------------------------------


def test_save_no_deja_temporales(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    assert p.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_crea_el_directorio(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "annotations" / "spar.annot.json"
    save(doc_rich, p)
    assert p.exists()


def test_save_no_pisa_el_anterior_si_falla_la_serializacion(
    doc_rich: AnnotationDoc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Se serializa entero en memoria antes de tocar el disco. Un fallo ahi no puede costar
    la anotacion previa, que es horas de trabajo.
    """
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    original = p.read_text(encoding="utf-8")

    import boxtwin.core.annotations as mod

    def _explota(_doc):
        raise RuntimeError("fallo simulado al serializar")

    monkeypatch.setattr(mod, "dumps", _explota)
    with pytest.raises(RuntimeError):
        save(doc_rich, p)

    assert p.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_no_deja_temporales_si_falla_la_escritura(
    doc_rich: AnnotationDoc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "spar.annot.json"
    import boxtwin.core.annotations as mod

    def _explota(*a, **kw):
        raise OSError("disco lleno simulado")

    monkeypatch.setattr(mod.os, "replace", _explota)
    with pytest.raises(OSError):
        save(doc_rich, p)
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_reemplaza_el_contenido_previo(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    doc_rich.events = doc_rich.events[:1]
    save(doc_rich, p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert len(payload["events"]) == 1


def test_save_guarda_documento_con_advertencias(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    """
    Guardar no valida. Si una inconsistencia impidiera guardar, el anotador quedaria
    encerrado justo cuando mas necesita persistir.
    """
    from boxtwin.core.validation import validate_document

    doc_rich.events[0].quality = doc_rich.events[0].quality  # no-op explicito
    doc_rich.identity.assignments[0].end_frame_excl = 53412  # crea colision de rol
    assert validate_document(doc_rich) != []

    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    assert p.exists()


# -- orden canonico --------------------------------------------------------


def test_canonicalize_ordena_eventos_por_frame(doc_rich: AnnotationDoc) -> None:
    doc_rich.events = list(reversed(doc_rich.events))
    canonicalize(doc_rich)
    starts = [e.start_frame for e in doc_rich.events]
    assert starts == sorted(starts)


def test_canonicalize_ordena_peleadores(doc_rich: AnnotationDoc) -> None:
    doc_rich.fighters = {
        FighterId.B: doc_rich.fighters[FighterId.B],
        FighterId.A: doc_rich.fighters[FighterId.A],
    }
    canonicalize(doc_rich)
    assert list(doc_rich.fighters) == [FighterId.A, FighterId.B]


def test_canonicalize_ordena_tracks_manuales_por_orden_de_creacion(doc_rich: AnnotationDoc) -> None:
    from boxtwin.core.schema import ManualBox, ManualTrack
    from boxtwin.core.types import TrackRole

    doc_rich.identity.manual_tracks.append(
        ManualTrack(track_id=-3, role=TrackRole.A,
                    boxes=[ManualBox(frame=100, xyxy=[1.0, 1.0, 2.0, 2.0])],
                    annotator="lucas", created_at=T0)
    )
    canonicalize(doc_rich)
    assert [m.track_id for m in doc_rich.identity.manual_tracks] == [-1, -3]


# -- touch -----------------------------------------------------------------


def test_touch_actualiza_updated_at(doc_min: AnnotationDoc) -> None:
    despues = T0 + timedelta(hours=2)
    touch(doc_min, despues)
    assert doc_min.generator.updated_at == despues
    assert doc_min.generator.created_at == T0


def test_updated_at_llega_al_archivo(doc_min: AnnotationDoc, tmp_path: Path) -> None:
    touch(doc_min, T0 + timedelta(hours=2))
    p = tmp_path / "spar.annot.json"
    save(doc_min, p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["generator"]["updated_at"] == "2026-08-11T16:03:11-03:00"


def test_timestamps_con_ancho_fijo(doc_rich: AnnotationDoc) -> None:
    """timespec=seconds mantiene el ancho estable y no ensucia los diffs."""
    payload = json.loads(dumps(doc_rich))
    assert payload["generator"]["created_at"] == "2026-08-11T14:03:11-03:00"
    assert payload["video"]["mtime"] == "2026-03-11T21:10:03-03:00"
