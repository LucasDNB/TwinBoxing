"""
Round-trip de serializacion.

Es el test que garantiza que los diffs de git sirvan. Si guardar dos veces el mismo
contenido produce archivos distintos, el historial deja de decir que se anoto y el
archivo pierde la mitad de su valor.
"""

from __future__ import annotations

import json
from pathlib import Path

from boxtwin.core.annotations import canonicalize, dumps, load, save
from boxtwin.core.schema import AnnotationDoc


def test_roundtrip_byte_a_byte(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p1 = tmp_path / "spar.annot.json"
    save(doc_rich, p1)

    recargado, migradas = load(p1)
    assert migradas == []

    p2 = tmp_path / "spar2.annot.json"
    save(recargado, p2)

    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_roundtrip_preserva_el_modelo(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    recargado, _ = load(p)
    assert recargado == canonicalize(doc_rich)


def test_roundtrip_preserva_fps_exacto(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    """
    30000/1001 no es representable en decimal corto. Si el guardado lo redondeara, los
    timestamps calculados sobre un video largo se irian varios cuadros.
    """
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    recargado, _ = load(p)
    assert recargado.video.fps == doc_rich.video.fps == 30000 / 1001


def test_canonicalize_es_idempotente(doc_rich: AnnotationDoc) -> None:
    una = dumps(canonicalize(doc_rich))
    dos = dumps(canonicalize(doc_rich))
    assert una == dos


def test_canonicalize_ordena_sin_importar_el_orden_de_creacion(doc_rich: AnnotationDoc) -> None:
    esperado = dumps(canonicalize(doc_rich))

    desordenado = doc_rich.model_copy(deep=True)
    desordenado.events = list(reversed(desordenado.events))
    desordenado.identity.assignments = list(reversed(desordenado.identity.assignments))
    desordenado.identity.manual_tracks[0].boxes = list(
        reversed(desordenado.identity.manual_tracks[0].boxes)
    )
    desordenado.process.event_metrics = dict(
        reversed(list(desordenado.process.event_metrics.items()))
    )

    assert dumps(canonicalize(desordenado)) == esperado


def test_formato_del_archivo(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    """Indentacion fija, newline final y sin escapes de unicode: requisitos de diffeo."""
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    texto = p.read_text(encoding="utf-8")

    assert texto.endswith("}\n")
    assert '\n  "schema_version": 1,' in texto or texto.startswith('{\n  "schema_version": 1,')
    assert "\\u" not in texto


def test_claves_en_orden_canonico_no_alfabetico(doc_rich: AnnotationDoc) -> None:
    """
    start_frame tiene que quedar al lado de end_frame. Alfabetico los separaria con side,
    quality y punch_type en el medio.
    """
    payload = json.loads(dumps(doc_rich))
    claves_evento = list(payload["events"][0])
    assert claves_evento[:5] == ["id", "fighter", "start_frame", "peak_frame", "end_frame"]
    assert claves_evento != sorted(claves_evento)

    raiz = list(payload)
    assert raiz[:6] == ["schema_version", "kind", "generator", "counters", "video", "pose"]


def test_derivados_no_llegan_al_archivo(doc_rich: AnnotationDoc) -> None:
    payload = json.loads(dumps(doc_rich))
    ev = payload["events"][0]
    assert "arm_role" not in ev
    assert "duration_frames" not in ev
