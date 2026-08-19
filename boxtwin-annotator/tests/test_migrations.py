"""
Mecanismo de migracion.

Con una sola version publicada no hay migracion real que testear, asi que se inyecta una
sintetica 0 -> 1. Es la unica forma honesta de ejercitar la cadena, el backup y los modos
de falla antes de que exista la version 2, que es cuando importan de verdad.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boxtwin.core import migrations
from boxtwin.core.annotations import load, save
from boxtwin.core.migrations import (
    SCHEMA_VERSION,
    FutureSchemaError,
    MigrationError,
    backup_path_for,
    migrate,
)
from boxtwin.core.schema import AnnotationDoc


def _v0_desde(doc: AnnotationDoc) -> dict:
    """
    Documento ficticio de la version 0: sin bloque counters y con la guardia del peleador
    en una clave vieja. Representa el tipo de cambio que va a haber de verdad.
    """
    raw = doc.model_dump(mode="json")
    raw["schema_version"] = 0
    raw.pop("counters")
    for fd in raw["fighters"].values():
        fd["stance"] = fd.pop("guard")
    return raw


def _migracion_0_a_1(raw: dict) -> dict:
    out = dict(raw)
    out["schema_version"] = 1
    out.setdefault("counters", {})
    out["fighters"] = {
        k: {**{kk: vv for kk, vv in v.items() if kk != "stance"}, "guard": v["stance"]}
        for k, v in out["fighters"].items()
    }
    return out


REGISTRO = {0: _migracion_0_a_1}


# -- camino feliz ----------------------------------------------------------


def test_documento_actual_no_migra(doc_rich: AnnotationDoc) -> None:
    raw = doc_rich.model_dump(mode="json")
    salida, aplicadas = migrate(raw, registry=REGISTRO)
    assert aplicadas == []
    assert salida == raw


def test_migracion_sintetica_produce_documento_valido(doc_rich: AnnotationDoc) -> None:
    raw, aplicadas = migrate(_v0_desde(doc_rich), registry=REGISTRO)
    assert aplicadas == [0]
    doc = AnnotationDoc.model_validate(raw)
    assert doc.schema_version == SCHEMA_VERSION
    assert doc.fighters == doc_rich.fighters


def test_migracion_escribe_backup(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    p = tmp_path / "spar.annot.json"
    original = json.dumps(_v0_desde(doc_rich), indent=2)
    p.write_text(original, encoding="utf-8")

    raw = json.loads(p.read_text(encoding="utf-8"))
    migrate(raw, source_path=p, registry=REGISTRO)

    bak = backup_path_for(p, 0)
    assert bak.name == "spar.annot.json.v0.bak"
    assert bak.read_text(encoding="utf-8") == original


def test_backup_no_se_pisa_en_una_segunda_pasada(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    """Si el backup ya existe es el original de verdad; sobrescribirlo lo perderia."""
    p = tmp_path / "spar.annot.json"
    p.write_text(json.dumps(_v0_desde(doc_rich)), encoding="utf-8")
    bak = backup_path_for(p, 0)
    bak.write_text("EL ORIGINAL DE VERDAD", encoding="utf-8")

    migrate(json.loads(p.read_text(encoding="utf-8")), source_path=p, registry=REGISTRO)
    assert bak.read_text(encoding="utf-8") == "EL ORIGINAL DE VERDAD"


def test_sin_source_path_no_escribe_nada(doc_rich: AnnotationDoc, tmp_path: Path) -> None:
    migrate(_v0_desde(doc_rich), registry=REGISTRO)
    assert list(tmp_path.iterdir()) == []


# -- modos de falla --------------------------------------------------------


def test_version_futura_falla(doc_rich: AnnotationDoc) -> None:
    raw = doc_rich.model_dump(mode="json")
    raw["schema_version"] = SCHEMA_VERSION + 5
    with pytest.raises(FutureSchemaError, match="bajar de version"):
        migrate(raw, registry=REGISTRO)


def test_sin_migracion_registrada_falla(doc_rich: AnnotationDoc) -> None:
    with pytest.raises(MigrationError, match="no hay migracion registrada"):
        migrate(_v0_desde(doc_rich), registry={})


def test_migracion_que_no_actualiza_la_version_falla(doc_rich: AnnotationDoc) -> None:
    """Una migracion que se olvida de subir schema_version haria un bucle infinito."""
    with pytest.raises(MigrationError, match="toda migracion tiene que actualizar"):
        migrate(_v0_desde(doc_rich), registry={0: lambda raw: dict(raw)})


def test_sin_schema_version_falla(doc_rich: AnnotationDoc) -> None:
    raw = doc_rich.model_dump(mode="json")
    del raw["schema_version"]
    with pytest.raises(MigrationError, match="no declara schema_version"):
        migrate(raw, registry=REGISTRO)


def test_kind_ajeno_falla(doc_rich: AnnotationDoc) -> None:
    """Evita que un reanno.json o cualquier otro JSON entre por error."""
    raw = doc_rich.model_dump(mode="json")
    raw["kind"] = "boxtwin.reanno"
    with pytest.raises(MigrationError, match="kind inesperado"):
        migrate(raw, registry=REGISTRO)


def test_registro_global_esta_vacio_en_la_version_1() -> None:
    """La version 1 es la primera. Si esto cambia, hay que agregar tests de la cadena real."""
    assert migrations.MIGRATIONS == {}


# -- integracion con load --------------------------------------------------


def test_load_de_archivo_actual_no_reporta_migraciones(
    doc_rich: AnnotationDoc, tmp_path: Path
) -> None:
    p = tmp_path / "spar.annot.json"
    save(doc_rich, p)
    _, aplicadas = load(p)
    assert aplicadas == []


def test_load_rechaza_json_que_no_es_objeto(tmp_path: Path) -> None:
    p = tmp_path / "raro.annot.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(MigrationError, match="objeto JSON"):
        load(p)
