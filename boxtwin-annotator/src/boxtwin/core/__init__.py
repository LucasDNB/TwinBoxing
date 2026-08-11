"""
Nucleo de boxtwin-annotator: modelo de datos, validaciones e IO.

Este paquete no importa torch, ultralytics ni opencv, ni directa ni indirectamente. La
regla la verifica tests/test_core_dependencies.py y existe para que todo el pipeline de
export se pueda correr en una maquina sin GPU y sin el stack de video, por ejemplo dentro
de un notebook o en el entorno de entrenamiento.
"""

from boxtwin.core.annotations import canonicalize, dumps, load, new_id, save, touch
from boxtwin.core.migrations import FutureSchemaError, MigrationError, migrate
from boxtwin.core.schema import SCHEMA_VERSION, AnnotationDoc, Event, new_document
from boxtwin.core.types import (
    ArmRole,
    Completeness,
    FighterId,
    Guard,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
    TrackRole,
    arm_role,
)
from boxtwin.core.validation import Issue, has_errors, validate_document

__all__ = [
    "SCHEMA_VERSION",
    "AnnotationDoc",
    "Event",
    "new_document",
    "load",
    "save",
    "dumps",
    "canonicalize",
    "new_id",
    "touch",
    "migrate",
    "MigrationError",
    "FutureSchemaError",
    "validate_document",
    "has_errors",
    "Issue",
    "arm_role",
    "ArmRole",
    "Completeness",
    "FighterId",
    "Guard",
    "Landed",
    "PunchType",
    "Quality",
    "Side",
    "Target",
    "TrackRole",
]
