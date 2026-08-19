"""
BoxTwin - Migracion de esquema del archivo de anotacion.

POR QUE EXISTE
  La anotacion de un dataset propio dura meses y el esquema va a cambiar en el medio. Sin
  migracion, cada cambio obliga a elegir entre romper los archivos ya anotados o arrastrar
  campos muertos para siempre. Las dos salidas son malas cuando lo anotado es el activo
  mas caro del proyecto.
  El backup antes de migrar no es cortesia: una migracion con un bug puede corromper horas
  de anotacion, y sin copia del original no hay vuelta atras.

QUE HACE
  Aplica en cadena las migraciones registradas desde la version del archivo hasta la
  actual, escribiendo un backup del original antes de tocar nada. Se niega a abrir un
  archivo de version futura en vez de adivinar que significan sus campos.
  Trabaja sobre el dict crudo y no sobre modelos pydantic: los modelos tienen
  extra="forbid" y rechazarian cualquier archivo viejo antes de que el migrador lo vea.

USO
  raw = json.loads(path.read_text())
  raw, aplicadas = migrate(raw, source_path=path)
  doc = AnnotationDoc.model_validate(raw)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

from boxtwin.core.schema import DOC_KIND, SCHEMA_VERSION

__all__ = [
    "Migration",
    "MIGRATIONS",
    "MigrationError",
    "FutureSchemaError",
    "register",
    "migrate",
    "backup_path_for",
]

# Una migracion recibe el dict de la version n y devuelve el de la version n+1.
Migration = Callable[[dict], dict]

# Registro global. Hoy vacio: la version 1 es la primera y no tiene predecesora.
MIGRATIONS: dict[int, Migration] = {}


class MigrationError(Exception):
    """No hay camino de migracion o una migracion fallo."""


class FutureSchemaError(MigrationError):
    """
    El archivo es de una version mas nueva que la que entiende este binario.

    No se degrada ni se intenta leer igual: significaria descartar en silencio campos cuyo
    significado esta definido en una version que este codigo no conoce.
    """


def register(from_version: int) -> Callable[[Migration], Migration]:
    """Decorador para registrar la migracion de `from_version` a `from_version + 1`."""

    def _deco(fn: Migration) -> Migration:
        if from_version in MIGRATIONS:
            raise MigrationError(f"ya hay una migracion registrada desde la version {from_version}")
        MIGRATIONS[from_version] = fn
        return fn

    return _deco


def backup_path_for(source_path: Path, version: int) -> Path:
    """Ruta del backup del original, antes de migrar."""
    return source_path.with_suffix(source_path.suffix + f".v{version}.bak")


def migrate(
    raw: MutableMapping,
    *,
    source_path: Path | None = None,
    registry: Mapping[int, Migration] | None = None,
) -> tuple[dict, list[int]]:
    """
    Lleva `raw` de su version a SCHEMA_VERSION.

    Devuelve el dict migrado y la lista de versiones de origen aplicadas. Si no hacia
    falta migrar, la lista viene vacia y no se escribe ningun backup.
    `registry` permite inyectar un registro propio; se usa en los tests, que es la unica
    forma honesta de ejercitar el mecanismo cuando existe una sola version.
    """
    regs = MIGRATIONS if registry is None else registry
    data = dict(raw)

    kind = data.get("kind")
    if kind is not None and kind != DOC_KIND:
        raise MigrationError(f"kind inesperado: {kind!r}, se esperaba {DOC_KIND!r}")

    version = data.get("schema_version")
    if not isinstance(version, int):
        raise MigrationError("el archivo no declara schema_version entero, no se puede migrar")

    if version > SCHEMA_VERSION:
        raise FutureSchemaError(
            f"el archivo declara schema_version={version} y este binario entiende hasta "
            f"{SCHEMA_VERSION}. Actualiza boxtwin-annotator; bajar de version no se hace."
        )

    if version == SCHEMA_VERSION:
        return data, []

    # Backup del original antes de transformar nada.
    if source_path is not None:
        bak = backup_path_for(source_path, version)
        if not bak.exists():
            shutil.copy2(source_path, bak)

    aplicadas: list[int] = []
    while version < SCHEMA_VERSION:
        fn = regs.get(version)
        if fn is None:
            raise MigrationError(
                f"no hay migracion registrada de la version {version} a la {version + 1}"
            )
        data = fn(data)
        nueva = data.get("schema_version")
        if nueva != version + 1:
            raise MigrationError(
                f"la migracion {version}->{version + 1} dejo schema_version={nueva!r}; "
                "toda migracion tiene que actualizar el campo"
            )
        aplicadas.append(version)
        version = nueva

    return data, aplicadas


def load_raw(path: Path) -> dict:
    """Lee el JSON crudo. Separado para que los tests puedan armar dicts sin tocar disco."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise MigrationError(f"{path} no contiene un objeto JSON en la raiz")
    return data
