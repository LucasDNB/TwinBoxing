"""
BoxTwin - Lectura y escritura del archivo de anotacion.

POR QUE EXISTE
  El annot.json se guarda cada 30 segundos y en cada evento confirmado, o sea cientos de
  veces por sesion. Dos cosas se vuelven criticas a esa frecuencia.
  Primero, la atomicidad: una escritura interrumpida a mitad deja un JSON truncado y se
  pierde el video entero, no el ultimo evento. Se serializa completo en memoria, se
  escribe a un temporal y recien ahi se reemplaza.
  Segundo, la estabilidad del formato: si el orden de las listas depende de en que orden
  se fueron creando los objetos, cada guardado produce un diff distinto sin que haya
  cambiado nada y el archivo deja de servir para revisar que se anoto.

QUE HACE
  Carga con migracion previa, ordena de forma canonica, serializa con orden de claves
  estable y escribe de forma atomica. Ademas emite los IDs, que salen de contadores
  persistidos y nunca se reusan.

USO
  from boxtwin.core.annotations import load, save, new_id
  doc, migradas = load(Path("annotations/spar.annot.json"))
  ev_id = new_id(doc, "event")     # "ev_0043"
  save(doc, Path("annotations/spar.annot.json"))
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from boxtwin.core import migrations
from boxtwin.core.schema import AnnotationDoc

__all__ = ["load", "save", "dumps", "canonicalize", "new_id", "touch", "ID_PREFIXES"]


# Prefijo de ID por tipo de objeto. La clave es el campo de Counters.
ID_PREFIXES: dict[str, str] = {
    "event": "ev",
    "assignment": "as",
    "unreliable": "un",
    "interpolation": "in",
    "combo": "co",
    "session": "se",
    "op": "op",
}


def new_id(doc: AnnotationDoc, kind: str) -> str:
    """
    Emite el proximo ID de ese tipo y avanza el contador.

    El contador se persiste en el documento en vez de derivarse del maximo existente.
    Derivarlo fallaria al borrar el ultimo objeto: el siguiente tomaria el mismo ID y una
    referencia externa, por ejemplo la de un reanno.json, pasaria a apuntar a otra cosa.
    """
    if kind not in ID_PREFIXES:
        raise KeyError(f"tipo de ID desconocido: {kind!r}")
    n = getattr(doc.counters, kind) + 1
    setattr(doc.counters, kind, n)
    return f"{ID_PREFIXES[kind]}_{n:04d}"


def touch(doc: AnnotationDoc, now: datetime) -> None:
    """Marca la fecha de ultima modificacion. Se llama antes de guardar."""
    doc.generator.updated_at = now


def canonicalize(doc: AnnotationDoc) -> AnnotationDoc:
    """
    Ordena todas las colecciones del documento. Idempotente.

    Sin esto, dos guardados con el mismo contenido pueden producir archivos distintos
    segun el orden en que se crearon los objetos en memoria, y el diff de git deja de
    significar algo.
    """
    doc.events = sorted(doc.events, key=lambda e: (e.start_frame, e.fighter.value, e.id))

    ident = doc.identity
    ident.assignments = sorted(ident.assignments, key=lambda a: (a.track_id, a.start_frame, a.id))
    for mt in ident.manual_tracks:
        mt.boxes = sorted(mt.boxes, key=lambda b: b.frame)
    # Los manuales son negativos: descendente los deja como -1, -2, -3, o sea en orden de
    # creacion, que es como el anotador los piensa.
    ident.manual_tracks = sorted(ident.manual_tracks, key=lambda m: -m.track_id)
    ident.interpolations = sorted(ident.interpolations, key=lambda i: (i.gap_start_frame, i.id))

    doc.unreliable_segments = sorted(
        doc.unreliable_segments, key=lambda s: (s.fighter.value, s.start_frame, s.id)
    )
    doc.combo_overrides = sorted(doc.combo_overrides, key=lambda c: c.id)

    doc.fighters = {k: doc.fighters[k] for k in sorted(doc.fighters, key=lambda f: f.value)}
    for fd in doc.fighters.values():
        fd.guard_overrides = sorted(fd.guard_overrides, key=lambda o: o.start_frame)

    proc = doc.process
    proc.annotators = sorted(proc.annotators, key=lambda a: a.id)
    proc.sessions = sorted(proc.sessions, key=lambda s: s.id)
    proc.event_metrics = {k: proc.event_metrics[k] for k in sorted(proc.event_metrics)}

    return doc


def dumps(doc: AnnotationDoc) -> str:
    """
    Serializa a texto con el formato estable del proyecto.

    sort_keys queda en False a proposito: el orden es el de declaracion de los modelos,
    que agrupa start_frame con end_frame. Alfabetico los separaria con side y quality en
    el medio y el archivo dejaria de leerse solo.
    """
    payload = doc.model_dump(mode="json")
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def save(doc: AnnotationDoc, path: Path, *, canonical: bool = True) -> None:
    """
    Escribe el documento de forma atomica.

    No valida a nivel documento. Guardar un archivo con advertencias tiene que funcionar
    siempre: si no, una inconsistencia a mitad de sesion dejaria al anotador sin poder
    guardar, que es peor que la inconsistencia.
    """
    if canonical:
        canonicalize(doc)

    # Se serializa entero antes de tocar el disco. Si esto falla, no se creo ningun
    # archivo y el annot.json anterior queda intacto.
    text = dumps(doc)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    # fsync del directorio para que el rename sobreviva a un corte de energia.
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load(path: Path) -> tuple[AnnotationDoc, list[int]]:
    """
    Carga el documento, migrando si hace falta.

    Devuelve el documento y las versiones de origen migradas. El orden importa: primero se
    migra el dict crudo y recien despues se parsea, porque los modelos tienen
    extra="forbid" y rechazarian cualquier archivo viejo antes de que el migrador lo vea.
    """
    path = Path(path)
    raw = migrations.load_raw(path)
    raw, aplicadas = migrations.migrate(raw, source_path=path)
    doc = AnnotationDoc.model_validate(raw)
    return doc, aplicadas
