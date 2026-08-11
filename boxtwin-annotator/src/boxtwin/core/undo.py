"""
BoxTwin - Historial de deshacer y rehacer.

POR QUE EXISTE
  Anotar es marcar cientos de eventos seguidos con el teclado, y equivocarse es parte del
  proceso: una tecla de mas, un peleador mal elegido, un evento marcado dos veces. Sin
  deshacer, cada error obliga a buscar el evento en la lista y corregirlo a mano, y ese
  costo hace que el anotador termine dejando errores pasar.

  Se implementa con comandos y no guardando copias del documento entero. Un annot.json con
  mil eventos y sus metricas pesa varios megas; cincuenta copias en memoria serian cientos
  de megas y cada operacion costaria una serializacion. Un comando guarda solo lo que hace
  falta para revertirse.

  Los comandos operan sobre el documento y no sobre la interfaz, asi que el bloque 5 puede
  agregar operaciones de identidad sin tocar nada de aca: alcanza con que cumplan el
  protocolo.

QUE HACE
  Define el protocolo Command, los comandos de evento y una pila con tope configurable que
  descarta lo mas viejo en vez de crecer sin limite.

USO
  pila = UndoStack(doc)
  pila.do(AddEvent(evento, metricas))
  pila.undo()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from boxtwin.core.schema import AnnotationDoc, Event, EventMetrics

__all__ = [
    "Command",
    "UndoStack",
    "AddEvent",
    "DeleteEvent",
    "EditEvent",
    "CompositeCommand",
    "DEFAULT_DEPTH",
]

# El enunciado pide al menos 50 pasos. Se toma el doble: un comando pesa unos cientos de
# bytes y el margen no cuesta nada.
DEFAULT_DEPTH = 100


@runtime_checkable
class Command(Protocol):
    """Una operacion reversible sobre el documento."""

    label: str

    def do(self, doc: AnnotationDoc) -> None: ...

    def undo(self, doc: AnnotationDoc) -> None: ...


# ---------------------------------------------------------------------------
# Comandos de evento
# ---------------------------------------------------------------------------


@dataclass
class AddEvent:
    """
    Alta de un evento con sus metricas de proceso.

    Las metricas viajan con el evento y no aparte porque deshacer un alta tiene que dejar
    el documento exactamente como estaba: si las metricas quedaran, el reporte de tiempo de
    anotacion contaria un evento que ya no existe.
    """

    event: Event
    metrics: EventMetrics | None = None
    label: str = "agregar evento"

    def do(self, doc: AnnotationDoc) -> None:
        doc.events = [*doc.events, self.event]
        if self.metrics is not None:
            doc.process.event_metrics = {
                **doc.process.event_metrics,
                self.event.id: self.metrics,
            }

    def undo(self, doc: AnnotationDoc) -> None:
        doc.events = [e for e in doc.events if e.id != self.event.id]
        if self.event.id in doc.process.event_metrics:
            restante = dict(doc.process.event_metrics)
            restante.pop(self.event.id, None)
            doc.process.event_metrics = restante


@dataclass
class DeleteEvent:
    """
    Baja de un evento.

    Guarda el evento y sus metricas al ejecutarse, no al construirse: asi el comando se
    puede armar con solo el id y siempre revierte lo que de verdad se borro.
    """

    event_id: str
    label: str = "borrar evento"
    _event: Event | None = field(default=None, repr=False)
    _metrics: EventMetrics | None = field(default=None, repr=False)

    def do(self, doc: AnnotationDoc) -> None:
        self._event = doc.event_by_id(self.event_id)
        if self._event is None:
            raise KeyError(f"no existe el evento {self.event_id}")
        self._metrics = doc.process.event_metrics.get(self.event_id)
        doc.events = [e for e in doc.events if e.id != self.event_id]
        if self._metrics is not None:
            restante = dict(doc.process.event_metrics)
            restante.pop(self.event_id, None)
            doc.process.event_metrics = restante

    def undo(self, doc: AnnotationDoc) -> None:
        if self._event is None:
            raise RuntimeError("no se puede deshacer un borrado que nunca se ejecuto")
        doc.events = [*doc.events, self._event]
        if self._metrics is not None:
            doc.process.event_metrics = {
                **doc.process.event_metrics,
                self.event_id: self._metrics,
            }


@dataclass
class EditEvent:
    """
    Cambio de un campo de un evento.

    El valor anterior se lee al ejecutar y no se pide al llamador, para que no pueda pasar
    un `antes` que no coincida con la realidad y deje el documento en un estado que nunca
    existio al deshacer.
    """

    event_id: str
    campo: str
    despues: Any
    label: str = ""
    _antes: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"editar {self.campo}"

    def do(self, doc: AnnotationDoc) -> None:
        ev = doc.event_by_id(self.event_id)
        if ev is None:
            raise KeyError(f"no existe el evento {self.event_id}")
        if self.campo not in type(ev).model_fields:
            raise KeyError(f"el evento no tiene el campo {self.campo!r}")
        self._antes = getattr(ev, self.campo)
        setattr(ev, self.campo, self.despues)

    def undo(self, doc: AnnotationDoc) -> None:
        ev = doc.event_by_id(self.event_id)
        if ev is None:
            raise KeyError(f"no existe el evento {self.event_id}")
        setattr(ev, self.campo, self._antes)


@dataclass
class CompositeCommand:
    """
    Varios comandos como uno solo.

    Un intercambio de identidad son dos assignments truncados y dos nuevos; si cada uno
    fuera un paso, deshacer dejaria el documento a mitad de camino, con un peleador
    corregido y el otro no.
    """

    comandos: list[Command]
    label: str = "operacion compuesta"

    def do(self, doc: AnnotationDoc) -> None:
        hechos: list[Command] = []
        try:
            for c in self.comandos:
                c.do(doc)
                hechos.append(c)
        except BaseException:
            # Si uno falla, se revierte lo ya aplicado: un compuesto es todo o nada.
            for c in reversed(hechos):
                c.undo(doc)
            raise

    def undo(self, doc: AnnotationDoc) -> None:
        for c in reversed(self.comandos):
            c.undo(doc)


# ---------------------------------------------------------------------------
# Pila
# ---------------------------------------------------------------------------


class UndoStack:
    """Pila de comandos con tope. Al llenarse descarta lo mas viejo."""

    def __init__(self, doc: AnnotationDoc, *, depth: int = DEFAULT_DEPTH) -> None:
        if depth < 1:
            raise ValueError("la profundidad tiene que ser al menos 1")
        self.doc = doc
        self.depth = depth
        self._hechos: list[Command] = []
        self._deshechos: list[Command] = []

    # -- consultas ---------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._hechos)

    @property
    def can_redo(self) -> bool:
        return bool(self._deshechos)

    @property
    def undo_label(self) -> str | None:
        return self._hechos[-1].label if self._hechos else None

    @property
    def redo_label(self) -> str | None:
        return self._deshechos[-1].label if self._deshechos else None

    def __len__(self) -> int:
        return len(self._hechos)

    # -- operaciones -------------------------------------------------------

    def do(self, cmd: Command) -> Command:
        """
        Ejecuta y apila.

        Una operacion nueva invalida el rehacer: la rama que se habia deshecho ya no
        aplica sobre este documento.
        """
        cmd.do(self.doc)
        self._hechos.append(cmd)
        self._deshechos.clear()
        if len(self._hechos) > self.depth:
            del self._hechos[0 : len(self._hechos) - self.depth]
        return cmd

    def undo(self) -> Command | None:
        if not self._hechos:
            return None
        cmd = self._hechos.pop()
        cmd.undo(self.doc)
        self._deshechos.append(cmd)
        return cmd

    def redo(self) -> Command | None:
        if not self._deshechos:
            return None
        cmd = self._deshechos.pop()
        cmd.do(self.doc)
        self._hechos.append(cmd)
        return cmd

    def clear(self) -> None:
        self._hechos.clear()
        self._deshechos.clear()
