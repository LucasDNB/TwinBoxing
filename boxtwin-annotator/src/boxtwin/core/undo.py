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
from boxtwin.core.types import FighterId, Guard

__all__ = [
    "Command",
    "UndoStack",
    "AddEvent",
    "DeleteEvent",
    "EditEvent",
    "SetGuard",
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
class SetGuard:
    """
    Cambia la guardia base de un peleador.

    POR QUE TIENE UN INTERRUPTOR
      El evento guarda la guardia vigente en su inicio, y `arm_role` se deriva de ESA y no
      de la del peleador. Eso es a proposito: permite que un peleador cambie de guardia a
      mitad del combate sin reescribir lo ya anotado.

      Pero hace que cambiar la guardia base signifique dos cosas distintas segun por que se
      la cambia:

      - SE ANOTO MAL desde el principio. Entonces los eventos ya anotados tambien estan mal
        y hay que reescribirles la instantanea: `resync_events=True`.
      - EL PELEADOR CAMBIO de guardia de verdad. Entonces lo anotado estaba bien y no hay
        que tocarlo; lo que corresponde es un GuardOverride sobre el tramo.

      Conflatearlas es como se pierde trabajo en silencio, asi que el llamador elige. Con
      `resync_events=False` los eventos viejos quedan con su guardia y la validacion los
      marca con EV_GUARD_MISMATCH, que es informacion y no un error.

    El estado anterior se lee al ejecutar y no se pide al llamador, igual que en EditEvent.
    """

    fighter: FighterId
    guard: Guard
    resync_events: bool = True
    label: str = ""
    _antes: Guard | None = field(default=None, repr=False)
    _eventos: list[tuple[str, Guard]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"guardia {self.fighter.value} = {self.guard.value}"

    def do(self, doc: AnnotationDoc) -> None:
        fd = doc.fighters.get(self.fighter)
        if fd is None:
            raise KeyError(f"no existe el peleador {self.fighter}")
        self._antes = fd.guard
        fd.guard = self.guard
        self._eventos = []
        if not self.resync_events:
            return
        for ev in doc.events:
            if ev.fighter is not self.fighter:
                continue
            vigente = doc.guard_at(ev.fighter, ev.start_frame)
            if ev.guard is not vigente:
                self._eventos.append((ev.id, ev.guard))
                ev.guard = vigente

    def undo(self, doc: AnnotationDoc) -> None:
        fd = doc.fighters.get(self.fighter)
        if fd is None:
            raise KeyError(f"no existe el peleador {self.fighter}")
        fd.guard = self._antes
        for eid, antes in self._eventos:
            ev = doc.event_by_id(eid)
            if ev is not None:
                ev.guard = antes
        self._eventos = []


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
