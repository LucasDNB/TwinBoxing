"""
BoxTwin - Medicion del tiempo de anotacion.

POR QUE EXISTE
  Las metricas de proceso alimentan el analisis metodologico del proyecto, no son un extra.
  La productividad medida sobre la reanotacion previa fue de 2,2 segundos por decision, y
  ese numero solo significa algo si se mide igual siempre.

  Lo que se cuenta es tiempo ACTIVO, no tiempo transcurrido. Si se mide de reloj de pared,
  una sesion de tres horas con dos de almuerzo en el medio reporta tres horas y el numero
  deja de servir para nada: ni para estimar cuanto falta, ni para comparar la dificultad de
  dos videos, ni para escribirlo en el capitulo. Por eso el contador se detiene cuando la
  ventana pierde el foco y cuando no hay actividad por un rato.

  El umbral de inactividad es un compromiso declarado: demasiado corto y se descuenta el
  tiempo en que el anotador mira el cuadro pensando, que es trabajo real; demasiado largo y
  se cuenta el tiempo en que se levanto a buscar un cafe. Quince segundos deja pasar la
  reflexion sobre un golpe dudoso y corta las pausas de verdad.

QUE HACE
  Acumula milisegundos activos, cuenta cuantas veces se reviso un clip antes de confirmar y
  arma las metricas por evento y por sesion.

USO
  reloj = ActiveTimeTracker()
  reloj.actividad(); reloj.foco(False); ms = reloj.tomar()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from boxtwin.core.schema import EventMetrics, SessionMetrics

__all__ = ["ActiveTimeTracker", "EventTimer", "IDLE_SECONDS", "new_session_metrics"]

# Segundos sin actividad tras los cuales se deja de contar. Ver el razonamiento arriba.
IDLE_SECONDS = 15.0


class ActiveTimeTracker:
    """
    Cronometro que solo corre con la ventana en foco y con actividad reciente.

    El reloj se inyecta para que los tests no dependan del tiempo real: medir tiempo con
    sleeps hace suites lentas e intermitentes.
    """

    def __init__(self, *, idle_seconds: float = IDLE_SECONDS, clock=time.monotonic) -> None:
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._acumulado_ms = 0.0
        self._desde: float | None = None
        self._ultima_actividad: float = clock()
        self._en_foco = True

    # -- entradas ----------------------------------------------------------

    def actividad(self) -> None:
        """Se llama con cada tecla o click. Reanuda el conteo si estaba detenido."""
        ahora = self._clock()
        self._cortar(ahora)
        self._ultima_actividad = ahora
        if self._en_foco:
            self._desde = ahora

    def foco(self, activo: bool) -> None:
        ahora = self._clock()
        self._cortar(ahora)
        self._en_foco = activo
        if activo:
            self._ultima_actividad = ahora
            self._desde = ahora
        else:
            self._desde = None

    def _cortar(self, ahora: float) -> None:
        """
        Cierra el tramo abierto acumulando solo lo que cuenta como activo.

        Si desde la ultima actividad paso mas que el umbral, se acumula hasta el umbral y
        no hasta ahora: el tiempo posterior no fue trabajo.
        """
        if self._desde is None:
            return
        limite = min(ahora, self._ultima_actividad + self.idle_seconds)
        if limite > self._desde:
            self._acumulado_ms += (limite - self._desde) * 1000.0
        self._desde = None

    # -- salidas -----------------------------------------------------------

    @property
    def activo(self) -> bool:
        return self._en_foco and (self._clock() - self._ultima_actividad) <= self.idle_seconds

    def leer_ms(self) -> int:
        """Milisegundos acumulados sin detener el conteo."""
        ahora = self._clock()
        parcial = self._acumulado_ms
        if self._desde is not None:
            limite = min(ahora, self._ultima_actividad + self.idle_seconds)
            if limite > self._desde:
                parcial += (limite - self._desde) * 1000.0
        return int(round(parcial))

    def tomar_ms(self) -> int:
        """
        Devuelve lo acumulado y reinicia. Se usa al confirmar un evento.

        El orden importa: primero se cierra el tramo abierto y despues se pone el
        acumulador en cero. Al reves, el tramo se sumaba DESPUES del reinicio y el tiempo
        del evento anterior se volvia a contar en el siguiente, inflando el active_ms de
        todos menos el primero.
        """
        ahora = self._clock()
        self._cortar(ahora)
        ms = int(round(self._acumulado_ms))
        self._acumulado_ms = 0.0
        if self._en_foco:
            self._desde = ahora
        return ms


@dataclass
class EventTimer:
    """
    Estado de medicion de un evento en curso, desde que se marca el inicio.

    `replays` cuenta las vueltas del preview antes de confirmar y es el proxy directo de
    dificultad de la decision: un golpe claro se confirma en la primera, uno dudoso se mira
    cinco veces. Es el dato que distingue un evento facil de uno que quiza haya que revisar.
    """

    created_at: datetime
    reloj: ActiveTimeTracker = field(default_factory=ActiveTimeTracker)
    replays: int = 0

    def replay(self) -> None:
        self.replays += 1

    def confirmar(
        self, *, annotator: str, session_id: str, ahora: datetime
    ) -> EventMetrics:
        return EventMetrics(
            annotator=annotator,
            session_id=session_id,
            created_at=self.created_at,
            confirmed_at=ahora,
            active_ms=self.reloj.tomar_ms(),
            replays=self.replays,
            edits=0,
        )


def new_session_metrics(
    *, session_id: str, annotator: str, ahora: datetime, app_version: str
) -> SessionMetrics:
    return SessionMetrics(
        id=session_id,
        annotator=annotator,
        started_at=ahora,
        app_version=app_version,
    )
