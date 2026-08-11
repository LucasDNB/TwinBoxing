"""
BoxTwin - Timeline con dos carriles.

POR QUE EXISTE
  Dos carriles y no uno porque los eventos de los dos peleadores se solapan casi siempre:
  en un intercambio los dos estan pegando. En un solo carril las marcas se apilarian y no
  se podria ver de quien es cada golpe, que es lo primero que uno quiere saber mirando la
  distribucion de un round.
  Las marcas van coloreadas por tipo de golpe para detectar desbalance de clases a ojo
  mientras se anota, sin abrir el reporte.

QUE HACE
  Barra navegable por click y arrastre, con las marcas de eventos anotados, los tramos no
  confiables y las costuras de reanudacion del preproceso, que son los puntos donde hay un
  corte de identidad garantizado.

USO
  tl.set_total(53412); tl.set_events(doc.events); tl.seekRequested.connect(ctl.seek)
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from boxtwin.core.constants import PUNCH_COLORS, ROLE_COLOR_A, ROLE_COLOR_B
from boxtwin.core.schema import Event, UnreliableSegment
from boxtwin.core.types import FighterId

__all__ = ["Timeline"]

ALTO_CARRIL = 14
ALTO_EJE = 10
MARGEN = 2


class Timeline(QWidget):
    seekRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total = 1
        self._cursor = 0
        self._events: list[Event] = []
        self._unreliable: list[UnreliableSegment] = []
        self._seams: list[int] = []
        self._seleccionado: str | None = None
        self.setMinimumHeight(2 * ALTO_CARRIL + ALTO_EJE + 3 * MARGEN)
        self.setMouseTracking(True)

    # -- datos -------------------------------------------------------------

    def set_total(self, total: int) -> None:
        self._total = max(1, total)
        self.update()

    def set_cursor(self, frame: int) -> None:
        self._cursor = frame
        self.update()

    def set_events(self, events: list[Event]) -> None:
        self._events = events
        self.update()

    def set_unreliable(self, segments: list[UnreliableSegment]) -> None:
        self._unreliable = segments
        self.update()

    def set_seams(self, frames: list[int]) -> None:
        self._seams = frames
        self.update()

    def set_selected(self, event_id: str | None) -> None:
        self._seleccionado = event_id
        self.update()

    # -- geometria ---------------------------------------------------------

    def _x(self, frame: int) -> float:
        return frame / self._total * max(1, self.width() - 1)

    def _frame(self, x: float) -> int:
        return int(round(x / max(1, self.width() - 1) * self._total))

    def _lane_rect(self, fighter: FighterId) -> QRectF:
        fila = 0 if fighter is FighterId.A else 1
        y = MARGEN + fila * (ALTO_CARRIL + MARGEN)
        return QRectF(0, y, self.width(), ALTO_CARRIL)

    # -- pintado -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 28, 30))

        for fighter, color in ((FighterId.A, ROLE_COLOR_A), (FighterId.B, ROLE_COLOR_B)):
            carril = self._lane_rect(fighter)
            p.fillRect(carril, QColor(44, 44, 48))
            # Franja de color a la izquierda para saber de quien es el carril sin leyenda.
            p.fillRect(QRectF(carril.left(), carril.top(), 3, carril.height()), QColor(*color))

        self._pintar_no_confiables(p)
        self._pintar_eventos(p)
        self._pintar_costuras(p)
        self._pintar_cursor(p)
        p.end()

    def _pintar_no_confiables(self, p: QPainter) -> None:
        for seg in self._unreliable:
            carril = self._lane_rect(seg.fighter)
            x1 = self._x(seg.start_frame)
            x2 = self._x(seg.end_frame_excl)
            p.fillRect(
                QRectF(x1, carril.top(), max(1.0, x2 - x1), carril.height()),
                QColor(120, 120, 120, 90),
            )

    def _pintar_eventos(self, p: QPainter) -> None:
        for ev in self._events:
            carril = self._lane_rect(ev.fighter)
            x1 = self._x(ev.start_frame)
            x2 = self._x(ev.end_frame + 1)
            color = QColor(*PUNCH_COLORS.get(ev.punch_type.value, (200, 200, 200)))
            rect = QRectF(x1, carril.top() + 2, max(2.0, x2 - x1), carril.height() - 4)
            p.fillRect(rect, color)
            if ev.id == self._seleccionado:
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect.adjusted(-1, -1, 1, 1))

    def _pintar_costuras(self, p: QPainter) -> None:
        """
        Las costuras del preproceso son cortes de identidad garantizados. Se marcan para
        que el anotador sepa que ahi hay que re-sembrar, en vez de descubrirlo.
        """
        p.setPen(QPen(QColor(245, 194, 66), 1, Qt.PenStyle.DashLine))
        for frame in self._seams:
            x = self._x(frame)
            p.drawLine(int(x), 0, int(x), self.height())

    def _pintar_cursor(self, p: QPainter) -> None:
        x = self._x(self._cursor)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(int(x), 0, int(x), self.height())

    # -- interaccion -------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.seekRequested.emit(self._frame(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.seekRequested.emit(self._frame(event.position().x()))
        event.accept()
