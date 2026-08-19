"""
BoxTwin - Vista del cuadro con zoom y paneo.

POR QUE EXISTE
  Hay un solo sistema de coordenadas y son pixeles del video ORIGINAL. La imagen que se
  muestra viene del proxy, que esta a otra escala, y los keypoints vienen del cache, que
  esta en resolucion original. Si cada capa aplicara su propia conversion, el esqueleto se
  correria del cuerpo apenas se toque el zoom, y peor: se correria poco, lo suficiente para
  no notarlo y anotar mal.
  Por eso la imagen se escala al tamano original al dibujarla y todo lo demas se pinta en
  esas coordenadas. Una sola transformacion, la del zoom y el paneo, aplicada al final.

QUE HACE
  Dibuja el cuadro y delega el overlay, con rueda para zoom sobre el puntero y arrastre
  para paneo. Expone la conversion entre pantalla y video para que el bloque 5 pueda
  dibujar cajas a mano.

USO
  vista.set_frame(img, poses)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from boxtwin.core.identity import ResolvedPose
from boxtwin.gui.widgets.overlay import OverlayOptions, paint_poses

__all__ = ["VideoView"]

ZOOM_MIN = 0.1
ZOOM_MAX = 12.0
ZOOM_STEP = 1.25


class VideoView(QWidget):
    """Cuadro mas overlay, en coordenadas del video original."""

    zoomChanged = Signal(float)
    clickedAt = Signal(QPointF)  # en coordenadas del video
    boxDrawn = Signal(QRectF)  # caja dibujada a mano, en coordenadas del video

    def __init__(self, video_size: tuple[int, int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_w, self.video_h = video_size
        self.options = OverlayOptions()

        self._pixmap: QPixmap | None = None
        self._poses: list[ResolvedPose] = []
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)  # esquina del video visible, en coords de video
        self._fit = True
        self._arrastrando = False
        self._ultimo_mouse = QPointF()
        self._extra_painter: Callable[[QPainter, float], None] | None = None
        self._dibujando = False
        self._caja_desde: QPointF | None = None
        self._caja_hasta: QPointF | None = None

        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- contenido ---------------------------------------------------------

    def set_frame(self, img: np.ndarray | None, poses: list[ResolvedPose]) -> None:
        """Recibe BGR de opencv. La conversion a RGB se hace aca y no antes."""
        self._poses = poses
        if img is None:
            self._pixmap = None
        else:
            alto, ancho = img.shape[:2]
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            qimg = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_extra_painter(self, fn: Callable[[QPainter, float], None] | None) -> None:
        """Gancho para que otros bloques dibujen encima sin tocar esta clase."""
        self._extra_painter = fn

    # -- dibujo de caja a mano ---------------------------------------------

    def set_draw_mode(self, activo: bool) -> None:
        """
        Modo de dibujo de caja, para re-sembrar un peleador donde el tracker lo perdio.

        La caja se devuelve en coordenadas del VIDEO, no de la pantalla: es lo que se
        compara por IoU contra las cajas del cache, que estan en esa escala. Convertir en
        otro lado obligaria a repetir la transformacion y a mantener dos copias sincronizadas.
        """
        self._dibujando = activo
        self._caja_desde = self._caja_hasta = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if activo else Qt.CursorShape.ArrowCursor
        )
        self.update()

    @property
    def draw_mode(self) -> bool:
        return self._dibujando

    def _caja_actual(self) -> QRectF | None:
        if self._caja_desde is None or self._caja_hasta is None:
            return None
        return QRectF(self._caja_desde, self._caja_hasta).normalized()

    # -- zoom y paneo ------------------------------------------------------

    @property
    def zoom(self) -> float:
        return self._effective_zoom()

    def fit_to_window(self) -> None:
        self._fit = True
        self._pan = QPointF(0.0, 0.0)
        self.update()
        self.zoomChanged.emit(self.zoom)

    def set_zoom(self, z: float, ancla: QPointF | None = None) -> None:
        """Zoom manteniendo fijo el punto del video que esta bajo el ancla en pantalla."""
        nuevo = max(ZOOM_MIN, min(ZOOM_MAX, z))
        if ancla is None:
            ancla = QPointF(self.width() / 2, self.height() / 2)
        antes = self.widget_to_video(ancla)

        self._fit = False
        self._zoom = nuevo
        despues = self.widget_to_video(ancla)
        self._pan += antes - despues
        self._clamp_pan()
        self.update()
        self.zoomChanged.emit(nuevo)

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom / ZOOM_STEP)

    def _fit_zoom(self) -> float:
        if not self.video_w or not self.video_h:
            return 1.0
        return min(self.width() / self.video_w, self.height() / self.video_h)

    def _effective_zoom(self) -> float:
        return self._fit_zoom() if self._fit else self._zoom

    def _clamp_pan(self) -> None:
        """El paneo se limita al video. Si sobra lugar, queda en cero y centra el offset."""
        z = self._effective_zoom()
        max_x = max(0.0, self.video_w - self.width() / z)
        max_y = max(0.0, self.video_h - self.height() / z)
        self._pan.setX(min(max(0.0, self._pan.x()), max_x))
        self._pan.setY(min(max(0.0, self._pan.y()), max_y))

    # -- conversion de coordenadas ----------------------------------------

    def _offset(self) -> QPointF:
        """
        Centrado cuando el video ocupa menos que el widget.

        Va en la transformacion y no en el rectangulo de destino del pixmap para que el
        overlay reciba exactamente el mismo desplazamiento que la imagen. Calcularlo en dos
        lugares es como se corre el esqueleto del cuerpo.
        """
        z = self._effective_zoom()
        return QPointF(
            max(0.0, (self.width() - self.video_w * z) / 2),
            max(0.0, (self.height() - self.video_h * z) / 2),
        )

    def transform(self) -> QTransform:
        z = self._effective_zoom()
        off = self._offset()
        t = QTransform()
        t.translate(off.x(), off.y())
        t.scale(z, z)
        t.translate(-self._pan.x(), -self._pan.y())
        return t

    def video_to_widget(self, p: QPointF) -> QPointF:
        return self.transform().map(p)

    def widget_to_video(self, p: QPointF) -> QPointF:
        inversa, ok = self.transform().inverted()
        return inversa.map(p) if ok else QPointF()

    # -- eventos -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap is None:
            painter.end()
            return

        painter.setTransform(self.transform())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # El pixmap viene del proxy, a menor resolucion. Se estira al tamano del video
        # original para que los keypoints, que estan en esa escala, caigan donde deben.
        painter.drawPixmap(QRectF(0, 0, self.video_w, self.video_h), self._pixmap,
                           QRectF(self._pixmap.rect()))

        paint_poses(painter, self._poses, self.options, zoom=self._effective_zoom())
        caja = self._caja_actual()
        if caja is not None:
            z = self._effective_zoom()
            painter.setPen(QPen(QColor(245, 194, 66), max(0.5, 2.0 / z), Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(caja)
        if self._extra_painter is not None:
            self._extra_painter(painter, self._effective_zoom())
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        pasos = event.angleDelta().y() / 120.0
        if pasos:
            self.set_zoom(self.zoom * (ZOOM_STEP**pasos), event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._dibujando and event.button() == Qt.MouseButton.LeftButton:
            self._caja_desde = self.widget_to_video(event.position())
            self._caja_hasta = self._caja_desde
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._arrastrando = True
            self._ultimo_mouse = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clickedAt.emit(self.widget_to_video(event.position()))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dibujando and self._caja_desde is not None:
            self._caja_hasta = self.widget_to_video(event.position())
            self.update()
            event.accept()
            return
        if self._arrastrando:
            delta = event.position() - self._ultimo_mouse
            self._ultimo_mouse = event.position()
            z = self._effective_zoom()
            self._pan -= QPointF(delta.x() / z, delta.y() / z)
            self._fit = False
            self._clamp_pan()
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dibujando and self._caja_desde is not None:
            caja = self._caja_actual()
            self._caja_desde = self._caja_hasta = None
            # Un click sin arrastre no es una caja: sin este minimo, cualquier click
            # accidental crearia un track manual de un pixel.
            if caja is not None and caja.width() >= 8 and caja.height() >= 8:
                self.boxDrawn.emit(caja)
            self.update()
            event.accept()
            return
        self._arrastrando = False
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_to_window()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_pan()
