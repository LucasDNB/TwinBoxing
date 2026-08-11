"""
BoxTwin - Panel de gestion de identidad.

POR QUE EXISTE
  Es donde se corrige el problema central del dominio. Un track_id no significa nada
  estable: cambia en cada oclusion, en cada clinch y en cada reanudacion del preproceso. Si
  el rol no se corrige, los keypoints se exportan bajo el peleador equivocado, y eso no se
  ve mirando el overlay porque el esqueleto sigue estando sobre un cuerpo.

  Las uniones de tracks se PROPONEN y se confirman de a una. Aplicarlas solas seria comodo y
  peligroso: en un clinch las cajas de los dos peleadores se superponen casi por completo, y
  ahi es donde la heuristica de IoU se equivoca. Confirmar de a una cuesta unos segundos y
  evita atribuir un tramo entero a la persona equivocada.

  Los tramos no confiables se marcan, no se borran. Que entren o no al dataset es politica
  del export, y esa politica puede cambiar sin volver a mirar el video.

QUE HACE
  Muestra los tracks presentes en el cuadro con su rol, permite asignarlos, corregir un
  intercambio desde el cuadro actual, re-sembrar sobre una caja dibujada, marcar tramos y
  aceptar las uniones propuestas.

USO
  panel.refrescar(frame); panel.asignar.connect(...)
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from boxtwin.core.interpolation import GapCandidate
from boxtwin.core.types import FighterId, TrackRole, UnreliableReason

__all__ = ["IdentityPanel"]


class IdentityPanel(QWidget):
    asignar = Signal(int, object)  # track_id, TrackRole
    intercambiar = Signal()
    resembrar = Signal(object)  # TrackRole; activa el modo de dibujo
    marcarTramo = Signal(object, int, int, object)  # fighter, ini, fin_excl, motivo
    aceptarUnion = Signal(object, object)  # GapCandidate, TrackRole
    buscarUniones = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candidatos: list[GapCandidate] = []
        self._build_ui()

    # -- construccion ------------------------------------------------------

    def _build_ui(self) -> None:
        raiz = QVBoxLayout(self)

        # -- tracks del cuadro
        g_tracks = QGroupBox("Tracks en este cuadro")
        v = QVBoxLayout(g_tracks)
        self.lista_tracks = QListWidget()
        self.lista_tracks.setMaximumHeight(120)
        v.addWidget(self.lista_tracks)
        fila = QHBoxLayout()
        for rol, etiqueta in (
            (TrackRole.A, "→ A"), (TrackRole.B, "→ B"), (TrackRole.IGNORE, "→ ignorar"),
        ):
            b = QPushButton(etiqueta)
            b.clicked.connect(lambda _, r=rol: self._asignar(r))
            fila.addWidget(b)
        v.addLayout(fila)
        raiz.addWidget(g_tracks)

        # -- correcciones
        g_corr = QGroupBox("Correcciones desde el cuadro actual")
        v = QVBoxLayout(g_corr)
        b_swap = QPushButton("Intercambiar A ↔ B desde acá")
        b_swap.clicked.connect(self.intercambiar.emit)
        v.addWidget(b_swap)
        fila = QHBoxLayout()
        fila.addWidget(QLabel("re-sembrar:"))
        for rol, etiqueta in ((TrackRole.A, "A"), (TrackRole.B, "B")):
            b = QPushButton(etiqueta)
            b.setToolTip("Dibujar una caja sobre el peleador en el cuadro actual")
            b.clicked.connect(lambda _, r=rol: self.resembrar.emit(r))
            fila.addWidget(b)
        v.addLayout(fila)
        self.lbl_modo = QLabel("")
        self.lbl_modo.setStyleSheet("color: #e08a3c;")
        v.addWidget(self.lbl_modo)
        raiz.addWidget(g_corr)

        # -- tramo no confiable
        g_tramo = QGroupBox("Marcar tramo no confiable")
        f = QFormLayout(g_tramo)
        self.cb_pel = QComboBox()
        for x in FighterId:
            self.cb_pel.addItem(x.value, x)
        self.sp_ini, self.sp_fin = QSpinBox(), QSpinBox()
        for sp in (self.sp_ini, self.sp_fin):
            sp.setRange(0, 10_000_000)
        self.cb_motivo = QComboBox()
        for r in UnreliableReason:
            self.cb_motivo.addItem(r.value, r)
        botones = QHBoxLayout()
        b_ini = QPushButton("inicio = cuadro actual")
        b_fin = QPushButton("fin = cuadro actual")
        b_ini.clicked.connect(lambda: self.sp_ini.setValue(self._frame))
        b_fin.clicked.connect(lambda: self.sp_fin.setValue(self._frame))
        botones.addWidget(b_ini)
        botones.addWidget(b_fin)
        b_marcar = QPushButton("Marcar tramo")
        b_marcar.clicked.connect(self._marcar)
        f.addRow("peleador", self.cb_pel)
        f.addRow("desde", self.sp_ini)
        f.addRow("hasta (excl.)", self.sp_fin)
        f.addRow("motivo", self.cb_motivo)
        f.addRow(botones)
        f.addRow(b_marcar)
        raiz.addWidget(g_tramo)

        # -- uniones propuestas
        g_union = QGroupBox("Uniones propuestas")
        v = QVBoxLayout(g_union)
        b_buscar = QPushButton("Buscar tracks para unir")
        b_buscar.clicked.connect(self.buscarUniones.emit)
        v.addWidget(b_buscar)
        self.lista_uniones = QListWidget()
        v.addWidget(self.lista_uniones)
        fila = QHBoxLayout()
        for rol, etiqueta in ((TrackRole.A, "unir como A"), (TrackRole.B, "unir como B")):
            b = QPushButton(etiqueta)
            b.clicked.connect(lambda _, r=rol: self._aceptar(r))
            fila.addWidget(b)
        v.addLayout(fila)
        raiz.addWidget(g_union, 1)

        self._frame = 0

    # -- datos -------------------------------------------------------------

    def refrescar(self, frame: int, tracks: list[tuple[int, TrackRole | None, bool]]) -> None:
        """`tracks` son (track_id, rol o None, es_interpolado) del cuadro actual."""
        self._frame = frame
        seleccionado = self.track_seleccionado()
        self.lista_tracks.clear()
        for tid, rol, interpolado in tracks:
            etiqueta = f"#{tid}  {rol.value if rol else 'sin asignar'}"
            if interpolado:
                etiqueta += "  (interpolado)"
            item = QListWidgetItem(etiqueta)
            item.setData(Qt.ItemDataRole.UserRole, tid)
            self.lista_tracks.addItem(item)
            if tid == seleccionado:
                item.setSelected(True)
        if self.sp_fin.value() <= self.sp_ini.value():
            self.sp_fin.setValue(frame + 1)

    def set_candidatos(self, candidatos: list[GapCandidate]) -> None:
        self._candidatos = candidatos
        self.lista_uniones.clear()
        for c in candidatos:
            que = "continuacion" if c.es_continuacion else f"hueco de {c.gap_len}"
            self.lista_uniones.addItem(
                f"#{c.from_track_id} → #{c.to_track_id}  cuadro {c.last_frame}  "
                f"IoU {c.iou:.2f}  {que}"
            )
        if not candidatos:
            self.lista_uniones.addItem("no hay tracks compatibles para unir")

    def set_modo_dibujo(self, activo: bool, rol: TrackRole | None = None) -> None:
        self.lbl_modo.setText(
            f"dibujá la caja de {rol.value} sobre el cuadro" if activo and rol else ""
        )

    # -- interaccion -------------------------------------------------------

    def track_seleccionado(self) -> int | None:
        item = self.lista_tracks.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _asignar(self, rol: TrackRole) -> None:
        tid = self.track_seleccionado()
        if tid is not None:
            self.asignar.emit(tid, rol)

    def _marcar(self) -> None:
        self.marcarTramo.emit(
            self.cb_pel.currentData(),
            self.sp_ini.value(),
            self.sp_fin.value(),
            self.cb_motivo.currentData(),
        )

    def _aceptar(self, rol: TrackRole) -> None:
        fila = self.lista_uniones.currentRow()
        if 0 <= fila < len(self._candidatos):
            self.aceptarUnion.emit(self._candidatos[fila], rol)
