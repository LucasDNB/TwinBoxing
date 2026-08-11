"""
BoxTwin - Lista de eventos anotados.

POR QUE EXISTE
  Anotar sin poder revisar lo anotado obliga a confiar en la memoria, y a los doscientos
  eventos eso no funciona. La lista es donde se detecta que un golpe quedo con el peleador
  cambiado, que hay dos eventos casi identicos, o que una tanda entera salio con la altura
  mal.

  Se edita en la propia celda y no reabriendo el dialogo. Corregir un solo campo de un
  evento viejo no deberia costar volver a mirar el clip entero: lo que se esta arreglando
  es un error de tipeo, no una decision.

  Cada cambio pasa por el historial de deshacer, igual que el alta. Una correccion tambien
  se puede errar, y es el momento en que menos se esta prestando atencion.

QUE HACE
  Tabla filtrable y ordenable, con salto al evento al hacer click y edicion in situ de
  cualquier campo mediante desplegables para los enums y contadores para los cuadros.

USO
  lista.saltarA.connect(...); lista.editar.connect(...); lista.refrescar(doc)
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from boxtwin.core.constants import PUNCH_COLORS
from boxtwin.core.schema import AnnotationDoc, Event
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)

__all__ = ["EventList"]

# columna -> (titulo, campo del evento, enum o None si es entero)
COLUMNAS: list[tuple[str, str, Any]] = [
    ("id", "id", str),
    ("pel", "fighter", FighterId),
    ("ini", "start_frame", int),
    ("pico", "peak_frame", int),
    ("fin", "end_frame", int),
    ("lado", "side", Side),
    ("tipo", "punch_type", PunchType),
    ("alt", "target", Target),
    ("compl", "completeness", Completeness),
    ("result", "landed", Landed),
    ("calidad", "quality", Quality),
    ("notas", "notes", str),
]

NO_EDITABLES = {"id"}


class _Delegate(QStyledItemDelegate):
    """Desplegable para los enums y contador para los cuadros."""

    def createEditor(self, parent, option, index):  # noqa: N802
        _, campo, tipo = COLUMNAS[index.column()]
        if campo in NO_EDITABLES:
            return None
        if tipo is int:
            sb = QSpinBox(parent)
            sb.setRange(-1, 10_000_000)
            sb.setSpecialValueText("—")  # -1 significa sin pico
            return sb
        if tipo is str:
            return QLineEdit(parent)
        cb = QComboBox(parent)
        for v in tipo:
            cb.addItem(v.value, v)
        return cb

    def setEditorData(self, editor, index):  # noqa: N802
        texto = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if isinstance(editor, QSpinBox):
            editor.setValue(int(texto) if texto.strip("—").strip() else -1)
        elif isinstance(editor, QComboBox):
            i = editor.findText(texto)
            editor.setCurrentIndex(max(0, i))
        elif isinstance(editor, QLineEdit):
            editor.setText(texto)

    def setModelData(self, editor, model, index):  # noqa: N802
        if isinstance(editor, QSpinBox):
            valor: Any = editor.value()
            valor = None if valor < 0 else valor
        elif isinstance(editor, QComboBox):
            valor = editor.currentData()
        else:
            valor = editor.text()
        model.setData(index, valor, Qt.ItemDataRole.UserRole + 1)


class EventList(QWidget):
    saltarA = Signal(int)  # frame
    seleccionado = Signal(str)  # event_id
    editar = Signal(str, str, object)  # event_id, campo, valor
    borrar = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc: AnnotationDoc | None = None
        self._cargando = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.f_peleador = QComboBox()
        self.f_peleador.addItem("todos", None)
        for f in FighterId:
            self.f_peleador.addItem(f.value, f)
        self.f_peleador.currentIndexChanged.connect(self._repoblar)

        self.f_texto = QLineEdit()
        self.f_texto.setPlaceholderText("filtrar por tipo, lado, notas...")
        self.f_texto.textChanged.connect(self._repoblar)

        filtros = QHBoxLayout()
        filtros.addWidget(QLabel("peleador"))
        filtros.addWidget(self.f_peleador)
        filtros.addWidget(self.f_texto, 1)

        self.tabla = QTableWidget(0, len(COLUMNAS))
        self.tabla.setHorizontalHeaderLabels([c[0] for c in COLUMNAS])
        self.tabla.setSortingEnabled(True)
        self.tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabla.setItemDelegate(_Delegate(self.tabla))
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.tabla.itemSelectionChanged.connect(self._al_seleccionar)
        self.tabla.itemChanged.connect(self._al_cambiar)

        self.lbl_total = QLabel("sin eventos")

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.addLayout(filtros)
        raiz.addWidget(self.tabla, 1)
        raiz.addWidget(self.lbl_total)

    # -- datos -------------------------------------------------------------

    def refrescar(self, doc: AnnotationDoc) -> None:
        self._doc = doc
        self._repoblar()

    def _visibles(self) -> list[Event]:
        if self._doc is None:
            return []
        eventos = self._doc.events
        pel = self.f_peleador.currentData()
        if pel is not None:
            eventos = [e for e in eventos if e.fighter is pel]
        texto = self.f_texto.text().strip().lower()
        if texto:
            eventos = [e for e in eventos if texto in self._buscable(e)]
        return eventos

    @staticmethod
    def _buscable(e: Event) -> str:
        return " ".join(
            [
                e.id, e.fighter.value, e.side.value, e.punch_type.value, e.target.value,
                e.completeness.value, e.landed.value, e.quality.value, e.notes,
            ]
        ).lower()

    def _repoblar(self) -> None:
        self._cargando = True
        self.tabla.setSortingEnabled(False)
        eventos = self._visibles()
        self.tabla.setRowCount(len(eventos))
        for fila, ev in enumerate(eventos):
            for col, (_, campo, _) in enumerate(COLUMNAS):
                valor = getattr(ev, campo)
                texto = "" if valor is None else (
                    valor.value if hasattr(valor, "value") else str(valor)
                )
                item = QTableWidgetItem(texto)
                item.setData(Qt.ItemDataRole.UserRole, ev.id)
                if campo in NO_EDITABLES:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if campo == "punch_type":
                    item.setForeground(QColor(*PUNCH_COLORS.get(texto, (200, 200, 200))))
                self.tabla.setItem(fila, col, item)
        self.tabla.setSortingEnabled(True)
        self._cargando = False

        total = 0 if self._doc is None else len(self._doc.events)
        plural = "evento" if total == 1 else "eventos"
        self.lbl_total.setText(
            f"{len(eventos)} de {total} {plural}" if len(eventos) != total
            else f"{total} {plural}"
        )

    # -- interaccion -------------------------------------------------------

    def _fila_id(self, fila: int) -> str | None:
        item = self.tabla.item(fila, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _al_seleccionar(self) -> None:
        if self._cargando or self._doc is None:
            return
        filas = self.tabla.selectionModel().selectedRows()
        if not filas:
            return
        eid = self._fila_id(filas[0].row())
        ev = self._doc.event_by_id(eid) if eid else None
        if ev is not None:
            self.seleccionado.emit(ev.id)
            self.saltarA.emit(ev.start_frame)

    def _al_cambiar(self, item: QTableWidgetItem) -> None:
        if self._cargando:
            return
        eid = item.data(Qt.ItemDataRole.UserRole)
        _, campo, _ = COLUMNAS[item.column()]
        if eid is None or campo in NO_EDITABLES:
            return
        nuevo = item.data(Qt.ItemDataRole.UserRole + 1)
        if nuevo is None and item.text() == "":
            nuevo = None
        elif nuevo is None:
            nuevo = item.text()
        self.editar.emit(eid, campo, nuevo)

    def seleccionar(self, event_id: str) -> None:
        """Marca la fila de un evento sin disparar el salto, para sincronizar desde afuera."""
        self._cargando = True
        for fila in range(self.tabla.rowCount()):
            if self._fila_id(fila) == event_id:
                self.tabla.selectRow(fila)
                break
        self._cargando = False

    def id_seleccionado(self) -> str | None:
        filas = self.tabla.selectionModel().selectedRows()
        return self._fila_id(filas[0].row()) if filas else None
