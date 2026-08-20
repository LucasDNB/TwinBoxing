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
    QDoubleSpinBox,
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
    rellenarInternos = Signal()
    previsualizarChicos = Signal(float)   # cambio el umbral, recalcular el conteo
    ignorarChicos = Signal(float)         # aplicar
    ignorarResto = Signal()               # todo lo que no sea peleador en este plano

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

        # -- publico
        #
        # En metraje de transmision el detector encuentra a todo el mundo. Sobre 20 s de una
        # pelea a 1080p: 9,4 personas por cuadro y 108 tracks, de los cuales dos son los
        # boxeadores. Asignarlos de a uno no es trabajo, es imposible.
        g_pub = QGroupBox("Descartar público")
        v = QVBoxLayout(g_pub)
        self.lbl_chicos = QLabel("sin analizar")
        self.lbl_chicos.setWordWrap(True)
        v.addWidget(self.lbl_chicos)
        fila = QHBoxLayout()
        fila.addWidget(QLabel("alto mínimo:"))
        self.sp_alto = QDoubleSpinBox()
        self.sp_alto.setRange(0.05, 0.90)
        self.sp_alto.setSingleStep(0.05)
        self.sp_alto.setDecimals(2)
        self.sp_alto.setValue(0.30)
        self.sp_alto.setSuffix("  del alto de imagen")
        self.sp_alto.valueChanged.connect(self.previsualizarChicos.emit)
        fila.addWidget(self.sp_alto, 1)
        v.addLayout(fila)
        self.b_chicos = QPushButton("Ignorar los tracks más chicos")
        self.b_chicos.setToolTip(
            "Marca como ignore los tracks que nunca llegan a ese alto.\n"
            "Nunca toca un track que ya es un peleador. Se deshace con Ctrl+Z."
        )
        self.b_chicos.clicked.connect(lambda: self.ignorarChicos.emit(self.sp_alto.value()))
        v.addWidget(self.b_chicos)

        # Lo anterior filtra por tamano en todo el video. Esto es lo que se sabe con certeza
        # despues de asignar a los dos boxeadores de un plano: el resto es publico, arbitro o
        # esquina. Acotado al plano porque en el siguiente los track_id son otros.
        self.b_resto = QPushButton("Ignorar todo lo demás de este plano")
        self.b_resto.setToolTip(
            "Marca como ignore todos los tracks del plano actual que no sean A ni B.\n"
            "Hacelo despues de asignar a los dos boxeadores. Se deshace con Ctrl+Z."
        )
        self.b_resto.clicked.connect(self.ignorarResto.emit)
        v.addWidget(self.b_resto)
        raiz.addWidget(g_pub)

        # -- huecos internos
        #
        # Grupo aparte de las uniones, y no es cosmetica. Las uniones piden una decision de
        # identidad y se confirman de a una; esto rellena cuadros que faltan dentro de un
        # track que ya es un peleador, no decide nada, y por eso puede ir en un solo boton.
        # Mezclarlos en la misma lista invitaria a aplicar las uniones con el mismo criterio.
        g_int = QGroupBox("Huecos internos de los tracks")
        v = QVBoxLayout(g_int)
        self.lbl_internos = QLabel("sin analizar")
        self.lbl_internos.setWordWrap(True)
        v.addWidget(self.lbl_internos)
        self.b_internos = QPushButton("Rellenar huecos internos")
        self.b_internos.setToolTip(
            "Interpola los cuadros que faltan dentro de un mismo track.\n"
            "No cambia ninguna asignacion de rol y se puede deshacer."
        )
        self.b_internos.clicked.connect(self.rellenarInternos.emit)
        v.addWidget(self.b_internos)
        raiz.addWidget(g_int)

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

    def set_chicos(self, a_ignorar: int, total: int) -> None:
        """Cuantos tracks caerian con el umbral actual."""
        if total:
            self.lbl_chicos.setText(
                f"{a_ignorar} de {total} tracks nunca llegan a ese alto"
            )
        else:
            self.lbl_chicos.setText("sin tracks")
        self.b_chicos.setEnabled(bool(a_ignorar))

    def set_internos(self, pendientes: int, cuadros: int) -> None:
        """Cuantos huecos internos quedan por rellenar y cuantos cuadros recuperan."""
        if pendientes:
            self.lbl_internos.setText(
                f"{pendientes} huecos sin rellenar, {cuadros} cuadros de pose que faltan"
            )
        else:
            self.lbl_internos.setText("sin huecos pendientes")
        self.b_internos.setEnabled(bool(pendientes))

    def set_modo_dibujo(self, activo: bool, rol: TrackRole | None = None) -> None:
        self.lbl_modo.setText(
            f"dibujá la caja de {rol.value} sobre el cuadro" if activo and rol else ""
        )

    # -- interaccion -------------------------------------------------------

    def seleccionar(self, track_id: int) -> None:
        """Marca un track en la lista, si esta presente en el cuadro actual."""
        for i in range(self.lista_tracks.count()):
            item = self.lista_tracks.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == track_id:
                self.lista_tracks.setCurrentItem(item)
                return

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
