"""
BoxTwin - Panel de reanotacion ciega.

POR QUE EXISTE
  Es el instrumento que produce el numero de acuerdo, y ese numero es lo que hace defendible
  un dataset propio. Todo el panel esta armado para que sea dificil hacer trampa sin querer.

  Mientras el modo esta activo, la interfaz oculta las marcas del timeline y la lista de
  eventos. No es paranoia: la marca del timeline dice exactamente donde empieza y termina el
  golpe, y con eso a la vista el error de fronteras mide cero por construccion.

  La ventana de cada intento viene con relleno aleatorio desde el archivo de muestra, asi
  que los bordes tampoco delatan las fronteras.

  Revelar la etiqueta previa es posible y queda registrado. Prohibirlo no serviria, porque
  el archivo esta ahi para abrirlo; registrarlo hace que el intento no cuente como ciego y
  el reporte lo diga.

QUE HACE
  Muestra el avance, lleva al siguiente intento pendiente y permite revelar la etiqueta
  original dejando constancia.

USO
  panel.siguiente.connect(...); panel.refrescar(doc_re, actual)
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ReannoPanel"]


class ReannoPanel(QWidget):
    empezar = Signal()
    siguiente = Signal()
    revelar = Signal()
    salir = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        raiz = QVBoxLayout(self)

        self.lbl_estado = QLabel("sin muestra sorteada")
        self.lbl_estado.setWordWrap(True)
        raiz.addWidget(self.lbl_estado)

        self.barra = QProgressBar()
        self.barra.setFormat("%v de %m")
        raiz.addWidget(self.barra)

        self.b_empezar = QPushButton("Empezar reanotación ciega")
        self.b_empezar.clicked.connect(self.empezar.emit)
        raiz.addWidget(self.b_empezar)

        self.b_siguiente = QPushButton("Ir al siguiente intento")
        self.b_siguiente.clicked.connect(self.siguiente.emit)
        self.b_siguiente.setEnabled(False)
        raiz.addWidget(self.b_siguiente)

        g = QGroupBox("Intento en curso")
        v = QVBoxLayout(g)
        self.lbl_intento = QLabel("—")
        self.lbl_intento.setWordWrap(True)
        v.addWidget(self.lbl_intento)
        self.b_revelar = QPushButton("Revelar la etiqueta original")
        self.b_revelar.setToolTip(
            "El intento queda marcado y no cuenta como ciego en el reporte"
        )
        self.b_revelar.clicked.connect(self.revelar.emit)
        self.b_revelar.setEnabled(False)
        v.addWidget(self.b_revelar)
        self.lbl_revelado = QLabel("")
        self.lbl_revelado.setWordWrap(True)
        self.lbl_revelado.setStyleSheet("color: #e08a3c;")
        v.addWidget(self.lbl_revelado)
        raiz.addWidget(g)

        self.b_salir = QPushButton("Salir del modo ciego")
        self.b_salir.clicked.connect(self.salir.emit)
        self.b_salir.setEnabled(False)
        raiz.addWidget(self.b_salir)

        self.ayuda = QLabel(
            "Mientras el modo está activo se ocultan las marcas del timeline y la lista de "
            "eventos: con la marca a la vista, el error de fronteras mediría cero por "
            "construcción.<br><br>"
            "En cada intento hay que marcar el inicio y el final con <b>[</b> y <b>]</b> "
            "dentro de la ventana, igual que al anotar. La ventana lleva relleno aleatorio, "
            "así que sus bordes no dicen dónde está el golpe."
        )
        self.ayuda.setWordWrap(True)
        self.ayuda.setTextFormat(Qt.TextFormat.RichText)
        self.ayuda.setStyleSheet("color: #888;")
        raiz.addWidget(self.ayuda)
        raiz.addStretch(1)

    # -- estado ------------------------------------------------------------

    def refrescar(self, doc_re, activo: bool, actual: str | None, revelado: bool) -> None:
        if doc_re is None:
            self.lbl_estado.setText(
                "sin muestra sorteada.<br>Sortearla con "
                "<code>boxtwin-annotator reanno &lt;video&gt;</code>"
            )
            self.lbl_estado.setTextFormat(Qt.TextFormat.RichText)
            self.barra.setMaximum(1)
            self.barra.setValue(0)
            self.b_empezar.setEnabled(False)
            return

        total = doc_re.sample.n
        hechos = len(doc_re.trials)
        self.barra.setMaximum(max(1, total))
        self.barra.setValue(hechos)
        self.lbl_estado.setText(
            f"muestra de {total} eventos, semilla {doc_re.sample.seed}, "
            f"sorteada por {doc_re.sample.drawn_by}.<br>"
            f"{hechos} reanotados, {len(doc_re.pendientes())} pendientes."
        )
        self.lbl_estado.setTextFormat(Qt.TextFormat.RichText)

        self.b_empezar.setEnabled(not activo and bool(doc_re.pendientes()))
        self.b_siguiente.setEnabled(activo)
        self.b_salir.setEnabled(activo)
        self.b_revelar.setEnabled(activo and actual is not None and not revelado)

        if actual is None:
            self.lbl_intento.setText("—" if not activo else "sin intentos pendientes")
        else:
            ini, fin = doc_re.ventana(actual)
            self.lbl_intento.setText(
                f"ventana <b>{ini}–{fin}</b><br>marcar inicio y final adentro"
            )
            self.lbl_intento.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_revelado.setText(
            "etiqueta revelada: este intento no cuenta como ciego" if revelado else ""
        )
