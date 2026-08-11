"""
BoxTwin - Ventana principal del anotador.

POR QUE EXISTE
  Junta el reproductor, el overlay y el timeline, y cablea el teclado. El panel de
  definiciones operacionales esta siempre visible y no escondido en un menu de ayuda a
  proposito: la consistencia entre sesiones depende de que start_frame signifique lo mismo
  el lunes y el jueves, y una definicion que hay que ir a buscar deja de consultarse a la
  media hora.
  El indicador permanente muestra el numero de cuadro y no el timestamp como dato
  principal. Todo el sistema indexa por cuadro; mostrar el tiempo como referencia primaria
  invitaria a razonar en segundos, que es donde se cuela el error de fps.

QUE HACE
  Arma la ventana, conecta las acciones del keymap y refresca la vista cuando cambia el
  cursor.

USO
  ventana = MainWindow(Session.open(video)); ventana.show()
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from boxtwin.gui.keymap import Keymap
from boxtwin.gui.player.controller import SPEEDS, PlayerController
from boxtwin.gui.state import Session
from boxtwin.gui.widgets.timeline import Timeline
from boxtwin.gui.widgets.video_view import VideoView

__all__ = ["MainWindow"]

DEFINICIONES = """<b>Fronteras temporales</b><br><br>
<b>start_frame</b> (onset)<br>
Primer cuadro en que el puño inicia el desplazamiento hacia el objetivo, con el codo
empezando a extenderse o el hombro rotando.
<i>No</i> el cuadro en que se carga el peso.<br><br>
<b>peak_frame</b><br>
Máxima extensión del brazo o contacto, lo que ocurra primero.<br><br>
<b>end_frame</b> (offset)<br>
Cuadro en que el puño retrocedió aproximadamente la mitad del camino de vuelta a la
guardia.<br><br>
<b>feint</b><br>
El movimiento inicia pero se aborta antes del 60% de la extensión esperada y no hay
retracción de recuperación completa.
"""


class MainWindow(QMainWindow):
    def __init__(self, session: Session, keymap: Keymap | None = None) -> None:
        super().__init__()
        self.session = session
        self.keymap = keymap or Keymap.load(session.paths.config)
        self.player = PlayerController(session.source, session.fps, self)

        self.setWindowTitle(f"boxtwin-annotator — {session.paths.video.name}")
        self._build_ui()
        self._build_actions()

        self._fuente_actual = "proxy" if session.using_proxy else "original"
        self.player.frameChanged.connect(self._on_frame)
        # Cambiar el zoom puede cruzar el umbral de resolucion, asi que hay que rearmar el
        # cuadro y no solo repintar.
        self.view.zoomChanged.connect(lambda _: self._on_frame(self.player.cursor))
        # Al soltar play, volver al proxy; al pausar, subir a original si el zoom lo pide.
        self.player.playingChanged.connect(lambda _: self._on_frame(self.player.cursor))
        self.player.speedChanged.connect(lambda _: self._refresh_status())
        self.player.playingChanged.connect(lambda _: self._refresh_status())
        self.timeline.seekRequested.connect(self.player.seek)

        self._on_frame(0)

    # -- construccion ------------------------------------------------------

    def _build_ui(self) -> None:
        self.view = VideoView(self.session.video_size, self)
        self.timeline = Timeline(self)
        self.timeline.set_total(self.session.total_frames)
        self.timeline.set_events(self.session.doc.events)
        self.timeline.set_unreliable(self.session.doc.unreliable_segments)
        self.timeline.set_seams(self.session.seams)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.timeline, 0)
        self.setCentralWidget(central)

        self.lbl_frame = QLabel()
        self.lbl_time = QLabel()
        self.lbl_fps = QLabel()
        self.lbl_evento = QLabel()
        self.lbl_fuente = QLabel()
        barra = self.statusBar()
        for w in (self.lbl_frame, self.lbl_time, self.lbl_fps, self.lbl_evento):
            barra.addWidget(w)
        barra.addPermanentWidget(self.lbl_fuente)

        self._build_dock()

    def _build_dock(self) -> None:
        dock = QDockWidget("Anotación", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        contenido = QWidget()
        layout = QVBoxLayout(contenido)

        toggles = QWidget()
        fila = QVBoxLayout(toggles)
        fila.setContentsMargins(0, 0, 0, 0)
        self.chk = {}
        for clave, etiqueta, inicial in (
            ("skeleton", "Esqueleto", True),
            ("boxes", "Cajas", True),
            ("ids", "IDs y rol", True),
            ("gloves", "Guantes derivados", True),
        ):
            cb = QCheckBox(etiqueta)
            cb.setChecked(inicial)
            cb.toggled.connect(lambda v, k=clave: self._set_option(k, v))
            fila.addWidget(cb)
            self.chk[clave] = cb
        layout.addWidget(toggles)

        defs = QLabel(DEFINICIONES)
        defs.setWordWrap(True)
        defs.setTextFormat(Qt.TextFormat.RichText)
        defs.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(defs, 1)

        dock.setWidget(contenido)
        dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_actions(self) -> None:
        def add(accion: str, fn) -> QAction:
            act = QAction(accion, self)
            act.setShortcut(QKeySequence(self.keymap.sequence(accion)))
            act.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act.triggered.connect(fn)
            self.addAction(act)
            return act

        p = self.player
        fps = self.session.fps
        add("player.play_pause", lambda: p.toggle(1))
        add("player.play_backward", lambda: p.toggle(-1))
        add("player.step_forward", lambda: p.step(1))
        add("player.step_back", lambda: p.step(-1))
        add("player.step_forward_5", lambda: p.step(5))
        add("player.step_back_5", lambda: p.step(-5))
        add("player.step_forward_1s", lambda: p.step(int(round(fps))))
        add("player.step_back_1s", lambda: p.step(-int(round(fps))))
        add("player.go_start", p.go_start)
        add("player.go_end", p.go_end)
        add("player.goto_frame", self._pedir_frame)
        add("player.speed_up", lambda: p.cycle_speed(1))
        add("player.speed_down", lambda: p.cycle_speed(-1))

        add("view.zoom_in", self.view.zoom_in)
        add("view.zoom_out", self.view.zoom_out)
        add("view.fit", self.view.fit_to_window)
        add("view.toggle_skeleton", lambda: self._flip("skeleton"))
        add("view.toggle_boxes", lambda: self._flip("boxes"))
        add("view.toggle_ids", lambda: self._flip("ids"))
        add("view.toggle_gloves", lambda: self._flip("gloves"))
        add("file.save", self._guardar)

    # -- reacciones --------------------------------------------------------

    # Por encima de este zoom el proxy se ve borroso y conviene pagar la decodificacion
    # del original. Solo en pausa: en reproduccion no habria tiempo.
    ZOOM_HIRES = 1.6

    def _on_frame(self, frame: int) -> None:
        img = self._imagen(frame)
        poses = self.session.resolver.resolve_frame(frame)
        self.view.set_frame(img, poses)
        self.timeline.set_cursor(frame)
        self._refresh_status()

    def _imagen(self, frame: int):
        """
        Elige la fuente del cuadro.

        La vista estira el pixmap al tamano del video original pase lo que pase, asi que
        cambiar de fuente no mueve ni un keypoint: es el beneficio de tener un solo sistema
        de coordenadas.
        """
        if not self.player.playing and self.view.zoom >= self.ZOOM_HIRES:
            hires = self.session.hires()
            if hires is not None:
                img = hires.frame(frame)
                if img is not None:
                    self._fuente_actual = "original"
                    return img
        self._fuente_actual = "proxy" if self.session.using_proxy else "original"
        return self.session.source.frame(frame)

    def _refresh_status(self) -> None:
        f = self.player.cursor
        total = self.session.total_frames
        t = self.player.timestamp()
        self.lbl_frame.setText(f"  cuadro {f} / {total - 1}")
        self.lbl_time.setText(f"| {int(t // 60):02d}:{t % 60:06.3f}")
        estado = "▶" if self.player.playing else "❚❚"
        direccion = "→" if self.player.direction > 0 else "←"
        self.lbl_fps.setText(
            f"| {self.session.fps:.3f} fps nativo | {estado}{direccion} {self.player.speed:g}x"
        )
        self.lbl_evento.setText("| sin evento abierto")
        fuente = getattr(self, "_fuente_actual", "proxy" if self.session.using_proxy else "original")
        self.lbl_fuente.setText(f"zoom {self.view.zoom:.2f}x | fuente: {fuente}  ")

    def _set_option(self, clave: str, valor: bool) -> None:
        setattr(self.view.options, clave, valor)
        self.view.update()

    def _flip(self, clave: str) -> None:
        self.chk[clave].setChecked(not self.chk[clave].isChecked())

    def _pedir_frame(self) -> None:
        self.player.pause()
        n, ok = QInputDialog.getInt(
            self, "Ir al cuadro", "Número de cuadro:",
            self.player.cursor, 0, self.session.total_frames - 1,
        )
        if ok:
            self.player.seek(n)

    def _guardar(self) -> None:
        self.session.save()
        self.statusBar().showMessage(f"guardado en {self.session.paths.annot.name}", 2000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.player.pause()
        self.session.close()
        super().closeEvent(event)
