"""
BoxTwin - Ventana principal del anotador.

POR QUE EXISTE
  Junta el reproductor, el overlay, el timeline y la anotacion, y cablea el teclado.

  El panel de definiciones operacionales esta siempre visible y no escondido en un menu de
  ayuda a proposito: la consistencia entre sesiones depende de que start_frame signifique
  lo mismo el lunes y el jueves, y una definicion que hay que ir a buscar deja de
  consultarse a la media hora.

  El indicador permanente muestra el numero de cuadro y no el timestamp como dato
  principal. Todo el sistema indexa por cuadro; mostrar el tiempo como referencia primaria
  invitaria a razonar en segundos, que es donde se cuela el error de fps.

  Se guarda cada 30 segundos y ademas en cada evento confirmado. Un corte no puede costar
  mas de un evento, y con guardado solo por temporizador costaria hasta treinta segundos de
  trabajo, que a 2,2 s por decision son una docena de golpes.

QUE HACE
  Arma la ventana, conecta las acciones del keymap, lleva el evento en curso y aplica todos
  los cambios al documento a traves del historial de deshacer.

USO
  ventana = MainWindow(Session.open(video)); ventana.show()
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from boxtwin.core.annotations import new_id
from boxtwin.core.metrics import ActiveTimeTracker, EventTimer, new_session_metrics
from boxtwin.core.types import FighterId, IssueLevel
from boxtwin.core.undo import AddEvent, DeleteEvent, EditEvent
from boxtwin.core.validation import validate_document
from boxtwin.gui.keymap import Keymap
from boxtwin.gui.player.controller import PlayerController
from boxtwin.gui.state import Session
from boxtwin.gui.widgets.class_counter import ClassCounter
from boxtwin.gui.widgets.classify_dialog import ClassifyDialog
from boxtwin.gui.widgets.event_list import EventList
from boxtwin.gui.widgets.timeline import Timeline
from boxtwin.gui.widgets.video_view import VideoView
from boxtwin.version import __version__

__all__ = ["MainWindow", "AUTOSAVE_MS"]

AUTOSAVE_MS = 30_000

DEFINICIONES = """<b>Fronteras temporales</b><br><br>
<b>start_frame</b> (onset)<br>
Primer cuadro en que el puño inicia el desplazamiento hacia el objetivo, con el codo
empezando a extenderse o el hombro rotando. <i>No</i> el cuadro en que se carga el peso.
<br><br>
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
    ZOOM_HIRES = 1.6

    def __init__(self, session: Session, keymap: Keymap | None = None) -> None:
        super().__init__()
        self.session = session
        self.keymap = keymap or Keymap.load(session.paths.config)
        self.player = PlayerController(session.source, session.fps, self)

        self.fighter = FighterId.A
        self._abierto: int | None = None  # start_frame del evento en curso
        self._timer_evento: EventTimer | None = None
        self._seleccionado: str | None = None
        self._fuente_actual = "proxy" if session.using_proxy else "original"
        self._sucio = False

        self.reloj = ActiveTimeTracker()
        self.session_id = session.begin_session(__version__)

        self.setWindowTitle(f"boxtwin-annotator — {session.paths.video.name}")
        self._build_ui()
        self._build_actions()

        self.player.frameChanged.connect(self._on_frame)
        self.player.speedChanged.connect(lambda _: self._refresh_status())
        self.player.playingChanged.connect(lambda _: self._on_frame(self.player.cursor))
        self.view.zoomChanged.connect(lambda _: self._on_frame(self.player.cursor))
        self.timeline.seekRequested.connect(self.player.seek)
        self.lista.saltarA.connect(self.player.seek)
        self.lista.seleccionado.connect(self._seleccionar_evento)
        self.lista.editar.connect(self._editar_campo)

        self._autosave = QTimer(self)
        self._autosave.timeout.connect(self._guardar_si_hace_falta)
        self._autosave.start(AUTOSAVE_MS)

        self._refrescar_anotacion()
        self._on_frame(0)

    # -- construccion ------------------------------------------------------

    def _build_ui(self) -> None:
        self.view = VideoView(self.session.video_size, self)
        self.timeline = Timeline(self)
        self.timeline.set_total(self.session.total_frames)
        self.timeline.set_seams(self.session.seams)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.timeline, 0)
        self.setCentralWidget(central)

        self.lbl_frame, self.lbl_time = QLabel(), QLabel()
        self.lbl_fps, self.lbl_evento = QLabel(), QLabel()
        self.lbl_pel, self.lbl_fuente = QLabel(), QLabel()
        barra = self.statusBar()
        for w in (self.lbl_frame, self.lbl_time, self.lbl_fps, self.lbl_pel, self.lbl_evento):
            barra.addWidget(w)
        barra.addPermanentWidget(self.lbl_fuente)

        self._build_dock()

    def _build_dock(self) -> None:
        dock = QDockWidget("Anotación", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        pestanas = QTabWidget()

        # -- eventos
        self.lista = EventList()
        pestanas.addTab(self.lista, "Eventos")

        # -- balance de clases
        self.contador = ClassCounter()
        pestanas.addTab(self._envolver(self.contador), "Balance")

        # -- vista y definiciones
        vista = QWidget()
        vlayout = QVBoxLayout(vista)
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
            vlayout.addWidget(cb)
            self.chk[clave] = cb
        defs = QLabel(DEFINICIONES)
        defs.setWordWrap(True)
        defs.setTextFormat(Qt.TextFormat.RichText)
        defs.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.addWidget(defs, 1)
        pestanas.addTab(vista, "Vista")

        # -- validacion
        self.lbl_issues = QLabel("sin observaciones")
        self.lbl_issues.setWordWrap(True)
        self.lbl_issues.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_issues.setAlignment(Qt.AlignmentFlag.AlignTop)
        pestanas.addTab(self._envolver(self.lbl_issues), "Validación")

        dock.setWidget(pestanas)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    @staticmethod
    def _envolver(w: QWidget) -> QWidget:
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.addWidget(w)
        lay.addStretch(1)
        return cont

    def _build_actions(self) -> None:
        def add(accion: str, fn) -> None:
            act = QAction(accion, self)
            act.setShortcut(QKeySequence(self.keymap.sequence(accion)))
            act.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act.triggered.connect(lambda: (self.reloj.actividad(), fn()))
            self.addAction(act)

        p, fps = self.player, self.session.fps
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

        add("fighter.select_a", lambda: self._elegir_peleador(FighterId.A))
        add("fighter.select_b", lambda: self._elegir_peleador(FighterId.B))

        add("event.mark_start", self._marcar_inicio)
        add("event.mark_end", self._marcar_fin)
        add("event.cancel_open", self._cancelar_abierto)
        add("event.goto_start", lambda: self._ir_al_borde(inicio=True))
        add("event.goto_end", lambda: self._ir_al_borde(inicio=False))
        add("event.delete", self._borrar_seleccionado)

        add("edit.undo", self._deshacer)
        add("edit.redo", self._rehacer)
        add("file.save", self._guardar)

    # -- reproductor -------------------------------------------------------

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
        f, total = self.player.cursor, self.session.total_frames
        t = self.player.timestamp()
        self.lbl_frame.setText(f"  cuadro {f} / {total - 1}")
        self.lbl_time.setText(f"| {int(t // 60):02d}:{t % 60:06.3f}")
        estado = "▶" if self.player.playing else "❚❚"
        direccion = "→" if self.player.direction > 0 else "←"
        self.lbl_fps.setText(
            f"| {self.session.fps:.3f} fps | {estado}{direccion} {self.player.speed:g}x"
        )
        self.lbl_pel.setText(f"| anotando {self.fighter.value}")
        if self._abierto is None:
            self.lbl_evento.setText("| sin evento abierto")
            self.lbl_evento.setStyleSheet("")
        else:
            # Cuadros transcurridos desde el inicio del evento en curso: es lo que dice si
            # la duracion se esta yendo del rango razonable mientras se busca el final.
            self.lbl_evento.setText(
                f"| ABIERTO en {self._abierto}  (+{f - self._abierto} cuadros)"
            )
            self.lbl_evento.setStyleSheet("color: #e08a3c; font-weight: 600;")
        sucio = " ·" if self._sucio else ""
        self.lbl_fuente.setText(
            f"zoom {self.view.zoom:.2f}x | {self._fuente_actual}{sucio}  "
        )

    # -- anotacion ---------------------------------------------------------

    def _elegir_peleador(self, f: FighterId) -> None:
        self.fighter = f
        self._refresh_status()

    def _marcar_inicio(self) -> None:
        self.player.pause()
        self._abierto = self.player.cursor
        self._timer_evento = EventTimer(created_at=datetime.now().astimezone())
        self._refresh_status()

    def _cancelar_abierto(self) -> None:
        """
        Descarta el evento en curso.

        Es distinto de deshacer: no toca el documento porque el evento todavia no existe.
        Deshacer con Ctrl+Z opera sobre lo ya confirmado.
        """
        if self._abierto is None:
            return
        self._abierto = None
        self._timer_evento = None
        self.statusBar().showMessage("evento en curso descartado", 2000)
        self._refresh_status()

    def _marcar_fin(self) -> None:
        if self._abierto is None:
            self.statusBar().showMessage(
                f"no hay evento abierto: marcar el inicio con "
                f"{self.keymap.sequence('event.mark_start')}", 3000
            )
            return
        inicio, fin = self._abierto, self.player.cursor
        if fin <= inicio:
            self.statusBar().showMessage(
                "el final tiene que ser posterior al inicio", 3000
            )
            return

        self.player.pause()
        timer = self._timer_evento or EventTimer(created_at=datetime.now().astimezone())
        dlg = ClassifyDialog(
            self.session, self.keymap,
            fighter=self.fighter, start_frame=inicio, end_frame=fin,
            timer=timer, event_id=new_id(self.session.doc, "event"),
            annotator=self.session.annotator, session_id=self.session_id, parent=self,
        )
        acepto = dlg.exec()
        self._abierto = None
        self._timer_evento = None

        if not acepto or dlg.resultado() is None:
            # El id emitido no se reusa: los contadores no retroceden a proposito, para que
            # una referencia externa nunca apunte a otro evento.
            self._refresh_status()
            return

        evento, metricas = dlg.resultado()
        self.session.undo.do(AddEvent(evento, metricas))
        self.session.doc.process.sessions[-1].events_created += 1
        self._seleccionado = evento.id
        self._refrescar_anotacion()
        # Al confirmar, el foco vuelve al reproductor en el final del evento.
        self.player.seek(evento.end_frame)
        self._guardar()

    def _ir_al_borde(self, *, inicio: bool) -> None:
        ev = self.session.doc.event_by_id(self._seleccionado or "")
        if ev is None:
            return
        self.player.seek(ev.start_frame if inicio else ev.end_frame)

    def _seleccionar_evento(self, event_id: str) -> None:
        self._seleccionado = event_id
        self.timeline.set_selected(event_id)

    def _borrar_seleccionado(self) -> None:
        eid = self.lista.id_seleccionado() or self._seleccionado
        ev = self.session.doc.event_by_id(eid or "")
        if ev is None:
            self.statusBar().showMessage("no hay evento seleccionado", 2000)
            return
        self.session.undo.do(DeleteEvent(ev.id))
        self._seleccionado = None
        self._refrescar_anotacion()
        self.statusBar().showMessage(
            f"borrado {ev.id} · {self.keymap.sequence('edit.undo')} para deshacer", 4000
        )

    def _editar_campo(self, event_id: str, campo: str, valor) -> None:
        ev = self.session.doc.event_by_id(event_id)
        if ev is None or getattr(ev, campo) == valor:
            return
        try:
            self.session.undo.do(EditEvent(event_id, campo, valor))
        except Exception as exc:  # noqa: BLE001
            # El esquema rechaza, por ejemplo, un fin anterior al inicio. Se avisa y se
            # repuebla la tabla para que la celda vuelva al valor real.
            QMessageBox.warning(self, "Cambio rechazado", str(exc))
        metricas = self.session.doc.process.event_metrics.get(event_id)
        if metricas is not None:
            metricas.edits += 1
            metricas.last_edited_at = datetime.now().astimezone()
        self._refrescar_anotacion()

    # -- deshacer y guardar ------------------------------------------------

    def _deshacer(self) -> None:
        cmd = self.session.undo.undo()
        if cmd is None:
            self.statusBar().showMessage("nada que deshacer", 2000)
            return
        self._refrescar_anotacion()
        self.statusBar().showMessage(f"deshecho: {cmd.label}", 3000)

    def _rehacer(self) -> None:
        cmd = self.session.undo.redo()
        if cmd is None:
            self.statusBar().showMessage("nada que rehacer", 2000)
            return
        self._refrescar_anotacion()
        self.statusBar().showMessage(f"rehecho: {cmd.label}", 3000)

    def _guardar(self) -> None:
        self.session.doc.process.sessions[-1].active_ms = self.reloj.leer_ms()
        self.session.save()
        self._sucio = False
        self._refresh_status()
        self.statusBar().showMessage(f"guardado en {self.session.paths.annot.name}", 2000)

    def _guardar_si_hace_falta(self) -> None:
        if self._sucio:
            self._guardar()

    # -- refresco ----------------------------------------------------------

    def _refrescar_anotacion(self) -> None:
        doc = self.session.doc
        self.lista.refrescar(doc)
        self.contador.refrescar(doc)
        self.timeline.set_events(doc.events)
        self.timeline.set_unreliable(doc.unreliable_segments)
        self.timeline.set_selected(self._seleccionado)
        self._pintar_issues()
        self._sucio = True
        self._refresh_status()

    def _pintar_issues(self) -> None:
        issues = validate_document(self.session.doc)
        if not issues:
            self.lbl_issues.setText("sin observaciones")
            return
        colores = {
            IssueLevel.ERROR: "#ff8080",
            IssueLevel.WARNING: "#e08a3c",
            IssueLevel.INFO: "#888",
        }
        filas = [
            f"<div style='color:{colores[i.level]}; margin-bottom:6px'>"
            f"<b>{i.code}</b> {i.ref or ''}<br>{i.message}</div>"
            for i in issues[:40]
        ]
        if len(issues) > 40:
            filas.append(f"<i>y {len(issues) - 40} mas</i>")
        self.lbl_issues.setText("".join(filas))

    # -- varios ------------------------------------------------------------

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

    def event(self, e):
        """El cronometro solo corre con la ventana en foco: ver core/metrics.py."""
        tipo = e.type()
        if tipo == e.Type.WindowActivate:
            self.reloj.foco(True)
        elif tipo == e.Type.WindowDeactivate:
            self.reloj.foco(False)
        return super().event(e)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.player.pause()
        self.session.end_session(self.reloj.leer_ms())
        if self._sucio:
            self.session.save()
        self.session.close()
        super().closeEvent(event)
