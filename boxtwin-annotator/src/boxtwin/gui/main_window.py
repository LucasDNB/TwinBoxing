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
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from boxtwin.core.annotations import new_id
from boxtwin.core.metrics import ActiveTimeTracker, EventTimer, new_session_metrics
from boxtwin.core.identity_ops import (
    AcceptJoin,
    AssignRole,
    FillInternalGaps,
    IgnoreSmallTracks,
    MarkUnreliable,
    Reseed,
    SwapFromFrame,
)
from boxtwin.core.identity import altura_maxima_por_track, rol_en_track
from boxtwin.core.interpolation import detectar_huecos, detectar_huecos_internos, iou
from boxtwin.core.types import FighterId, IssueLevel, TrackRole
from boxtwin.core.undo import AddEvent, DeleteEvent, EditEvent
from boxtwin.core.reanno import ReannoTrial, TrialLabels, cargar as cargar_reanno
from boxtwin.core.reanno import guardar as guardar_reanno
from boxtwin.core.validation import validate_document
from boxtwin.gui.keymap import Keymap
from boxtwin.gui.player.controller import PlayerController
from boxtwin.gui.state import Session
from boxtwin.gui.widgets.class_counter import ClassCounter
from boxtwin.gui.widgets.classify_dialog import ClassifyDialog
from boxtwin.gui.widgets.event_list import EventList
from boxtwin.gui.widgets.identity_panel import IdentityPanel
from boxtwin.gui.widgets.reanno_panel import ReannoPanel
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
retracción de recuperación completa.<br><br>
<hr>
<b>Tipo de golpe</b><br><br>
<b>straight</b> (jab, cross)<br>
El puño viaja hacia el blanco trazando una recta o casi recta. Si se traza una recta entre
el puño y el hombro durante el recorrido, <b>el codo siempre está por debajo de esa línea</b>,
salvo en la extensión máxima. El codo queda detrás del puño todo el recorrido y se extiende
siguiéndolo. La potencia viene del empuje de la pierna trasera.<br>
<i>Vista frontal o trasera:</i> codo, hombro y puño casi en el mismo punto.<br><br>
<b>hook</b><br>
El puño viaja sobre un arco alrededor del eje vertical del cuerpo. El codo se mantiene
flexionado en un ángulo aproximadamente constante y va <b>al costado, no detrás</b>. La
potencia viene de la rotación de tronco y cadera.<br>
<i>Vista frontal o trasera:</i> codo y hombro en puntos cercanos, el puño un poco más
alejado.<br><br>
<b>Cuando no se distingue</b><br>
Si el golpe se tira hacia la cámara y no se ve el plano del recorrido, mirar el
<b>hombro contrario</b>: en el hook rota visiblemente, en el straight acompaña menos.
También se observa mayor amplitud del brazo y mayor rotación de torso en el hook.
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
        self._resembrando: TrackRole | None = None
        self._reanno = self._cargar_reanno()
        self._reanno_golpes: list[TrialLabels] = []
        self._reanno_ms = 0
        self._reanno_replays = 0
        self._reanno_activo = False
        self._reanno_actual: str | None = None
        self._reanno_revelado = False

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
        self.identidad.asignar.connect(self._asignar_rol)
        self.identidad.intercambiar.connect(self._intercambiar)
        self.identidad.resembrar.connect(self._activar_resiembra)
        self.identidad.marcarTramo.connect(self._marcar_tramo)
        self.identidad.buscarUniones.connect(self._buscar_uniones)
        self.identidad.aceptarUnion.connect(self._aceptar_union)
        self.identidad.rellenarInternos.connect(self._rellenar_internos)
        self.identidad.previsualizarChicos.connect(self._previsualizar_chicos)
        self.identidad.ignorarChicos.connect(self._ignorar_chicos)
        self.view.boxDrawn.connect(self._caja_dibujada)
        self.reanno.empezar.connect(self._reanno_empezar)
        self.reanno.siguiente.connect(self._reanno_siguiente)
        self.reanno.revelar.connect(self._reanno_revelar)
        self.reanno.salir.connect(self._reanno_salir)
        self.reanno.confirmarVentana.connect(self._reanno_confirmar_ventana)

        self._autosave = QTimer(self)
        self._autosave.timeout.connect(self._guardar_si_hace_falta)
        self._autosave.start(AUTOSAVE_MS)

        self._refrescar_anotacion()
        self._refrescar_reanno()
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
        pestanas.addTab(self._scroll(self.lista), "Eventos")

        # -- identidad
        self.identidad = IdentityPanel()
        pestanas.addTab(self._scroll(self.identidad), "Identidad")

        # -- reanotacion ciega
        self.reanno = ReannoPanel()
        pestanas.addTab(self._scroll(self.reanno), "Reanotación")

        # -- balance de clases
        self.contador = ClassCounter()
        pestanas.addTab(self._scroll(self._envolver(self.contador)), "Balance")

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
        pestanas.addTab(self._scroll(vista), "Vista")

        # -- validacion
        self.lbl_issues = QLabel("sin observaciones")
        self.lbl_issues.setWordWrap(True)
        self.lbl_issues.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_issues.setAlignment(Qt.AlignmentFlag.AlignTop)
        pestanas.addTab(self._scroll(self._envolver(self.lbl_issues)), "Validación")

        dock.setWidget(pestanas)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        """
        Mete un panel en un area desplazable.

        Sin esto, la altura minima del contenido del dock manda sobre el tamano de la
        ventana entera. Con una anotacion real de 143 observaciones, la pestana de
        validacion exigia 1635 px y la ventana salia de 1709 de alto sobre una pantalla de
        1080: el timeline y la barra de estado quedaban abajo del borde, invisibles, y el
        video aparecia chico y corrido porque se centraba en una vista mucho mas alta que
        lo que se veia.
        """
        area = QScrollArea()
        area.setWidget(w)
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        # Horizontal no: el contenido se adapta al ancho del dock y una barra horizontal
        # solo robaria alto.
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return area

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
        add("view.only_selected", self._solo_seleccionado)

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
        self.identidad.refrescar(
            frame, [(p.track_id, p.role, p.interpolated) for p in poses]
        )
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
        if self._reanno_activo:
            self.lbl_pel.setText(f"| CIEGO · {self.fighter.value}")
            self.lbl_pel.setStyleSheet("color: #e08a3c; font-weight: 600;")
        else:
            self.lbl_pel.setText(f"| anotando {self.fighter.value}")
            self.lbl_pel.setStyleSheet("")
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
        if self._reanno_activo and self._reanno_actual is not None:
            # En modo ciego el peleador lo fija el intento. Dejarlo cambiar por un atajo
            # apretado de costumbre reintroduce el error que este bloqueo existe para evitar,
            # y es un error mudo: el golpe queda registrado bajo el peleador equivocado y el
            # reporte lo cuenta como desacuerdo de etiqueta.
            self.statusBar().showMessage(
                "en modo ciego el peleador lo fija el intento; no se puede cambiar", 3000
            )
            return
        self.fighter = f
        # Si el filtro de un solo peleador esta puesto, sigue al que se acaba de elegir.
        if self.view.options.only_fighter is not None:
            self.view.options.only_fighter = f
            self.view.update()
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
        if self._reanno_activo:
            return self._reanno_clasificar(inicio, fin, timer)
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

    # -- identidad ---------------------------------------------------------

    def _aplicar_identidad(self, comando) -> bool:
        """
        Ejecuta una operacion de identidad y rehace el indice del resolver.

        Sin el refresh, el overlay seguiria mostrando los roles viejos y el anotador
        confirmaria una correccion que en pantalla parece no haber pasado.
        """
        try:
            self.session.undo.do(comando)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Operación rechazada", str(exc))
            return False
        self.session.resolver.refresh()
        self._refrescar_anotacion()
        self._on_frame(self.player.cursor)
        self.statusBar().showMessage(comando.label, 3000)
        return True

    def _asignar_rol(self, track_id: int, rol: TrackRole) -> None:
        """
        Asigna desde el cuadro actual hasta la proxima decision manual.

        Desde el cuadro actual y no desde el principio del video: el track pudo haber sido
        otra persona antes, y pisar todo el rango borraria correcciones ya hechas.
        """
        from boxtwin.core.identity_ops import boundary_after

        frame = self.player.cursor
        self._aplicar_identidad(
            AssignRole(
                track_id=track_id, role=rol, start_frame=frame,
                end_frame_excl=boundary_after(self.session.doc, frame),
                annotator=self.session.annotator,
            )
        )

    def _intercambiar(self) -> None:
        self._aplicar_identidad(
            SwapFromFrame(self.player.cursor, annotator=self.session.annotator)
        )

    def _activar_resiembra(self, rol: TrackRole) -> None:
        self.player.pause()
        self._resembrando = rol
        self.view.set_draw_mode(True)
        self.identidad.set_modo_dibujo(True, rol)

    def _caja_dibujada(self, rect) -> None:
        if self._resembrando is None:
            return
        rol = self._resembrando
        self._resembrando = None
        self.view.set_draw_mode(False)
        self.identidad.set_modo_dibujo(False)

        caja = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        frame = self.player.cursor
        # Si la caja cae sobre un track existente, se le asigna el rol a ese track: es el
        # caso comun, porque el tracker no perdio al peleador sino que le cambio el id.
        import numpy as np

        dets = self.session.cache.detections(frame)
        mejor_id, mejor_iou = None, 0.0
        for i in range(len(dets)):
            solape = iou(np.asarray(caja, np.float32), dets.bbox[i])
            if solape > mejor_iou:
                mejor_id, mejor_iou = int(dets.track_id[i]), solape

        umbral = self.session.doc.settings_snapshot.interp_min_iou
        self._aplicar_identidad(
            Reseed(
                frame=frame, role=rol, bbox=caja,
                track_id=mejor_id if mejor_iou >= umbral else None,
                iou=round(mejor_iou, 4) if mejor_id is not None else None,
                annotator=self.session.annotator,
            )
        )

    def _marcar_tramo(self, fighter: FighterId, ini: int, fin: int, motivo) -> None:
        if fin <= ini:
            self.statusBar().showMessage("el fin del tramo tiene que ser posterior", 3000)
            return
        self._aplicar_identidad(
            MarkUnreliable(
                fighter=fighter, start_frame=ini, end_frame_excl=fin,
                reason=motivo, annotator=self.session.annotator,
            )
        )

    def _buscar_uniones(self) -> None:
        st = self.session.doc.settings_snapshot
        candidatos = detectar_huecos(
            self.session.cache,
            max_gap=st.interp_max_gap_frames,
            min_iou=st.interp_min_iou,
            rol_en=rol_en_track(self.session.doc),
        )
        self.identidad.set_candidatos(candidatos)
        self.statusBar().showMessage(f"{len(candidatos)} uniones propuestas", 3000)

    def _aceptar_union(self, candidato, rol: TrackRole) -> None:
        if self._aplicar_identidad(
            AcceptJoin(candidato=candidato, role=rol, annotator=self.session.annotator)
        ):
            self._buscar_uniones()

    def _huecos_todos(self):
        """
        Todos los huecos internos del cache, calculados una sola vez.

        Depende solo del cache, que es inmutable, asi que recalcularlo en cada refresco es
        puro desperdicio. Y no es poco: sobre la pelea, con 11.100 tracks y 1,1 millones de
        detecciones, recorrerlo tarda 3,5 s. Como el refresco corre despues de cada evento
        confirmado, sin este cache anotar 1400 golpes costaria 82 minutos de espera.
        """
        if getattr(self, "_huecos_cache", None) is None:
            self._huecos_cache = detectar_huecos_internos(self.session.cache)
        return self._huecos_cache

    def _huecos_internos_pendientes(self):
        """Los huecos internos que todavia no tienen interpolacion declarada."""
        ya = {
            (i.to_track_id, i.gap_start_frame)
            for i in self.session.doc.identity.interpolations
        }
        rol_en = rol_en_track(self.session.doc)
        return [
            c
            for c in self._huecos_todos()
            if (c.to_track_id, c.gap_start) not in ya
            and rol_en(c.from_track_id, c.last_frame) in (TrackRole.A, TrackRole.B)
        ]

    def _alturas(self) -> dict[int, float]:
        """Cacheado: recorrer el cache entero cuesta segundos sobre un video largo."""
        if getattr(self, "_alturas_cache", None) is None:
            self._alturas_cache = altura_maxima_por_track(
                self.session.cache, self.session.video_size[1]
            )
        return self._alturas_cache

    def _tracks_chicos(self, umbral: float) -> list[int]:
        """
        Tracks que nunca llegan al umbral y todavia no estan resueltos.

        Se descuentan los que ya tienen rol de peleador, que no se tocan nunca, y tambien los
        que ya estan ignorados de punta a punta: si no, el conteo seguiria mostrando los
        mismos 75 despues de aplicarlos y el boton quedaria invitando a repetir una operacion
        que ya no hace nada.
        """
        peleadores = {TrackRole.A, TrackRole.B}
        resueltos = {
            a.track_id
            for a in self.session.doc.identity.assignments
            if a.role in peleadores
            or (
                a.role is TrackRole.IGNORE
                and a.start_frame == 0
                and a.end_frame_excl >= self.session.total_frames
            )
        }
        return [
            t for t, alto in self._alturas().items()
            if alto < umbral and t not in resueltos
        ]

    def _previsualizar_chicos(self, umbral: float) -> None:
        self.identidad.set_chicos(len(self._tracks_chicos(umbral)), len(self._alturas()))

    def _ignorar_chicos(self, umbral: float) -> None:
        chicos = self._tracks_chicos(umbral)
        if not chicos:
            return
        if (
            QMessageBox.question(
                self,
                "Descartar público",
                f"Se van a marcar como ignore {len(chicos)} tracks que nunca llegan al "
                f"{umbral:.0%} del alto de imagen.\n\n"
                "No se toca ningún track que ya sea un peleador. Se deshace con Ctrl+Z.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        comando = IgnoreSmallTracks(
            track_ids=chicos, total_frames=self.session.total_frames,
            umbral=umbral, annotator=self.session.annotator,
        )
        if self._aplicar_identidad(comando):
            self.statusBar().showMessage(f"{comando.aplicados} tracks ignorados", 5000)

    def _refrescar_internos(self) -> None:
        pend = self._huecos_internos_pendientes()
        self.identidad.set_internos(len(pend), sum(c.gap_len for c in pend))
        self._previsualizar_chicos(self.identidad.sp_alto.value())

    def _rellenar_internos(self) -> None:
        pend = self._huecos_internos_pendientes()
        if not pend:
            return
        cuadros = sum(c.gap_len for c in pend)
        if (
            QMessageBox.question(
                self,
                "Rellenar huecos internos",
                f"Se van a interpolar {cuadros} cuadros en {len(pend)} huecos.\n\n"
                "Son huecos dentro de un mismo track, así que no cambia ninguna asignación "
                "de rol. Los cuadros quedan marcados como interpolados y el export decide "
                "si los usa. Se puede deshacer con Ctrl+Z.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        comando = FillInternalGaps(candidatos=pend, annotator=self.session.annotator)
        if self._aplicar_identidad(comando):
            self.statusBar().showMessage(
                f"{comando.aplicados} huecos rellenados, {cuadros} cuadros interpolados", 5000
            )

    # -- reanotacion ciega -------------------------------------------------

    def _cargar_reanno(self):
        ruta = self.session.paths.annot.with_name(
            self.session.paths.annot.name.replace(".annot.json", ".reanno.json")
        )
        self._reanno_path = ruta
        if not ruta.is_file():
            return None
        try:
            return cargar_reanno(ruta, self.session.doc)
        except Exception:  # noqa: BLE001
            return None

    def _reanno_empezar(self) -> None:
        if self._reanno is None:
            return
        self._reanno_activo = True
        # Se ocultan las marcas y la lista: con el golpe senalado en pantalla, el error de
        # fronteras del reporte mediria cero por construccion.
        self.timeline.set_blind(True)
        self.lista.setEnabled(False)
        self._reanno_siguiente()

    def _reanno_salir(self) -> None:
        self._reanno_activo = False
        self._reanno_actual = None
        self._reanno_revelado = False
        self._reanno_golpes = []
        self._reanno_ms = 0
        self._reanno_replays = 0
        self.timeline.set_blind(False)
        self.lista.setEnabled(True)
        self._cancelar_abierto()
        self._refrescar_reanno()

    def _reanno_siguiente(self) -> None:
        if self._reanno is None:
            return
        pendientes = self._reanno.pendientes()
        self._reanno_revelado = False
        self._cancelar_abierto()
        if not pendientes:
            self._reanno_actual = None
            self.statusBar().showMessage("no quedan intentos pendientes", 4000)
            self._refrescar_reanno()
            return
        self._reanno_actual = pendientes[0]
        self._reanno_golpes = []
        self._reanno_ms = 0
        self._reanno_replays = 0
        # El peleador lo fija el intento, no la seleccion de la interfaz. Sin esto el
        # reanotador marca el golpe del que tenia seleccionado, que en general no es el del
        # intento, y el reporte compara contra el evento equivocado.
        ev = self.session.doc.event_by_id(self._reanno_actual)
        if ev is not None:
            self.fighter = ev.fighter
        inicio, _ = self._reanno.ventana(self._reanno_actual)
        self.player.seek(inicio)
        self._refrescar_reanno()

    def _reanno_revelar(self) -> None:
        if self._reanno_actual is None:
            return
        ev = self.session.doc.event_by_id(self._reanno_actual)
        if ev is None:
            return
        self._reanno_revelado = True
        QMessageBox.information(
            self, "Etiqueta original",
            f"{ev.side.value} {ev.punch_type.value} {ev.target.value} "
            f"{ev.completeness.value}\ncuadros {ev.start_frame}–{ev.end_frame}\n\n"
            "Este intento queda marcado como no ciego y no entra en el reporte.",
        )
        self._refrescar_reanno()

    def _reanno_clasificar(self, inicio: int, fin: int, timer: EventTimer) -> None:
        """Clasifica un intento y lo escribe en el archivo de reanotacion, no en el annot."""
        if self._reanno is None or self._reanno_actual is None:
            return
        dlg = ClassifyDialog(
            self.session, self.keymap,
            fighter=self.fighter, start_frame=inicio, end_frame=fin,
            timer=timer, event_id=self._reanno_actual,
            annotator=self.session.annotator, session_id=self.session_id, parent=self,
        )
        acepto = dlg.exec()
        self._abierto = None
        self._timer_evento = None
        if not acepto or dlg.resultado() is None:
            self._refresh_status()
            return

        evento, metricas = dlg.resultado()
        # Se acumula, no se cierra el intento. La ventana puede contener varios golpes del
        # mismo peleador: sobre la muestra real de Sparring.mp4 son 26 de 35. Cerrarla al
        # primero es lo que hacia la v1, y ahi el reporte terminaba comparando contra el
        # golpe de al lado.
        self._reanno_golpes.append(
            TrialLabels(
                start_frame=evento.start_frame,
                end_frame=evento.end_frame,
                peak_frame=evento.peak_frame,
                side=evento.side,
                punch_type=evento.punch_type,
                target=evento.target,
                completeness=evento.completeness,
                landed=evento.landed,
                quality=evento.quality,
            )
        )
        self._reanno_ms += metricas.active_ms
        self._reanno_replays += metricas.replays
        self.player.seek(evento.end_frame)
        self.statusBar().showMessage(
            f"{len(self._reanno_golpes)} marcados en esta ventana; confirmá cuando no "
            "quede ninguno", 4000
        )
        self._refrescar_reanno()

    def _reanno_confirmar_ventana(self) -> None:
        """
        Cierra el intento con los golpes marcados, que pueden ser cero.

        Cero es una respuesta valida y significativa: "no vi ningun golpe aca". Sin esa
        opcion el reanotador queda obligado a inventar uno y el numero deja de medir.
        """
        if self._reanno is None or self._reanno_actual is None:
            return
        self._cancelar_abierto()
        ev = self.session.doc.event_by_id(self._reanno_actual)
        if ev is None:
            return
        self._reanno.trials = [
            *self._reanno.trials,
            ReannoTrial(
                event_id=self._reanno_actual,
                fighter=ev.fighter,
                annotator=self.session.annotator,
                annotated_at=datetime.now().astimezone(),
                active_ms=self._reanno_ms,
                replays=self._reanno_replays,
                revealed=self._reanno_revelado,
                punches=list(self._reanno_golpes),
            ),
        ]
        guardar_reanno(self._reanno, self._reanno_path)
        self.statusBar().showMessage(
            f"ventana cerrada con {len(self._reanno_golpes)} golpes "
            f"({len(self._reanno.trials)} de {self._reanno.sample.n})", 3000
        )
        self._reanno_siguiente()

    def _refrescar_reanno(self) -> None:
        objetivo = None
        if self._reanno_actual is not None:
            ev = self.session.doc.event_by_id(self._reanno_actual)
            objetivo = ev.fighter.value if ev is not None else None
        self.reanno.refrescar(
            self._reanno, self._reanno_activo, self._reanno_actual, self._reanno_revelado,
            objetivo, len(self._reanno_golpes),
        )
        self._refresh_status()

    # -- deshacer y guardar ------------------------------------------------

    def _deshacer(self) -> None:
        cmd = self.session.undo.undo()
        if cmd is None:
            self.statusBar().showMessage("nada que deshacer", 2000)
            return
        # Deshacer puede haber tocado identidad: el indice del resolver hay que rehacerlo
        # igual, y rehacerlo de mas no cuesta nada.
        self.session.resolver.refresh()
        self._refrescar_anotacion()
        self._on_frame(self.player.cursor)
        self.statusBar().showMessage(f"deshecho: {cmd.label}", 3000)

    def _rehacer(self) -> None:
        cmd = self.session.undo.redo()
        if cmd is None:
            self.statusBar().showMessage("nada que rehacer", 2000)
            return
        self.session.resolver.refresh()
        self._refrescar_anotacion()
        self._on_frame(self.player.cursor)
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
        self._refrescar_internos()
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

    def _solo_seleccionado(self) -> None:
        """
        Alterna entre ver todas las detecciones y solo las del peleador elegido.

        Sirve en el clinch, donde los dos esqueletos se superponen y no se distingue cual
        keypoint es de quien. Alterna contra el peleador vigente, asi que cambiar con 1 o 2
        mientras esta activo cambia a quien se mira.
        """
        opciones = self.view.options
        opciones.only_fighter = None if opciones.only_fighter is not None else self.fighter
        self.view.update()
        quien = opciones.only_fighter.value if opciones.only_fighter else "todos"
        self.statusBar().showMessage(f"mostrando {quien}", 2000)

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
