"""
BoxTwin - Dialogo de clasificacion de un golpe.

POR QUE EXISTE
  Es donde se decide la etiqueta, o sea donde se genera el dato. Tres cosas de su diseno no
  son cosmeticas.

  Se completa entero con el teclado. Con mouse, cada evento cuesta seis clicks sobre
  desplegables y la anotacion de un video largo deja de ser viable; la medicion previa del
  proyecto dio 2,2 s por decision y ese numero exige que la mano no se mueva.

  El clip se reproduce en loop a 0,25x mientras se clasifica. Decidir si un golpe fue hook
  o uppercut mirando un cuadro congelado es adivinar: lo que distingue los dos es la
  trayectoria. El loop tambien cuenta las vueltas, que es el proxy directo de dificultad de
  la decision y queda en las metricas.

  Ningun campo obligatorio arranca elegido salvo los que el esquema define con default.
  Preseleccionar el tipo mas frecuente ahorraria teclas y sesgaria el dataset hacia esa
  clase cada vez que el anotador confirme sin mirar.

QUE HACE
  Muestra el clip en loop con su overlay, acepta las teclas mnemonicas, valida antes de
  confirmar y devuelve el evento junto con sus metricas de proceso.

USO
  dlg = ClassifyDialog(session, keymap, fighter, start, end, parent)
  if dlg.exec(): evento, metricas = dlg.resultado()
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxtwin.core.metrics import EventTimer
from boxtwin.core.schema import Event, EventMetrics
from boxtwin.core.types import (
    Completeness,
    FighterId,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)
from boxtwin.core.validation import Issue
from boxtwin.gui.keymap import Keymap
from boxtwin.gui.widgets.video_view import VideoView

__all__ = ["ClassifyDialog"]

PREVIEW_SPEED = 0.25

# Cuadros de margen a cada lado del evento en el preview. El golpe se entiende mejor con un
# poco de contexto antes del inicio y despues del final, y el margen no altera el evento.
MARGEN = 4


class ClassifyDialog(QDialog):
    """Clasificacion por teclado con preview en loop."""

    def __init__(
        self,
        session,
        keymap: Keymap,
        *,
        fighter: FighterId,
        start_frame: int,
        end_frame: int,
        timer: EventTimer,
        event_id: str,
        annotator: str,
        session_id: str,
        parent: QWidget | None = None,
        editando: Event | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.keymap = keymap
        self.timer = timer
        self.event_id = event_id
        self.annotator = annotator
        self.session_id = session_id

        self.fighter = fighter
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.peak_frame: int | None = None

        # Sin preseleccion en lo que define la clase: ver el docstring.
        self.side: Side | None = None
        self.punch_type: PunchType | None = None
        self.target: Target | None = None
        self.completeness = Completeness.FULL
        self.landed = Landed.UNKNOWN
        self.quality = Quality.CLEAN
        self.notes = ""

        if editando is not None:
            self._precargar(editando)

        self._resultado: tuple[Event, EventMetrics] | None = None
        self._cursor = start_frame
        self._acciones = self._armar_acciones()

        self.setWindowTitle(f"Clasificar golpe — {fighter.value}")
        self.setModal(True)
        self._build_ui()

        self._loop = QTimer(self)
        self._loop.timeout.connect(self._avanzar)
        self._loop.setInterval(max(1, int(round(1000 / (session.fps * PREVIEW_SPEED)))))
        self._loop.start()
        self._pintar()

    def _precargar(self, ev: Event) -> None:
        self.side, self.punch_type, self.target = ev.side, ev.punch_type, ev.target
        self.completeness, self.landed, self.quality = ev.completeness, ev.landed, ev.quality
        self.peak_frame, self.notes = ev.peak_frame, ev.notes

    # -- interfaz ----------------------------------------------------------

    def _build_ui(self) -> None:
        self.vista = VideoView(self.session.video_size, self)
        self.vista.setMinimumSize(560, 315)
        self.vista.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.lbl_campos = QLabel()
        self.lbl_campos.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_ayuda = QLabel(self._texto_ayuda())
        self.lbl_ayuda.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_ayuda.setStyleSheet("color: #888;")
        self.lbl_estado = QLabel()

        lateral = QVBoxLayout()
        lateral.addWidget(self.lbl_campos)
        lateral.addStretch(1)
        lateral.addWidget(self.lbl_ayuda)

        arriba = QHBoxLayout()
        arriba.addWidget(self.vista, 3)
        arriba.addLayout(lateral, 2)

        raiz = QVBoxLayout(self)
        raiz.addLayout(arriba, 1)
        raiz.addWidget(self.lbl_estado)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(980, 460)

    def _texto_ayuda(self) -> str:
        k = self.keymap.sequence
        filas = [
            ("lado", f"{k('event.side_left')} izq · {k('event.side_right')} der"),
            ("tipo", f"{k('event.type_straight')} recto · {k('event.type_hook')} hook · "
                     f"{k('event.type_uppercut')} uppercut"),
            ("altura", f"{k('event.target_head')} cabeza · {k('event.target_body')} cuerpo"),
            ("completitud", f"{k('event.feint')} amague · {k('event.aborted')} abortado"),
            ("resultado", f"{k('event.landed')} conecta · {k('event.blocked')} bloqueado · "
                          f"{k('event.slipped')} esquivado · {k('event.missed')} falla · "
                          f"{k('event.landed_unknown')} sin datos"),
            ("calidad", f"{k('event.quality_clean')} limpia · "
                        f"{k('event.quality_partial')} oclusion · "
                        f"{k('event.quality_ambiguous')} ambigua"),
            ("pico", f"{k('event.mark_peak')} en el cuadro del preview"),
            ("fronteras", f"{self.keymap.sequence('event.mark_start')} / "
                          f"{self.keymap.sequence('event.mark_end')} remarcar en el preview"),
            ("", f"{k('event.preview_toggle')} pausar · {k('event.confirm')} confirmar · "
                 f"{k('event.abort_dialog')} cancelar"),
        ]
        return "<br>".join(
            f"<b>{n}</b> {t}" if n else f"<br>{t}" for n, t in filas
        )

    # -- preview -----------------------------------------------------------

    @property
    def _desde(self) -> int:
        return max(0, self.start_frame - MARGEN)

    @property
    def _hasta(self) -> int:
        return min(self.session.total_frames - 1, self.end_frame + MARGEN)

    def _avanzar(self) -> None:
        self._cursor += 1
        if self._cursor > self._hasta:
            self._cursor = self._desde
            # Una vuelta completa del loop: el anotador volvio a mirar el golpe.
            self.timer.replay()
        self._pintar()

    def _pintar(self) -> None:
        img = self.session.source.frame(self._cursor)
        poses = self.session.resolver.resolve_frame(self._cursor)
        self.vista.set_frame(img, poses)
        self._refrescar_textos()

    def _refrescar_textos(self) -> None:
        def v(x) -> str:
            return f"<b style='color:#f5c242'>{x.value}</b>" if x is not None else "<i>—</i>"

        rel = self._cursor - self.start_frame
        self.lbl_campos.setText(
            f"peleador <b>{self.fighter.value}</b><br>"
            f"cuadros <b>{self.start_frame}–{self.end_frame}</b> "
            f"({self.end_frame - self.start_frame + 1} de duracion)<br>"
            f"pico {self.peak_frame if self.peak_frame is not None else '—'}<br><br>"
            f"lado {v(self.side)}<br>"
            f"tipo {v(self.punch_type)}<br>"
            f"altura {v(self.target)}<br>"
            f"completitud {v(self.completeness)}<br>"
            f"resultado {v(self.landed)}<br>"
            f"calidad {v(self.quality)}<br><br>"
            f"vueltas del preview <b>{self.timer.replays}</b>"
        )
        falta = self._faltantes()
        if falta:
            self.lbl_estado.setText(f"falta elegir: {', '.join(falta)}")
            self.lbl_estado.setStyleSheet("color: #e08a3c;")
        else:
            self.lbl_estado.setText(
                f"listo para confirmar · preview en el cuadro {self._cursor} ({rel:+d})"
            )
            self.lbl_estado.setStyleSheet("color: #6fbf73;")

    def _faltantes(self) -> list[str]:
        falta = []
        if self.side is None:
            falta.append("lado")
        if self.punch_type is None:
            falta.append("tipo")
        if self.target is None:
            falta.append("altura")
        return falta

    # -- teclado -----------------------------------------------------------

    def _armar_acciones(self) -> dict[str, str]:
        """Tecla normalizada -> accion, solo del contexto del dialogo mas las fronteras."""
        acciones = self.keymap.acciones_de("dialog")
        for extra in ("event.mark_start", "event.mark_end"):
            acciones[extra] = self.keymap.sequence(extra)
        return {
            QKeySequence(t).toString(QKeySequence.SequenceFormat.PortableText).lower(): a
            for a, t in acciones.items()
        }

    def keyPressEvent(self, event) -> None:  # noqa: N802
        self.timer.reloj.actividad()
        secuencia = QKeySequence(event.keyCombination())
        clave = secuencia.toString(QKeySequence.SequenceFormat.PortableText).lower()
        accion = self._acciones.get(clave)
        if accion is None:
            return super().keyPressEvent(event)
        event.accept()
        self._ejecutar(accion)

    def _ejecutar(self, accion: str) -> None:
        setters = {
            "event.side_left": ("side", Side.LEFT),
            "event.side_right": ("side", Side.RIGHT),
            "event.type_straight": ("punch_type", PunchType.STRAIGHT),
            "event.type_hook": ("punch_type", PunchType.HOOK),
            "event.type_uppercut": ("punch_type", PunchType.UPPERCUT),
            "event.target_head": ("target", Target.HEAD),
            "event.target_body": ("target", Target.BODY),
            "event.feint": ("completeness", Completeness.FEINT),
            "event.aborted": ("completeness", Completeness.ABORTED),
            "event.landed": ("landed", Landed.LANDED),
            "event.blocked": ("landed", Landed.BLOCKED),
            "event.slipped": ("landed", Landed.SLIPPED),
            "event.missed": ("landed", Landed.MISSED),
            "event.landed_unknown": ("landed", Landed.UNKNOWN),
            "event.quality_clean": ("quality", Quality.CLEAN),
            "event.quality_partial": ("quality", Quality.PARTIAL_OCCLUSION),
            "event.quality_ambiguous": ("quality", Quality.AMBIGUOUS),
        }
        if accion in setters:
            campo, valor = setters[accion]
            # Volver a apretar la misma tecla de completitud vuelve a full: es un toggle,
            # porque marcar un amague por error no puede obligar a cancelar el dialogo.
            if campo == "completeness" and getattr(self, campo) is valor:
                valor = Completeness.FULL
            setattr(self, campo, valor)
            self._refrescar_textos()
            return

        if accion == "event.mark_peak":
            self.peak_frame = self._cursor
        elif accion == "event.mark_start":
            self.start_frame = min(self._cursor, self.end_frame - 1)
            self.peak_frame = self._recortar_pico()
        elif accion == "event.mark_end":
            self.end_frame = max(self._cursor, self.start_frame + 1)
            self.peak_frame = self._recortar_pico()
        elif accion == "event.preview_toggle":
            self._loop.stop() if self._loop.isActive() else self._loop.start()
        elif accion == "event.confirm":
            self._confirmar()
            return
        elif accion == "event.abort_dialog":
            self.reject()
            return
        self._refrescar_textos()

    def _recortar_pico(self) -> int | None:
        """Mover una frontera puede dejar el pico afuera, y el esquema lo rechazaria."""
        if self.peak_frame is None:
            return None
        return min(max(self.peak_frame, self.start_frame), self.end_frame)

    # -- salida ------------------------------------------------------------

    def _confirmar(self) -> None:
        if self._faltantes():
            self._refrescar_textos()
            return
        ahora = datetime.now().astimezone()
        evento = Event(
            id=self.event_id,
            fighter=self.fighter,
            start_frame=self.start_frame,
            peak_frame=self.peak_frame,
            end_frame=self.end_frame,
            side=self.side,
            punch_type=self.punch_type,
            target=self.target,
            completeness=self.completeness,
            landed=self.landed,
            # La guardia se hereda de la vigente para ese peleador en el inicio del golpe.
            # El rol lead/rear se deriva de aca, nunca al reves.
            guard=self.session.doc.guard_at(self.fighter, self.start_frame),
            quality=self.quality,
            notes=self.notes,
        )
        metricas = self.timer.confirmar(
            annotator=self.annotator, session_id=self.session_id, ahora=ahora
        )
        self._resultado = (evento, metricas)
        self._loop.stop()
        self.accept()

    def resultado(self) -> tuple[Event, EventMetrics] | None:
        return self._resultado

    def closeEvent(self, event) -> None:  # noqa: N802
        self._loop.stop()
        super().closeEvent(event)
