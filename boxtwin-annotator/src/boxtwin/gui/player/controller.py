"""
BoxTwin - Reloj de reproduccion y cursor.

POR QUE EXISTE
  Un reproductor comun descarta cuadros para mantener el ritmo. Aca eso seria un error: si
  la reproduccion a 1,0x saltea cuadros, el anotador ve una version del video que no es la
  que va a etiquetar y las fronteras salen corridas.
  Por eso el reloj avanza EXACTAMENTE un cuadro por tic. Si la maquina no llega, la
  reproduccion se pone lenta, que es visible y honesto, en vez de saltear cuadros, que es
  invisible y corrompe el dato. A 0,25x, que es la velocidad por defecto para anotar, hay
  133 ms por cuadro y sobra tiempo.

QUE HACE
  Mantiene el cursor, la velocidad y la direccion, y emite senales cuando algo cambia.
  Reproduce hacia adelante y hacia atras a las mismas velocidades.

USO
  ctl = PlayerController(source, fps=29.97)
  ctl.frameChanged.connect(vista.mostrar)
  ctl.play(direction=-1)
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from boxtwin.gui.player.decoder import FrameSource

__all__ = ["PlayerController", "SPEEDS", "DEFAULT_SPEED"]

# 0,25x es el default porque es la velocidad a la que se distingue el inicio de la
# extension del codo, que es la definicion de start_frame.
SPEEDS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0)
DEFAULT_SPEED = 0.25


class PlayerController(QObject):
    frameChanged = Signal(int)
    playingChanged = Signal(bool)
    speedChanged = Signal(float)

    def __init__(self, source: FrameSource, fps: float, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.fps = fps
        self.total_frames = source.total_frames

        self._cursor = 0
        self._speed = DEFAULT_SPEED
        self._direction = 1

        self._timer = QTimer(self)
        # PreciseTimer: con el default, a 1,0x el intervalo de 33 ms se redondea y la
        # reproduccion se va de ritmo de forma perceptible.
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

    # -- estado ------------------------------------------------------------

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def timestamp(self, frame: int | None = None) -> float:
        """Segundos desde el inicio. Se calcula del numero de cuadro, nunca al reves."""
        return (self._cursor if frame is None else frame) / self.fps

    # -- navegacion --------------------------------------------------------

    def seek(self, frame: int, *, emit: bool = True) -> int:
        objetivo = max(0, min(int(frame), self.total_frames - 1))
        if objetivo != self._cursor:
            self._cursor = objetivo
            if emit:
                self.frameChanged.emit(objetivo)
        elif emit:
            self.frameChanged.emit(objetivo)
        return self._cursor

    def step(self, delta: int) -> int:
        """Avance o retroceso relativo. Pausa si estaba reproduciendo."""
        self.pause()
        return self.seek(self._cursor + delta)

    def step_seconds(self, segundos: float) -> int:
        return self.step(int(round(segundos * self.fps)))

    def go_start(self) -> int:
        return self.seek(0)

    def go_end(self) -> int:
        return self.seek(self.total_frames - 1)

    # -- reproduccion ------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"velocidad {speed} fuera de {SPEEDS}")
        self._speed = speed
        self.speedChanged.emit(speed)
        if self.playing:
            self._timer.setInterval(self._interval_ms())

    def cycle_speed(self, paso: int = 1) -> float:
        i = (SPEEDS.index(self._speed) + paso) % len(SPEEDS)
        self.set_speed(SPEEDS[i])
        return self._speed

    def play(self, direction: int = 1) -> None:
        self._direction = 1 if direction >= 0 else -1
        if not self._timer.isActive():
            self._timer.start(self._interval_ms())
            self.playingChanged.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playingChanged.emit(False)

    def toggle(self, direction: int = 1) -> None:
        if self.playing:
            self.pause()
        else:
            self.play(direction)

    def _interval_ms(self) -> int:
        return max(1, int(round(1000.0 / (self.fps * self._speed))))

    def _tick(self) -> None:
        siguiente = self._cursor + self._direction
        if not 0 <= siguiente < self.total_frames:
            self.pause()
            return
        self._cursor = siguiente
        self.frameChanged.emit(siguiente)
