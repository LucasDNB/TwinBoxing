"""
Medicion del tiempo de anotacion.

El reloj se inyecta para no depender del tiempo real: una suite con sleeps es lenta e
intermitente, y ademas no permite probar el caso que importa, que es una pausa de veinte
minutos.

Lo que se fija es que el contador mida trabajo y no reloj de pared. Si midiera reloj de
pared, una sesion con un almuerzo en el medio reportaria el almuerzo, y el numero dejaria
de servir para estimar, comparar videos o escribirlo en el capitulo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from boxtwin.core.metrics import IDLE_SECONDS, ActiveTimeTracker, EventTimer

AR = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 11, 14, 0, 0, tzinfo=AR)


class RelojFalso:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def avanzar(self, segundos: float) -> None:
        self.t += segundos


@pytest.fixture
def reloj() -> RelojFalso:
    return RelojFalso()


# -- conteo basico ---------------------------------------------------------


def test_cuenta_el_tiempo_con_actividad(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    reloj.avanzar(5)
    t.actividad()
    assert t.leer_ms() == pytest.approx(5000, abs=2)


def test_la_inactividad_corta_el_conteo(reloj: RelojFalso) -> None:
    """
    Veinte minutos sin tocar nada son veinte minutos que no se trabajo. Se acumula solo
    hasta el umbral, que es el margen que se le da a mirar el cuadro pensando.
    """
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    reloj.avanzar(20 * 60)
    t.actividad()
    assert t.leer_ms() == pytest.approx(IDLE_SECONDS * 1000, abs=2)


def test_actividad_repetida_mantiene_el_conteo(reloj: RelojFalso) -> None:
    """Trabajando de verdad, cada tecla renueva el margen y no se pierde tiempo."""
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    for _ in range(20):
        reloj.avanzar(5)
        t.actividad()
    assert t.leer_ms() == pytest.approx(100_000, abs=5)


def test_sin_foco_no_cuenta(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    reloj.avanzar(3)
    t.foco(False)
    reloj.avanzar(600)
    t.foco(True)
    reloj.avanzar(2)
    t.actividad()
    assert t.leer_ms() == pytest.approx(5000, abs=5)


def test_activo_refleja_el_estado(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    assert t.activo
    reloj.avanzar(IDLE_SECONDS + 1)
    assert not t.activo
    t.foco(False)
    t.foco(True)
    assert t.activo


# -- toma ------------------------------------------------------------------


def test_tomar_reinicia(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    reloj.avanzar(4)
    primero = t.tomar_ms()
    assert primero == pytest.approx(4000, abs=5)
    reloj.avanzar(3)
    t.actividad()
    assert t.tomar_ms() == pytest.approx(3000, abs=5)


def test_leer_no_reinicia(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(clock=reloj)
    t.actividad()
    reloj.avanzar(4)
    assert t.leer_ms() == pytest.approx(t.leer_ms(), abs=2)
    assert t.leer_ms() > 0


def test_umbral_configurable(reloj: RelojFalso) -> None:
    t = ActiveTimeTracker(idle_seconds=2.0, clock=reloj)
    t.actividad()
    reloj.avanzar(60)
    assert t.leer_ms() == pytest.approx(2000, abs=5)


# -- por evento ------------------------------------------------------------


def test_el_timer_cuenta_las_vueltas_del_preview(reloj: RelojFalso) -> None:
    """
    Las vueltas son el proxy de dificultad de la decision: un golpe claro se confirma en la
    primera, uno dudoso se mira cinco veces.
    """
    et = EventTimer(created_at=T0, reloj=ActiveTimeTracker(clock=reloj))
    for _ in range(3):
        et.replay()
    et.reloj.actividad()
    reloj.avanzar(7)

    m = et.confirmar(annotator="lucas", session_id="se_0001", ahora=T0)
    assert m.replays == 3
    assert m.active_ms == pytest.approx(7000, abs=10)
    assert m.annotator == "lucas"
    assert m.session_id == "se_0001"
    assert m.created_at == T0
    assert m.confirmed_at == T0
    assert m.edits == 0


def test_el_timer_arranca_en_cero() -> None:
    et = EventTimer(created_at=T0)
    assert et.replays == 0
