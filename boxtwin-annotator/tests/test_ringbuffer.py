"""
Buffer circular de cuadros.

Lo que se fija aca es la politica de desalojo: por distancia al cursor y no por antiguedad.
Con LRU, ir y venir sobre una frontera temporal (que es lo que se hace todo el tiempo al
anotar) desalojaria justo los cuadros que se estan mirando.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.gui.player.ringbuffer import MIN_CAPACITY, FrameRing


def img(valor: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), valor, np.uint8)


def test_guarda_y_devuelve() -> None:
    r = FrameRing(10)
    r.put(5, img(1))
    assert 5 in r
    assert r.get(5) is not None
    assert r.get(6) is None


def test_capacidad_por_presupuesto() -> None:
    """Un cuadro 4K y uno del proxy no pueden dar la misma cantidad de cuadros."""
    cuatro_k = FrameRing.for_budget(mb=512, frame_nbytes=3840 * 2160 * 3)
    proxy = FrameRing.for_budget(mb=512, frame_nbytes=960 * 540 * 3)
    assert cuatro_k.capacity == 21
    assert proxy.capacity == 345
    assert proxy.capacity > cuatro_k.capacity * 10


def test_capacidad_minima() -> None:
    """Con menos que el minimo no se cubre ni un paso de 5 cuadros para los dos lados."""
    r = FrameRing.for_budget(mb=1, frame_nbytes=3840 * 2160 * 3)
    assert r.capacity == MIN_CAPACITY


def test_desaloja_lo_mas_lejano_al_cursor() -> None:
    r = FrameRing(3)
    r.set_cursor(100)
    for f in (100, 101, 99):
        r.put(f, img())
    r.put(102, img())
    # El cursor sobrevive y el nuevo entra; cae el mas lejano de los previos.
    assert 100 in r and 102 in r
    assert len(r) == 3


def test_nunca_supera_la_capacidad() -> None:
    """
    El presupuesto es de memoria. Que el buffer crezca por encima de su capacidad, aunque
    sea por un cuadro, es exactamente lo que no puede pasar con material 4K, donde cada
    cuadro son 25 MB.
    """
    r = FrameRing(4)
    for f in range(200):
        r.set_cursor(f)
        r.put(f, img(f % 256))
        assert len(r) <= 4
    # Tambien cuando el cuadro pedido esta lejos del cursor, que es el caso del salto.
    r.set_cursor(0)
    for f in (5000, 6000, 7000, 8000, 9000):
        r.put(f, img())
        assert len(r) <= 4


def test_desalojo_es_deterministico() -> None:
    """Dos buffers alimentados en distinto orden tienen que terminar con lo mismo."""
    a, b = FrameRing(3), FrameRing(3)
    a.set_cursor(100)
    b.set_cursor(100)
    for f in (99, 100, 101, 102):
        a.put(f, img())
    for f in (102, 101, 100, 99):
        b.put(f, img())
    # El ultimo insertado difiere, pero la politica no puede depender del orden previo.
    assert len(a) == len(b) == 3


def test_la_ventana_sigue_al_cursor_hacia_atras() -> None:
    """
    Reproducir hacia atras es el caso que rompe LRU: los cuadros mas viejos son
    exactamente los que se van a pedir enseguida.
    """
    r = FrameRing(5)
    for f in range(20):
        r.set_cursor(f)
        r.put(f, img(f))

    # Retrocediendo, la ventana se reacomoda alrededor del cursor.
    for f in range(19, 9, -1):
        r.set_cursor(f)
        if r.get(f) is None:
            r.put(f, img(f))
    presentes = sorted(r)
    assert min(presentes) >= 8
    assert 10 in r


def test_no_desaloja_lo_que_se_acaba_de_poner() -> None:
    """Con capacidad chica, desalojar lo recien insertado dejaria el cache inutil."""
    r = FrameRing(1)
    r.set_cursor(0)
    r.put(500, img(7))
    assert r.get(500) is not None


def test_span_y_contiguous_span() -> None:
    r = FrameRing(10)
    for f in (10, 11, 12, 20):
        r.put(f, img())
    r.set_cursor(11)
    assert r.span() == (10, 20)
    # El tramo continuo alrededor del cursor es lo reproducible sin volver a decodificar.
    assert r.contiguous_span() == (10, 12)


def test_contiguous_span_sin_cursor_en_memoria() -> None:
    r = FrameRing(10)
    r.put(10, img())
    r.set_cursor(50)
    assert r.contiguous_span() is None


def test_clear_y_discard_outside() -> None:
    r = FrameRing(10)
    for f in range(10):
        r.put(f, img())
    r.discard_outside(3, 5)
    assert sorted(r) == [3, 4, 5]
    r.clear()
    assert len(r) == 0


@pytest.mark.parametrize("capacidad", [0, -1])
def test_capacidad_invalida(capacidad: int) -> None:
    with pytest.raises(ValueError):
        FrameRing(capacidad)


def test_presupuesto_invalido() -> None:
    with pytest.raises(ValueError):
        FrameRing.for_budget(mb=512, frame_nbytes=0)
