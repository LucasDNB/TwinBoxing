"""T7: precision y cobertura por cuadro de la identidad, con un track mixto."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

import medir_e11 as m

A, B, IG = "fighter_A", "fighter_B", "ignore"


def _asig(track, rol, ini, fin):
    return NS(track_id=track, role=rol, start_frame=ini, end_frame_excl=fin)


# La verdad: el track 3 es A hasta el 50 y B despues (mixto); el 1 es A, el 2 es B y el 9 es
# el arbitro. Cada track tiene un cuadro por cuadro de 0 a 99.
VERDAD = [_asig(1, A, 0, 100), _asig(2, B, 0, 100), _asig(3, A, 0, 50), _asig(3, B, 50, 100),
          _asig(9, IG, 0, 100)]
FRAMES = {t: np.arange(100) for t in (1, 2, 3, 9)}


def test_track_mixto_sin_corte():
    prop = [(1, 0, 100, A), (2, 0, 100, B), (3, 0, 100, A)]
    r = m.por_cuadro(m.rangos_verdad(VERDAD), prop, FRAMES.get)
    assert r["asignados"] == 300 and r["bien"] == 250 and r["peleador"] == 300
    assert r["precision"] == pytest.approx(250 / 300) and r["cobertura"] == pytest.approx(250 / 300)


def test_track_mixto_con_corte():
    prop = [(1, 0, 100, A), (2, 0, 100, B), (3, 0, 48, A), (3, 48, 100, B)]
    r = m.por_cuadro(m.rangos_verdad(VERDAD), prop, FRAMES.get)
    assert r["bien"] == 298 and r["precision"] == pytest.approx(298 / 300)


def test_sin_asignar_baja_la_cobertura_y_no_la_precision():
    prop = [(1, 0, 100, A), (2, 0, 100, B)]
    r = m.por_cuadro(m.rangos_verdad(VERDAD), prop, FRAMES.get)
    assert r["precision"] == 1.0 and r["cobertura"] == pytest.approx(200 / 300)


def test_un_rol_sobre_el_arbitro_es_un_error_de_precision():
    prop = [(1, 0, 100, A), (2, 0, 100, B), (9, 0, 100, A)]
    r = m.por_cuadro(m.rangos_verdad(VERDAD), prop, FRAMES.get)
    assert r["asignados"] == 300 and r["bien"] == 200 and r["peleador"] == 300


def test_solo_cuenta_cuadros_donde_el_track_esta():
    frames = {**FRAMES, 3: np.arange(0, 100, 2)}
    prop = [(3, 0, 100, A)]
    r = m.por_cuadro(m.rangos_verdad(VERDAD), prop, frames.get)
    assert r["asignados"] == 50 and r["bien"] == 25


def test_p15d_precision_y_recall_del_corte():
    verdad_cuadros = {1: {A: 100, B: 0}, 3: {A: 50, B: 50}, 4: {A: 30, B: 20}}
    r = m.p15d({3, 1}, verdad_cuadros)
    assert r["partidos"] == 2 and r["partidos_mixtos"] == 1 and r["mixtos"] == 2
    assert r["precision_del_corte"] == 0.5 and r["recall_del_corte"] == 0.5


def test_unidades_con_su_verdad_mayoritaria():
    # con cortes, cada segmento se puntua contra el rol mayoritario dentro de su rango
    prop = [(3, 0, 48, A), (3, 48, 100, B)]
    r = m.por_unidad(m.rangos_verdad(VERDAD), prop, FRAMES.get)
    assert r == {"evaluables": 2, "aciertos": 2}


def test_total_suma_los_videos():
    v = [{"bien": 9, "asignados": 10, "peleador": 12}, {"bien": 1, "asignados": 10, "peleador": 8}]
    t = m.total(v)
    assert t["precision"] == 0.5 and t["cobertura"] == 0.5 and t["wilson95_precision"][0] < 0.5
