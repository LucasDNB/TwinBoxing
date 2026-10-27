"""
B11: la regla de altura del golpe (prototipo/specs/ampliacion-13-10/03 §2), con poses
sinteticas. Que la regla sirva sobre material real lo dice C8, no esta prueba.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.mvp.objetivo import confianza, objetivo_de

T = 20


def _cuerpo(cx: float, hombros_y: float = 200.0, torso: float = 100.0):
    kp = np.zeros((17, 2), np.float32)
    kp[0] = (cx, hombros_y - 60)
    kp[5], kp[6] = (cx - 40, hombros_y), (cx + 40, hombros_y)
    kp[7], kp[8] = (cx - 50, hombros_y + 50), (cx + 50, hombros_y + 50)
    kp[9], kp[10] = (cx - 30, hombros_y + 20), (cx + 30, hombros_y + 20)   # guardia
    kp[11], kp[12] = (cx - 30, hombros_y + torso), (cx + 30, hombros_y + torso)
    return kp


def _escena():
    kp = np.zeros((2, T, 17, 2), np.float32)
    for f in range(T):
        kp[0, f] = _cuerpo(300.0)
        kp[1, f] = _cuerpo(500.0)
    sc = np.full((2, T, 17), 0.9, np.float32)
    valid = np.ones((2, T), bool)
    return kp, sc, valid


def _golpe(kp, f, y, brazo="izq", extension=180.0):
    i = 9 if brazo == "izq" else 10
    hombro = 5 if brazo == "izq" else 6
    kp[0, f, i] = (kp[0, f, hombro, 0] + extension, y)


def test_mano_a_la_altura_de_la_cara_del_rival_es_cabeza():
    kp, sc, valid = _escena()
    _golpe(kp, 10, y=170.0)                       # 30 px por encima de los hombros: h = -0,3
    r = objetivo_de(kp, sc, valid, 0, "izq", 8, 12, tau=0.2)
    assert (r.objetivo, r.h, r.motivo, r.cuadro_pico) == ("cabeza", -0.3, None, 10)


def test_mano_a_mitad_del_torso_del_rival_es_cuerpo():
    kp, sc, valid = _escena()
    _golpe(kp, 10, y=250.0, brazo="der")          # h = 0,5
    r = objetivo_de(kp, sc, valid, 0, "right", 8, 12, tau=0.2)
    assert (r.objetivo, r.h) == ("cuerpo", 0.5)


def test_el_pico_es_el_cuadro_de_mayor_extension():
    kp, sc, valid = _escena()
    _golpe(kp, 9, y=260.0, extension=100.0)       # mas abajo pero menos extendido
    _golpe(kp, 11, y=170.0, extension=200.0)
    assert objetivo_de(kp, sc, valid, 0, "izq", 8, 12, tau=0.2).cuadro_pico == 11


def test_la_extension_se_mide_en_anchos_de_hombro_del_que_pega():
    kp, sc, valid = _escena()
    _golpe(kp, 9, y=170.0, extension=150.0)
    _golpe(kp, 11, y=260.0, extension=150.0)
    kp[0, 11, 5, 0] -= 40.0                        # hombros mas anchos en 11: menos anchos de hombro
    assert objetivo_de(kp, sc, valid, 0, "izq", 8, 12, tau=0.2).cuadro_pico == 9


def test_el_rival_en_un_vecino_alcanza_y_mas_lejos_no():
    kp, sc, valid = _escena()
    _golpe(kp, 10, y=170.0)
    valid[1, 9:12] = False
    assert objetivo_de(kp, sc, valid, 0, "izq", 8, 12, tau=0.2).objetivo == "cabeza"
    valid[1, 8:13] = False
    r = objetivo_de(kp, sc, valid, 0, "izq", 10, 10, tau=0.2)
    assert (r.objetivo, r.motivo) == (None, "sin_rival")


def test_caderas_del_rival_con_score_bajo_cuentan_como_sin_rival():
    kp, sc, valid = _escena()
    _golpe(kp, 10, y=170.0)
    sc[1, :, 11] = 0.1
    assert objetivo_de(kp, sc, valid, 0, "izq", 10, 10, tau=0.2).motivo == "sin_rival"


def test_sin_pose_del_que_pega():
    kp, sc, valid = _escena()
    valid[0] = False
    r = objetivo_de(kp, sc, valid, 0, "izq", 8, 12, tau=0.2)
    assert (r.objetivo, r.motivo) == (None, "sin_pose")


def test_B_contra_A_usa_el_cuerpo_de_A():
    kp, sc, valid = _escena()
    kp[0, :, :, 1] += 100.0                        # A esta agachado: sus hombros a 300
    kp[1, 10, 9] = (kp[1, 10, 5, 0] - 180.0, 260.0)
    r = objetivo_de(kp, sc, valid, 1, "izq", 10, 10, tau=0.2)
    assert (r.objetivo, r.h) == ("cabeza", -0.4), "a la altura de la cabeza del agachado"


def test_guante_reemplaza_a_la_muneca_cuando_hay_caja():
    kp, sc, valid = _escena()
    _golpe(kp, 10, y=170.0)
    r = objetivo_de(kp, sc, valid, 0, "izq", 10, 10, tau=0.2, guantes={(10, "izq"): (480.0, 260.0)})
    assert r.objetivo == "cuerpo"


def test_confianza():
    assert confianza(0.2, 0.2) == 0.0
    assert confianza(0.3, 0.2) == confianza(0.1, 0.2)
    assert 0 < confianza(0.25, 0.2) < confianza(0.6, 0.2) < 1
    assert confianza(-0.3, 0.2) == pytest.approx(0.987, abs=1e-3)
