"""
Los dos indicadores de guardia, sobre esqueletos sinteticos.

El indicador entra al MVP condicionado a validarse contra 100 golpes marcados a mano
(criterio C3). Lo que estos tests fijan no es que acierte -eso se mide, no se testea- sino
que no confunda "la mano esta abajo" con "no se ve la mano", que es el modo de falla que
convertiria una oclusion en un descuido tactico.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.mvp.guardia import ConfigGuardia, en_guardia, medir

NARIZ, L_HOM, R_HOM, L_MUN, R_MUN = 0, 5, 6, 9, 10


class _Golpe:
    def __init__(self, peleador, brazo, inicio, fin, fps=30.0):
        self.peleador, self.brazo = peleador, brazo
        self.inicio, self.fin = inicio, fin
        self.t_inicio, self.t_fin = inicio / fps, (fin + 1) / fps
        self.score = 0.9
        self.id = f"{peleador}-{brazo}-{inicio}"


def _cuerpo(T, ancho_hombros=100.0):
    """Un cuerpo con las dos manos en guardia todo el tiempo."""
    kp = np.zeros((2, T, 17, 2), np.float32)
    sc = np.ones((2, T, 17), np.float32)
    for p in range(2):
        cx = 300.0 + p * 500.0
        kp[p, :, NARIZ] = (cx, 100.0)
        kp[p, :, L_HOM] = (cx - ancho_hombros / 2, 150.0)
        kp[p, :, R_HOM] = (cx + ancho_hombros / 2, 150.0)
        kp[p, :, L_MUN] = (cx - 20.0, 120.0)     # a 20 px de la nariz: adentro de 0,6*100
        kp[p, :, R_MUN] = (cx + 20.0, 120.0)
    return kp, sc


# -- la zona ----------------------------------------------------------------


def test_la_zona_se_mide_en_anchos_de_hombro_y_no_en_pixeles():
    # El mismo gesto, el doble de cerca de la camara. Un umbral en pixeles diria que la
    # guardia empeoro cuando lo que cambio es el zoom.
    chico, sc = _cuerpo(5, ancho_hombros=100.0)
    grande, _ = _cuerpo(5, ancho_hombros=200.0)
    grande[0, :, L_MUN] = (grande[0, 0, NARIZ][0] - 40.0, 120.0)   # el doble de lejos
    assert en_guardia(chico[0], sc[0], L_MUN, 0.6).tolist() == [1] * 5
    assert en_guardia(grande[0], sc[0], L_MUN, 0.6).tolist() == [1] * 5


def test_una_mano_lejos_de_la_nariz_esta_afuera():
    kp, sc = _cuerpo(5)
    kp[0, :, L_MUN] = (kp[0, 0, NARIZ][0] - 90.0, 300.0)
    assert en_guardia(kp[0], sc[0], L_MUN, 0.6).tolist() == [0] * 5


def test_una_mano_que_no_se_ve_no_es_una_mano_abajo():
    # El modo de falla que importa: tratar la oclusion como descuido. Son tres estados y no
    # dos justamente por esto.
    kp, sc = _cuerpo(5)
    sc[0, :, L_MUN] = 0.05
    assert en_guardia(kp[0], sc[0], L_MUN, 0.6).tolist() == [-1] * 5


# -- retorno ----------------------------------------------------------------


def test_la_mano_que_pega_y_vuelve_rapido_da_un_retorno_corto():
    kp, sc = _cuerpo(60)
    # El golpe va de 10 a 17 y la mano vuelve a la zona en el cuadro 20: 3 cuadros = 100 ms
    kp[0, 10:20, L_MUN] = (kp[0, 0, NARIZ][0] - 200.0, 120.0)
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0)[0]
    assert ev.retorno_ms == pytest.approx(100.0, abs=1.0)


def test_una_mano_que_no_vuelve_no_es_un_numero_grande():
    # Reportar "2000 ms" cuando la mano nunca volvio inventa una medicion. None y el
    # indicador en True dicen otra cosa, que es la que pasa.
    kp, sc = _cuerpo(90)
    kp[0, 10:, L_MUN] = (kp[0, 0, NARIZ][0] - 300.0, 400.0)
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0,
               cfg=ConfigGuardia(retorno_lento_ms=200.0))[0]
    assert ev.retorno_ms is None
    assert ev.retorno_lento is True


def test_sin_umbral_calibrado_se_reporta_el_tiempo_y_no_se_marca_nada():
    # El umbral sale de la mediana medida sobre sparring-3 y esa medicion es parte de C3.
    # Hasta entonces, marcar "guardia lenta" seria afirmar algo que nadie comprobo.
    kp, sc = _cuerpo(60)
    kp[0, 10:25, L_MUN] = (kp[0, 0, NARIZ][0] - 200.0, 120.0)
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0)[0]
    assert ev.retorno_ms is not None
    assert ev.retorno_lento is None


def test_el_retorno_no_se_busca_mas_alla_del_proximo_golpe_del_mismo_brazo():
    # Si el peleador vuelve a tirar, lo que pase despues es el proximo golpe y no el
    # retorno de este.
    kp, sc = _cuerpo(90)
    kp[0, 10:, L_MUN] = (kp[0, 0, NARIZ][0] - 300.0, 400.0)
    golpes = [_Golpe("A", "left", 10, 17), _Golpe("A", "left", 30, 37)]
    ev = medir(kp, sc, golpes, fps=30.0)[0]
    assert ev.retorno_ms is None


# -- mano opuesta -----------------------------------------------------------


def test_la_mano_opuesta_caida_durante_el_golpe():
    kp, sc = _cuerpo(60)
    kp[0, 10:18, L_MUN] = (kp[0, 0, NARIZ][0] - 200.0, 120.0)   # pega con la izquierda
    kp[0, 10:18, R_MUN] = (kp[0, 0, NARIZ][0] + 30.0, 400.0)    # la derecha, abajo
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0)[0]
    assert ev.fraccion_opuesta_afuera == 1.0
    assert ev.mano_opuesta_caida is True


def test_la_mano_opuesta_arriba_no_se_marca():
    kp, sc = _cuerpo(60)
    kp[0, 10:18, L_MUN] = (kp[0, 0, NARIZ][0] - 200.0, 120.0)
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0)[0]
    assert ev.fraccion_opuesta_afuera == 0.0
    assert ev.mano_opuesta_caida is False


def test_si_no_se_vio_la_mano_opuesta_no_se_afirma_nada():
    kp, sc = _cuerpo(60)
    kp[0, 10:18, L_MUN] = (kp[0, 0, NARIZ][0] - 200.0, 120.0)
    sc[0, 10:18, R_MUN] = 0.05
    ev = medir(kp, sc, [_Golpe("A", "left", 10, 17)], fps=30.0)[0]
    assert ev.mano_opuesta_caida is None
    assert "mano opuesta" in (ev.motivo or "")


def test_cada_peleador_se_mide_sobre_su_propio_cuerpo():
    kp, sc = _cuerpo(60)
    kp[1, 10:18, R_MUN] = (kp[1, 0, NARIZ][0] + 200.0, 120.0)
    kp[1, 10:18, L_MUN] = (kp[1, 0, NARIZ][0] - 30.0, 400.0)
    ev = medir(kp, sc, [_Golpe("B", "right", 10, 17)], fps=30.0)[0]
    assert ev.peleador == "B"
    assert ev.mano_opuesta_caida is True


def test_un_fps_invalido_se_rechaza():
    kp, sc = _cuerpo(10)
    with pytest.raises(ValueError, match="positivo"):
        medir(kp, sc, [], fps=0.0)
