import numpy as np
import pytest

from boxtwin_detector.bio import B, FUERA, I, Segmento, etiquetas_de, segmentos_de


def test_un_segmento_simple():
    # B-clase0 = 1, I-clase0 = 2
    carril = np.array([0, 0, 1, 2, 2, 2, 0, 0], np.int16)
    assert segmentos_de(carril) == [Segmento(2, 5, 0)]


def test_dos_golpes_pegados_sin_fondo_en_el_medio():
    # Es el caso por el que se conserva B: sin ella esto seria un solo golpe largo.
    carril = np.array([0, 1, 2, 2, 1, 2, 2, 0], np.int16)
    assert segmentos_de(carril) == [Segmento(1, 3, 0), Segmento(4, 6, 0)]


def test_clase_se_recupera_del_indice_bio():
    # B-clase3 = 7, I-clase3 = 8
    carril = np.array([0, 7, 8, 0], np.int16)
    assert segmentos_de(carril) == [Segmento(1, 2, 3)]


def test_golpe_de_un_solo_cuadro():
    carril = np.array([0, 1, 0], np.int16)
    segs = segmentos_de(carril)
    assert segs == [Segmento(1, 1, 0)]
    assert segs[0].largo == 1


def test_golpe_pegado_al_final():
    carril = np.array([0, 0, 1, 2], np.int16)
    assert segmentos_de(carril) == [Segmento(2, 3, 0)]


def test_i_sin_b_previa_no_pierde_el_golpe():
    carril = np.array([2, 2, 0], np.int16)
    assert segmentos_de(carril) == [Segmento(0, 1, 0)]


def test_ida_y_vuelta():
    segs = [Segmento(2, 5, 0), Segmento(9, 9, 4)]
    e = etiquetas_de(segs, 12)
    esperado = [FUERA, FUERA, B, I, I, I, FUERA, FUERA, FUERA, B, FUERA, FUERA]
    assert e.tolist() == esperado


def test_etiquetas_recortan_fuera_de_rango():
    e = etiquetas_de([Segmento(8, 12, 0)], 10)
    assert e[8] == B and e[9] == I
    assert e[:8].sum() == 0


def test_segmento_completamente_fuera_se_descarta():
    e = etiquetas_de([Segmento(20, 25, 0)], 10)
    assert e.sum() == 0


def test_segmento_invertido_es_error():
    with pytest.raises(ValueError):
        Segmento(5, 2, 0)


def test_segmentos_de_tambien_lee_el_espacio_O_B_I():
    # Load-bearing: el detector emite O/B/I y se recuperan segmentos con la misma funcion.
    # Funciona porque B=1 es impar e I=2 es par distinto de cero, igual que en el export.
    carril = np.array([0, 1, 2, 2, 0, 1, 2, 0], np.int8)
    assert segmentos_de(carril) == [Segmento(1, 3, 0), Segmento(5, 6, 0)]


def test_el_segmento_habla_el_idioma_del_emparejador():
    from boxtwin.core.agreement import emparejar, iou_temporal

    s = Segmento(10, 20, 0)
    assert (s.start_frame, s.end_frame) == (10, 20)
    assert iou_temporal(s.start_frame, s.end_frame, 10, 20) == 1.0
    parejas, om, ag = emparejar([Segmento(10, 20, 0)], [Segmento(12, 22, 0)])
    assert len(parejas) == 1 and not om and not ag
