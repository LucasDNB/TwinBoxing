import numpy as np
import pytest

from boxtwin_detector.bio import Segmento
from boxtwin_detector.decodificacion import (
    HUECO_MAXIMO, LARGO_MINIMO, decodificar, probabilidad_de_golpe,
)


def logits(T=60, tramos=(), p_golpe=0.95):
    """Logits sinteticos: fondo salvo en los tramos dados."""
    import math
    alto = math.log(p_golpe / (1 - p_golpe))
    L = np.zeros((T, 3), np.float64)
    L[:, 0] = alto
    for a, b in tramos:
        L[a : b + 1, 0] = -alto
        L[a : b + 1, 2] = alto
        L[a, 1] = alto + 1.0
    return L


def test_probabilidad_de_golpe_es_uno_menos_p_de_O():
    L = np.array([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    p = probabilidad_de_golpe(L)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isclose(p[1], 2 / 3)


def test_un_tramo_activo_es_un_segmento():
    assert decodificar(logits(tramos=[(10, 20)]), 0.5) == [Segmento(10, 20, 0)]


def test_los_tramos_demasiado_cortos_se_descartan():
    # Los golpes duran 7 cuadros de mediana, p10 en 5. Dos cuadros es ruido.
    assert decodificar(logits(tramos=[(10, 12)]), 0.5) == []
    assert len(decodificar(logits(tramos=[(10, 14)]), 0.5)) == 1


def test_un_hueco_corto_no_parte_el_golpe():
    L = logits(tramos=[(10, 15), (17, 22)])   # hueco de 1 cuadro
    assert decodificar(L, 0.5) == [Segmento(10, 22, 0)]


def test_un_hueco_largo_si_lo_parte():
    L = logits(tramos=[(10, 16), (25, 32)])
    assert len(decodificar(L, 0.5)) == 2


def test_el_hueco_maximo_es_configurable():
    L = logits(tramos=[(10, 16), (22, 30)])   # hueco de 5
    assert len(decodificar(L, 0.5, hueco_maximo=2)) == 2
    assert len(decodificar(L, 0.5, hueco_maximo=6)) == 1


def test_subir_el_umbral_no_agrega_segmentos():
    # Es la perilla que el disparador heuristico no tenia.
    rng = np.random.default_rng(0)
    L = rng.normal(size=(500, 3)) * 2
    n = [len(decodificar(L, u)) for u in (0.2, 0.4, 0.6, 0.8)]
    cubiertos = [sum(s.largo for s in decodificar(L, u)) for u in (0.2, 0.4, 0.6, 0.8)]
    assert cubiertos == sorted(cubiertos, reverse=True)
    assert all(x >= 0 for x in n)


def test_valido_apaga_los_cuadros_fuera_del_tramo_anotado():
    L = logits(T=100, tramos=[(10, 20), (60, 70)])
    v = np.zeros(100, bool)
    v[50:] = True
    assert decodificar(L, 0.5, valido=v) == [Segmento(60, 70, 0)]


def test_cortar_en_b_separa_dos_golpes_pegados_del_mismo_brazo():
    L = logits(tramos=[(10, 26)])
    L[18, 1] = L[18, 2] + 2.0          # una B fuerte en el medio
    assert len(decodificar(L, 0.5)) == 1, "por defecto no corta"
    assert len(decodificar(L, 0.5, cortar_en_b=0.3)) == 2


def test_cortar_en_b_no_crea_segmentos_demasiado_cortos():
    L = logits(tramos=[(10, 26)])
    L[12, 1] = L[12, 2] + 2.0          # cortar aca dejaria un trozo de 2 cuadros
    assert len(decodificar(L, 0.5, cortar_en_b=0.3)) == 1


def test_umbral_fuera_de_rango_es_error():
    for u in (0.0, 1.0, -1.0, 2.0):
        with pytest.raises(ValueError):
            decodificar(logits(), u)


def test_sin_nada_activo_devuelve_lista_vacia():
    assert decodificar(logits(), 0.5) == []


def test_un_golpe_pegado_al_final_no_se_pierde():
    assert decodificar(logits(T=30, tramos=[(20, 29)]), 0.5) == [Segmento(20, 29, 0)]


def test_decodificar_score_usa_el_mismo_criterio_que_los_logits():
    # Es lo que hace comparable la heuristica con el modelo: mismo decodificador.
    from boxtwin_detector.decodificacion import decodificar_score
    L = logits(tramos=[(10, 20)])
    p = probabilidad_de_golpe(L)
    assert decodificar_score(p, 0.5) == decodificar(L, 0.5)


def test_decodificar_score_aplica_largo_minimo_y_huecos():
    from boxtwin_detector.decodificacion import decodificar_score
    s = np.zeros(60)
    s[10:13] = 1.0                     # muy corto
    s[30:36] = 1.0
    s[38:44] = 1.0                     # hueco de 1
    segs = decodificar_score(s, 0.5)
    assert segs == [Segmento(30, 43, 0)]
