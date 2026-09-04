import numpy as np
import pytest

from boxtwin_detector.bio import Segmento
from boxtwin_detector.evaluacion import (
    alcanzables, barrer, evaluar, segmentos_anotados, sobre_pose_medida,
)


def test_prediccion_perfecta():
    gt = [[Segmento(10, 20, 0)], [Segmento(30, 40, 0)]]
    r = evaluar(gt, gt)
    assert r["recall"] == 1.0 and r["precision"] == 1.0
    assert r["error_inicio"] == 0.0 and r["iou_medio"] == 1.0


def test_sin_predicciones_el_recall_es_cero():
    r = evaluar([[], []], [[Segmento(10, 20, 0)], []])
    assert r["recall"] == 0.0 and r["golpes"] == 1
    assert r["precision"] == 0.0


def test_el_emparejamiento_es_por_carril():
    # Dos golpes simultaneos de peleadores distintos no son el mismo golpe. Emparejando
    # global, el recall se inflaria justo en los intercambios.
    gt = [[Segmento(10, 20, 0)], []]
    pred = [[], [Segmento(10, 20, 0)]]
    r = evaluar(pred, gt)
    assert r["recall"] == 0.0
    assert r["precision"] == 0.0


def test_solapamiento_insuficiente_no_empareja():
    r = evaluar([[Segmento(19, 29, 0)]], [[Segmento(10, 20, 0)]])
    assert r["emparejados"] == 0


def test_error_de_fronteras_con_signo():
    r = evaluar([[Segmento(12, 23, 0)]], [[Segmento(10, 20, 0)]])
    assert r["emparejados"] == 1
    assert r["error_inicio"] == 2.0 and r["error_fin"] == 3.0
    assert r["sesgo_inicio"] == 2.0 and r["sesgo_fin"] == 3.0


def test_un_predicho_largo_no_se_lleva_dos_golpes():
    gt = [[Segmento(10, 16, 0), Segmento(20, 26, 0)]]
    r = evaluar([[Segmento(10, 26, 0)]], gt)
    assert r["emparejados"] <= 1, "uno a uno: si no, el recall sale inflado"


def test_recall_sobre_los_alcanzables():
    labels = np.zeros((1, 100), np.int8)
    labels[0, 10] = 1; labels[0, 11:17] = 2
    labels[0, 50] = 1; labels[0, 51:57] = 2
    usable = np.ones((1, 100), bool)
    usable[0, 45:60] = False           # el segundo golpe no tiene pose
    gt = segmentos_anotados(labels)
    assert alcanzables(gt, usable) == [True, False]
    r = evaluar([[Segmento(10, 16, 0)]], gt, usable)
    assert r["recall"] == 0.5, "sobre todos los anotados"
    assert r["recall_alcanzables"] == 1.0, "sobre los que el sistema podia encontrar"


def test_carriles_desparejos_es_error():
    with pytest.raises(ValueError):
        evaluar([[]], [[], []])


def test_barrer_da_un_renglon_por_umbral():
    import math
    T = 200
    L = np.zeros((T, 3)); L[:, 0] = 3.0
    L[50:60, 0] = -3.0; L[50:60, 2] = 3.0
    labels = np.zeros((1, T), np.int8)
    labels[0, 50] = 1; labels[0, 51:60] = 2
    usable = np.ones((1, T), bool)
    filas = barrer([L], labels, usable, [0.2, 0.5, 0.8])
    assert [f["umbral"] for f in filas] == [0.2, 0.5, 0.8]
    assert all(f["recall"] == 1.0 for f in filas)


def test_barrer_respeta_la_cobertura():
    T = 300
    L = np.zeros((T, 3)); L[:, 0] = 3.0
    L[20:30, 0] = -3.0; L[20:30, 2] = 3.0      # golpe fuera de cobertura
    L[150:160, 0] = -3.0; L[150:160, 2] = 3.0
    labels = np.zeros((1, T), np.int8)
    labels[0, 150] = 1; labels[0, 151:160] = 2
    usable = np.ones((1, T), bool)
    filas = barrer([L], labels, usable, [0.5], cobertura=(100, 250))
    assert filas[0]["predichos"] == 1, "lo de afuera del tramo anotado no se cuenta"
    assert filas[0]["precision"] == 1.0


def test_la_verdad_se_recorta_a_la_region_evaluada():
    # En distribucion, el tensor de etiquetas es el de la fuente entera y solo la mascara
    # separa train de val. Sin recortar, los golpes de la mitad que no se evaluo se
    # contarian como no encontrados y el recall saldria dividido por cuatro.
    T = 400
    labels = np.zeros((1, T), np.int8)
    for s in (20, 100, 300, 350):
        labels[0, s] = 1
        labels[0, s + 1 : s + 7] = 2
    assert len(segmentos_anotados(labels)[0]) == 4
    assert len(segmentos_anotados(labels, (250, 399))[0]) == 2


def test_un_golpe_a_caballo_del_borde_no_se_cuenta():
    T = 200
    labels = np.zeros((1, T), np.int8)
    labels[0, 95] = 1
    labels[0, 96:103] = 2
    assert segmentos_anotados(labels, (100, 199))[0] == []


def test_barrer_no_castiga_por_los_golpes_de_afuera():
    T = 400
    L = np.zeros((T, 3)); L[:, 0] = 3.0
    L[300:310, 0] = -3.0; L[300:310, 2] = 3.0
    labels = np.zeros((1, T), np.int8)
    for s in (20, 100, 300):
        labels[0, s] = 1
        labels[0, s + 1 : s + 10] = 2
    usable = np.ones((1, T), bool)
    fila = barrer([L], labels, usable, [0.5], cobertura=(250, 399))[0]
    assert fila["golpes"] == 1
    assert fila["recall"] == 1.0


# -- pose medida contra pose rellenada -------------------------------------


def test_un_solo_cuadro_interpolado_contamina_el_golpe():
    # El golpe dura 7 cuadros de mediana: uno solo ya es una fraccion grande.
    I = np.zeros((1, 100), bool)
    I[0, 13] = True
    gt = [[Segmento(10, 16, 0), Segmento(50, 56, 0)]]
    assert sobre_pose_medida(gt, I) == [False, True]


def test_recall_separado_por_pose_medida_y_rellenada():
    I = np.zeros((1, 100), bool)
    I[0, 12:15] = True
    gt = [[Segmento(10, 16, 0), Segmento(50, 56, 0)]]
    r = evaluar([[Segmento(50, 56, 0)]], gt, interpolado=I)
    assert r["golpes_medidos"] == 1 and r["golpes_rellenados"] == 1
    assert r["recall_medidos"] == 1.0
    assert r["recall_rellenados"] == 0.0
    assert r["recall"] == 0.5, "el recall global sigue siendo el de siempre"


def test_sin_interpolado_no_se_reporta_la_division():
    gt = [[Segmento(10, 16, 0)]]
    r = evaluar([[Segmento(10, 16, 0)]], gt)
    assert "recall_medidos" not in r


def test_una_fuente_sin_interpolacion_reporta_todo_como_medido():
    # Es el caso de las cuatro fuentes anotadas con el flujo nuevo: 0% interpolado.
    I = np.zeros((1, 100), bool)
    gt = [[Segmento(10, 16, 0), Segmento(50, 56, 0)]]
    r = evaluar([[Segmento(10, 16, 0)]], gt, interpolado=I)
    assert r["golpes_medidos"] == 2 and r["golpes_rellenados"] == 0
    assert r["recall_medidos"] == 0.5
    assert r["recall_rellenados"] is None


def test_el_barrido_propaga_la_mascara_de_interpolacion():
    T = 200
    L = np.zeros((T, 3)); L[:, 0] = 3.0
    L[50:60, 0] = -3.0; L[50:60, 2] = 3.0
    labels = np.zeros((1, T), np.int8)
    labels[0, 50] = 1; labels[0, 51:60] = 2
    labels[0, 100] = 1; labels[0, 101:110] = 2
    usable = np.ones((1, T), bool)
    I = np.zeros((1, T), bool); I[0, 100:110] = True
    fila = barrer([L], labels, usable, [0.5], interpolado=I)[0]
    assert fila["golpes_medidos"] == 1 and fila["golpes_rellenados"] == 1
    assert fila["recall_medidos"] == 1.0, "el golpe con pose medida se encuentra"
    assert fila["recall_rellenados"] == 0.0
