"""
La medicion del producto contra la anotacion manual.

Lo que se fija aca es la separacion que da sentido al numero: un golpe que ocurre en
cuadros sin identidad resuelta no lo puede encontrar ningun detector, y contarlo como
fallo del detector manda a trabajar sobre el modelo equivocado.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.mvp.medicion import carriles_de_annot, medir_sesion


def _annot(eventos):
    return {
        "events": [
            {"fighter": f"fighter_{f}", "side": lado, "start_frame": a, "end_frame": b}
            for f, lado, a, b in eventos
        ]
    }


def _fightcard(golpes):
    fc = {"peleadores": {"A": {"golpes": []}, "B": {"golpes": []}}}
    for f, brazo, a, b in golpes:
        fc["peleadores"][f]["golpes"].append(
            {"brazo": brazo, "cuadro_inicio": a, "cuadro_fin": b}
        )
    return fc


def _todo_visible(T=500):
    return np.ones((2, T), bool)


# -- lo basico --------------------------------------------------------------


def test_un_golpe_encontrado_cuenta_una_vez():
    r = medir_sesion(
        _fightcard([("A", "izq", 100, 109)]),
        _annot([("A", "left", 100, 109)]),
        _todo_visible(),
    )
    assert (r.anotados, r.marcas, r.emparejados) == (1, 1, 1)
    assert r.recall == 1.0 and r.precision == 1.0


def test_una_marca_en_otro_carril_no_empareja():
    # Dos golpes simultaneos de brazos distintos no son el mismo golpe.
    r = medir_sesion(
        _fightcard([("A", "der", 100, 109)]),
        _annot([("A", "left", 100, 109)]),
        _todo_visible(),
    )
    assert r.emparejados == 0


def test_una_marca_lejos_no_empareja():
    r = medir_sesion(
        _fightcard([("A", "izq", 300, 309)]),
        _annot([("A", "left", 100, 109)]),
        _todo_visible(),
    )
    assert r.emparejados == 0
    assert r.recall == 0.0 and r.precision == 0.0


# -- la separacion que importa ----------------------------------------------


def test_un_golpe_sin_identidad_no_se_le_cobra_al_detector():
    # El peleador no estaba resuelto en esos cuadros: el detector no lo pudo ver. Contarlo
    # como fallo suyo manda a trabajar sobre el modelo equivocado.
    valid = _todo_visible()
    valid[0, 100:110] = False
    r = medir_sesion(
        _fightcard([]),
        _annot([("A", "left", 100, 109)]),
        valid,
    )
    assert r.sin_identidad == 1
    assert r.visibles == 0
    assert r.recall == 0.0, "el recall de punta a punta si lo cuenta"
    assert r.recall_sobre_visible == 0.0


def test_visible_y_no_encontrado_si_es_del_detector():
    r = medir_sesion(
        _fightcard([]),
        _annot([("A", "left", 100, 109)]),
        _todo_visible(),
    )
    assert r.sin_identidad == 0
    assert r.visibles == 1
    assert r.recall_sobre_visible == 0.0


def test_el_recall_sobre_visible_descuenta_lo_invisible():
    # Cuatro golpes, dos sin identidad, uno de los visibles encontrado.
    valid = _todo_visible()
    valid[0, 300:360] = False
    r = medir_sesion(
        _fightcard([("A", "izq", 100, 109)]),
        _annot([("A", "left", 100, 109), ("A", "left", 200, 209),
                ("A", "left", 300, 309), ("A", "left", 340, 349)]),
        valid,
    )
    assert r.anotados == 4
    assert r.sin_identidad == 2 and r.visibles == 2
    assert r.recall == pytest.approx(0.25), "de punta a punta encontro uno de cuatro"
    assert r.recall_sobre_visible == pytest.approx(0.5), "de los que pudo ver, uno de dos"


def test_media_ventana_con_identidad_todavia_cuenta_como_visible():
    valid = _todo_visible()
    valid[0, 105:110] = False        # la mitad de la ventana
    r = medir_sesion(
        _fightcard([]), _annot([("A", "left", 100, 109)]), valid
    )
    assert r.visibles == 1 and r.sin_identidad == 0


# -- A y B no significan lo mismo en los dos documentos ---------------------


def test_los_roles_invertidos_se_detectan_y_no_dan_cero():
    # Los roles de la anotacion los puso una persona; los de la sesion salen de la semilla
    # que eligio el usuario. No tienen por que coincidir.
    r = medir_sesion(
        _fightcard([("B", "izq", 100, 109), ("B", "izq", 200, 209)]),
        _annot([("A", "left", 100, 109), ("A", "left", 200, 209)]),
        _todo_visible(),
    )
    assert r.invertido is True
    assert r.emparejados == 2


def test_sin_inversion_no_se_marca_invertido():
    r = medir_sesion(
        _fightcard([("A", "izq", 100, 109)]),
        _annot([("A", "left", 100, 109)]),
        _todo_visible(),
    )
    assert r.invertido is False


def test_carriles_de_annot_traduce_el_lado():
    c = carriles_de_annot(_annot([("A", "left", 1, 2), ("B", "right", 3, 4)]))
    assert set(c) == {"A-izq", "B-der"}


# -- sin mascara ------------------------------------------------------------


def test_sin_mascara_se_mide_igual_pero_no_se_separa():
    r = medir_sesion(
        _fightcard([("A", "izq", 100, 109)]),
        _annot([("A", "left", 100, 109)]),
        None,
    )
    assert r.emparejados == 1
    assert r.visibles == 0 and r.sin_identidad == 0
