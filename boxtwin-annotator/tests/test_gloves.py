"""
Keypoints derivados de guante.

Estos tests fijan la propiedad que mas importa y que es facil de romper sin notarlo: el
guante cae del lado de la muneca opuesto al codo, en la prolongacion del antebrazo. Un
signo invertido lo pondria dentro del brazo y el overlay se veria casi igual.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.core.constants import (
    GLOVE_LEFT_IDX,
    GLOVE_RIGHT_IDX,
    LEFT_ELBOW_IDX,
    LEFT_WRIST_IDX,
    RIGHT_ELBOW_IDX,
    RIGHT_WRIST_IDX,
)
from boxtwin.core.gloves import DEFAULT_K, append_gloves, derive_gloves


def pose_con_brazos(
    codo_izq=(100.0, 100.0), muneca_izq=(140.0, 100.0),
    codo_der=(200.0, 100.0), muneca_der=(200.0, 140.0),
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((1, 17, 2), np.float32)
    sc = np.full((1, 17), 0.9, np.float32)
    xy[0, LEFT_ELBOW_IDX] = codo_izq
    xy[0, LEFT_WRIST_IDX] = muneca_izq
    xy[0, RIGHT_ELBOW_IDX] = codo_der
    xy[0, RIGHT_WRIST_IDX] = muneca_der
    return xy, sc


def test_guante_extiende_el_antebrazo() -> None:
    """Antebrazo horizontal de 40 px con k=0,35: el guante queda 14 px mas alla."""
    xy, sc = pose_con_brazos()
    g_xy, _ = derive_gloves(xy, sc, k=DEFAULT_K)
    assert g_xy[0, 0] == pytest.approx([154.0, 100.0])


def test_guante_sigue_la_direccion_del_antebrazo() -> None:
    """El derecho apunta hacia abajo en esta pose: el guante tiene que seguir hacia abajo."""
    xy, sc = pose_con_brazos()
    g_xy, _ = derive_gloves(xy, sc, k=DEFAULT_K)
    assert g_xy[0, 1] == pytest.approx([200.0, 154.0])


def test_guante_esta_del_lado_opuesto_al_codo() -> None:
    """
    La propiedad que no se puede romper. Con el signo invertido el guante caeria dentro
    del brazo y el overlay se veria casi igual, asi que el bug pasaria desapercibido.
    """
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, (32, 17, 2)).astype(np.float32)
    sc = np.full((32, 17), 0.9, np.float32)
    g_xy, _ = derive_gloves(xy, sc)

    for lado, codo, muneca in ((0, LEFT_ELBOW_IDX, LEFT_WRIST_IDX),
                               (1, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX)):
        v_antebrazo = xy[:, muneca] - xy[:, codo]
        v_guante = g_xy[:, lado] - xy[:, muneca]
        producto = (v_antebrazo * v_guante).sum(axis=1)
        assert np.all(producto > 0), "el guante quedo del lado del codo"


def test_k_cero_deja_el_guante_en_la_muneca() -> None:
    xy, sc = pose_con_brazos()
    g_xy, _ = derive_gloves(xy, sc, k=0.0)
    assert g_xy[0, 0] == pytest.approx(xy[0, LEFT_WRIST_IDX])


def test_confianza_es_el_minimo_de_sus_insumos() -> None:
    """No se puede estar mas seguro de una extrapolacion que de los puntos que la generan."""
    xy, sc = pose_con_brazos()
    sc[0, LEFT_ELBOW_IDX] = 0.2
    sc[0, LEFT_WRIST_IDX] = 0.95
    sc[0, RIGHT_ELBOW_IDX] = 0.8
    sc[0, RIGHT_WRIST_IDX] = 0.4
    _, g_sc = derive_gloves(xy, sc)
    assert g_sc[0, 0] == pytest.approx(0.2)
    assert g_sc[0, 1] == pytest.approx(0.4)


def test_append_pone_los_guantes_en_17_y_18() -> None:
    xy, sc = pose_con_brazos()
    xy19, sc19 = append_gloves(xy, sc)
    assert xy19.shape == (1, 19, 2) and sc19.shape == (1, 19)
    # Los 17 observados no se tocan.
    np.testing.assert_array_equal(xy19[:, :17], xy)
    g_xy, _ = derive_gloves(xy, sc)
    np.testing.assert_allclose(xy19[:, GLOVE_LEFT_IDX], g_xy[:, 0])
    np.testing.assert_allclose(xy19[:, GLOVE_RIGHT_IDX], g_xy[:, 1])


def test_lote_vacio() -> None:
    xy = np.empty((0, 17, 2), np.float32)
    sc = np.empty((0, 17), np.float32)
    g_xy, g_sc = derive_gloves(xy, sc)
    assert g_xy.shape == (0, 2, 2) and g_sc.shape == (0, 2)


@pytest.mark.parametrize("forma", [(1, 16, 2), (1, 17, 3), (17, 2)])
def test_rechaza_formas_que_no_son_coco17(forma) -> None:
    xy = np.zeros(forma, np.float32)
    sc = np.zeros(forma[:2] if len(forma) == 3 else (1, 17), np.float32)
    with pytest.raises(ValueError):
        derive_gloves(xy, sc)
