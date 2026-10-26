"""Pruebas de E7 (diseno 2, puntos 1 a 5). Corren en twinboxing_env, sin hmr2 ni GPU."""

import importlib
import json

import numpy as np

import elevar3d
import features3d
from test_entrenar import _trabajo_sintetico
from test_features3d import _esqueleto


def _smpl_desde_h36m(P):
    """Esqueleto SMPL de 24 con las articulaciones del pre-registro E3 puestas desde un H36M."""
    J = np.zeros(P.shape[:-2] + (24, 3))
    for h, s in {0: 0, 1: 2, 4: 1, 8: 12, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21}.items():
        J[..., s, :] = P[..., h, :]
    return J


# 1. Mapeo SMPL -> H36M ---------------------------------------------------------------

def test_mapeo_smpl_a_h36m_y_features_iguales():
    import elevar_hmr2 as eh
    P = _esqueleto().astype(np.float32)      # pose3d.npz de E5 y de E7 se guardan en float32
    J = _smpl_desde_h36m(P)
    Q = eh.smpl_a_h36m(J)
    usadas = [0, 1, 4, 8, 11, 12, 13, 14, 15, 16]
    assert Q.shape == P.shape
    np.testing.assert_allclose(Q[:, usadas], P[:, usadas])
    assert np.all(Q[:, [2, 3, 5, 6, 7, 9, 10]] == 0)
    for side in ("left", "right"):
        np.testing.assert_allclose(features3d.features_de(Q, side, 0.5), features3d.features_de(P, side, 0.5),
                                   atol=1e-6)


def test_mapeo_respeta_izquierda_y_derecha_de_smpl():
    import elevar_hmr2 as eh
    J = np.zeros((24, 3))
    J[16], J[17], J[20], J[21] = (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)   # hombro y muneca izq / der
    Q = eh.smpl_a_h36m(J)
    assert tuple(Q[11]) == (1, 0, 0) and tuple(Q[14]) == (2, 0, 0)
    assert tuple(Q[13]) == (3, 0, 0) and tuple(Q[16]) == (4, 0, 0)


# 2. Caja ------------------------------------------------------------------------------

def test_caja_usa_solo_keypoints_con_score_suficiente():
    import elevar_hmr2 as eh
    kp = np.array([[10, 20], [30, 40], [500, 600]], np.float32)
    sc = np.array([0.9, 0.3, 0.29])
    assert eh.caja(kp, sc).tolist() == [10, 20, 30, 40]
    assert eh.caja(kp, np.zeros(3)) is None


def test_cajas_rellenan_huecos_cortos_y_enmascaran_los_largos():
    import elevar_hmr2 as eh
    T = 30
    kp = np.zeros((T, 17, 2), np.float32)
    kp[:, :, 0] = np.arange(T)[:, None] + np.arange(17)[None]
    kp[:, :, 1] = 100
    sc = np.full((T, 17), 0.9)
    valid = np.ones(T, bool)
    valid[5:8] = False                      # hueco corto: se interpola
    sc[15] = 0.1                            # cuadro presente sin keypoints utiles: hueco de 1
    valid[18:30] = False                    # hueco largo en el borde: mascara 0
    cajas, mascara = eh.cajas_de(kp, sc, valid)
    assert cajas.shape == (T, 4)
    assert mascara[:18].all() and not mascara[18:].any()
    np.testing.assert_allclose(cajas[6], [6, 100, 22, 100])
    np.testing.assert_allclose(cajas[15], [15, 100, 31, 100])


# 3. Ventana ---------------------------------------------------------------------------

def test_cuadros_de_la_ventana_coinciden_con_los_de_e5():
    import elevar_hmr2 as eh
    for fps, ini, fin in ((29.97, 101, 132), (59.94, 501, 540), (59.94, 0, 30)):
        paso = elevar3d.paso_de(fps)
        crudo_e5, _ = elevar3d.indices_contexto(ini, fin, 5000, paso)
        sel, idx = eh.cuadros_ventana(ini, fin, 5000, fps)
        assert set(sel) <= set(crudo_e5)
        assert sel[0] <= ini and sel[-1] >= fin
        assert sel[1] > ini and sel[-2] < fin            # minima: un cuadro afuera como mucho
        assert np.all(np.diff(sel) == paso)
        assert idx.min() >= 0 and np.array_equal(idx, np.clip(sel, 0, 4999))


def test_a_ventana_desde_cuadros_salteados():
    import elevar_hmr2 as eh
    sel, _ = eh.cuadros_ventana(501, 540, 5000, 59.94)
    X = np.zeros((len(sel), 17, 3), np.float32)
    X[..., 0] = sel[:, None]
    out = elevar3d.a_ventana(X, sel, 501, 540)
    assert np.allclose(out[:, 0, 0], np.linspace(501, 540, 32))


# 4. entrenar.py --exp E7 ----------------------------------------------------------------

def _features3d(dest, frame_dir, valor):
    dest.mkdir(parents=True, exist_ok=True)
    X3 = np.full((len(frame_dir), 32, 16), valor, np.float32)
    np.savez(dest / "features3d.npz", frame_dir=frame_dir, X3=X3, M3=np.ones((len(frame_dir), 32), bool))
    (dest / "features3d.meta.json").write_text(json.dumps({"features": [f"g{i}" for i in range(16)]}))


def test_entrenar_e7_lee_su_propio_3d_y_escribe_en_su_trabajo(trabajo, monkeypatch):
    _trabajo_sintetico(trabajo)
    e7 = trabajo / "e7"
    monkeypatch.setenv("BOXTWIN_E7", str(e7))
    import geo
    importlib.reload(geo)
    import entrenar
    importlib.reload(entrenar)
    fd = np.load(trabajo / "datos" / "golpes.npz")["frame_dir"]
    _features3d(trabajo / "datos", fd, 1.0)      # el de E5
    _features3d(e7 / "datos", fd, 7.0)           # el de E7
    d5 = entrenar.cargar_entrada("E5", trabajo)
    d7 = entrenar.cargar_entrada("E7", trabajo)
    assert np.all(d5["X"][..., 26:] == 1.0) and np.all(d7["X"][..., 26:] == 7.0)
    np.testing.assert_array_equal(d5["X"][..., :26], d7["X"][..., :26])
    assert entrenar.trabajo_de("E7", trabajo) == e7 and entrenar.trabajo_de("E5", trabajo) == trabajo
    out = entrenar.correr("E7", "F1", 0, d7, "cpu", entrenar.trabajo_de("E7", trabajo))
    assert out == e7 / "corridas" / "E7" / "F1" / "s0" and (out / "pred.npz").exists()
    assert not (trabajo / "corridas" / "E7").exists()


# 5. Vibracion -------------------------------------------------------------------------

def _quieto(T=32):
    P = np.zeros((T, 17, 3))
    P[:, 1], P[:, 4] = (-0.15, 0, 0), (0.15, 0, 0)       # caderas: x del cuerpo = x
    P[:, 8] = (0, 1, 0)                                   # torax: escala 1
    P[:, 11], P[:, 12], P[:, 13] = (0.2, 1, 0), (0.3, 0.8, 0.2), (0.3, 0.9, 0.5)
    P[:, 14], P[:, 15], P[:, 16] = (-0.2, 1, 0), (-0.3, 0.8, 0.2), (-0.3, 0.9, 0.5)
    return P


def test_vibracion_cero_si_quieto_y_4a_si_alterna():
    import evaluar
    importlib.reload(evaluar)
    P = _quieto()
    M = np.ones(32, bool)
    assert evaluar.vibracion(P, M, "left") == 0.0
    a = 0.01
    Q = P.copy()
    Q[:, 13, 0] += a * (-1) ** np.arange(32)
    assert np.isclose(evaluar.vibracion(Q, M, "left"), 4 * a)
    assert evaluar.vibracion(Q, M, "right") == 0.0          # la derecha no se mueve
    M2 = M.copy()
    M2[::2] = False                                          # ningun trio de pasos con 3D
    assert evaluar.vibracion(Q, M2, "left") is None
