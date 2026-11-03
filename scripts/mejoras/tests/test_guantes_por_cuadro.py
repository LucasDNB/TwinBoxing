"""T8: a que mano va cada guante (E12 §2.3) y la extraccion de un cuadro."""

import numpy as np

import guantes_por_cuadro as g

IZQ, DER = np.array([100.0, 100.0]), np.array([300.0, 100.0])


def test_dos_guantes_cada_uno_a_su_mano():
    r = g.asignar_mano([(110, 100, 0.9), (290, 105, 0.8)], [IZQ, DER], [True, True])
    assert np.allclose(r, [[110, 100, 0.9], [290, 105, 0.8]])


def test_un_guante_va_a_la_mano_mas_cercana():
    r = g.asignar_mano([(280, 100, 0.7)], [IZQ, DER], [True, True])
    assert r[0, 2] == 0 and np.allclose(r[1], [280, 100, 0.7])


def test_ningun_guante():
    assert not g.asignar_mano([], [IZQ, DER], [True, True]).any()


def test_dos_a_la_misma_mano_queda_el_de_mayor_confianza():
    r = g.asignar_mano([(105, 100, 0.5), (95, 100, 0.9)], [IZQ, DER], [True, True])
    assert np.allclose(r[0], [95, 100, 0.9]) and r[1, 2] == 0


def test_con_una_sola_mano_esperada_todo_va_a_esa():
    r = g.asignar_mano([(290, 100, 0.6)], [IZQ, DER], [True, False])
    assert np.allclose(r[0], [290, 100, 0.6]) and r[1, 2] == 0


def test_sin_manos_esperadas_se_descarta():
    assert not g.asignar_mano([(290, 100, 0.6)], [IZQ, DER], [False, False]).any()


class _Caja:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = xyxy, conf


class _Detector:
    def __init__(self, cajas):
        self.cajas, self.llamadas = cajas, []

    def detectar_con_color(self, frame, bbox):
        self.llamadas.append(tuple(bbox))
        return [(c, (0.0, 0.0, 0.0)) for c in self.cajas]


def _pose():
    kp = np.zeros((17, 2), np.float32)
    kp[5], kp[6] = (180, 100), (220, 100)          # hombros
    kp[11], kp[12] = (185, 250), (215, 250)        # caderas: torso 150
    kp[7], kp[9] = (150, 150), (120, 150)          # codo y muneca izquierdos
    kp[8], kp[10] = (250, 150), (280, 150)         # derechos
    return kp, np.ones(17, np.float32)


def test_extraer_cuadro_por_peleador():
    kp, sc = _pose()
    det = _Detector([_Caja((100, 140, 120, 160), 0.9), _Caja((290, 140, 310, 160), 0.8),
                     _Caja((900, 900, 920, 920), 0.95)])          # lejos de las manos: ajeno
    poses = [(np.array([50, 50, 350, 400]), kp, sc), None]
    r = g.extraer_cuadro(np.zeros((10, 10, 3), np.uint8), poses, det)
    assert r.shape == (2, 2, 3)
    assert np.allclose(r[0, 0], [110, 150, 0.9]) and np.allclose(r[0, 1], [300, 150, 0.8])
    assert not r[1].any() and len(det.llamadas) == 1


def test_lista_k_mano_es_reproducible():
    arr = np.zeros((2, 100, 2, 3), np.float32)
    arr[0, ::2, 0, 2] = 0.9
    arr[1, 1::3, 1, 2] = 0.8
    a = g.lista_k_mano(arr, n=20, semilla=0)
    assert a == g.lista_k_mano(arr, n=20, semilla=0) and len(a) == 20
    assert all(arr[x["peleador"], x["cuadro"], x["mano"], 2] > 0 for x in a)


def test_cobertura_solo_sobre_cuadros_con_el_peleador(tmp_path):
    # P16c es por cuadro-carril con peleador: un video con medio round anotado no puede dar 4%
    g = np.zeros((2, 10, 2, 3), np.float32)
    g[0, :4, 0, 2] = 0.9                       # A, mano izquierda, 4 cuadros con guante
    presente = np.zeros((2, 10), bool)
    presente[0, :5] = True                     # A esta en 5 cuadros
    np.savez_compressed(tmp_path / "F.npz", guantes=g, ms=np.ones(10))
    r = g_mod().resumir(tmp_path, ["F"], {"F": presente})
    assert r["P16c"]["F"]["cobertura_por_mano"]["A-izq"] == 0.8
    assert r["P16c"]["F"]["cuadros_con_peleador"] == {"A": 5, "B": 0}


def g_mod():
    import guantes_por_cuadro
    return guantes_por_cuadro


def test_el_costo_es_sobre_cuadros_con_algun_peleador(tmp_path):
    g = np.zeros((2, 10, 2, 3), np.float32)
    ms = np.array([20.0] * 4 + [0.2] * 6)      # sin peleador la extraccion no detecta nada
    presente = np.zeros((2, 10), bool)
    presente[0, :4] = True
    np.savez_compressed(tmp_path / "F.npz", guantes=g, ms=ms)
    r = g_mod().resumir(tmp_path, ["F"], {"F": presente})
    assert r["P16d"]["mediana_ms"] == 20.0 and r["P16d"]["cuadros"] == 4
