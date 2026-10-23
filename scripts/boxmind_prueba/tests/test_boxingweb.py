from dataclasses import dataclass

import numpy as np


def _ev(b, e, tec="lstraight", tgt="head", side="red", dist="long", eff="effective"):
    return {"frame_begin": str(b) if b != "" else "", "frame_end": str(e) if e != "" else "",
            "technique": tec, "target": tgt, "side": side, "distance": dist, "effect": eff}


def test_etiquetas_mapean_al_vocabulario_de_boxtwin(trabajo):
    import boxingweb as bw
    e = bw.etiquetas(_ev(0, 10, tec="ruppercut", tgt="abdomen", side="blue", dist="close", eff="no"))
    assert e == {"tipo": "uppercut", "altura": "body", "distancia": 2, "efecto": 0, "side": "right", "p": 1}
    e = bw.etiquetas(_ev(0, 10, tec="lhook", tgt="chest", dist="medium"))
    assert (e["tipo"], e["altura"], e["distancia"], e["efecto"], e["side"], e["p"]) == ("hook", "body", 1, 1, "left", 0)
    assert bw.etiquetas(_ev(0, 10, tec="rstraight"))["tipo"] == "straight"


def test_filtro_de_largo_es_el_de_make_dataset(trabajo):
    import boxingweb as bw
    evs = [_ev(10, 13), _ev(10, 14), _ev(10, 40), _ev(10, 41), _ev("", 20), _ev(5, "")]
    assert [(int(e["frame_begin"]), int(e["frame_end"])) for e in bw.eventos_validos(evs)] == [(10, 14), (10, 40)]


def test_a_pixeles_deshace_el_cuadrado_con_relleno(trabajo):
    import boxingweb as bw
    ancho, alto = 640, 360
    px = np.array([[100.0, 50.0], [600.0, 300.0]])
    lado = max(ancho, alto)
    pad = (lado - np.array([ancho, alto])) // 2
    np.testing.assert_allclose(bw.a_pixeles((px + pad) / lado, ancho, alto), px, atol=1e-3)


def test_referencia_reordena_pose_gt(trabajo):
    import boxingweb as bw
    T = 2
    gt = np.zeros((T, 15, 2), np.float32)
    gt[:, 5] = 0.25      # hombro izquierdo en el orden de pose_gt
    gt[:, 2] = 0.75      # hombro derecho
    pose = {"pose_red_2d": gt, "pose_blue_2d": gt}
    rojo, _ = bw.referencia(pose, True, 100, 100)
    np.testing.assert_allclose(rojo[:, 16], 25.0)     # SMPL 16 = hombro izquierdo
    np.testing.assert_allclose(rojo[:, 17], 75.0)


@dataclass
class _Det:
    keypoints: np.ndarray
    kp_score: np.ndarray

    def __len__(self):
        return len(self.keypoints)


class _Cache:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def detections(self, t):
        return self.frames[t]


def _coco_en(x, y):
    kp = np.zeros((17, 2), np.float32)
    kp[5], kp[6] = (x - 20, y), (x + 20, y)
    kp[7], kp[8] = (x - 25, y + 20), (x + 25, y + 20)
    kp[11], kp[12] = (x - 10, y + 50), (x + 10, y + 50)
    return kp


def _ref_en(x, y):
    r = np.zeros((25, 2), np.float32)
    r[16], r[17] = (x - 20, y), (x + 20, y)
    r[18], r[19] = (x - 25, y + 20), (x + 25, y + 20)
    r[1], r[2] = (x - 10, y + 50), (x + 10, y + 50)
    return r


def test_emparejar_asigna_por_cercania_y_descarta_lejanos(trabajo):
    import boxingweb as bw
    sc = np.ones(17, np.float32)
    # cuadro 0: azul, distractor y rojo, en ese orden; cuadro 1: solo uno cerca del rojo
    f0 = _Det(np.stack([_coco_en(300, 100), _coco_en(500, 500), _coco_en(100, 100)]), np.stack([sc] * 3))
    f1 = _Det(np.stack([_coco_en(102, 101)]), np.stack([sc]))
    cache = _Cache([f0, f1])
    rojo = np.stack([_ref_en(100, 100), _ref_en(100, 100)])
    azul = np.stack([_ref_en(300, 100), _ref_en(300, 100)])
    kp, sc_, valid = bw.emparejar(cache, rojo, azul)
    assert valid.tolist() == [[True, True], [True, False]]
    np.testing.assert_allclose(kp[0, 0], _coco_en(100, 100))
    np.testing.assert_allclose(kp[1, 0], _coco_en(300, 100))


def test_asignar_no_repite_deteccion(trabajo):
    import boxingweb as bw
    costos = np.array([[0.1, 0.4], [0.05, 0.9]])
    assert bw.asignar(costos, 0.5) == (1, 0)


def test_golpes_de_ronda_calcula_cobertura(trabajo):
    import boxingweb as bw
    valid = np.zeros((2, 50), bool)
    valid[0, 10:13] = True
    gs = bw.golpes_de_ronda([_ev(10, 15)], "r1", "train", valid, 30.0)
    assert len(gs) == 1 and gs[0]["cobertura_atacante"] == 0.5 and gs[0]["frame_dir"] == "bw_r1_10_15_r"
