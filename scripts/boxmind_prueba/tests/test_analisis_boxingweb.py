import csv
import json
import pickle

import numpy as np


def _brazos(T, extiende, idx, J):
    """Pose 3D quieta salvo la muneca de `extiende` ('l' o 'r'), que se aleja del hombro."""
    hi, mi, hd, md = idx
    P = np.zeros((T, J, 3))
    P[:, hi], P[:, hd] = [-0.2, 0, 1.4], [0.2, 0, 1.4]
    P[:, mi], P[:, md] = [-0.2, 0.1, 1.2], [0.2, 0.1, 1.2]
    m = mi if extiende == "l" else md
    P[:, m, 1] += np.linspace(0, 0.5, T)
    return P


def test_mano_que_extiende(trabajo):
    import analisis_boxingweb as ab
    assert ab.mano_que_extiende(_brazos(8, "l", ab.BRAZOS_GT, 45), ab.BRAZOS_GT) == "l"
    assert ab.mano_que_extiende(_brazos(8, "r", ab.BRAZOS_GT, 45), ab.BRAZOS_GT) == "r"
    assert ab.mano_que_extiende(_brazos(8, "r", ab.BRAZOS_H36M, 17), ab.BRAZOS_H36M) == "r"
    assert ab.mano_que_extiende(_brazos(1, "r", ab.BRAZOS_GT, 45), ab.BRAZOS_GT) is None
    P = _brazos(8, "l", ab.BRAZOS_GT, 45)
    P[3, 0, 0] = np.nan
    assert ab.mano_que_extiende(P, ab.BRAZOS_GT) is None


def test_particion_cuenta_peleadores_repetidos(trabajo):
    import analisis_boxingweb as ab
    r = ab.particion({"2023-01-01_A_B_1": "train", "2023-01-01_A_B_2": "train",
                      "2024-01-01_A_C_1": "test", "2024-02-02_D_E_3": "test"})
    assert r == {"peleadores": 5, "peleadores_test": 4, "peleadores_test_en_train": 1,
                 "lista_test_en_train": ["A"], "combates_partidos": 0}
    r = ab.particion({"2023-01-01_A_B_1": "train", "2023-01-01_A_B_2": "test"})
    assert r["combates_partidos"] == 1


def _ronda(base, sub, nombre, T, eventos, poses):
    d = base / sub / nombre
    d.mkdir(parents=True)
    (d / f"{nombre}_1.mp4").write_bytes(b"")
    (d / "video_event.json").write_text(json.dumps(eventos))
    with (d / f"{nombre}_1_pose_gt.pkl").open("wb") as fh:
        pickle.dump({"pose_red_2d": np.zeros((T, 45, 2)), "pose_blue_2d": np.zeros((T, 45, 2)),
                     "pose_red_3d": poses["red"], "pose_blue_3d": poses["blue"]}, fh)


def _ev(b, f, tec, side="red"):
    return {"name": "punching", "frame_begin": str(b), "frame_end": str(f), "technique": tec,
            "target": "head", "distance": "long", "effect": "ineffective", "side": side,
            "foul": False, "type": 1}


def test_analizar_mide_base_de_tiempo_y_mano(trabajo):
    import analisis_boxingweb as ab
    base = trabajo / "bw"
    T = 40
    # ronda a 60 fps: el rojo extiende la izquierda en [10, 19]; en [20, 39] no hace nada
    rojo = np.zeros((T, 45, 3))
    rojo[10:20] = _brazos(10, "l", ab.BRAZOS_GT, 45)
    rojo[20:] = _brazos(1, "r", ab.BRAZOS_GT, 45)[0]
    rojo[20:, ab.BRAZOS_GT[1]] = rojo[20:, ab.BRAZOS_GT[3]] * [-1, 1, 1]
    rojo[:10] = rojo[10]
    ev = [{"name": "set_side", "frame_begin": "0", "frame_end": "", "technique": "", "target": "",
           "distance": "", "effect": "", "side": ""},
          _ev(10, 19, "lstraight"), _ev(30, 30, "rhook")]
    _ronda(base, "data_train", "2023-01-01_A_B_1", T, ev, {"red": rojo, "blue": rojo})
    _ronda(base, "data_test", "2024-01-01_A_C_1", T, [_ev(10, 19, "rstraight")],
           {"red": rojo, "blue": rojo})
    fps = {"2023-01-01_A_B_1_1.mp4": (60.0, 1280, 720), "2024-01-01_A_C_1_1.mp4": (24.0, 1920, 1080)}
    r = ab.analizar(base, info_video=lambda p: fps[p.name], duracion=lambda p: 120.0)
    assert r["estructura"]["particion"] == {"train": 1, "test": 1}
    assert r["estructura"]["tipo_evento"] == {"punching": 3, "set_side": 1}
    assert r["anomalias"]["largo<=0"] == 1
    assert r["duracion_ventana_por_fps"]["60"] == {"golpes": 2, "mediana_cuadros": 5.5,
                                                   "mediana_ms": 91.7, "fuera_del_filtro": 0.5}
    assert r["duracion_ventana_por_fps"]["24"]["mediana_ms"] == 416.7
    assert r["ultimo_evento_sobre_largo"]["60"] == [0.75]
    # indices tal cual: la etiqueta coincide en 60 fps (l) y no en 24 (r)
    assert r["mano_boxingweb"]["straight@60"] == {"aciertos": 1, "n": 1, "fraccion": 1.0}
    assert r["mano_boxingweb"]["straight@24"] == {"aciertos": 0, "n": 1, "fraccion": 0.0}
    assert r["mano_boxingweb"]["straight"]["n"] == 2
    assert r["particion"]["peleadores_test_en_train"] == 1


def test_calibrar_boxtwin(trabajo, tmp_path):
    import analisis_boxingweb as ab
    P3 = np.zeros((3, 6, 17, 3))
    P3[0] = _brazos(6, "l", ab.BRAZOS_H36M, 17)
    P3[1] = _brazos(6, "r", ab.BRAZOS_H36M, 17)
    P3[2] = _brazos(6, "l", ab.BRAZOS_H36M, 17)
    np.savez(tmp_path / "p.npz", frame_dir=np.array(["a", "b", "a"]), persona=np.array([0, 0, 1]),
             largo=np.array([6, 6, 6]), P3=P3)
    with (tmp_path / "g.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["frame_dir", "tipo", "side"])
        w.writeheader()
        w.writerows([{"frame_dir": "a", "tipo": "straight", "side": "left"},
                     {"frame_dir": "b", "tipo": "straight", "side": "left"}])
    assert ab.calibrar_boxtwin(tmp_path / "p.npz", tmp_path / "g.csv") == {
        "straight": {"aciertos": 1, "n": 2, "fraccion": 0.5}}
    assert ab.calibrar_boxtwin(tmp_path / "no.npz", tmp_path / "g.csv") == {}
