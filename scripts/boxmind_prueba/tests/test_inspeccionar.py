import json
import pickle

import numpy as np


def _pose(T, rojo, azul):
    return {"pose_red_2d": rojo, "pose_blue_2d": azul,
            "pose_red_3d": np.zeros((T, 25, 3)), "pose_blue_3d": np.zeros((T, 25, 3))}


def test_comparar_identidad_detecta_cruce(trabajo):
    import inspeccionar as ins
    rng = np.random.default_rng(0)
    T = 10
    a, b = rng.uniform(0, 0.4, (T, 25, 2)), rng.uniform(0.6, 1, (T, 25, 2))
    r = ins.comparar_identidad(_pose(T, a, b), _pose(T, a, b))
    assert r["cruzado"] == 0.0 and r["identicos"] == 1.0
    ur, ub = a.copy(), b.copy()
    ur[:4], ub[:4] = b[:4], a[:4]                 # el seguidor cruzo 4 de 10 cuadros
    r = ins.comparar_identidad(_pose(T, a, b), _pose(T, ur, ub))
    assert r["cruzado"] == 0.4 and r["identicos"] == 0.6
    assert ins.comparar_identidad({"x": 1}, _pose(T, a, b)) is None


def test_main_sobre_un_paquete_sintetico(trabajo, monkeypatch):
    import importlib

    datos = trabajo / "datos" / "boxingweb"
    for part, sub in (("train", "data_train"), ("test", "data_test")):
        d = datos / sub / f"2024_{part}"
        d.mkdir(parents=True)
        (d / "x.mp4").write_bytes(b"")
        ev = [{"frame_begin": "1", "frame_end": "9", "technique": "lhook", "target": "head",
               "side": "red", "distance": "long", "effect": "effective"},
              {"frame_begin": "1", "frame_end": "2", "technique": "lhook", "target": "head",
               "side": "red", "distance": "long", "effect": "effective"}]
        (d / "video_event.json").write_text(json.dumps(ev))
        p = _pose(5, np.zeros((5, 25, 2)), np.ones((5, 25, 2)))
        for n in ("r_pose_gt.pkl", "r_uve.pkl"):
            with (d / n).open("wb") as fh:
                pickle.dump(p, fh)
    (datos.parent / "LICENSE.txt").write_text("x")
    monkeypatch.setenv("BOXTWIN_BOXINGWEB", str(datos))
    import bm
    importlib.reload(bm)
    import boxingweb
    importlib.reload(boxingweb)
    import inspeccionar
    importlib.reload(inspeccionar)
    assert inspeccionar.main() == 0
    inf = json.loads((bm.TRABAJO_BM / "inspeccion.json").read_text())
    assert inf["por_particion"] == {"train": {"rondas": 1, "eventos_filtro_largo": 1},
                                    "test": {"rondas": 1, "eventos_filtro_largo": 1}}
    assert any(t.endswith("LICENSE.txt") for t in inf["terminos_o_licencia"])
    assert inf["resumen_gt_contra_uve"]["identicos_mediana"] == 1.0
