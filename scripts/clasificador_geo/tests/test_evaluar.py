import numpy as np
import pytest

import evaluar

F7 = [f"f{i}" for i in range(7)]


def _golpes(n=24):
    tipos = ["straight", "hook", "uppercut"]
    return [{"frame_dir": f"x_{i}", "fuente": "Sparring", "tipo": tipos[i % 3],
             "altura": "body" if i % 4 == 0 else "head", "rol": "lead" if i % 2 else "rear"} for i in range(n)]


def _onehot(idx, k):
    return np.eye(k)[idx]


def test_predicciones_perfectas_dan_uno():
    gs = _golpes()
    pt = _onehot([evaluar.TIPOS.index(g["tipo"]) for g in gs], 3)
    pa = _onehot([evaluar.ALTURAS.index(g["altura"]) for g in gs], 2)
    m = evaluar.metricas(pt, pa, gs)
    for b in ("familia", "altura", "ocho_clases"):
        assert m[b]["exactitud"] == 1.0 and m[b]["balanceada"] == 1.0
    assert m["altura"]["recall_cuerpo"] == 1.0


def test_ocho_clases_parte_el_recto_por_rol():
    gs = [{"tipo": "straight", "altura": "head", "rol": "lead"}, {"tipo": "straight", "altura": "body", "rol": "rear"}]
    assert evaluar.clases_ocho([g["tipo"] for g in gs], [g["altura"] for g in gs], [g["rol"] for g in gs]) == ["jab-head", "cross-body"]


def _folds(base, gana):
    return {f: base + (0.01 if i < gana else -0.01) for i, f in enumerate(F7)}


@pytest.mark.parametrize("dif,gana,esperado", [
    (0.029, 7, "no demostrada"),
    (0.031, 4, "no demostrada"),
    (0.031, 5, "mejora"),
    (0.030, 5, "mejora"),
])
def test_regla_003_y_5_de_7(dif, gana, esperado):
    base = {f: 0.6 for f in F7}
    v = evaluar.veredicto_mejora(0.6 + dif, 0.6, _folds(0.6, gana), base)
    assert v["veredicto"] == esperado and v["folds_ganados"] == gana


def test_desvio_entre_semillas_alto_da_no_concluyente():
    base = {f: 0.6 for f in F7}
    v = evaluar.veredicto_mejora(0.65, 0.6, _folds(0.6, 6), base, desvios=[0.031])
    assert v["veredicto"] == "no concluyente"
    v = evaluar.veredicto_mejora(0.65, 0.6, _folds(0.6, 6), base, desvios=[0.029])
    assert v["veredicto"] == "mejora"
    # sin mejora, el desvio no la convierte en no concluyente
    v = evaluar.veredicto_mejora(0.61, 0.6, _folds(0.6, 6), base, desvios=[0.2])
    assert v["veredicto"] == "no demostrada"


def test_regla_de_altura_3_de_4():
    base = {f: 0.7 for f in F7[:4]}
    assert evaluar.veredicto_mejora(0.74, 0.7, _folds(0.7, 3), base, 3)["veredicto"] == "mejora"
    assert evaluar.veredicto_mejora(0.74, 0.7, _folds(0.7, 2), base, 3)["veredicto"] == "no demostrada"


def test_lectura_ocho_clases():
    assert evaluar.lectura_ocho({"E4": 0.79, "E5": 0.5})["lectura"] == "a"
    assert evaluar.lectura_ocho({"E4": 0.80})["lectura"] == "b"
    assert evaluar.lectura_ocho({})["lectura"] == "no realizado"


def test_agrupar_e3_por_tipo_y_altura():
    clases = ["straight-head", "straight-body", "hook-head", "hook-body", "uppercut-head", "uppercut-body"]
    P = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
    pt, pa = evaluar.agrupar_e3(P, clases)
    assert np.allclose(pt, [[0.3, 0.4, 0.3]]) and np.allclose(pa, [[0.6, 0.4]])


def test_leer_corridas_ensambla_y_frena(trabajo, monkeypatch):
    monkeypatch.setattr(evaluar, "FUENTES", ["A", "B"])
    gs = [{"frame_dir": f"{f}_{i}", "fuente": f, "tipo": "hook", "altura": "head", "rol": "lead"}
          for f in ("A", "B") for i in range(3)]
    for f in ("A", "B"):
        for s in (0, 1):
            d = trabajo / "corridas" / "E4" / f / f"s{s}"
            d.mkdir(parents=True)
            fd = [f"{f}_{i}" for i in (2, 0, 1)]          # orden distinto al de golpes.csv
            pt = np.tile([0.1, 0.8, 0.1] if s == 0 else [0.5, 0.4, 0.1], (3, 1))
            np.savez(d / "pred.npz", frame_dir=np.array(fd), p_tipo=pt, p_altura=np.tile([0.9, 0.1], (3, 1)))
    c = evaluar.leer_corridas("E4", gs, trabajo, semillas=(0, 1))
    assert not c["faltan"]
    r = evaluar.evaluar_probs(c["preds"], gs, semillas=(0, 1))
    assert r["ensamble_total"]["familia"]["exactitud"] == 1.0       # promedio: hook 0,6 gana
    assert r["por_semilla_total"]["s1"]["familia"]["exactitud"] == 0.0
    assert r["dispersion_entre_semillas"]["familia_exactitud"]["desvio"] > 0.5
    # un pred.npz con golpes de otra fuente frena
    d = trabajo / "corridas" / "E4" / "B" / "s0"
    np.savez(d / "pred.npz", frame_dir=np.array(["A_0", "B_0", "B_1"]), p_tipo=np.ones((3, 3)), p_altura=np.ones((3, 2)))
    with pytest.raises(SystemExit):
        evaluar.leer_corridas("E4", gs, trabajo, semillas=(0, 1))
