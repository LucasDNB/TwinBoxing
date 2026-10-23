import json

import numpy as np
import pytest


def test_f1_macro_conocido(trabajo):
    import evaluar_bm as ev
    real = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    # clase 0: tp1 fp1 fn1 -> 0,5; clase 1: tp2 fp1 fn0 -> 0,8; clase 2: tp1 fp0 fn1 -> 2/3
    assert ev.f1_macro(real, pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_gatillo(trabajo):
    import evaluar_bm as ev
    ok = {"veredicto": "valida"}
    assert ev.gatillo({"veredicto": "no valida"}, {"veredicto": "mejora"}, 0.9)["resultado"] == "no respaldada"
    assert ev.gatillo(ok, {"veredicto": "mejora"}, 0.6)["resultado"] == "respaldada"
    assert ev.gatillo(ok, {"veredicto": "no demostrada"}, 0.81)["resultado"] == "respaldada"
    assert ev.gatillo(ok, {"veredicto": "no concluyente"}, 0.6)["resultado"] == "no concluyente"
    assert ev.gatillo(ok, {"veredicto": "no demostrada"}, 0.6)["resultado"] == "no respaldada"


def test_v0_mediana_contra_el_piso(trabajo):
    import bm
    import evaluar_bm as ev
    for s, acierto in ((0, 1.0), (1, 0.0), (2, 1.0)):
        d = bm.TRABAJO_BM / "corridas" / "BM-0b" / "boxingweb" / f"s{s}"
        d.mkdir(parents=True)
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.eye(3)[y if acierto else (y + 1) % 3]
        np.savez(d / "pred.npz", y_tipo=y, p_tipo=p, y_altura=y % 2, p_altura=np.eye(2)[y % 2])
    r = ev.v0(bm.TRABAJO_BM)
    assert r["mediana"]["tecnica"] == 1.0 and r["veredicto"] == "valida"


def _res(total, folds, desvio=0.01):
    hook = {"recall": 0.5, "n": 10}
    return {"ensamble_total": {"familia": {"exactitud": total, "recall": {"hook": hook}}},
            "por_fold": {f: {"familia": {"exactitud": v}} for f, v in folds.items()},
            "dispersion_entre_semillas": {"familia_exactitud": {"desvio": desvio}}}


def test_comparar_aplica_la_regla_de_e4(trabajo):
    import evaluar_bm as ev
    base = _res(0.70, {f"f{i}": 0.70 for i in range(7)})
    nuevo = _res(0.74, {f"f{i}": 0.75 if i < 5 else 0.6 for i in range(7)})
    assert ev.comparar(nuevo, base, "x")["veredicto"] == "mejora"
    nuevo = _res(0.74, {f"f{i}": 0.75 if i < 4 else 0.6 for i in range(7)})
    assert ev.comparar(nuevo, base, "x")["veredicto"] == "no demostrada"
    assert ev.comparar(None, base, "x")["veredicto"] == "no realizado"


def test_bm0_lee_la_ultima_epoca(trabajo):
    import bm0
    log = ("epoch: 10\ntest_f1_score: 0.61 [np.float64(0.6), np.float64(0.7), np.float64(0.5), np.float64(0.64)]\n"
           "epoch: 11\ntest_f1_score: 0.66 [np.float64(0.69), np.float64(0.75), np.float64(0.55), np.float64(0.65)]\n")
    r = bm0.ultima_epoca(log)
    assert r["epocas_vistas"] == 2 and r["tecnica"] == 0.69 and r["objetivo"] == 0.75
    assert bm0.ultima_epoca("sin nada") is None


def test_fuera_del_repo(trabajo):
    import bm
    with pytest.raises(SystemExit):
        bm.fuera_del_repo(bm.REPO / "docs" / "x")
    assert bm.fuera_del_repo(trabajo / "afuera") == (trabajo / "afuera").resolve()
    with pytest.raises(SystemExit):
        bm.verificar_clon(trabajo / "no-existe")
