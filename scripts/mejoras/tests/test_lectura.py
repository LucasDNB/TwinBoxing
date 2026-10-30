"""T11: las reglas de lectura de E8b, E10, E11 y E12 en sus bordes."""

import json

import pytest

import lectura

F6 = [f"f{i}" for i in range(6)]
F7 = [f"f{i}" for i in range(7)]


def _folds(base, dif, ganados, nombres=F6):
    b = {f: base for f in nombres}
    n = {f: base + (0.01 if i < ganados else -0.01) for i, f in enumerate(nombres)}
    return n, b, base + dif, base


@pytest.mark.parametrize("dif,ganados,esperado", [
    (0.030, 5, "mejora"), (0.029, 5, "iguala"), (0.030, 4, "no demostrada"),
    (-0.029, 0, "iguala"), (-0.030, 0, "peor"), (0.10, 6, "mejora"),
])
def test_regla_del_detector_con_tres_categorias(dif, ganados, esperado):
    n, b, tn, tb = _folds(0.6, dif, ganados)
    v = lectura.regla(tn, tb, n, b, folds_min=5, tres=True)
    assert v["veredicto"] == esperado


@pytest.mark.parametrize("dif,ganados,esperado", [
    (0.030, 5, "mejora"), (0.029, 6, "no demostrada"), (0.030, 4, "no demostrada"),
])
def test_regla_de_e8_sin_categorias(dif, ganados, esperado):
    n, b, tn, tb = _folds(0.6, dif, ganados)
    assert lectura.regla(tn, tb, n, b, folds_min=5)["veredicto"] == esperado


@pytest.mark.parametrize("desvio,esperado", [(0.030, "mejora"), (0.031, "no concluyente")])
def test_desvio_entre_semillas_de_e4(desvio, esperado):
    n, b, tn, tb = _folds(0.7, 0.04, 5, F7)
    v = lectura.regla(tn, tb, n, b, folds_min=5, desvios=[0.01, desvio], tres=True)
    assert v["veredicto"] == esperado


@pytest.mark.parametrize("dp,dc,esperado", [
    (0.030, -0.020, "mejora"), (0.030, -0.021, "sin diferencia demostrada"),
    (0.029, 0.0, "sin diferencia demostrada"), (-0.030, 0.0, "empeora"),
    (0.05, -0.051, "empeora"), (0.0, -0.050, "sin diferencia demostrada"),
])
def test_regla_de_e11(dp, dc, esperado):
    sin = {"precision": 0.80, "cobertura": 0.90}
    con = {"precision": 0.80 + dp, "cobertura": 0.90 + dc}
    assert lectura.regla_e11(con, sin)["veredicto"] == esperado


def _reporte(f1_por_fold):
    """Un folds.json minimo: el ensamble en la curva de 0,70 a 0,90 y en su mejor umbral."""
    res = []
    for f, v in f1_por_fold.items():
        curva = [{"umbral": u, "f1": v, "recall": v, "precision": v} for u in lectura.CURVA]
        res.append({"fold": f, "ensamble_curva": curva, "ensamble": {"f1": v}, "corridas": [{"f1": v}]})
    return {"resultados": res}


def test_detector_desde_los_reportes(tmp_path):
    base = _reporte({f: 0.60 for f in F6})
    nuevo = _reporte({f: (0.64 if i < 5 else 0.59) for i, f in enumerate(F6)})
    v = lectura.detector(base, nuevo, tres=True)
    assert v["veredicto"] == "mejora" and v["folds_ganados"] == 5
    assert v["nuevo"]["curva"]["0.85"] == pytest.approx(v["nuevo"]["f1_080"])


def test_main_e11_escribe_veredictos(tmp_path):
    d = tmp_path / "e11"
    d.mkdir()
    (d / "confirmacion.json").write_text(json.dumps({
        "sin_corte": {"total": {"precision": 0.80, "cobertura": 0.90}},
        "con_corte": {"total": {"precision": 0.85, "cobertura": 0.89}}}))
    assert lectura.main(["e11", "--mejoras", str(tmp_path)]) == 0
    v = json.loads((d / "veredictos.json").read_text())
    assert v["P15ab"]["veredicto"] == "mejora"


def test_main_sin_datos_no_inventa_veredictos(tmp_path):
    assert lectura.main(["e10", "--mejoras", str(tmp_path), "--e8", str(tmp_path / "e8")]) == 0
    v = json.loads((tmp_path / "e10" / "veredictos.json").read_text())
    assert v["P14a"] == {"veredicto": "no realizado"}
