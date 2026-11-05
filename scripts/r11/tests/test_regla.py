"""Pruebas de las reglas de R-11 (00-preregistro.md §3, §4, §5 y §6)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regla  # noqa: E402


def _ok(n, huecos=()):
    a = np.ones(n, bool)
    for i, j in huecos:
        a[i:j] = False
    return a


def test_tramos_saltea_los_cortos_y_recorta_el_ultimo():
    ok = _ok(1000, [(100, 110), (300, 310)])
    assert regla.elegir_tramos(ok, [], min_largo=150, total=400) == [(110, 300), (310, 520)]


def test_tramos_no_cruzan_un_corte():
    assert regla.elegir_tramos(_ok(1000), [200], min_largo=150, total=300) == [(0, 200), (200, 300)]


def test_tramos_sin_cuadros_suficientes_avisa():
    with pytest.raises(ValueError):
        regla.elegir_tramos(_ok(100), [], min_largo=50, total=200)


def test_estadisticos():
    e = regla.estadisticos(np.r_[np.full(90, 10.0), np.full(10, 50.0)])
    assert e["mediana"] == 10.0
    assert e["pct_sobre_40"] == 10.0 and e["pct_sobre_66"] == 0.0


def test_pasa_gana_el_peor_video():
    assert regla.pasa({"a": {"p95": 40.0}, "b": {"p95": 39.0}}, 40.0)
    assert not regla.pasa({"a": {"p95": 10.0}, "b": {"p95": 40.1}}, 40.0)
    assert not regla.pasa({}, 40.0)


def _e1(fallan):
    return {regla.clave(f, t, 640): {"v": {"p95": 50.0 if (f, t) in fallan else 20.0}}
            for f in regla.FORMATOS for t in regla.TAMANOS}


def test_etapa2_baja_los_que_no_pasan():
    s = regla.celdas_etapa2(_e1({("onnx", "m"), ("onnx", "l")}))
    assert set(s) == {"onnx-m-480", "onnx-m-416", "onnx-l-480", "onnx-l-416",
                      "engine-l-480", "engine-l-416"}


def test_etapa2_todos_pasan_solo_la_curva_de_l():
    assert set(regla.celdas_etapa2(_e1(set()))) == {"engine-l-480", "engine-l-416",
                                                    "onnx-l-480", "onnx-l-416"}


def _c(formato, tamano, imgsz, p95, mapa=None, mediana=None):
    return {"clave": regla.clave(formato, tamano, imgsz), "formato": formato, "tamano": tamano,
            "imgsz": imgsz, "por_video": {"v": {"p95": p95}, "w": {"p95": p95}},
            "mediana": p95 if mediana is None else mediana, "map": mapa}


def test_asignar_a_b_c_distintos():
    r = regla.asignar([_c("engine", "m", 640, 35, 0.65), _c("engine", "l", 640, 50, 0.68),
                       _c("onnx", "s", 640, 60, 0.59), _c("onnx", "l", 640, 90, 0.68)])
    assert (r["A"], r["B"], r["C"]) == ("engine-m-640", "engine-l-640", "onnx-s-640")
    assert not r["B_colapsa_en_A"] and r["C_colapsa_en"] is None and not r["provisional"]


def test_asignar_colapsa_y_no_fuerza_otra():
    r = regla.asignar([_c("engine", "l", 640, 20, 0.68), _c("engine", "m", 640, 15, 0.65)])
    assert r["A"] == r["B"] == "engine-l-640" and r["B_colapsa_en_A"]
    assert r["C"] is None


def test_asignar_vacio_no_relaja():
    r = regla.asignar([_c("engine", "l", 640, 70, 0.68)])
    assert r["A"] is None and r["B"] is None and r["C"] is None


def test_empate_de_map_gana_la_mas_rapida():
    r = regla.asignar([_c("engine", "l", 640, 30, 0.680, 25), _c("onnx", "l", 640, 35, 0.678, 30),
                       _c("engine", "l", 480, 20, 0.676, 15)])
    assert r["A"] == "engine-l-480"


def test_sin_map_es_provisional_y_ordena_por_tamano():
    r = regla.asignar([_c("engine", "m", 640, 20), _c("engine", "l", 416, 30)])
    assert r["A"] == "engine-l-416" and r["provisional"]


def test_concordancia():
    caja = np.array([[0, 0, 100, 200.0]])
    kp = np.zeros((1, 17, 2))
    sc = np.ones((1, 17))
    kp2 = kp.copy()
    kp2[0, 9] = [20, 0]
    sc2 = sc.copy()
    sc2[0, 10] = 0.1
    c = regla.concordancia([(caja, kp, sc, caja, kp2, sc2)])
    assert c["emparejadas"] == 1 and c["n_error"] == 1
    assert c["error_muneca_mediana"] == pytest.approx(0.1)
    assert c["fraccion_munecas_visibles"] == 0.5


def test_concordancia_sin_solape_no_empareja():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[50, 50, 60, 60.0]])
    z = np.zeros((1, 17, 2))
    s = np.ones((1, 17))
    assert regla.concordancia([(a, z, s, b, z, s)])["emparejadas"] == 0


def test_720p_pide_los_dos_umbrales_en_los_dos_videos():
    bien = {"fraccion_emparejada": 0.98, "error_muneca_mediana": 0.02}
    assert regla.veredicto_720p({"a": bien, "b": bien})["apto"]
    assert not regla.veredicto_720p({"a": bien, "b": {**bien, "fraccion_emparejada": 0.97}})["apto"]
    assert not regla.veredicto_720p({"a": {**bien, "error_muneca_mediana": 0.021}, "b": bien})["apto"]
    assert not regla.veredicto_720p({})["apto"]


def test_factor_4k_suma_el_transcodificado_solo_en_uno():
    f = regla.factor_4k(10.0, 80.0, 10.0, 50.0)
    assert f == {"con_transcodificado": 2.0, "sin_transcodificado": 1.8}


def test_veredicto_4k_pide_tiempo_y_pose():
    bien = {"fraccion_emparejada": 0.98, "error_muneca_mediana": 0.02}
    assert regla.veredicto_4k(2.0, bien)["sirve"]
    assert not regla.veredicto_4k(2.01, bien)["sirve"]
    v = regla.veredicto_4k(1.0, {**bien, "error_muneca_mediana": 0.03})
    assert v["entra_en_tiempo"] and not v["conserva_pose"] and not v["sirve"]

