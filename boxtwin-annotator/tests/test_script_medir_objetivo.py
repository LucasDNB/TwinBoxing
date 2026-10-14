"""Las cuentas de C8 (ampliacion 03 §4): metrica estricta, linea de base y eleccion de tau."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "medir_objetivo", Path(__file__).resolve().parents[1] / "scripts" / "medir_objetivo.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_todo_cabeza_da_balanceada_0_5_aunque_la_simple_sea_alta():
    pares = [("cabeza", "cabeza")] * 83 + [("cuerpo", "cabeza")] * 17
    r = m.metricas(pares)
    assert r["estricta"]["exactitud_simple"] == 0.83
    assert r["estricta"]["exactitud_balanceada"] == 0.5
    assert r["linea_de_base_todo_cabeza"] == 0.5


def test_sin_dato_es_error_en_la_estricta_y_no_entra_en_con_dato():
    pares = [("cabeza", "cabeza"), ("cabeza", None), ("cuerpo", "cuerpo"), ("cuerpo", None)]
    r = m.metricas(pares)
    assert r["estricta"]["recall_cabeza"] == 0.5 and r["estricta"]["recall_cuerpo"] == 0.5
    assert r["con_dato"]["exactitud_balanceada"] == 1.0
    assert r["sin_dato"] == 2
    assert r["confusion"]["cuerpo"] == {"cabeza": 0, "cuerpo": 1, "None": 1}


def test_pasa_necesita_las_dos_condiciones():
    bien = [("cabeza", "cabeza")] * 8 + [("cuerpo", "cuerpo")] * 6 + [("cuerpo", "cabeza")] * 4
    assert m.pasa(m.metricas(bien)), "balanceada 0,8 y cuerpo 0,6"
    poco_cuerpo = [("cabeza", "cabeza")] * 10 + [("cuerpo", "cuerpo")] * 5 + [("cuerpo", "cabeza")] * 5
    assert not m.pasa(m.metricas(poco_cuerpo)), "balanceada 0,75 pero cuerpo 0,5"


def test_elegir_tau_desempata_por_recall_de_cuerpo_y_despues_por_el_menor():
    def fila(bal, cue):
        return {"estricta": {"exactitud_balanceada": bal, "recall_cuerpo": cue}}
    assert m.elegir_tau({0.1: fila(0.7, 0.5), 0.2: fila(0.8, 0.5)}) == 0.2
    assert m.elegir_tau({0.1: fila(0.8, 0.5), 0.2: fila(0.8, 0.6)}) == 0.2
    assert m.elegir_tau({0.1: fila(0.8, 0.6), 0.2: fila(0.8, 0.6)}) == 0.1


def test_wilson_del_recall_de_cuerpo():
    lo, hi = m.wilson(80, 132)
    assert hi - lo == pytest.approx(0.164, abs=0.01), "con 132 golpes ronda +-0,08"


def test_las_particiones_son_las_del_spec():
    assert m.CALIBRACION == ("01-sparring", "02-sparring")
    assert set(m.MEDICION) == {"03-sparring", "04-sparring", "sparring-3", "pacquiao_margarito"}
    assert not set(m.CALIBRACION) & set(m.MEDICION)


def test_nivel2_empareja_en_la_mejor_orientacion_y_mide_la_altura_de_lo_encontrado():
    annot = {"events": [
        {"fighter": "fighter_A", "side": "left", "start_frame": 10, "end_frame": 16, "target": "head"},
        {"fighter": "fighter_A", "side": "left", "start_frame": 40, "end_frame": 46, "target": "body"},
        {"fighter": "fighter_B", "side": "right", "start_frame": 70, "end_frame": 76, "target": "head"},
    ]}
    # La sesion tiene los roles al reves: el A anotado es su B.
    fc = {"peleadores": {
        "A": {"golpes": [{"brazo": "der", "cuadro_inicio": 70, "cuadro_fin": 76, "objetivo": "cuerpo"}]},
        "B": {"golpes": [{"brazo": "izq", "cuadro_inicio": 11, "cuadro_fin": 16, "objetivo": "cabeza"},
                         {"brazo": "izq", "cuadro_inicio": 100, "cuadro_fin": 106, "objetivo": "cuerpo"}]},
    }}
    pares, extra = m.pares_nivel2(fc, annot)
    assert extra["roles_invertidos"] is True
    assert sorted(pares) == [("cabeza", "cabeza"), ("cabeza", "cuerpo")]
    assert extra["recall_detector_por_altura"]["cuerpo"] == {"anotados": 1, "encontrados": 0, "recall": 0.0}
    assert extra["recall_detector_por_altura"]["cabeza"]["recall"] == 1.0
