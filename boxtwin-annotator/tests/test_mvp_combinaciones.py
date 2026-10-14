"""
B10: combinaciones izq-der-izq (prototipo/specs/ampliacion-13-10/01 §3, pruebas 1 a 6).

La prueba 6 es la que ata la funcion a lo medido: con los golpes de las seis `annot.json`
(fijados en tests/datos por scripts/fijar_golpes_anotados.py) tiene que dar los numeros de la
tabla de 01 §2, y en particular 28 con las decisiones tomadas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boxtwin.mvp.combinaciones import REGLA, resumen, ternas
from boxtwin.mvp.fightcard import NO_GOLPE, aplicar_correcciones, construir, correccion
from boxtwin.mvp.sesion import Sesion

DATOS = Path(__file__).parent / "datos" / "golpes_anotados.json"


def g(brazo, t, gid=None, dur=0.2, **extra):
    return {"id": gid or f"{brazo}-{t}", "brazo": brazo, "t_inicio": t, "t_fin": t + dur, **extra}


def test_1_menos_de_medio_segundo_es_terna_y_medio_segundo_justo_no():
    assert len(ternas([g("izq", 0.0), g("der", 0.4), g("izq", 0.8)])) == 1
    assert ternas([g("izq", 0.0), g("der", 0.5), g("izq", 0.9)]) == []


def test_la_separacion_es_de_inicio_a_inicio_y_no_de_fin_a_inicio():
    # Fin a inicio daria 0,1 s y seria terna; inicio a inicio da 0,6 y no lo es.
    largos = [g("izq", 0.0, dur=0.5), g("der", 0.6, dur=0.5), g("izq", 1.2, dur=0.5)]
    assert ternas(largos) == []
    assert len(ternas(largos, {**REGLA, "separacion": "fin_a_inicio"})) == 1


def test_2_izq_der_izq_der_izq_cuenta_una():
    serie = [g("izq", 0.0), g("der", 0.3), g("izq", 0.6), g("der", 0.9), g("izq", 1.2)]
    assert len(ternas(serie)) == 1
    assert len(ternas(serie, solapadas=True)) == 2


def test_otro_patron_no_es_combinacion():
    assert ternas([g("der", 0.0), g("izq", 0.3), g("der", 0.6)]) == []
    assert ternas([g("izq", 0.0), g("izq", 0.3), g("izq", 0.6)]) == []


def test_3_un_golpe_sin_tipo_no_corta_y_uno_descartado_si():
    con_hueco = [g("izq", 0.0, tipo="jab"), g("der", 0.3, tipo=None), g("izq", 0.6, tipo="hook")]
    assert len(ternas(con_hueco)) == 1
    descartado = [g("izq", 0.0), g("der", 0.3, descartado=True), g("izq", 0.6)]
    assert ternas(descartado) == []


def test_el_orden_de_entrada_no_importa():
    assert len(ternas([g("izq", 0.8), g("izq", 0.0), g("der", 0.4)])) == 1


class _Golpe:
    def __init__(self, peleador, brazo, t, dur=0.2, fps=30.0):
        self.peleador, self.brazo = peleador, brazo
        self.t_inicio, self.t_fin = t, t + dur
        self.inicio, self.fin = int(round(t * fps)), int(round((t + dur) * fps))
        self.score = 0.9


VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": 400.0, "total_frames": 12000}


@pytest.fixture
def fc(tmp_path):
    s = Sesion.nueva(tmp_path, VIDEO, round_s=180.0, descanso_s=20.0)
    golpes = [
        _Golpe("A", "left", 10.0), _Golpe("A", "right", 10.3), _Golpe("A", "left", 10.6),
        _Golpe("A", "left", 50.0), _Golpe("A", "right", 50.4), _Golpe("A", "left", 50.8),
        # cruza el fin del round 1 (180 s): cuenta en el round de su primer golpe
        _Golpe("A", "left", 179.8), _Golpe("A", "right", 180.1), _Golpe("A", "left", 180.4),
        _Golpe("B", "right", 30.0), _Golpe("B", "left", 30.2), _Golpe("B", "right", 30.4),
    ]
    return construir(s, golpes)


def test_la_fightcard_trae_el_bloque_con_su_regla(fc):
    c = fc["peleadores"]["A"]["combinaciones"]
    assert (c["patron"], c["separacion"], c["max_s"]) == ("izq-der-izq", "inicio_a_inicio", 0.5)
    assert c["total"] == 3
    assert c["lista"][0]["golpes"] == ["A-izq-300", "A-der-309", "A-izq-318"]
    assert fc["peleadores"]["B"]["combinaciones"]["total"] == 0, "der-izq-der no es combinacion"


def test_4_no_golpe_en_el_medio_baja_el_total_en_uno(fc):
    medio = fc["peleadores"]["A"]["combinaciones"]["lista"][1]["golpes"][1]
    fc = aplicar_correcciones(fc, [correccion(medio, NO_GOLPE, fc)])
    assert fc["peleadores"]["A"]["combinaciones"]["total"] == 2
    assert fc["peleadores"]["A"]["total"]["total"] == 9, "los detectados no cambian"


def test_una_correccion_de_tipo_se_ve_en_los_tipos_de_la_terna(fc):
    primero = fc["peleadores"]["A"]["combinaciones"]["lista"][0]["golpes"][0]
    fc = aplicar_correcciones(fc, [correccion(primero, "jab", fc)])
    assert fc["peleadores"]["A"]["combinaciones"]["lista"][0]["tipos"][0] == "jab"


def test_5_por_round_suma_el_total_y_la_terna_que_cruza_va_al_round_del_primer_golpe(fc):
    c = fc["peleadores"]["A"]["combinaciones"]
    assert sum(r["total"] for r in c["por_round"]) == c["total"]
    assert [r["total"] for r in c["por_round"]] == [3, 0]
    assert c["lista"][-1]["round"] == 1


def test_una_terna_en_el_descanso_va_al_round_que_termino():
    rondas = [{"round": 1, "inicio_s": 0.0, "fin_s": 180.0},
              {"round": 2, "inicio_s": 200.0, "fin_s": 380.0}]
    c = resumen([g("izq", 185.0), g("der", 185.3), g("izq", 185.6)], rondas)
    assert c["lista"][0]["round"] == 1
    assert sum(r["total"] for r in c["por_round"]) == c["total"] == 1


def _contar(regla, solapadas):
    datos = json.loads(DATOS.read_text())
    n = 0
    for fuente in datos.values():
        for p in ("A", "B"):
            suyos = [x for x in fuente["golpes"] if x["peleador"] == p and not x["abortado"]]
            n += len(ternas(suyos, regla, solapadas=solapadas))
    return n


@pytest.mark.parametrize(
    ("separacion", "solapadas", "esperado"),
    [("inicio_a_inicio", False, 28), ("inicio_a_inicio", True, 29),
     ("fin_a_inicio", False, 34), ("fin_a_inicio", True, 36)],
)
def test_6_concordancia_con_la_tabla_del_spec(separacion, solapadas, esperado):
    assert _contar({**REGLA, "separacion": separacion}, solapadas) == esperado


def test_6_con_las_decisiones_tomadas_son_28():
    assert _contar(REGLA, solapadas=False) == 28
