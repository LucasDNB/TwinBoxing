"""
Combinaciones: cualquier secuencia de dos o mas golpes con menos de 0,5 s entre cada uno
(pedido del 14-10, que reemplaza al izq-der-izq de B10, prototipo/specs/ampliacion-13-10/01).

La prueba de concordancia ata la funcion a lo anotado: con los golpes de las seis `annot.json`
(fijados en tests/datos por scripts/fijar_golpes_anotados.py) da 179 combinaciones con la regla
tomada. El numero se midio el 14-10 al cambiar la regla; si se mueve, cambio la regla o el
archivo fijo, no el material.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boxtwin.mvp.combinaciones import REGLA, cadenas, resumen
from boxtwin.mvp.fightcard import NO_GOLPE, aplicar_correcciones, construir, correccion
from boxtwin.mvp.sesion import Sesion

DATOS = Path(__file__).parent / "datos" / "golpes_anotados.json"


def g(brazo, t, gid=None, dur=0.2, **extra):
    return {"id": gid or f"{brazo}-{t}", "brazo": brazo, "t_inicio": t, "t_fin": t + dur, **extra}


def test_menos_de_medio_segundo_es_combinacion_y_medio_segundo_justo_no():
    assert len(cadenas([g("izq", 0.0), g("der", 0.4)])) == 1
    assert cadenas([g("izq", 0.0), g("der", 0.5)]) == []


def test_un_golpe_solo_no_es_combinacion():
    assert cadenas([g("izq", 0.0)]) == []
    assert cadenas([g("izq", 0.0), g("der", 1.0), g("izq", 2.0)]) == []


def test_cualquier_secuencia_de_brazos_cuenta():
    for brazos in (("izq", "der"), ("der", "izq"), ("izq", "izq"), ("der", "der", "izq"),
                   ("der", "izq", "der"), ("izq", "der", "izq")):
        serie = [g(b, 0.3 * k) for k, b in enumerate(brazos)]
        assert len(cadenas(serie)) == 1, brazos


def test_la_separacion_es_de_inicio_a_inicio_y_no_de_fin_a_inicio():
    # Fin a inicio daria 0,1 s y seria combinacion; inicio a inicio da 0,6 y no lo es.
    largos = [g("izq", 0.0, dur=0.5), g("der", 0.6, dur=0.5)]
    assert cadenas(largos) == []
    assert len(cadenas(largos, {**REGLA, "separacion": "fin_a_inicio"})) == 1


def test_una_serie_larga_es_una_sola_combinacion():
    serie = [g("izq", 0.0), g("der", 0.3), g("izq", 0.6), g("der", 0.9), g("izq", 1.2)]
    c = cadenas(serie)
    assert len(c) == 1 and len(c[0]) == 5


def test_un_hueco_de_medio_segundo_parte_la_serie_en_dos():
    serie = [g("izq", 0.0), g("der", 0.3), g("izq", 1.0), g("izq", 1.2)]
    assert [len(c) for c in cadenas(serie)] == [2, 2]


def test_un_golpe_sin_tipo_no_corta_y_uno_descartado_si():
    con_hueco = [g("izq", 0.0, tipo="jab"), g("der", 0.3, tipo=None), g("izq", 0.6, tipo="hook")]
    assert len(cadenas(con_hueco)[0]) == 3
    descartado = [g("izq", 0.0), g("der", 0.3, descartado=True), g("izq", 0.6)]
    assert cadenas(descartado) == []


def test_el_orden_de_entrada_no_importa():
    assert len(cadenas([g("izq", 0.8), g("izq", 0.0), g("der", 0.4)])[0]) == 3


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
    assert (c["patron"], c["separacion"], c["max_s"], c["min_golpes"]) == (
        "cualquiera", "inicio_a_inicio", 0.5, 2)
    assert c["total"] == 3
    assert c["lista"][0]["golpes"] == ["A-izq-300", "A-der-309", "A-izq-318"]
    assert c["lista"][0]["brazos"] == ["izq", "der", "izq"]
    b = fc["peleadores"]["B"]["combinaciones"]
    assert b["total"] == 1 and b["lista"][0]["brazos"] == ["der", "izq", "der"]


def test_no_golpe_en_el_medio_parte_la_combinacion(fc):
    medio = fc["peleadores"]["A"]["combinaciones"]["lista"][1]["golpes"][1]
    fc = aplicar_correcciones(fc, [correccion(medio, NO_GOLPE, fc)])
    assert fc["peleadores"]["A"]["combinaciones"]["total"] == 2, "0,8 s entre los que quedan"
    assert fc["peleadores"]["A"]["total"]["total"] == 9, "los detectados no cambian"


def test_una_correccion_de_tipo_se_ve_en_los_tipos_de_la_combinacion(fc):
    primero = fc["peleadores"]["A"]["combinaciones"]["lista"][0]["golpes"][0]
    fc = aplicar_correcciones(fc, [correccion(primero, "jab", fc)])
    assert fc["peleadores"]["A"]["combinaciones"]["lista"][0]["tipos"][0] == "jab"


def test_por_round_suma_el_total_y_la_que_cruza_va_al_round_del_primer_golpe(fc):
    c = fc["peleadores"]["A"]["combinaciones"]
    assert sum(r["total"] for r in c["por_round"]) == c["total"]
    assert [r["total"] for r in c["por_round"]] == [3, 0]
    assert c["lista"][-1]["round"] == 1


def test_una_combinacion_en_el_descanso_va_al_round_que_termino():
    rondas = [{"round": 1, "inicio_s": 0.0, "fin_s": 180.0},
              {"round": 2, "inicio_s": 200.0, "fin_s": 380.0}]
    c = resumen([g("izq", 185.0), g("der", 185.3)], rondas)
    assert c["lista"][0]["round"] == 1
    assert sum(r["total"] for r in c["por_round"]) == c["total"] == 1


def _contar(regla):
    datos = json.loads(DATOS.read_text())
    n = 0
    for fuente in datos.values():
        for p in ("A", "B"):
            suyos = [x for x in fuente["golpes"] if x["peleador"] == p and not x["abortado"]]
            n += len(cadenas(suyos, regla))
    return n


@pytest.mark.parametrize(("separacion", "esperado"),
                         [("inicio_a_inicio", 179), ("fin_a_inicio", 209)])
def test_concordancia_con_lo_anotado(separacion, esperado):
    assert _contar({**REGLA, "separacion": separacion}) == esperado
