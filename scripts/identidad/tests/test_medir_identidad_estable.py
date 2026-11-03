"""IE5: errores silenciosos, usuario simulado y la regla de §6, con verdad sintetica."""

import numpy as np
import pytest

import medir_identidad_estable as m

A, B = m.A, m.B
# La verdad: el 1 es A, el 2 es B, el 3 es A hasta el 50 y B despues (mixto), el 9 nadie.
VERDAD = {1: [(0, 100, A)], 2: [(0, 100, B)], 3: [(0, 50, A), (50, 100, B)], 9: []}
FRAMES = {t: np.arange(100) for t in (1, 2, 3, 9)}


def _item(track, desde=0, hasta=100):
    return {"track": track, "desde": desde, "hasta": hasta, "revision_pendiente": True}


def test_el_mixto_fuera_de_la_cola_es_error_silencioso():
    prop = [(1, 0, 100, "A"), (2, 0, 100, "B"), (3, 0, 100, "A")]
    e = m.errores(VERDAD, prop, [], FRAMES.get)
    assert (e["silenciosos"], e["visibles"], e["peleador"]) == (50, 0, 300)
    e = m.errores(VERDAD, prop, [_item(3)], FRAMES.get)
    assert (e["silenciosos"], e["visibles"]) == (0, 50)
    # un item que cubre solo parte del error deja el resto silencioso
    e = m.errores(VERDAD, prop, [_item(3, 0, 70)], FRAMES.get)
    assert (e["silenciosos"], e["visibles"]) == (30, 20)


def test_el_usuario_simulado_parte_el_mixto_con_dos_decisiones():
    corr, dec = m.oraculo([_item(3)], VERDAD, FRAMES.get)
    assert dec == 2
    assert [(c.desde, c.hasta, c.rol) for c in corr] == [(0, 50, "A"), (50, 100, "B")]


def test_un_item_de_un_solo_rol_es_una_decision_y_lo_ajeno_se_ignora():
    corr, dec = m.oraculo([_item(1), _item(9)], VERDAD, FRAMES.get)
    assert dec == 2 and [(c.track, c.rol) for c in corr] == [(1, "A"), (9, "ninguno")]
    # lo que ya decidio alguien no se vuelve a contestar
    hecho = {**_item(2), "revision_pendiente": False}
    assert m.oraculo([hecho], VERDAD, FRAMES.get) == ([], 0)


def test_con_el_oraculo_aplicado_la_precision_vuelve_a_uno():
    from boxtwin.mvp.revision import Tramo, aplicar_correcciones

    auto = [Tramo(1, 0, 100, "A"), Tramo(2, 0, 100, "B"), Tramo(3, 0, 100, "A")]
    corr, _ = m.oraculo([_item(3)], VERDAD, FRAMES.get)
    fin = aplicar_correcciones(auto, corr)
    r = m.por_cuadro(VERDAD, m.en_rangos(fin), FRAMES.get)
    assert r["precision"] == 1.0 and r["cobertura"] == 1.0


def _tot(sil, peor, prec, dpm):
    return {"errores": {"silenciosos": sil, "fraccion_silenciosa": sil / 10000, "peor_fuente": peor},
            "oraculo": {"precision": prec}, "decisiones_por_minuto": dpm}


def test_la_regla_elige_la_que_pasa_con_menos_decisiones_y_desempata_por_banderas():
    lec = m.regla({"base": _tot(200, 0.05, 0.999, 3.0), "corte": _tot(40, 0.01, 0.995, 2.0),
                   "corte+estricta": _tot(10, 0.003, 0.996, 2.05)})
    assert lec["veredictos"]["base"] == {"E1": False, "E2": True}
    assert lec["elegida"] == "corte" and lec["estable"]       # empate a 0,1: menos banderas
    assert lec["banderas"] == {"BOXTWIN_CORTE_MIXTOS": 1, "BOXTWIN_IDENTIDAD_ESTRICTA": 0}


def test_si_ninguna_pasa_e1_se_declara_no_estable():
    lec = m.regla({"base": _tot(900, 0.2, 0.999, 3.0), "corte": _tot(300, 0.08, 0.999, 2.0),
                   "corte+estricta": _tot(300, 0.07, 0.999, 2.5)})
    assert not lec["estable"] and lec["elegida"] == "corte"


def test_e1_mira_tambien_la_peor_fuente():
    lec = m.regla({"base": _tot(10, 0.03, 0.999, 1.0)})
    assert lec["veredictos"]["base"]["E1"] is False


def test_medir_condicion_de_punta_a_punta_sobre_evidencia_sintetica():
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, EvidenciaTrack

    rojo, azul = (175.0, 200.0, 120.0), (105.0, 200.0, 120.0)

    def tr(tid, partes, x):
        cols = [(f, c) for fs, c in partes for f in fs]
        fs = [f for f, _ in cols]
        return EvidenciaTrack(tid, fs[0], fs[-1], 0.9, len(fs), len(fs), cols, fs,
                              [(f, x) for f in fs])

    ev = {1: tr(1, [(range(0, 400, 5), rojo)], 100.0),
          2: tr(2, [(range(0, 200, 5), azul), (range(200, 400, 5), rojo)], 500.0)}
    verdad = {1: [(0, 400, A)], 2: [(0, 200, B), (200, 400, A)]}
    frames = {1: np.arange(0, 400), 2: np.arange(0, 400)}
    base = m.medir_condicion(ev, ConfigIdentidadAuto(), 400, 25.0, (1, 2), None, None, verdad,
                             frames.get, cortar=False, estricta=False)
    # el 2 entero como B: 200 cuadros mal, pero la cola los ve (conflicto o mezcla)
    assert base["errores"]["silenciosos"] == 0 and base["errores"]["visibles"] > 0
    assert base["oraculo"]["precision"] == pytest.approx(1.0)
