"""IE7: usuario simulado v2, rondas, regla y diagnostico de senales, con verdad sintetica."""

import numpy as np

import diagnostico_senales_apariencia as ds
import medir_identidad_apariencia as ie7
from boxtwin.mvp.revision import Tramo

A, B = ie7.A, ie7.B
# 1 es A todo el video; 4 es A hasta el 80 y despues un tercero (ignore); 5 es A con 3 de B.
VERDAD = {1: [(0, 100, A)], 4: [(0, 80, A), (80, 100, "ignore")],
          5: [(0, 97, A), (97, 100, B)]}
FRAMES = {t: np.arange(100) for t in VERDAD}


def _item(track, desde=0, hasta=100):
    return {"track": track, "desde": desde, "hasta": hasta, "revision_pendiente": True}


def test_el_simulado_v2_separa_al_peleador_del_tercero():
    corr, dec = ie7.oraculo_v2([_item(4)], VERDAD, FRAMES.get)
    assert dec == 2
    assert [(c.desde, c.hasta, c.rol) for c in corr] == [(0, 80, "A"), (80, 100, "ninguno")]
    # el de IE6 no lo parte: contesta la mayoria y el tercero se queda con A
    corr, dec = ie7.m.oraculo([_item(4)], VERDAD, FRAMES.get)
    assert dec == 1 and [(c.desde, c.hasta, c.rol) for c in corr] == [(0, 100, "A")]


def test_con_menos_de_cinco_cuadros_del_otro_contesta_la_mayoria():
    corr, dec = ie7.oraculo_v2([_item(5), _item(1)], VERDAD, FRAMES.get)
    assert dec == 2 and [(c.track, c.rol) for c in corr] == [(5, "A"), (1, "A")]


def test_las_rondas_siguen_mientras_la_cola_pida_algo_y_suman_decisiones():
    auto = [Tramo(1, 0, 100, "B"), Tramo(4, 0, 100, "A")]
    colas = [[_item(1)], [_item(4)], []]
    vistas = []

    def pendientes_de(corregidos):
        vistas.append([(t.track, t.rol, t.origen) for t in corregidos])
        return colas[len(vistas) - 1]

    final, vistos, dec, rondas = ie7.revisar(auto, pendientes_de, ie7.oraculo_v2, VERDAD,
                                             FRAMES.get, 3)
    assert (rondas, dec, len(vistos)) == (2, 3, 2)
    # la segunda ronda ya ve la correccion de la primera
    assert vistas[1][0] == (1, "A", "manual")
    assert [(t.track, t.desde, t.hasta, t.rol) for t in final] == [
        (1, 0, 100, "A"), (4, 0, 80, "A"), (4, 80, 100, "ninguno")]


def test_las_rondas_tienen_tope():
    auto = [Tramo(1, 0, 100, "B")]
    _, _, _, rondas = ie7.revisar(auto, lambda c: [_item(1)], ie7.oraculo_v2, VERDAD,
                                  FRAMES.get, 3)
    assert rondas == 3


def _tot(sil, peor, prec, dpm):
    return {"errores": {"silenciosos": int(sil * 1000), "fraccion_silenciosa": sil,
                        "peor_fuente": peor},
            "oraculo": {"precision": prec}, "decisiones_por_minuto": dpm}


def test_la_regla_elige_y_dice_el_estado():
    r = ie7.regla({"base": _tot(0.015, 0.12, 0.97, 22.0), "apariencia": _tot(0.004, 0.01, 0.995, 25.0)})
    assert (r["elegida"], r["estado"], r["banderas"]) == (
        "apariencia", "estable", {"BOXTWIN_IDENTIDAD_APARIENCIA": 1})
    r = ie7.regla({"base": _tot(0.015, 0.12, 0.97, 22.0), "apariencia": _tot(0.004, 0.01, 0.98, 25.0)})
    assert (r["elegida"], r["estado"]) == ("apariencia", "E1 si, E2 no")
    r = ie7.regla({"base": _tot(0.015, 0.12, 0.97, 22.0), "apariencia": _tot(0.016, 0.12, 0.98, 25.0)})
    assert (r["elegida"], r["estado"]) == ("base", "no estable")
    # empate de decisiones a 0,1: la de menos banderas
    r = ie7.regla({"base": _tot(0.004, 0.01, 0.995, 22.05), "apariencia": _tot(0.001, 0.01, 0.999, 22.0)})
    assert r["elegida"] == "base"


def test_el_diagnostico_cuenta_aciertos_por_signo_y_altura_contra_las_semillas():
    assert ds.acierto([(A, 1.0), (B, -0.5), (A, -2.0), (B, 0.0), ("ignore", 1.0)]) == {
        "n": 3, "acierto": round(2 / 3, 4)}
    alt = ds.altura_relativa([(2.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0)], True)
    assert (alt["a_mas_alto"], alt["acierto_con_semillas"]) == (0.75, 0.75)
    assert ds.altura_relativa([(2.0, 1.0)], False)["acierto_con_semillas"] == 0.0
