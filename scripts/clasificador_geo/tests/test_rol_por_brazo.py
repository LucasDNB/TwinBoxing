import pytest

import rol_por_brazo as rb


@pytest.mark.parametrize("guardia,lado,rol", [
    ("orthodox", "left", "lead"), ("orthodox", "right", "rear"),
    ("southpaw", "right", "lead"), ("southpaw", "left", "rear"),
])
def test_regla_en_los_cuatro_casos(guardia, lado, rol):
    assert rb.rol_regla(guardia, lado) == rol


def _ev(i, f, ini, fin, side, tipo="straight", guard="orthodox", comp="full"):
    return {"id": f"ev_{i:04d}", "fighter": f"fighter_{f}", "start_frame": ini, "end_frame": fin, "side": side,
            "punch_type": tipo, "guard": guard, "completeness": comp}


def _g(i, ini, fin, brazo):
    return {"id": f"g{i}", "cuadro_inicio": ini, "cuadro_fin": fin, "brazo": brazo}


def test_emparejamiento_por_peleador_ve_el_brazo_cambiado():
    eventos = [_ev(1, "A", 10, 20, "left"), _ev(2, "A", 50, 60, "right"), _ev(3, "B", 10, 20, "left"),
               _ev(4, "A", 80, 90, "left", comp="feint")]
    golpes = {"A": [_g(1, 11, 20, "der"),        # brazo cambiado: anotado izquierdo
                    _g(2, 50, 61, "der"),
                    _g(3, 80, 90, "izq")],       # contra un amague: no cuenta
              "B": [_g(4, 10, 19, "izq")]}
    pares = rb.emparejar_por_peleador(golpes, eventos, invertido=False)
    assert len(pares) == 3
    r = rb.evaluar_pares(pares, {"F_ev_0001": "lead", "F_ev_0002": None, "F_ev_0003": "rear"}, "F")
    assert sorted(r["brazo"]) == sorted([("right", "left"), ("right", "right"), ("left", "left")])
    filas = {f["frame_dir"]: f for f in r["rectos"]}
    # ev 1: verdad lead (ortodoxo, izquierda); el detector dijo derecha -> la regla dice rear
    assert filas["F_ev_0001"]["verdad"] == "lead" and filas["F_ev_0001"]["regla"] == "rear"
    assert not filas["F_ev_0001"]["brazo_ok"]
    s = rb.resumen(r["rectos"], r["brazo"])
    assert s["acuerdo_brazo"]["aciertos"] == 2 and s["rectos"] == 3
    assert s["rol_regla"]["aciertos"] == 2                 # ev 2 y ev 3
    assert s["rol_clasificador"]["aciertos"] == 1          # ev 1; ev 2 dijo hook (None) = error


def test_inversion_de_a_y_b():
    eventos = [_ev(1, "A", 10, 20, "left")]
    golpes = {"B": [_g(1, 10, 20, "izq")]}
    assert len(rb.emparejar_por_peleador(golpes, eventos, invertido=True)) == 1
    assert len(rb.emparejar_por_peleador(golpes, eventos, invertido=False)) == 0


def test_veredicto_005():
    assert rb.veredicto(0.70, 0.65)["veredicto"] == "mejora"
    assert rb.veredicto(0.70, 0.651)["veredicto"] == "no demostrada"
    assert rb.veredicto(None, 0.6)["veredicto"] == "no realizado"
