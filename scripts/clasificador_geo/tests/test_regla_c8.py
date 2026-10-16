from types import SimpleNamespace as NS

import regla_c8


def test_sin_dato_cuenta_como_error():
    real = ["head", "head", "body", "body"]
    pred = ["head", "sin_dato", "body", "sin_dato"]
    m = regla_c8.metricas_altura(real, pred)
    assert m["exactitud"] == 0.5 and m["balanceada"] == 0.5 and m["sin_dato"] == 2
    assert m["recall_cuerpo"] == 0.5 and m["al_cuerpo"] == 2


def test_predecir_usa_ventana_anotada_y_tau():
    llamadas = []

    def falso(kp, sc, valid, p, side, ini, fin, tau):
        llamadas.append((p, side, ini, fin, tau))
        return NS(objetivo={10: "cabeza", 20: "cuerpo"}.get(ini))

    golpes = [{"fuente": "F", "p": 0, "side": "left", "start_frame": 10, "end_frame": 15, "ini": 2, "fin": 23},
              {"fuente": "F", "p": 1, "side": "right", "start_frame": 20, "end_frame": 25, "ini": 12, "fin": 33},
              {"fuente": "F", "p": 1, "side": "right", "start_frame": 30, "end_frame": 35, "ini": 22, "fin": 43}]
    serie = {"F": {"kp": None, "sc": None, "valid": None}}
    pred = regla_c8.predecir_regla(golpes, serie, objetivo_de=falso)
    assert pred == ["head", "body", "sin_dato"]
    assert llamadas[0] == (0, "left", 10, 15, 0.20)
