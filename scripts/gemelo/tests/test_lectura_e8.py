"""S15: lectura_e8.py aplica las reglas de 02-preregistro-e8.md §3 y §4 tal como estan escritas."""

import pytest


def _fold(nombre, f1_080, f1_mejor=None, val=(2,), train=(2,), semillas=(0.5, 0.52)):
    curva = [{"umbral": u, "golpes": 100, "predichos": 80, "emparejados": 60,
              "recall": 0.6, "precision": 0.75, "f1": (f1_080 if u == 0.8 else f1_080 - 0.01)}
             for u in (0.7, 0.75, 0.8, 0.85, 0.9)]
    return {"fold": nombre, "ensamble": {"f1": f1_mejor or f1_080 + 0.01, "recall": 0.5, "precision": 0.8,
                                         "recall_medidos": 0.55, "recall_rellenados": 0.3, "umbral": 0.6},
            "ensamble_curva": curva, "boundary_version_val": list(val), "boundary_version_train": list(train),
            "corridas": [{"semilla": i, "f1": s} for i, s in enumerate(semillas)], "f1_medio": sum(semillas) / 2}


FOLDS = ["sin-01", "sin-02", "sin-03", "sin-04", "sin-p", "sin-s3"]


def _detector(f1s, extra=None):
    res = [_fold(n, v) for n, v in zip(FOLDS, f1s)]
    res.append(_fold("sin-Sparring", 0.3, val=(1,), train=(2,)))      # el no comparable
    return {"resultados": res + (extra or [])}


def test_el_fold_no_comparable_queda_afuera(gemelo):
    import lectura_e8 as le
    assert [r["fold"] for r in le.folds_comparables(_detector([0.6] * 6))] == FOLDS


def test_p12a_mejora_con_003_y_cinco_de_seis(gemelo):
    import lectura_e8 as le
    base = _detector([0.60] * 6)
    y26 = _detector([0.64, 0.64, 0.64, 0.64, 0.64, 0.59])
    p = le.p12a(base, y26, k2_pasa=True)
    assert p["base"]["f1_080"] == pytest.approx(0.60) and p["veredicto"] == "mejora"
    assert p["folds_ganados"] == 5 and p["folds"] == 6


def test_p12a_cuatro_de_seis_no_alcanza(gemelo):
    import lectura_e8 as le
    p = le.p12a(_detector([0.60] * 6), _detector([0.66] * 4 + [0.59, 0.59]), k2_pasa=True)
    assert p["veredicto"] == "no demostrada"


def test_p12a_con_k2_fallado_es_no_concluyente(gemelo):
    import lectura_e8 as le
    p = le.p12a(_detector([0.60] * 6), _detector([0.70] * 6), k2_pasa=False)
    assert p["veredicto"] == "no concluyente"


def test_p12a_lee_el_080_y_no_el_mejor_umbral(gemelo):
    import lectura_e8 as le
    base = _detector([0.60] * 6)
    for r in base["resultados"]:
        r["ensamble"]["f1"] = 0.90                                   # mejor umbral muy alto
    assert le.p12a(base, _detector([0.64] * 6), True)["veredicto"] == "mejora"


def test_k2_del_detector_contra_0635(gemelo):
    import lectura_e8 as le
    d = _detector([0.62] * 6)                                         # mejor umbral 0,63
    assert le.k2_detector(d)["pasa"] is True
    d = _detector([0.70] * 6)
    assert le.k2_detector(d)["pasa"] is False


def _e4(fam_total, por_fold, desvio=0.01, ocho=0.6):
    return {"ensamble_total": {"golpes": 1047, "familia": {"exactitud": fam_total, "recall": {"hook": 0.6, "uppercut": 0.5}},
                               "altura": {"balanceada": 0.7, "recall": {"body": 0.4}},
                               "ocho_clases": {"exactitud": ocho}},
            "por_fold": {f: {"familia": {"exactitud": v}} for f, v in por_fold.items()},
            "dispersion_entre_semillas": {"familia_exactitud": {"desvio": desvio}}}


FUENTES = ["Sparring", "01-sparring", "02-sparring", "03-sparring", "04-sparring", "sparring-3-rounds",
           "pacquiao_margarito"]


def test_p12b_usa_la_regla_de_e7(gemelo):
    import lectura_e8 as le
    base = _e4(0.727, {f: 0.70 for f in FUENTES})
    y26 = _e4(0.760, {f: (0.75 if i < 5 else 0.69) for i, f in enumerate(FUENTES)})
    p = le.p12b(base, y26)
    assert p["veredicto"] == "mejora" and p["folds_ganados"] == 5
    y26 = _e4(0.760, {f: 0.75 for f in FUENTES}, desvio=0.05)
    assert le.p12b(base, y26)["veredicto"] == "no concluyente"


def test_k2_de_e4_contra_0727(gemelo):
    import lectura_e8 as le
    assert le.k2_e4(_e4(0.74, {}))["pasa"] is True
    assert le.k2_e4(_e4(0.70, {}))["pasa"] is False


def test_p12c_direccion_declarada(gemelo):
    import lectura_e8 as le
    tot = {"base": {"A": {"cambios": 10}, "B": {"cambios": 8}, "tracks_mixtos": 4},
           "y26": {"A": {"cambios": 7}, "B": {"cambios": 9}, "tracks_mixtos": 2},
           "denominador": {"A": 100, "B": 100}, "k3": {"base": 0.8, "y26": 0.85, "diferencia": 0.05}}
    p = le.p12c({"total": tot, "fuentes": {}})
    assert p["cambios"] == {"base": 18, "y26": 16, "en_la_direccion": True}
    assert p["tracks_mixtos"]["en_la_direccion"] is True


def test_k3_marca_el_mecanismo_con_mas_de_010(gemelo):
    import lectura_e8 as le
    assert le.k3({"total": {"k3": {"base": 0.70, "y26": 0.85, "diferencia": 0.15}}})["declarar"] is True
    assert le.k3({"total": {"k3": {"base": 0.70, "y26": 0.75, "diferencia": 0.05}}})["declarar"] is False


def test_k1_y_k4_por_fuente(gemelo):
    import lectura_e8 as le
    k = le.k1_k4({"a": {"k1": {"pasa": True}, "puente": {"puente_debil": False, "iou_mediana": 0.8}},
                  "b": {"k1": {"pasa": False}, "puente": {"puente_debil": True, "iou_mediana": 0.6}}})
    assert k["K1"]["pasa"] is False and k["K1"]["fallan"] == ["b"]
    assert k["K4"]["puente_debil"] == ["b"]


def test_json_de_e4_con_la_estructura_de_la_decision_11(gemelo):
    import lectura_e8 as le
    d = le.json_e4({"E4": _e4(0.7, {}), "E4a": _e4(0.69, {})}, {"E4": _e4(0.72, {}), "E4a": _e4(0.71, {})})
    assert set(d) >= {"base", "y26"} and d["y26"]["E4a"]["ensamble_total"]["familia"]["exactitud"] == 0.71


def test_main_escribe_veredictos_con_lo_que_haya(gemelo, tmp_path):
    """El 29-10 main() fallo por un import que las pruebas de las reglas no tocaban."""
    import json
    import lectura_e8 as le
    e8 = gemelo / "e8"
    for c, f1s in (("base", [0.625] * 6), ("y26", [0.55] * 6)):    # base: K2 pasa (0,635)
        (e8 / c).mkdir(parents=True)
        (e8 / c / "folds.json").write_text(json.dumps(_detector(f1s)))
    datos = tmp_path / "datos"
    assert le.main(["--fecha", "2026-10-29", "--e8", str(e8), "--datos", str(datos),
                    "--raices", str(gemelo / "raiz-y26"), str(gemelo / "raiz-v8")]) == 0
    v = json.loads((e8 / "veredictos.json").read_text())
    assert v["P12a"]["veredicto"] == "no demostrada"
    assert (datos / "2026-10-29-e8-yolo26-detector.json").is_file()
