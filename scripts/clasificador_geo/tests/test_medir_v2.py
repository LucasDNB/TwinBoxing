"""medir_v2.py (M1): guardia mas frecuente, orientacion, emparejamiento y metricas, con datos sinteticos."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("boxtwin")


def _ev(i, fighter, guard, ini, fin, tipo="straight", side="left", completeness="full"):
    return {"id": f"e{i}", "fighter": fighter, "guard": guard, "start_frame": ini, "end_frame": fin,
            "punch_type": tipo, "side": side, "completeness": completeness}


def test_guardia_mas_frecuente():
    import medir_v2 as m
    ev = [_ev(1, "fighter_A", "southpaw", 0, 5), _ev(2, "fighter_A", "southpaw", 10, 15),
          _ev(3, "fighter_A", "orthodox", 20, 25), _ev(4, "fighter_B", "orthodox", 0, 5),
          _ev(5, "fighter_B", "southpaw", 30, 35, completeness="feint")]
    assert m.guardia_mas_frecuente(ev, "fighter_A") == "zurda"
    assert m.guardia_mas_frecuente(ev, "fighter_B") == "ortodoxa"     # el amague no cuenta
    assert m.guardia_mas_frecuente(ev, "fighter_C") is None
    empate = [_ev(1, "fighter_A", "southpaw", 0, 5), _ev(2, "fighter_A", "orthodox", 10, 15)]
    assert m.guardia_mas_frecuente(empate, "fighter_A") == "ortodoxa"
    assert m.guardias_por_lado(ev, invertido=False) == {"A": "zurda", "B": "ortodoxa"}
    assert m.guardias_por_lado(ev, invertido=True) == {"A": "ortodoxa", "B": "zurda"}


def test_cuatro_clases():
    import medir_v2 as m
    assert m.cuatro_clases("straight", "lead") == "jab"
    assert m.cuatro_clases("straight", "rear") == "cross"
    assert m.cuatro_clases("hook", "rear") == "hook"


def test_emparejamiento_y_metricas():
    import medir_v2 as m
    from rol_por_brazo import emparejar_por_peleador

    def emparejar(a, b, min_iou):
        # parejas por posicion, como un emparejador trivial (el real se prueba en test_rol_por_brazo)
        return [(i, i, 1.0) for i in range(min(len(a), len(b)))], [], []

    eventos = [_ev(1, "fighter_A", "orthodox", 0, 10, "straight", "left"),     # jab
               _ev(2, "fighter_A", "southpaw", 20, 30, "straight", "left"),    # cross (zurdo, izquierda)
               _ev(3, "fighter_A", "orthodox", 40, 50, "hook", "right")]
    golpes = {"A": [{"id": "A-izq-0", "cuadro_inicio": 0, "cuadro_fin": 10, "brazo": "izq", "tipo": "jab", "tipo_clasificador": "straight"},
                    {"id": "A-izq-20", "cuadro_inicio": 20, "cuadro_fin": 30, "brazo": "izq", "tipo": "jab", "tipo_clasificador": "straight"},
                    {"id": "A-der-40", "cuadro_inicio": 40, "cuadro_fin": 50, "brazo": "der", "tipo": "uppercut", "tipo_clasificador": "uppercut"}]}
    pares = emparejar_por_peleador(golpes, eventos, False, emparejar=emparejar)
    filas = m.filas_de_pares(pares, "f")
    assert [f["cuatro"] for f in filas] == ["jab", "cross", "hook"]
    r = m.metricas(filas)
    assert r["golpes"] == 3
    assert r["familia"]["aciertos"] == 2 and r["cuatro_clases"]["aciertos"] == 1
    assert r["familia"]["recall"]["hook"]["exactitud"] == 0.0
    assert r["familia"]["recall"]["straight"]["exactitud"] == 1.0
    assert r["cuatro_clases"]["recall"]["cross"]["exactitud"] == 0.0
    assert r["familia"]["wilson95"] is not None


def test_copiar_no_toca_el_original(tmp_path):
    import medir_v2 as m
    origen = tmp_path / "orig" / "s"
    origen.mkdir(parents=True)
    for a in m.ARCHIVOS:
        (origen / a).write_text(a)
    d = m.copiar("s", "anotada", origen=tmp_path / "orig", destino=tmp_path / "m1")
    (d / "fightcard.json").write_text("cambiado")
    assert (origen / "fightcard.json").read_text() == "fightcard.json"
