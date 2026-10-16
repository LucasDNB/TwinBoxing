import pickle
from types import SimpleNamespace as NS

import datos


def _ev(i, fighter="fighter_A", ini=10, fin=20, comp="full", side="left", guard="orthodox"):
    rol = "lead" if (guard == "orthodox") == (side == "left") else "rear"
    return NS(id=f"ev_{i:04d}", fighter=fighter, start_frame=ini, end_frame=fin, side=side,
              punch_type="hook", target="body", completeness=comp, guard=guard, arm_role=rol)


def test_ventana_y_recorte_en_bordes():
    assert datos.ventana(10, 20, 100) == (2, 28)
    assert datos.ventana(3, 20, 100) == (0, 28)
    assert datos.ventana(90, 97, 100) == (82, 99)


def test_amagues_afuera_y_campos():
    evs = [_ev(1), _ev(2, comp="feint"), _ev(3, comp="aborted"),
           _ev(4, fighter="fighter_B", ini=2, fin=6, side="right")]
    gs = datos.golpes_de_eventos(evs, "F1", T=50, fps=30.0)
    assert [g["event_id"] for g in gs] == ["ev_0001", "ev_0004"]
    assert gs[0]["frame_dir"] == "F1_ev_0001" and gs[0]["p"] == 0 and gs[0]["rol"] == "lead"
    assert gs[1]["p"] == 1 and gs[1]["ini"] == 0 and gs[1]["fin"] == 14 and gs[1]["rol"] == "rear"


def test_frame_dir_unicos_entre_dos_fuentes_y_freno(tmp_path, monkeypatch):
    a = datos.golpes_de_eventos([_ev(1), _ev(2)], "F1", 50, 30.0)
    b = datos.golpes_de_eventos([_ev(1), _ev(2, comp="feint")], "F2", 50, 59.94)
    gs = a + b
    cuentas = {"F1": 2, "F2": 1}
    esperados = {"F1_ev_0001", "F1_ev_0002", "F2_ev_0001"}
    assert datos.chequear(gs, esperados, cuentas, 3) == []
    # un golpe que no esta en los folds frena
    errores = datos.chequear(gs, esperados - {"F2_ev_0001"} | {"F2_ev_0009"}, cuentas, 3)
    assert any("conjunto distinto" in e for e in errores)
    # conteo por fuente distinto frena
    assert datos.chequear(gs, esperados, {"F1": 2, "F2": 2}, 3)
    # repetidos frenan
    assert any("repetidos" in e for e in datos.chequear(gs + gs[:1], esperados, cuentas, 3))
    # sin folds tambien frena
    assert datos.chequear(gs, None, cuentas, 3)


def test_frame_dirs_de_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(datos.comun, "FUENTES", {"F1": "x", "F2": "y"})
    for f, ns in (("F1", ["F1_ev_0001"]), ("F2", ["F2_ev_0001", "F2_ev_0002"])):
        with (tmp_path / f"fold_{f}.pkl").open("wb") as fh:
            pickle.dump({"split": {"train": ["otro"], "test": ns}, "annotations": []}, fh)
    assert datos.frame_dirs_de_folds(tmp_path) == {"F1_ev_0001", "F2_ev_0001", "F2_ev_0002"}


def test_csv_ida_y_vuelta(tmp_path):
    gs = datos.golpes_de_eventos([_ev(1)], "F1", 50, 29.97)
    datos.escribir_csv(gs, tmp_path / "g.csv")
    assert datos.leer_csv(tmp_path / "g.csv") == gs
