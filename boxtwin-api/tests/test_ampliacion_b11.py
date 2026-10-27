"""
B11 de la ampliacion del 13-10 del lado de la API (prototipo/specs/ampliacion-13-10/03 §5 y
08 §3): la correccion de altura y las banderas nuevas. La regla y `enriquecer` se prueban en
boxtwin-annotator/tests/test_mvp_objetivo.py y test_mvp_enriquecer.py.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video
from test_api import _recargar


@pytest.fixture
def enriquecida(cliente, registrado, entorno):
    """Una sesion con Fight-Card 0.3 ya enriquecida: un golpe de A a la cabeza."""
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    g = {"id": "A-izq-300", "t_inicio": 10.0, "t_fin": 10.2, "cuadro_inicio": 300, "cuadro_fin": 306,
         "brazo": "izq", "score": 0.9, "tipo": "jab", "confianza_tipo": 0.7, "corregido": None,
         "descartado": False, "objetivo": "cabeza", "objetivo_confianza": 0.9, "objetivo_motivo": None,
         "objetivo_h": -0.3, "objetivo_corregido": None}
    fc = {"version": "0.3", "video": {"nombre": "v.mp4", "duracion_s": 60.0, "fps": 30.0,
                                      "rounds": [{"round": 1, "inicio_s": 0.0, "fin_s": 60.0}]},
          "metodos": {"objetivo": {"regla": "altura_relativa_v1", "tau": 0.2, "medido": None}},
          "peleadores": {"A": {"golpes": [g], "total": {"total": 1, "izq": 1, "der": 0},
                               "por_round": [{"round": 1, "inicio_s": 0.0, "fin_s": 60.0, "total": 1}],
                               "descartados": 0}},
          "clasificador": {"checkpoint": "clf"}}
    (d / "fightcard.json").write_text(json.dumps(fc))
    return sid


def test_CA_14_3_corregir_la_altura_cambia_la_fila_sin_reprocesar(cliente, enriquecida, entorno):
    r = cliente.patch(f"/fightcards/{enriquecida}/golpes/A-izq-300", json={"objetivo": "cuerpo"})
    assert r.status_code == 200, r.text
    a = cliente.get(f"/fightcards/{enriquecida}").json()["peleadores"]["A"]
    assert a["golpes"][0]["objetivo"] == "cuerpo"
    assert (a["total"]["cabeza"], a["total"]["cuerpo"]) == (0, 1)
    assert a["golpes"][0]["tipo"] == "jab", "la altura no toca el tipo"
    from boxtwin_api.modelos import Correccion, Trabajo

    with entorno["db"].hacer_sesion() as db:
        c = db.query(Correccion).one()
        assert (c.campo, c.tipo, c.tipo_original) == ("objetivo", "cuerpo", "cabeza")
        assert db.query(Trabajo).filter(Trabajo.etapa != "procesar").count() == 0, "no se reprocesa"


def test_una_correccion_lleva_tipo_u_objetivo_y_no_las_dos(cliente, enriquecida):
    url = f"/fightcards/{enriquecida}/golpes/A-izq-300"
    assert cliente.patch(url, json={}).status_code == 422
    assert cliente.patch(url, json={"tipo": "jab", "objetivo": "cuerpo"}).status_code == 422
    assert cliente.patch(url, json={"objetivo": "pierna"}).status_code == 422


def test_la_correccion_de_tipo_sigue_igual_y_queda_con_campo_tipo(cliente, enriquecida, entorno):
    assert cliente.patch(f"/fightcards/{enriquecida}/golpes/A-izq-300", json={"tipo": "hook"}).status_code == 200
    from boxtwin_api.modelos import Correccion

    with entorno["db"].hacer_sesion() as db:
        assert db.query(Correccion).one().campo == "tipo"


def test_banderas_nuevas_salen_de_los_criterios(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_C8="rechazado", BOXTWIN_C8_FECHA="2026-10-13",
                  BOXTWIN_OBJETIVO_ESTIMADO="0", BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["criterios"]["C8"] == {"estado": "rechazado", "fecha": "2026-10-13"}
    assert (d["banderas"]["objetivo"], d["banderas"]["exposicion"]) == (False, False)
    c = _recargar(monkeypatch, BOXTWIN_C8="aprobado", BOXTWIN_OBJETIVO_ESTIMADO="0",
                  BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["banderas"]["objetivo"] is True
    assert d["banderas"]["exposicion"] is False, "falta medir C9"
    c = _recargar(monkeypatch, BOXTWIN_C8="aprobado", BOXTWIN_C9="rechazado", BOXTWIN_C10="aprobado",
                  BOXTWIN_OBJETIVO_ESTIMADO="0", BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["banderas"]["exposicion"] is True, "C9 medido aunque no pase: el mapa va corregido"
    assert d["banderas"]["avatar_3d"] is True


def test_la_altura_se_muestra_por_decision_de_producto_sin_tocar_c8(entorno, monkeypatch):
    # 14-10: se muestra con la regla medida aunque C8 no paso. El criterio sigue diciendo lo
    # que se midio. El mapa de exposicion sale de la misma altura y va con ella (cambios-14-10,
    # B16), corregido por el sesgo del detector.
    c = _recargar(monkeypatch, BOXTWIN_C8="rechazado", BOXTWIN_C9="rechazado",
                  BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["criterios"]["C8"]["estado"] == "rechazado"
    assert d["banderas"]["objetivo"] is True
    assert d["banderas"]["exposicion"] is True
