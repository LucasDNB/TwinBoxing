"""
B16 de los cambios del 14-10 (prototipo/specs/cambios-14-10/01 §2): distribucion por mano y
altura, y exposicion dirigida con la correccion por sesgo del detector.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video
from test_api import _recargar

from boxtwin_api.zonas import RECALL_ALTURA, bloque, contar_zonas, corregir


def _g(i, brazo, objetivo, **extra):
    return {"id": f"g{i}", "t_inicio": float(i), "t_fin": i + 0.2, "brazo": brazo,
            "tipo": None, "corregido": None, "objetivo": objetivo, **extra}


def _fc(a, b):
    return {
        "version": "0.3", "video": {"duracion_s": 180.0, "rounds": []},
        "identidad": {"cobertura_A": 1.0, "cobertura_B": 1.0},
        "detector": {"checkpoint": "det"}, "clasificador": {"checkpoint": None},
        "peleadores": {"A": {"golpes": a, "por_round": [], "total": {"total": len(a)}},
                       "B": {"golpes": b, "por_round": [], "total": {"total": len(b)}}},
    }


def test_contar_zonas_suma_con_sin_dato_y_deja_afuera_lo_descartado():
    golpes = [_g(1, "izq", "cabeza"), _g(2, "der", "cuerpo"), _g(3, "izq", None),
              _g(4, "der", "cabeza", descartado=True), _g(5, "left", "cuerpo")]
    c = contar_zonas(golpes)
    assert c["zonas"] == {"cabeza_mano_izq": 1, "cabeza_mano_der": 0,
                          "cuerpo_mano_izq": 1, "cuerpo_mano_der": 1}
    assert sum(c["zonas"].values()) + c["sin_dato"] == c["total"] == 4, "CA-15.1"
    assert c["manos"] == {"izq": 3, "der": 1}, "la mano cuenta aunque no haya altura"


def test_la_correccion_pesa_el_cuerpo_por_su_recall_mas_bajo():
    f = corregir({"cabeza_mano_izq": 50, "cabeza_mano_der": 0, "cuerpo_mano_izq": 50,
                  "cuerpo_mano_der": 0})
    esperado = (50 / RECALL_ALTURA["cuerpo"]) / (50 / RECALL_ALTURA["cabeza"] + 50 / RECALL_ALTURA["cuerpo"])
    assert f["cuerpo_mano_izq"] == pytest.approx(esperado, abs=1e-4)
    assert f["cuerpo_mano_izq"] == pytest.approx(0.595, abs=0.001)


def test_con_recalls_parejos_no_se_corrige():
    f = corregir({"cabeza_mano_izq": 50, "cabeza_mano_der": 0, "cuerpo_mano_izq": 50,
                  "cuerpo_mano_der": 0}, {"cabeza": 0.5, "cuerpo": 0.45})
    assert f["cuerpo_mano_izq"] == 0.5
    assert bloque([], {"cabeza": 0.5, "cuerpo": 0.45})["sesgo_detector"]["corregido"] is False


@pytest.fixture
def ana(cliente, registrado):
    return cliente.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]


def _sesion(cliente, entorno, fc, a=None, b=None, contexto="sparring"):
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": a, "boxeador_b": b})
    return sid


def test_el_perfil_trae_ataque_y_exposicion_de_lo_que_le_tiran(cliente, entorno, ana):
    beto = cliente.post("/boxeadores", json={"nombre": "Beto"}).json()["id"]
    propios = [_g(1, "izq", "cabeza"), _g(2, "izq", "cuerpo"), _g(3, "der", "cabeza")]
    ajenos = [_g(10, "der", "cuerpo"), _g(11, "der", "cuerpo"), _g(12, "izq", None),
              _g(13, "izq", "cabeza", descartado=True)]
    _sesion(cliente, entorno, _fc(propios, ajenos), a=ana, b=beto)
    p = cliente.get(f"/boxeadores/{ana}/perfil").json()
    assert p["ataque"]["zonas"]["cabeza_mano_izq"]["n"] == 1
    assert p["ataque"]["manos"] == {"izq": 2, "der": 1}
    e = p["exposicion"]
    assert e["tipo"] == "dirigidos"
    assert e["zonas"]["cuerpo_mano_der"]["n"] == 2
    assert sum(z["n"] for z in e["zonas"].values()) + e["sin_dato"] == e["total"] == 3
    assert e["por_rival"] == [{"boxeador_id": beto, "nombre": "Beto", "total": 3}]
    assert e["sesgo_detector"]["corregido"] is True


def test_en_bolsa_no_hay_exposicion(cliente, entorno, ana):
    fc = _fc([_g(1, "izq", None)], [])
    del fc["peleadores"]["B"]
    _sesion(cliente, entorno, fc, a=ana, contexto="bolsa")
    p = cliente.get(f"/boxeadores/{ana}/perfil").json()
    assert p["exposicion"] is None


def test_con_la_altura_apagada_no_hay_zonas_pero_si_manos(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_OBJETIVO_ESTIMADO="0", BOXTWIN_INVITACION="abierto")
    t = c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"}).json()["token"]
    c.headers["Authorization"] = f"Bearer {t}"
    ana = c.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    _sesion(c, entorno, _fc([_g(1, "izq", "cabeza")], [_g(2, "der", "cuerpo")]), a=ana)
    p = c.get(f"/boxeadores/{ana}/perfil").json()
    assert p["ataque"]["zonas"] is None and p["ataque"]["manos"] == {"izq": 1, "der": 0}
    assert p["exposicion"] is None
    assert c.get("/producto").json()["banderas"]["exposicion"] is False


# -- B17: distancia de trabajo y estilo (cambios-14-10/01 §3) ------------------------------

def test_B17_distancia_y_estilo_se_cargan_editan_y_llegan_al_perfil(cliente, registrado):
    r = cliente.post("/boxeadores", json={"nombre": "Ceci", "distancia": "corta", "estilo": "fajador"})
    assert r.status_code == 201, r.text
    bid = r.json()["id"]
    assert (r.json()["distancia"], r.json()["estilo"]) == ("corta", "fajador")
    r = cliente.patch(f"/boxeadores/{bid}", json={"nombre": "Ceci", "distancia": "larga", "estilo": "estilista"})
    assert (r.json()["distancia"], r.json()["estilo"]) == ("larga", "estilista")
    b = cliente.get(f"/boxeadores/{bid}/perfil").json()["boxeador"]
    assert (b["distancia"], b["estilo"]) == ("larga", "estilista")
    lista = cliente.get("/boxeadores").json()
    assert lista[0]["distancia"] == "larga"


def test_B17_un_valor_fuera_de_la_lista_da_422(cliente, registrado):
    assert cliente.post("/boxeadores", json={"nombre": "X", "distancia": "lejos"}).status_code == 422
    assert cliente.post("/boxeadores", json={"nombre": "X", "estilo": "contragolpeador"}).status_code == 422
