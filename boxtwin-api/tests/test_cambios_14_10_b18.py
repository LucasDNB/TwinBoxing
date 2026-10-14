"""
B18 de los cambios del 14-10 (prototipo/specs/cambios-14-10/02 §1): ataque contra exposicion
en el Rival Fantasma. Las reglas se prueban sobre perfiles armados a mano; el endpoint, con
Fight-Cards en disco.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video
from test_api import _recargar

from boxtwin_api.rival import MINIMO_GOLPES, cruzar
from boxtwin_api.zonas import ZONAS

PALABRAS_PROHIBIDAS = ("conect", "vulnerable", "probabilidad", "va a entrar")


def _bloque(fr: dict[str, float], n: int = 100) -> dict:
    """Un bloque de zonas con fracciones corregidas dadas y n golpes con altura."""
    zonas = {z: {"n": round(fr.get(z, 0.0) * n), "fraccion": fr.get(z, 0.0),
                 "fraccion_corregida": fr.get(z, 0.0)} for z in ZONAS}
    return {"zonas": zonas, "con_dato": n}


def _perfil(nombre, ataque, exposicion, n=100):
    return {"boxeador": {"nombre": nombre}, "ataque": _bloque(ataque, n),
            "exposicion": _bloque(exposicion, n)}


PAREJO = {z: 0.25 for z in ZONAS}


def test_1_sin_altura_no_hay_cruces_ni_lecturas():
    r = cruzar({"boxeador": {"nombre": "A"}, "ataque": {"zonas": None}, "exposicion": None},
               {"boxeador": {"nombre": "B"}, "ataque": {"zonas": None}, "exposicion": None})
    assert r["cruces"] == [] and r["lecturas"] == []


def test_2_con_49_golpes_en_una_mitad_no_se_dispara_nada():
    pa = _perfil("Ana", {"cabeza_mano_izq": 0.6, "cuerpo_mano_der": 0.4}, PAREJO)
    pb = _perfil("Beto", PAREJO, {"cabeza_mano_izq": 0.6, "cuerpo_mano_der": 0.4})
    pb["exposicion"]["con_dato"] = MINIMO_GOLPES - 1
    r = cruzar(pa, pb)
    assert r["lecturas"] == []
    assert r["faltan"]["base"]["B"]["exposicion"] == 49
    assert len(r["cruces"]) == 8, "los cruces se muestran igual; lo que no hay son lecturas"


def test_3_cada_lectura_cita_una_zona_con_sus_dos_valores_en_los_cruces():
    pa = _perfil("Ana", {"cabeza_mano_izq": 0.5, "cuerpo_mano_der": 0.5}, PAREJO)
    pb = _perfil("Beto", PAREJO, {"cabeza_mano_izq": 0.5, "cabeza_mano_der": 0.4, "cuerpo_mano_der": 0.1})
    r = cruzar(pa, pb)
    assert r["lecturas"]
    zonas_cruzadas = {(c["atacante"], c["zona"]) for c in r["cruces"]}
    for x in r["lecturas"]:
        if x["zona"] != "cuerpo":
            assert (x["atacante"], x["zona"]) in zonas_cruzadas
        assert not any(p in x["texto"].lower() for p in PALABRAS_PROHIBIDAS), x["texto"]


def test_4_una_regla_es_oportunidad_para_el_atacante_y_amenaza_para_el_receptor():
    pa = _perfil("Ana", {"cabeza_mano_izq": 0.5, "cabeza_mano_der": 0.5}, PAREJO)
    pb = _perfil("Beto", PAREJO, {"cabeza_mano_izq": 0.5, "cabeza_mano_der": 0.5})
    r = cruzar(pa, pb)
    coincide = [x for x in r["lecturas"] if x["regla"] == "zona_coincidente" and x["zona"] == "cabeza_mano_izq"
                and x["atacante"] == "A"]
    assert {(x["tipo"], x["para"]) for x in coincide} == {("oportunidad", "A"), ("amenaza", "B")}
    assert coincide[0]["texto"] == ("Ana suele ir a la cabeza con la izquierda (50%), y ahí es "
                                    "donde a Beto suelen buscarlo (50%).")


@pytest.mark.parametrize(("ataque", "dispara"), [(0.25, True), (0.249, False)])
def test_5_zona_coincidente_en_el_umbral(ataque, dispara):
    resto = (1 - ataque) / 3
    at = {"cabeza_mano_izq": ataque, "cabeza_mano_der": resto, "cuerpo_mano_izq": resto, "cuerpo_mano_der": resto}
    pa = _perfil("Ana", at, PAREJO)
    pb = _perfil("Beto", PAREJO, PAREJO)
    r = cruzar(pa, pb)
    hay = any(x["regla"] == "zona_coincidente" and x["atacante"] == "A" and x["zona"] == "cabeza_mano_izq"
              for x in r["lecturas"])
    assert hay is dispara


@pytest.mark.parametrize(("ataque", "dispara"), [(0.35, True), (0.349, False)])
def test_5_zona_trabajada_diez_puntos(ataque, dispara):
    resto = (1 - ataque) / 3
    at = {"cabeza_mano_izq": ataque, "cabeza_mano_der": resto, "cuerpo_mano_izq": resto, "cuerpo_mano_der": resto}
    r = cruzar(_perfil("Ana", at, PAREJO), _perfil("Beto", PAREJO, PAREJO))
    hay = any(x["regla"] == "zona_trabajada" and x["atacante"] == "A" and x["zona"] == "cabeza_mano_izq"
              for x in r["lecturas"])
    assert hay is dispara


@pytest.mark.parametrize(("visto", "dispara"), [(0.149, True), (0.15, False)])
def test_5_cuerpo(visto, dispara):
    at = {"cabeza_mano_izq": 0.75, "cuerpo_mano_der": 0.25}
    ex = {"cabeza_mano_izq": 1 - visto, "cuerpo_mano_der": visto}
    r = cruzar(_perfil("Ana", at, PAREJO), _perfil("Beto", PAREJO, ex))
    hay = any(x["regla"] == "cuerpo" and x["atacante"] == "A" for x in r["lecturas"])
    assert hay is dispara


# -- Endpoint ------------------------------------------------------------------------------

def _g(i, brazo, objetivo):
    return {"id": f"g{i}", "t_inicio": float(i), "t_fin": i + 0.2, "brazo": brazo, "tipo": None,
            "corregido": None, "objetivo": objetivo}


def _fc(a, b):
    return {"version": "0.3", "video": {"duracion_s": 600.0, "rounds": []},
            "identidad": {"cobertura_A": 1.0, "cobertura_B": 1.0},
            "detector": {"checkpoint": "det"}, "clasificador": {"checkpoint": None},
            "peleadores": {"A": {"golpes": a, "por_round": [], "total": {"total": len(a)}},
                           "B": {"golpes": b, "por_round": [], "total": {"total": len(b)}}}}


@pytest.fixture
def dos(cliente, registrado, entorno):
    ana = cliente.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    beto = cliente.post("/boxeadores", json={"nombre": "Beto"}).json()["id"]
    # 60 de cada lado: Ana tira a la cabeza con la izquierda, Beto al cuerpo con la derecha.
    a = [_g(i, "izq", "cabeza") for i in range(60)]
    b = [_g(100 + i, "der", "cuerpo") for i in range(60)]
    sid = subir_video(cliente, contexto="sparring").json()["job_id"]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(_fc(a, b)))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": ana, "boxeador_b": beto})
    # una sesion de bolsa de Ana que no tiene que entrar
    sid = subir_video(cliente, contexto="bolsa").json()["job_id"]
    fc = _fc([_g(i, "der", "cuerpo") for i in range(200)], [])
    del fc["peleadores"]["B"]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": ana})
    return ana, beto


def test_el_endpoint_cruza_solo_sparring_y_combate(cliente, dos):
    ana, beto = dos
    r = cliente.get(f"/rival?a={ana}&b={beto}").json()
    assert r["base"]["A"] == {"ataque": 60, "exposicion": 60}, "la bolsa no entra"
    c = {(x["atacante"], x["zona"]): x for x in r["cruces"]}
    assert c[("A", "cabeza_mano_izq")]["ataque_fraccion"] == 1.0
    assert c[("B", "cuerpo_mano_der")]["exposicion_fraccion"] == 1.0, "a Ana le van al cuerpo por la derecha"
    assert {x["regla"] for x in r["lecturas"]} >= {"zona_coincidente"}


def test_6_el_mismo_peleador_en_los_dos_lados_da_422(cliente, dos):
    ana, _ = dos
    assert cliente.get(f"/rival?a={ana}&b={ana}").status_code == 422


def test_con_la_altura_apagada_el_endpoint_viene_vacio(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_OBJETIVO_ESTIMADO="0", BOXTWIN_INVITACION="abierto")
    t = c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"}).json()["token"]
    c.headers["Authorization"] = f"Bearer {t}"
    ana = c.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    beto = c.post("/boxeadores", json={"nombre": "Beto"}).json()["id"]
    r = c.get(f"/rival?a={ana}&b={beto}").json()
    assert r["cruces"] == [] and r["lecturas"] == []
