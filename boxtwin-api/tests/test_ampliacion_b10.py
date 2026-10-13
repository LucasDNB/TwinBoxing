"""
B10 de la ampliacion del 13-10: combinaciones izq-der-izq en la API
(prototipo/specs/ampliacion-13-10/01 §3).

La regla se prueba entera en el anotador (test_mvp_combinaciones.py). Aca se fija lo que es
de la API: que el perfil la use para sumar, por minuto con identidad y por contexto; que los
tipos vayan aparte como estimacion; que un "no fue un golpe" por la API baje la terna en la
Fight-Card; y que /producto publique la regla que usa el anotador.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


def _g(t, brazo, tipo="jab", **extra):
    return {"id": f"A-{brazo}-{int(t * 30)}", "t_inicio": t, "t_fin": t + 0.2, "brazo": brazo,
            "tipo": tipo, "confianza_tipo": 0.8, "corregido": None, **extra}


def _terna(t, tipos=("jab", "cross", "hook")):
    return [_g(t, "izq", tipos[0]), _g(t + 0.3, "der", tipos[1]), _g(t + 0.6, "izq", tipos[2])]


def _fc(golpes, cobertura=0.5, duracion=120.0):
    return {
        "version": "0.2",
        "video": {"duracion_s": duracion, "rounds": []},
        "identidad": {"cobertura_A": cobertura, "cobertura_B": 1.0},
        "detector": {"checkpoint": "det-v1"},
        "clasificador": {"checkpoint": "clf", "exactitud_familia_fuente_no_vista": None},
        "peleadores": {
            "A": {"golpes": golpes, "por_round": [], "total": {"total": len(golpes)}, "guardia": []},
            "B": {"golpes": [], "por_round": [], "total": {"total": 0}, "guardia": []},
        },
    }


def _sesion(cliente, entorno, fc, boxeador, contexto="sparring"):
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "fightcard.json").write_text(json.dumps(fc))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador})
    return sid


@pytest.fixture
def ana(cliente, registrado):
    return cliente.post("/boxeadores", json={"nombre": "Ana", "guardia": "zurda"}).json()["id"]


def test_el_perfil_suma_ternas_por_minuto_con_identidad_y_por_contexto(cliente, entorno, ana):
    # 120 s de video con el peleador visto la mitad: 1 minuto con identidad por sesion.
    _sesion(cliente, entorno, _fc(_terna(1) + _terna(10)), ana)
    _sesion(cliente, entorno, _fc(_terna(5, ("jab", "cross", "hook"))), ana, contexto="bolsa")
    _sesion(cliente, entorno, _fc([_g(1, "der"), _g(1.3, "izq"), _g(1.6, "der")]), ana)
    c = cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]
    assert c["patron"] == "izq-der-izq"
    assert c["total"] == 3
    assert c["por_minuto"] == 1.0, "3 ternas en 3 minutos con identidad, no en 6 de video"
    assert c["por_contexto"] == {"sparring": 2, "bolsa": 1}
    assert c["sesiones_con_combinaciones"] == 2
    assert c["tipos_frecuentes"][0] == {"tipos": ["jab", "cross", "hook"], "n": 3, "estimado": True}
    assert c["recall_medido"] is None, "estas Fight-Cards no traen recall"


def test_el_recall_del_margen_sale_de_la_fightcard_mas_reciente(cliente, entorno, ana):
    fc = _fc(_terna(1))
    fc["detector"]["recall_medido"] = 0.484
    _sesion(cliente, entorno, fc, ana)
    assert cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]["recall_medido"] == 0.484
    fc["detector"]["combinaciones"] = {"recall_medido": 0.2}
    _sesion(cliente, entorno, fc, ana)
    assert cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]["recall_medido"] == 0.2


def test_DA_3_un_zurdo_cuenta_izq_der_izq_literal(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc(_terna(1)), ana)
    assert cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]["total"] == 1


def test_el_filtro_de_contexto_se_aplica(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc(_terna(1)), ana)
    _sesion(cliente, entorno, _fc(_terna(1)), ana, contexto="bolsa")
    c = cliente.get(f"/boxeadores/{ana}/perfil?contexto=bolsa").json()["combinaciones"]
    assert c["total"] == 1 and c["por_contexto"] == {"bolsa": 1}


def test_lo_descartado_corta_y_lo_corregido_cambia_el_tipo(cliente, entorno, ana):
    rota = _terna(1)
    rota[1]["descartado"] = True
    corregida = _terna(10)
    corregida[2]["corregido"] = {"tipo": "uppercut"}
    _sesion(cliente, entorno, _fc(rota + corregida), ana)
    c = cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]
    assert c["total"] == 1
    assert c["tipos_frecuentes"] == [{"tipos": ["jab", "cross", "uppercut"], "n": 1, "estimado": True}]


def test_una_terna_con_un_tipo_sin_estimar_cuenta_pero_no_entra_en_tipos(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc(_terna(1, ("jab", None, "hook"))), ana)
    c = cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]
    assert c["total"] == 1 and c["tipos_frecuentes"] == []


def test_sin_cobertura_medida_no_hay_ritmo(cliente, entorno, ana):
    fc = _fc(_terna(1))
    del fc["identidad"]
    _sesion(cliente, entorno, fc, ana)
    c = cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]
    assert c["total"] == 1 and c["por_minuto"] is None


def test_4_no_golpe_por_la_api_baja_la_terna_de_la_fightcard(cliente, registrado, entorno):
    from boxtwin.mvp.fightcard import construir
    from boxtwin.mvp.sesion import Sesion

    class G:
        def __init__(self, brazo, t):
            self.peleador, self.brazo, self.t_inicio, self.t_fin = "A", brazo, t, t + 0.2
            self.inicio, self.fin, self.score = int(round(t * 30)), int(round((t + 0.2) * 30)), 0.9

    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    s = Sesion.nueva(d, {"nombre": "v.mp4", "fps": 30.0, "duracion_s": 60.0, "total_frames": 1800},
                     round_s=None, descanso_s=0.0)
    fc = construir(s, [G("left", 1.0), G("right", 1.3), G("left", 1.6)])
    (d / "fightcard.json").write_text(json.dumps(fc))
    assert cliente.get(f"/fightcards/{sid}").json()["peleadores"]["A"]["combinaciones"]["total"] == 1
    r = cliente.patch(f"/fightcards/{sid}/golpes/A-der-39", json={"tipo": "no_golpe"})
    assert r.status_code == 200, r.text
    a = cliente.get(f"/fightcards/{sid}").json()["peleadores"]["A"]
    assert a["combinaciones"]["total"] == 0
    assert a["total"]["total"] == 3, "los detectados no cambian"
