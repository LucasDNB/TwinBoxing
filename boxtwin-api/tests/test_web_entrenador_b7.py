"""
B7 de la web entrenador: el perfil extendido (prototipo/specs/05-api-y-datos.md).

Lo que se fija: que lo condicionado vuelva en nulo con la bandera apagada aunque haya datos,
que sin guardia cargada no se invente un lado, que las combinaciones respeten la pausa y
dejen afuera lo descartado, y que un peleador sume sus sesiones como A y como B.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video
from test_api import _recargar


def _g(t, tipo, brazo="izq", **extra):
    return {"id": f"g-{t}", "t_inicio": t, "t_fin": t + 0.2, "brazo": brazo, "tipo": tipo,
            "confianza_tipo": 0.8, "corregido": None, **extra}


def _fc(golpes, rounds=None, guardia=None, duracion=180.0):
    rounds = rounds or []
    return {
        "version": "0.2",
        "video": {"duracion_s": duracion, "rounds": [{"round": r["round"], "inicio_s": 0, "fin_s": 60} for r in rounds]},
        "detector": {"checkpoint": "det-v1"},
        "clasificador": {"checkpoint": "clf", "exactitud_familia_fuente_no_vista": None},
        "peleadores": {
            "A": {"golpes": golpes, "por_round": rounds, "total": {"total": len(golpes)}, "guardia": guardia or []},
            "B": {"golpes": [], "por_round": [], "total": {"total": 0}, "guardia": []},
        },
    }


def _sesion(cliente, entorno, fc, boxeador, lado="A", contexto="sparring"):
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={f"boxeador_{lado.lower()}": boxeador})
    return sid


RONDAS = [{"round": 1, "por_minuto": 20.0}, {"round": 2, "por_minuto": 18.0}, {"round": 3, "por_minuto": 12.6}]
GUARDIA = [{"golpe": "A-left-1", "peleador": "A", "brazo": "left", "t": 1.0, "mano_opuesta_caida": False, "medible": True},
           {"golpe": "A-left-2", "peleador": "A", "brazo": "left", "t": 2.0, "mano_opuesta_caida": True, "medible": True}]


@pytest.fixture
def ana(cliente, registrado):
    return cliente.post("/boxeadores", json={"nombre": "Ana", "guardia": "ortodoxa"}).json()["id"]


def test_con_las_banderas_apagadas_lo_condicionado_es_nulo_aunque_haya_datos(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc([_g(1, "jab")], rounds=RONDAS, guardia=GUARDIA), ana)
    p = cliente.get(f"/boxeadores/{ana}/perfil").json()
    ctx = p["por_contexto"][0]
    assert ctx["guardia_opuesta_arriba"] is None
    assert ctx["caida_ultimo_round"] is None
    assert p["rounds_promedio"] is None


def test_con_las_banderas_encendidas_aparecen(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_C2="aprobado", BOXTWIN_C3="aprobado", BOXTWIN_INVITACION="abierto")
    t = c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"}).json()["token"]
    c.headers["Authorization"] = f"Bearer {t}"
    ana = c.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    _sesion(c, entorno, _fc([_g(1, "jab")], rounds=RONDAS, guardia=GUARDIA), ana)
    p = c.get(f"/boxeadores/{ana}/perfil").json()
    ctx = p["por_contexto"][0]
    assert ctx["guardia_opuesta_arriba"] == 0.5
    assert ctx["caida_ultimo_round"] == pytest.approx(-0.3, abs=0.001), "de 18 a 12,6 por minuto"
    assert p["rounds_promedio"][2] == {"round": 3, "golpes_por_minuto": 12.6}


def test_sin_guardia_cargada_el_lado_es_nulo(cliente, registrado, entorno):
    sin = cliente.post("/boxeadores", json={"nombre": "Sin guardia"}).json()["id"]
    _sesion(cliente, entorno, _fc([_g(1, "jab"), _g(1.4, "cross", "der")]), sin)
    combo = cliente.get(f"/boxeadores/{sin}/perfil").json()["combinaciones"][0]
    assert combo["secuencia"] == [{"familia": "jab", "lado": None}, {"familia": "cross", "lado": None}]
    assert combo["estimado"] is True


def test_combinaciones_con_pausa_lado_y_correcciones(cliente, entorno, ana):
    golpes = [
        _g(1, "jab"), _g(1.4, "cross", "der"),                            # 1-2
        _g(5, "jab"), _g(5.5, "cross", "der", corregido={"tipo": "cross"}),  # 1-2 (corregido)
        _g(9, "jab"), _g(9.9, "cross", "der"),                            # pausa de 0,7: no
        _g(12, "jab"), _g(12.3, "hook", "der", descartado=True),          # descartado: no
    ]
    _sesion(cliente, entorno, _fc(golpes), ana)
    combos = cliente.get(f"/boxeadores/{ana}/perfil").json()["combinaciones"]
    assert combos[0] == {"secuencia": [{"familia": "jab", "lado": "adelante"}, {"familia": "cross", "lado": "atras"}],
                         "n": 2, "estimado": True}
    assert all(s["familia"] != "hook" for c in combos for s in c["secuencia"])


def test_la_mezcla_deja_afuera_lo_descartado(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc([_g(1, "jab"), _g(3, "hook", descartado=True)]), ana)
    p = cliente.get(f"/boxeadores/{ana}/perfil").json()
    assert p["totales"]["mezcla"] == {"jab": 1}
    assert p["totales"]["golpes"] == 2, "los detectados no cambian"
    assert p["por_contexto"][0]["jab_fraccion"] == 1.0


def test_filtrar_por_contexto(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc([_g(1, "jab")]), ana, contexto="sparring")
    _sesion(cliente, entorno, _fc([_g(1, "hook"), _g(2, "hook")]), ana, contexto="combate")
    todas = cliente.get(f"/boxeadores/{ana}/perfil?contexto=todas").json()
    assert todas["sesiones_por_contexto"] == {"bolsa": 0, "sombra": 0, "sparring": 1, "combate": 1}
    assert len(todas["sesiones"]) == 2
    combate = cliente.get(f"/boxeadores/{ana}/perfil?contexto=combate").json()
    assert len(combate["sesiones"]) == 1 and combate["totales"]["mezcla"] == {"hook": 2}
    assert combate["sesiones_por_contexto"]["sparring"] == 1, "los contadores son de todas"
    assert {c["contexto"] for c in combate["por_contexto"]} == {"sparring", "combate"}
    assert cliente.get(f"/boxeadores/{ana}/perfil?contexto=pelea").status_code == 422


def test_CA_06_1_un_peleador_como_A_en_una_y_como_B_en_otra_suma_las_dos(cliente, entorno, ana):
    fc_b = _fc([])
    fc_b["peleadores"]["B"]["golpes"] = [_g(1, "jab", "der")]
    _sesion(cliente, entorno, _fc([_g(1, "jab")]), ana, lado="A")
    _sesion(cliente, entorno, fc_b, ana, lado="B")
    p = cliente.get(f"/boxeadores/{ana}/perfil").json()
    assert [s["lado"] for s in p["sesiones"]] == ["A", "B"]
    assert p["totales"]["golpes"] == 2


def test_la_evolucion_dice_con_que_detector(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc([_g(1, "jab")]), ana)
    ev = cliente.get(f"/boxeadores/{ana}/perfil").json()["evolucion"][0]
    assert ev["detector"] == "det-v1" and ev["contexto"] == "sparring"


def test_rounds_total(cliente, entorno, ana):
    _sesion(cliente, entorno, _fc([_g(1, "jab")], rounds=RONDAS), ana)
    _sesion(cliente, entorno, _fc([_g(1, "jab")]), ana)
    assert cliente.get(f"/boxeadores/{ana}/perfil").json()["rounds_total"] == 4, "3 + 1 sin rounds"


def test_el_perfil_trae_la_nomenclatura_del_documento(cliente, entorno, ana):
    fc = _fc([_g(1, "jab")])
    fc["nomenclatura"] = {"jab": "jab", "cross": "gancho", "hook": "hook", "uppercut": "uppercut"}
    _sesion(cliente, entorno, fc, ana)
    assert cliente.get(f"/boxeadores/{ana}/perfil").json()["nomenclatura"]["cross"] == "gancho"
