"""
B5b de la web entrenador: GET /panel, el progreso en GET /jobs y el uso de cada boxeador en
GET /boxeadores (prototipo/specs/05-api-y-datos.md).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from conftest import subir_video

FC_REAL = Path(__file__).resolve().parents[2] / "prototipo/boxtwin-web-entrenador/tests/fixtures/sesion-real/fightcard-0.1.json"


def _estado(entorno, sid, estado):
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, sid).estado = estado
        db.commit()


def _lista(cliente, entorno, fc: dict | None = None, contexto="sparring"):
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    _estado(entorno, sid, "listo")
    if fc is not None:
        (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    return sid


def _fc(rounds=3, a=10, b=5, recall=0.484):
    return {
        "video": {"rounds": [{"round": i + 1, "inicio_s": 0, "fin_s": 1} for i in range(rounds)]},
        "detector": {"recall_medido": recall},
        "peleadores": {"A": {"total": {"total": a}}, "B": {"total": {"total": b}}},
    }


def test_panel_cuenta_lo_del_mes_desde_las_fightcards(cliente, registrado, entorno):
    _lista(cliente, entorno, _fc(rounds=3, a=10, b=5))
    _lista(cliente, entorno, _fc(rounds=0, a=2, b=1), contexto="combate")
    sid = subir_video(cliente).json()["job_id"]
    _estado(entorno, sid, "espera_siembra")
    subir_video(cliente)   # queda en_cola

    d = cliente.get("/panel").json()
    assert d["sesiones_mes"] == {"total": 4, "listo": 2, "en_curso": 1, "espera_siembra": 1, "fallo": 0}
    assert d["rounds_mes"] == {"total": 4, "sparring": 3, "combate": 1}, "sin rounds cuenta uno"
    assert d["golpes_detectados_mes"] == 18
    assert d["recall_detector"] == 0.484
    assert d["pendientes"] == [{"sesion_id": sid, "estado": "espera_siembra", "titulo": "sparring del martes"}]


def test_panel_con_la_fightcard_real(cliente, registrado, entorno):
    _lista(cliente, entorno, json.loads(FC_REAL.read_text()))
    d = cliente.get("/panel").json()
    assert d["golpes_detectados_mes"] == 51 + 32
    assert d["rounds_mes"]["total"] == 1


def test_una_sesion_lista_sin_fightcard_no_rompe_el_panel(cliente, registrado, entorno):
    _lista(cliente, entorno, None)
    sid = _lista(cliente, entorno, None)
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text("{ esto no es json")
    d = cliente.get("/panel").json()
    assert d["golpes_detectados_mes"] == 0
    assert d["recall_detector"] is None, "sin documento no se inventa un recall"


def test_un_usuario_no_ve_sesiones_de_otro_en_el_panel(cliente, registrado, entorno):
    _lista(cliente, entorno, _fc())
    r = cliente.post("/auth/registro", json={"email": "otro@b.com", "clave": "otraclave1"})
    cliente.headers["Authorization"] = f"Bearer {r.json()['token']}"
    d = cliente.get("/panel").json()
    assert d["sesiones_mes"]["total"] == 0
    assert d["golpes_detectados_mes"] == 0
    assert d["pendientes"] == []


def test_el_panel_ve_la_correccion_sin_reiniciar(cliente, registrado, entorno):
    sid = _lista(cliente, entorno, _fc(a=10, b=5))
    assert cliente.get("/panel").json()["golpes_detectados_mes"] == 15
    ruta = entorno["cfg"].dir_sesion(sid) / "fightcard.json"
    ruta.write_text(json.dumps(_fc(a=20, b=5)))
    os.utime(ruta, (time.time() + 5, time.time() + 5))
    assert cliente.get("/panel").json()["golpes_detectados_mes"] == 25


def test_activos_y_preliminares(cliente, registrado, entorno):
    a = cliente.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    b = cliente.post("/boxeadores", json={"nombre": "Bea"}).json()["id"]
    cliente.post("/boxeadores", json={"nombre": "Sin sesiones"})
    for _ in range(5):
        sid = subir_video(cliente).json()["job_id"]
        cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": a})
    sid = subir_video(cliente).json()["job_id"]
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": b, "boxeador_b": a})
    d = cliente.get("/panel").json()
    assert d["peleadores_activos"] == 2
    assert d["peleadores_preliminares"] == 1, "Bea tiene 1 sesion, menos de 5"


def test_jobs_trae_el_progreso_solo_de_las_que_estan_en_curso(cliente, registrado, entorno):
    en_curso = subir_video(cliente).json()["job_id"]
    (entorno["cfg"].dir_sesion(en_curso) / "progreso.json").write_text(
        json.dumps({"etapa": "pose", "hecho": 64, "total": 100, "fraccion": 0.64}))
    lista = _lista(cliente, entorno, _fc())
    filas = {s["id"]: s for s in cliente.get("/jobs").json()}
    assert filas[en_curso]["progreso"]["fraccion"] == 0.64
    assert "progreso" not in filas[lista]


def test_boxeadores_trae_sesiones_y_la_ultima(cliente, registrado, entorno):
    a = cliente.post("/boxeadores", json={"nombre": "Ana"}).json()["id"]
    b = cliente.post("/boxeadores", json={"nombre": "Bea"}).json()["id"]
    s1 = subir_video(cliente).json()["job_id"]
    s2 = subir_video(cliente).json()["job_id"]
    cliente.put(f"/sesiones/{s1}/boxeadores", json={"boxeador_a": a})
    cliente.put(f"/sesiones/{s2}/boxeadores", json={"boxeador_a": b, "boxeador_b": a})
    filas = {x["nombre"]: x for x in cliente.get("/boxeadores").json()}
    assert filas["Ana"]["sesiones"] == 2, "como A en una y como B en otra"
    assert filas["Bea"]["sesiones"] == 1
    assert filas["Ana"]["ultima_sesion"] is not None


def test_un_boxeador_sin_sesiones(cliente, registrado):
    cliente.post("/boxeadores", json={"nombre": "Nuevo"})
    f = cliente.get("/boxeadores").json()[0]
    assert f["sesiones"] == 0 and f["ultima_sesion"] is None
