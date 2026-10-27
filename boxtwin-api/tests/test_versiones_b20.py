"""
B20 de las versiones: horas del mes (prototipo/specs/versiones-26-10, 04 B20; I5 de 05 §2).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from versiones_util import H, crear_cuenta, crear_miembro, poner, subir


@pytest.fixture
def cuenta_individual(cliente, entorno):
    cid = crear_cuenta(entorno, "individual", "Lucas")
    _, t = crear_miembro(entorno, "lucas@x.ar", cid)
    return cid, t


def _horas(entorno, cid, ahora):
    from boxtwin_api.cupo import horas_del_mes
    from boxtwin_api.modelos import Cuenta

    with entorno["db"].hacer_sesion() as db:
        return horas_del_mes(db.get(Cuenta, cid), db, ahora)


def test_I5_el_mes_se_corta_en_hora_de_buenos_aires(cliente, entorno, cuenta_individual):
    cid, t = cuenta_individual
    a = subir(cliente, t, duracion_declarada_s="1800").json()["job_id"]
    b = subir(cliente, t, duracion_declarada_s="900").json()["job_id"]
    # 31-10 23:30 en Buenos Aires son las 02:30 del 01-11 en UTC.
    poner(entorno, a, creada=datetime(2026, 11, 1, 2, 30, tzinfo=timezone.utc))
    # 01-11 00:10 en Buenos Aires son las 03:10 en UTC.
    poner(entorno, b, creada=datetime(2026, 11, 1, 3, 10, tzinfo=timezone.utc))
    octubre = _horas(entorno, cid, datetime(2026, 10, 20, tzinfo=timezone.utc))
    noviembre = _horas(entorno, cid, datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert octubre["mes"] == "2026-10" and octubre["usadas"] == 0.5
    assert noviembre["mes"] == "2026-11" and noviembre["usadas"] == 0.25


def test_I5_cuenta_la_real_si_esta_y_la_declarada_si_no(cliente, entorno, cuenta_individual):
    cid, t = cuenta_individual
    a = subir(cliente, t, duracion_declarada_s="3600").json()["job_id"]
    subir(cliente, t, duracion_declarada_s="360")
    subir(cliente, t)   # sin ninguna de las dos
    poner(entorno, a, duracion_s=1800.0)
    h = _horas(entorno, cid, datetime.now(timezone.utc))
    assert h["usadas"] == 0.6 and h["sin_duracion"] == 1
    assert h["incluidas"] == 4 and h["restantes"] == 3.4


def test_I5_una_sesion_en_fallo_no_cuenta_y_un_reproceso_tampoco(cliente, entorno, cuenta_individual):
    cid, t = cuenta_individual
    a = subir(cliente, t, duracion_declarada_s="3600").json()["job_id"]
    b = subir(cliente, t, duracion_declarada_s="1800").json()["job_id"]
    poner(entorno, a, estado="fallo")
    antes = _horas(entorno, cid, datetime.now(timezone.utc))
    # Un reproceso (render) no sube video ni crea sesion.
    poner(entorno, b, estado="listo")
    cliente.post(f"/jobs/{b}/render", headers=H(t))
    despues = _horas(entorno, cid, datetime.now(timezone.utc))
    assert antes["usadas"] == despues["usadas"] == 0.5


def test_I5_con_informar_subir_de_mas_pasa_y_avisa(cliente, entorno, cuenta_individual):
    _, t = cuenta_individual
    r = subir(cliente, t, duracion_declarada_s=str(5 * 3600))
    assert r.status_code == 202
    assert r.json()["cupo"] == {"excedido": True, "restantes_h": 4.0}
    assert subir(cliente, t, duracion_declarada_s="60").json()["cupo"]["excedido"] is True


def test_I5_con_bloquear_409_y_no_queda_nada(cliente, entorno, monkeypatch, cuenta_individual):
    from boxtwin_api.modelos import Sesion, Trabajo

    monkeypatch.setattr(entorno["cfg"], "cupo", "bloquear")
    _, t = cuenta_individual
    assert subir(cliente, t, duracion_declarada_s=str(3 * 3600)).status_code == 202
    r = subir(cliente, t, duracion_declarada_s=str(int(1.5 * 3600)))
    assert r.status_code == 409
    assert r.json()["detail"] == "esta sesión usaría 1,5 h y quedan 1,0 h en el mes"
    assert cliente.get("/cuenta", headers=H(t)).json()["horas"]["modo"] == "bloquear"
    with entorno["db"].hacer_sesion() as db:
        assert db.scalar(select(func.count(Sesion.id))) == 1
        assert db.scalar(select(func.count(Trabajo.id))) == 1
    assert len(list((entorno["cfg"].datos / "sesiones").iterdir())) == 1


def test_get_cuenta_y_panel_traen_las_horas(cliente, entorno, cuenta_individual):
    _, t = cuenta_individual
    subir(cliente, t, duracion_declarada_s="720")
    h = cliente.get("/cuenta", headers=H(t)).json()["horas"]
    assert h == {"mes": h["mes"], "incluidas": 4.0, "usadas": 0.2, "restantes": 3.8,
                 "sin_duracion": 0, "modo": "informar"}
    assert cliente.get("/panel", headers=H(t)).json()["horas"] == h
    assert cliente.get("/producto").json()["cupo"] == "informar"


def test_duracion_declarada_invalida_da_422(cliente, cuenta_individual):
    _, t = cuenta_individual
    assert subir(cliente, t, duracion_declarada_s="mucho").status_code == 422
    assert subir(cliente, t, duracion_declarada_s="-1").status_code == 422


def test_el_worker_copia_duracion_y_fps_al_terminar_procesar(cliente, entorno, registrado):
    from boxtwin_api.modelos import Sesion, Trabajo

    from conftest import subir_video

    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps(
        {"estado": "espera_siembra", "video": {"duracion_s": 205.208, "fps": 59.94}}))
    entorno["worker"].correr(una_vuelta=True)
    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        assert (s.estado, s.duracion_s, s.fps) == ("espera_siembra", 205.208, 59.94)
        assert db.scalars(select(Trabajo).where(Trabajo.sesion_id == sid)).first().estado == "listo"
