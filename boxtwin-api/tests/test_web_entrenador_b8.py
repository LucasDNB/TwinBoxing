"""
B8 de la web entrenador: siembra de un solo peleador (prototipo/specs/05-api-y-datos.md).
La bandera queda apagada hasta medir el criterio del spec; esto prueba el cableado.
"""

from __future__ import annotations

import json

from conftest import subir_video


def _espera(cliente, entorno, contexto):
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps({"estado": "espera_siembra", "candidatos": [
        {"track": 3, "recorte": None}, {"track": 7, "recorte": None}]}))
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, sid).estado = "espera_siembra"
        db.commit()
    return sid


def test_en_bolsa_se_siembra_una_sola_persona(cliente, registrado, entorno):
    sid = _espera(cliente, entorno, "bolsa")
    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3})
    assert r.status_code == 202, r.text
    from boxtwin_api.modelos import Sesion, Trabajo
    from boxtwin_api.worker import comando_de

    with entorno["db"].hacer_sesion() as db:
        t = db.query(Trabajo).filter(Trabajo.etapa == "completar").one()
        cmd = comando_de(t, db.get(Sesion, sid))[0]
    assert "--semilla-a" in cmd and "--semilla-b" not in cmd


def test_en_bolsa_no_se_marcan_dos(cliente, registrado, entorno):
    sid = _espera(cliente, entorno, "sombra")
    assert cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3, "track_b": 7}).status_code == 422


def test_en_sparring_hacen_falta_los_dos(cliente, registrado, entorno):
    sid = _espera(cliente, entorno, "sparring")
    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3})
    assert r.status_code == 422 and "los dos peleadores" in r.json()["detail"]


def test_la_bandera_sigue_apagada_sin_medir(cliente):
    assert cliente.get("/producto").json()["banderas"]["sesiones_individuales"] is False
