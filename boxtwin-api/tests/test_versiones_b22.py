"""
B22 de las versiones: reglas de la version Individual (versiones-26-10, 04 B22; I4 de 05 §2).

El registro Individual (B21) quedo postergado por la version de prueba: la cuenta se crea
como la crea demo.py, con el peleador propio en cuentas.boxeador_propio_id.
"""

from __future__ import annotations

import json

import pytest

from versiones_util import (
    FC_MINIMA,
    H,
    crear_cuenta,
    crear_miembro,
    peleador_propio,
    poner,
    subir,
)


@pytest.fixture
def individual(cliente, entorno):
    cid = crear_cuenta(entorno, "individual", "Lucas")
    uid, t = crear_miembro(entorno, "lucas@x.ar", cid, nombre="Lucas")
    bid = peleador_propio(entorno, cid, uid, "Lucas")
    return {"cuenta": cid, "token": t, "propio": bid}


def _fc_bolsa():
    fc = json.loads(json.dumps(FC_MINIMA))
    fc["peleadores"].pop("B")
    fc["peleadores"]["A"]["golpes"] = [
        {"id": "A-30", "t_inicio": 1.0, "brazo": "izq", "tipo": "jab"},
        {"id": "A-60", "t_inicio": 2.0, "brazo": "der", "tipo": "cross"},
    ]
    fc["peleadores"]["A"]["total"] = {"total": 2}
    return fc


def test_I4_individual_de_punta_a_punta(cliente, entorno, individual):
    t, propio = individual["token"], individual["propio"]
    c = cliente.get("/cuenta", headers=H(t)).json()
    assert c["version"] == "individual" and c["boxeador_propio"] == {"id": propio, "nombre": "Lucas"}
    assert c["funciones"]["peleadores"] == "propio" and c["funciones"]["rival"] is False

    # Subir bolsa: A es el propio, sin elegirlo (RF-I2).
    sid = subir(cliente, t, contexto="bolsa").json()["job_id"]
    j = cliente.get(f"/jobs/{sid}", headers=H(t)).json()
    assert j["boxeadores"]["A"] == {"id": propio, "nombre": "Lucas"}
    assert j["boxeadores"]["B"] is None

    # Siembra de una persona: la ruta de B8 tal cual.
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps(
        {"estado": "espera_siembra", "candidatos": [{"track": 4}]}))
    poner(entorno, sid, estado="espera_siembra")
    assert cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 4},
                        headers=H(t)).status_code == 202

    # Fight-Card lista, y el perfil propio la suma a la evolucion.
    (d / "fightcard.json").write_text(json.dumps(_fc_bolsa()))
    (d / "sesion.json").write_text(json.dumps({"estado": "listo", "candidatos": []}))
    poner(entorno, sid, estado="listo")
    assert cliente.get(f"/fightcards/{sid}", headers=H(t)).status_code == 200
    perfil = cliente.get(f"/boxeadores/{propio}/perfil", headers=H(t)).json()
    assert len(perfil["evolucion"]) == 1

    # Lo que no esta en la version (RF-V2): la API lo rechaza aunque se pida directo.
    r = cliente.post("/boxeadores", json={"nombre": "Compañero"}, headers=H(t))
    assert r.status_code == 403 and r.json()["detail"] == "esta función no está en la versión Individual"
    r = cliente.put(f"/sesiones/{sid}/boxeadores",
                    json={"boxeador_a": propio, "boxeador_b": propio}, headers=H(t))
    assert r.status_code == 422   # el mismo boxeador de los dos lados, como hoy
    otro = _otro_peleador(entorno, individual)
    r = cliente.put(f"/sesiones/{sid}/boxeadores",
                    json={"boxeador_a": propio, "boxeador_b": otro}, headers=H(t))
    assert r.status_code == 403
    assert r.json()["detail"] == "en la versión Individual el compañero no tiene perfil"
    assert cliente.get(f"/rival?a={propio}&b={otro}", headers=H(t)).status_code == 403


def _otro_peleador(entorno, individual) -> str:
    """Un peleador de la cuenta que no es el propio, cargado a mano: la API no deja crearlo."""
    from boxtwin_api.modelos import Boxeador

    with entorno["db"].hacer_sesion() as db:
        b = Boxeador(usuario_id="x", cuenta_id=individual["cuenta"], nombre="Colado")
        db.add(b)
        db.commit()
        return b.id


def test_en_sparring_individual_se_marcan_los_dos_y_el_compañero_no_tiene_perfil(
        cliente, entorno, individual):
    t, propio = individual["token"], individual["propio"]
    sid = subir(cliente, t, contexto="sparring").json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    (d / "sesion.json").write_text(json.dumps(
        {"estado": "espera_siembra", "candidatos": [{"track": 4}, {"track": 9}]}))
    poner(entorno, sid, estado="espera_siembra")
    # "¿Cuál sos vos?" es track_a; el compañero, track_b (03 §I-03).
    assert cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 9, "track_b": 4},
                        headers=H(t)).status_code == 202
    j = cliente.get(f"/jobs/{sid}", headers=H(t)).json()
    assert j["siembra"] == {"track_a": 9, "track_b": 4}
    assert j["boxeadores"] == {"A": {"id": propio, "nombre": "Lucas"}, "B": None}
    assert [b["id"] for b in cliente.get("/boxeadores", headers=H(t)).json()] == [propio]


def test_en_individual_se_edita_el_propio_y_se_puede_desasignar(cliente, entorno, individual):
    t, propio = individual["token"], individual["propio"]
    r = cliente.patch(f"/boxeadores/{propio}", json={"nombre": "Lucas", "altura_m": 1.75},
                      headers=H(t))
    assert r.status_code == 200 and r.json()["altura_m"] == 1.75
    sid = subir(cliente, t, contexto="bolsa").json()["job_id"]
    assert cliente.put(f"/sesiones/{sid}/boxeadores", json={}, headers=H(t)).status_code == 200


def test_en_entrenador_nada_de_esto_cambia(cliente, registrado):
    b = cliente.post("/boxeadores", json={"nombre": "Tomás"}).json()["id"]
    c = cliente.post("/boxeadores", json={"nombre": "Nico"}).json()["id"]
    from conftest import subir_video

    sid = subir_video(cliente).json()["job_id"]
    assert cliente.get(f"/jobs/{sid}").json()["boxeadores"]["A"] is None
    assert cliente.put(f"/sesiones/{sid}/boxeadores",
                       json={"boxeador_a": b, "boxeador_b": c}).status_code == 200
    assert cliente.get(f"/rival?a={b}&b={c}").status_code == 200


def test_CA_I05_1_el_export_usa_el_nombre_propio_y_compañero(cliente, entorno, individual):
    t = individual["token"]
    sid = subir(cliente, t, contexto="sparring").json()["job_id"]
    fc = json.loads(json.dumps(FC_MINIMA))
    for p in fc["peleadores"].values():
        p.update({"izq": 0, "der": 0})
        p["total"].update({"izq": 0, "der": 0})
        p["por_round"] = [{"round": 1, "total": 1, "izq": 1, "der": 0, "por_minuto": 0.3}]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    poner(entorno, sid, estado="listo")
    html = cliente.get(f"/fightcards/{sid}/export?formato=html", headers=H(t)).text
    assert "Peleador B" not in html and "Peleador A" not in html
    assert "Lucas" in html and "Compañero" in html


def test_el_export_de_una_sesion_de_bolsa_no_se_rompe_sin_B(cliente, entorno, individual):
    t = individual["token"]
    sid = subir(cliente, t, contexto="bolsa").json()["job_id"]
    fc = _fc_bolsa()
    fc["peleadores"]["A"]["total"].update({"izq": 1, "der": 1})
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps(fc))
    poner(entorno, sid, estado="listo")
    r = cliente.get(f"/fightcards/{sid}/export?formato=html", headers=H(t))
    assert r.status_code == 200 and "Compañero" not in r.text.split("<h2>Volumen")[1].split("</table>")[0]
