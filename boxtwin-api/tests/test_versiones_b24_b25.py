"""
B24 y B25 de las versiones: plantel compartido, autoria y consentimiento por atleta
(versiones-26-10, 04 B24 y B25; I2 de 05 §2 y las pruebas de B25).

La baja de un miembro (B23) quedo postergada por la version de prueba: aca se marca a mano en
la base, que es lo unico que haria la ruta. Lo que se prueba es lo que pasa despues.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from versiones_util import H, crear_cuenta, crear_miembro, lista_con_fightcard, poner, subir

TEXTO = "El atleta acepta ser filmado y que sus golpes se analicen."


@pytest.fixture
def texto(entorno, tmp_path, monkeypatch):
    """Un texto de prueba en una carpeta aparte: el real no esta (DV-5) y no se inventa."""
    carpeta = tmp_path / "textos"
    carpeta.mkdir()
    (carpeta / "consentimiento-2026-10-08.md").write_text(TEXTO)
    (carpeta / "consentimiento-2026-09-01.md").write_text("version vieja")
    monkeypatch.setattr(entorno["cfg"], "textos", carpeta)
    return "2026-10-08"


@pytest.fixture
def gimnasio(cliente, entorno):
    cid = crear_cuenta(entorno, "gimnasio", "Gimnasio Norte")
    t_id, t = crear_miembro(entorno, "titular@norte.ar", cid, nombre="Tita")
    x_id, x = crear_miembro(entorno, "ana@norte.ar", cid, rol="entrenador", nombre="Ana")
    y_id, y = crear_miembro(entorno, "juan@norte.ar", cid, rol="entrenador", nombre="Juan")
    return {"cuenta": cid, "T": t, "X": x, "Y": y, "ids": {"T": t_id, "X": x_id, "Y": y_id}}


def _consentir(cliente, token, bid, version):
    return cliente.post(f"/boxeadores/{bid}/consentimiento",
                        json={"texto_version": version, "mayor_de_edad": True}, headers=H(token))


def _fc_con_golpe():
    return {
        "version": "0.3",
        "video": {"rounds": [{"round": 1, "inicio_s": 0, "fin_s": 180}], "duracion_s": 180},
        "detector": {"recall_medido": 0.484},
        "clasificador": {"checkpoint": "geo-v1"},
        "peleadores": {
            "A": {"total": {"total": 1}, "golpes": [{"id": "A-30", "t_inicio": 1.0,
                                                     "brazo": "izq", "tipo": "jab"}]},
            "B": {"total": {"total": 0}, "golpes": []},
        },
    }


# ---------------------------------------------------------------------------
# I2
# ---------------------------------------------------------------------------


def test_I2_plantel_compartido_autoria_y_baja(cliente, entorno, gimnasio, monkeypatch):
    g = gimnasio
    # X sube y marca.
    sid = lista_con_fightcard(cliente, entorno, g["X"], fc=_fc_con_golpe())
    # Y la ve, en la lista y por id.
    assert sid in {s["id"] for s in cliente.get("/jobs", headers=H(g["Y"])).json()}
    j = cliente.get(f"/jobs/{sid}", headers=H(g["Y"])).json()
    assert j["subida_por"] == {"id": g["ids"]["X"], "nombre": "Ana"}

    # Y la corrige, y la correccion queda con su usuario. El registro en la Fight-Card es del
    # anotador y tiene sus pruebas; aca se reemplaza por uno minimo.
    def registrar(directorio, golpe, tipo, por, objetivo=None):
        fc = json.loads((directorio / "fightcard.json").read_text())
        fc["peleadores"]["A"]["golpes"][0].update(
            {"tipo": tipo, "corregido": {"tipo_original": "jab", "por": por}})
        (directorio / "fightcard.json").write_text(json.dumps(fc))
        return fc

    import boxtwin.mvp.orquesta as orquesta

    monkeypatch.setattr(orquesta, "registrar_correccion", registrar)
    r = cliente.patch(f"/fightcards/{sid}/golpes/A-30", json={"tipo": "cross"}, headers=H(g["Y"]))
    assert r.status_code == 200, r.text
    from boxtwin_api.modelos import Correccion, Miembro, ahora

    with entorno["db"].hacer_sesion() as db:
        assert db.scalars(select(Correccion)).one().usuario_id == g["ids"]["Y"]
    fc = cliente.get(f"/fightcards/{sid}", headers=H(g["T"])).json()
    assert fc["sesion"]["autoria"] == {"subida_por": "Ana", "correcciones": {"A-30": "Juan"}}

    # T da de baja a X: el pedido siguiente de X da 401 con el mensaje de baja.
    with entorno["db"].hacer_sesion() as db:
        m = db.scalars(select(Miembro).where(Miembro.usuario_id == g["ids"]["X"])).one()
        m.baja_en, m.baja_por = ahora(), g["ids"]["T"]
        db.commit()
    r = cliente.get("/jobs", headers=H(g["X"]))
    assert r.status_code == 401 and r.json()["detail"] == "tu acceso a Gimnasio Norte fue dado de baja"
    # Y sus sesiones siguen visibles para Y y para T.
    for quien in ("Y", "T"):
        assert cliente.get(f"/jobs/{sid}", headers=H(g[quien])).status_code == 200


def test_CA_G04_1_el_perfil_trae_las_sesiones_de_todos_los_entrenadores(cliente, entorno, gimnasio, texto):
    g = gimnasio
    b = cliente.post("/boxeadores", json={"nombre": "Tomás Ríos"}, headers=H(g["T"])).json()["id"]
    assert _consentir(cliente, g["T"], b, texto).status_code == 201
    for quien in ("X", "Y"):
        sid = lista_con_fightcard(cliente, entorno, g[quien])
        assert cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": b},
                           headers=H(g[quien])).status_code == 200
    p = cliente.get(f"/boxeadores/{b}/perfil", headers=H(g["X"])).json()
    assert len(p["evolucion"]) == 2


def test_miembros_y_por_entrenador_solo_para_el_titular(cliente, entorno, gimnasio):
    g = gimnasio
    sid = subir(cliente, g["X"], duracion_declarada_s="1800").json()["job_id"]
    poner(entorno, sid, estado="espera_siembra")
    m = cliente.get("/cuenta/miembros", headers=H(g["T"])).json()
    assert [(x["nombre"], x["rol"]) for x in m] == [("Tita", "titular"), ("Ana", "entrenador"),
                                                     ("Juan", "entrenador")]
    assert m[1]["email"] == "ana@norte.ar"
    assert cliente.get("/cuenta/miembros", headers=H(g["X"])).status_code == 403
    pt = cliente.get("/panel", headers=H(g["T"])).json()
    ana = next(f for f in pt["por_entrenador"] if f["nombre"] == "Ana")
    assert (ana["sesiones_mes"], ana["horas_mes"], ana["pendientes"]) == (1, 0.5, 1)
    assert pt["pendientes"][0]["subida_por"]["nombre"] == "Ana"
    # CA-G01.1: el entrenador no ve el bloque, y la API se lo manda nulo.
    assert cliente.get("/panel", headers=H(g["X"])).json()["por_entrenador"] is None


def test_miembros_de_una_cuenta_de_prueba_no_traen_email(cliente, entorno):
    cid = crear_cuenta(entorno, "gimnasio", "Gimnasio de prueba", demo=True)
    _, t = crear_miembro(entorno, "gimnasio_titular@prueba.boxtwin", cid, nombre="Titular")
    assert "email" not in cliente.get("/cuenta/miembros", headers=H(t)).json()[0]


def test_entrenador_a_cargo_y_filtro_a_mi_cargo(cliente, gimnasio):
    g = gimnasio
    r = cliente.post("/boxeadores", json={"nombre": "Tomás", "entrenador_id": g["ids"]["X"]},
                     headers=H(g["T"]))
    assert r.status_code == 201 and r.json()["entrenador"] == {"id": g["ids"]["X"], "nombre": "Ana"}
    cliente.post("/boxeadores", json={"nombre": "Nico"}, headers=H(g["T"]))
    assert [b["nombre"] for b in cliente.get("/boxeadores?a_cargo=yo", headers=H(g["X"])).json()] == ["Tomás"]
    assert len(cliente.get("/boxeadores", headers=H(g["X"])).json()) == 2
    # Una edicion que no manda el campo no lo borra.
    bid = r.json()["id"]
    e = cliente.patch(f"/boxeadores/{bid}", json={"nombre": "Tomás Ríos"}, headers=H(g["Y"])).json()
    assert e["entrenador"]["nombre"] == "Ana"
    e = cliente.patch(f"/boxeadores/{bid}", json={"nombre": "Tomás Ríos", "entrenador_id": None},
                      headers=H(g["Y"])).json()
    assert e["entrenador"] is None
    # El entrenador tiene que ser miembro activo de la cuenta.
    r = cliente.patch(f"/boxeadores/{bid}", json={"nombre": "Tomás Ríos", "entrenador_id": "otro"},
                      headers=H(g["T"]))
    assert r.status_code == 422


def test_entrenador_a_cargo_no_existe_fuera_de_gimnasio(cliente, registrado):
    r = cliente.post("/boxeadores", json={"nombre": "Tomás", "entrenador_id": "x"})
    assert r.status_code == 403
    b = cliente.get("/boxeadores").json()
    assert b == []


# ---------------------------------------------------------------------------
# B25
# ---------------------------------------------------------------------------


def test_B25_sin_texto_cargado_no_se_inventa_uno(cliente, entorno, gimnasio, tmp_path, monkeypatch):
    monkeypatch.setattr(entorno["cfg"], "textos", tmp_path / "vacia")
    g = gimnasio
    r = cliente.get("/cuenta/consentimiento", headers=H(g["X"]))
    assert r.status_code == 409 and "DV-5" in r.json()["detail"]
    b = cliente.post("/boxeadores", json={"nombre": "Tomás"}, headers=H(g["T"])).json()["id"]
    assert _consentir(cliente, g["T"], b, "cualquiera").status_code == 409


def test_el_repo_no_trae_un_texto_de_consentimiento_inventado():
    from pathlib import Path

    carpeta = Path(__file__).resolve().parents[1] / "src" / "boxtwin_api" / "textos"
    assert not list(carpeta.glob("consentimiento-*.md")), "el texto lo decide Lucas (DV-5)"


def test_B25_registrar_ver_y_revocar(cliente, gimnasio, texto):
    g = gimnasio
    vig = cliente.get("/cuenta/consentimiento", headers=H(g["X"])).json()
    assert vig == {"version": "2026-10-08", "texto": TEXTO}
    b = cliente.post("/boxeadores", json={"nombre": "Tomás Ríos"}, headers=H(g["T"])).json()
    # CA-G07.1: recien cargado, falta.
    assert b["consentimiento"] == {"estado": "falta", "desde": None}
    bid = b["id"]
    assert cliente.post(f"/boxeadores/{bid}/consentimiento",
                        json={"texto_version": texto, "mayor_de_edad": False},
                        headers=H(g["X"])).status_code == 422
    assert _consentir(cliente, g["X"], bid, "2026-09-01").status_code == 422
    r = cliente.post(f"/boxeadores/{bid}/consentimiento",
                     json={"texto_version": texto, "mayor_de_edad": True, "nota": "en papel"},
                     headers=H(g["X"]))
    assert r.status_code == 201 and r.json()["estado"] == "vigente"
    assert _consentir(cliente, g["Y"], bid, texto).status_code == 409
    assert cliente.delete(f"/boxeadores/{bid}/consentimiento", headers=H(g["Y"])).status_code == 204
    # CA-G08.1: revocar no borra la constancia.
    d = cliente.get(f"/boxeadores/{bid}/consentimiento", headers=H(g["T"])).json()
    assert d["estado"] == "revocado"
    h = d["historia"][0]
    assert (h["registrado_por"], h["revocado_por"], h["nota"]) == ("Ana", "Juan", "en papel")
    lista = cliente.get("/boxeadores", headers=H(g["X"])).json()
    assert lista[0]["consentimiento"]["estado"] == "revocado"
    # Se puede volver a registrar.
    assert _consentir(cliente, g["Y"], bid, texto).status_code == 201
    assert len(cliente.get(f"/boxeadores/{bid}/consentimiento", headers=H(g["T"])).json()["historia"]) == 2


def test_B25_sin_consentimiento_vigente_no_se_asigna_y_desasignar_siempre(cliente, entorno, gimnasio, texto):
    g = gimnasio
    bid = cliente.post("/boxeadores", json={"nombre": "Tomás Ríos"}, headers=H(g["T"])).json()["id"]
    sid = lista_con_fightcard(cliente, entorno, g["X"])
    r = cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": bid}, headers=H(g["X"]))
    assert r.status_code == 422 and r.json()["detail"] == "Falta el consentimiento de Tomás Ríos"
    _consentir(cliente, g["X"], bid, texto)
    assert cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": bid},
                       headers=H(g["X"])).status_code == 200
    cliente.delete(f"/boxeadores/{bid}/consentimiento", headers=H(g["X"]))
    # Revocar frena las nuevas; la sesion anterior se queda con el atleta.
    assert cliente.get(f"/jobs/{sid}", headers=H(g["X"])).json()["boxeadores"]["A"]["id"] == bid
    otra = lista_con_fightcard(cliente, entorno, g["X"])
    assert cliente.put(f"/sesiones/{otra}/boxeadores", json={"boxeador_a": bid},
                       headers=H(g["X"])).status_code == 422
    # Desasignar siempre se puede.
    assert cliente.put(f"/sesiones/{sid}/boxeadores", json={}, headers=H(g["X"])).status_code == 200


def test_B25_el_csv_lo_baja_solo_el_titular(cliente, gimnasio, texto):
    g = gimnasio
    bid = cliente.post("/boxeadores", json={"nombre": "Tomás Ríos"}, headers=H(g["T"])).json()["id"]
    _consentir(cliente, g["X"], bid, texto)
    assert cliente.get("/cuenta/consentimientos.csv", headers=H(g["X"])).status_code == 403
    r = cliente.get("/cuenta/consentimientos.csv", headers=H(g["T"]))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lineas = r.text.strip().splitlines()
    assert lineas[0].startswith("atleta,estado,texto_version,registrado_por")
    assert lineas[1].startswith("Tomás Ríos,vigente,2026-10-08,Ana,")


def test_B25_en_una_cuenta_entrenador_nada_cambia(cliente, registrado, texto):
    b = cliente.post("/boxeadores", json={"nombre": "Tomás"}).json()
    assert b["consentimiento"] is None
    from conftest import subir_video

    sid = subir_video(cliente).json()["job_id"]
    assert cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": b["id"]}).status_code == 200
    assert cliente.get("/cuenta/consentimiento").status_code == 403
    assert _consentir(cliente, registrado, b["id"], texto).status_code == 403


def test_B25_un_consentimiento_de_la_semilla_no_se_revoca(cliente, entorno, texto):
    from boxtwin_api.modelos import Boxeador, Consentimiento

    cid = crear_cuenta(entorno, "gimnasio", "Gimnasio de prueba", demo=True)
    uid, t = crear_miembro(entorno, "gimnasio_titular@prueba.boxtwin", cid)
    with entorno["db"].hacer_sesion() as db:
        b = Boxeador(usuario_id=uid, cuenta_id=cid, nombre="Atleta 3")
        db.add(b)
        db.flush()
        db.add(Consentimiento(cuenta_id=cid, boxeador_id=b.id, texto_version="semilla",
                              registrado_por=uid, semilla=True))
        db.commit()
        bid = b.id
    r = cliente.delete(f"/boxeadores/{bid}/consentimiento", headers=H(t))
    assert r.status_code == 403
    assert r.json()["detail"] == "en la versión de prueba este consentimiento viene cargado"
    # Uno que carga un visitante si se revoca.
    b2 = cliente.post("/boxeadores", json={"nombre": "Atleta 9"}, headers=H(t)).json()["id"]
    assert _consentir(cliente, t, b2, texto).status_code == 201
    assert cliente.delete(f"/boxeadores/{b2}/consentimiento", headers=H(t)).status_code == 204
