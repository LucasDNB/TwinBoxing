"""
B19 de las versiones: cuentas, miembros y acceso (prototipo/specs/versiones-26-10, 04 B19).

I1 de 05 §2: aislamiento entre cuentas. Las rutas se enumeran solas desde la app, asi
que una ruta nueva con {sesion_id} o {boxeador_id} entra a la prueba sin que nadie se
acuerde. Si una ruta nueva pide cuerpo y no esta en CUERPOS, la prueba falla y lo dice: eso
es a proposito, no un olvido de la prueba.
"""

from __future__ import annotations

import re

import pytest

from versiones_util import H, crear_cuenta, crear_miembro, lista_con_fightcard, rutas, subir

# Un cuerpo valido por ruta que lo pide. Sin esto FastAPI contesta 422 antes de mirar la
# cuenta, y la prueba no veria la fuga.
CUERPOS = {
    ("POST", "/jobs/{sesion_id}/siembra"): {"track_a": 1, "track_b": 2},
    ("PUT", "/jobs/{sesion_id}/tramos"): {"asignaciones": [{"track": 1, "rol": "A"}]},
    ("PATCH", "/fightcards/{sesion_id}/golpes/{golpe}"): {"tipo": "jab"},
    ("PATCH", "/boxeadores/{boxeador_id}"): {"nombre": "Otro"},
    ("PUT", "/sesiones/{sesion_id}/boxeadores"): {},
    ("POST", "/boxeadores/{boxeador_id}/consentimiento"): {"texto_version": "v",
                                                          "mayor_de_edad": True},
}
OTROS = {"track": "1", "golpe": "A-1"}


@pytest.fixture
def dos_cuentas(cliente, entorno):
    c1 = crear_cuenta(entorno, "gimnasio", "Gimnasio Norte")
    c2 = crear_cuenta(entorno, "gimnasio", "Gimnasio Sur")
    _, t1 = crear_miembro(entorno, "ana@norte.ar", c1)
    _, t2 = crear_miembro(entorno, "beto@sur.ar", c2)
    s1 = lista_con_fightcard(cliente, entorno, t1)
    s2 = lista_con_fightcard(cliente, entorno, t2)
    b1 = cliente.post("/boxeadores", json={"nombre": "Tomás"}, headers=H(t1)).json()["id"]
    b2 = cliente.post("/boxeadores", json={"nombre": "Tomás"}, headers=H(t2)).json()["id"]
    return {"t1": t1, "t2": t2, "s1": s1, "s2": s2, "b1": b1, "b2": b2}


def _rutas_con_id(app):
    for r in rutas(app):
        params = set(re.findall(r"{(\w+)}", r.path))
        if params & {"sesion_id", "boxeador_id"}:
            yield r, params


def test_I1_toda_ruta_con_id_de_otra_cuenta_da_404(entorno, cliente, dos_cuentas):
    d = dos_cuentas
    revisadas = 0
    for r, params in _rutas_con_id(entorno["app"].app):
        url = r.path
        for p in params:
            valor = {"sesion_id": d["s1"], "boxeador_id": d["b1"]}.get(p, OTROS.get(p, "x"))
            url = url.replace("{" + p + "}", valor)
        for metodo in r.methods - {"HEAD"}:
            cuerpo = CUERPOS.get((metodo, r.path))
            if metodo in ("POST", "PUT", "PATCH") and r.body_field is not None and cuerpo is None:
                pytest.fail(f"{metodo} {r.path} pide cuerpo: agregalo a CUERPOS para probarla")
            resp = cliente.request(metodo, url, json=cuerpo, headers=H(d["t2"]))
            assert resp.status_code == 404, f"{metodo} {r.path} con otra cuenta dio {resp.status_code}"
            revisadas += 1
    # 19 al 26-10, con las tres de consentimiento de cuentas.py. Si baja, alguien saco una ruta o la prueba dejo de encontrarlas.
    assert revisadas >= 19, revisadas


def test_I1_las_listas_no_traen_nada_ajeno(cliente, dos_cuentas):
    d = dos_cuentas
    assert [s["id"] for s in cliente.get("/jobs", headers=H(d["t2"])).json()] == [d["s2"]]
    assert [b["id"] for b in cliente.get("/boxeadores", headers=H(d["t2"])).json()] == [d["b2"]]
    p = cliente.get("/panel", headers=H(d["t2"])).json()
    assert p["sesiones_mes"]["total"] == 1
    assert p["golpes_detectados_mes"] == 4


def test_I1_un_ticket_de_una_cuenta_no_sirve_para_la_otra(cliente, dos_cuentas):
    d = dos_cuentas
    t = cliente.get(f"/videos/{d['s1']}/ticket", headers=H(d["t1"])).json()["ticket"]
    assert cliente.get(f"/videos/{d['s1']}/stream?t={t}").status_code == 200
    # El mismo ticket sobre la sesion de la otra cuenta: no es su sesion.
    assert cliente.get(f"/videos/{d['s2']}/stream?t={t}").status_code in (401, 404)
    # Un ticket de la otra cuenta para su propia sesion, usado sobre esta.
    t2 = cliente.get(f"/videos/{d['s2']}/ticket", headers=H(d["t2"])).json()["ticket"]
    assert cliente.get(f"/videos/{d['s1']}/stream?t={t2}").status_code in (401, 404)
    assert cliente.get(f"/videos/{d['s1']}/procesado?t={t2}").status_code in (401, 404)


def test_un_miembro_dado_de_baja_no_entra_aunque_su_ticket_sea_valido(cliente, entorno, dos_cuentas):
    from boxtwin_api.modelos import Miembro, ahora
    from sqlalchemy import select

    d = dos_cuentas
    t = cliente.get(f"/videos/{d['s1']}/ticket", headers=H(d["t1"])).json()["ticket"]
    with entorno["db"].hacer_sesion() as db:
        m = db.scalars(select(Miembro)).first()
        m.baja_en = ahora()
        db.commit()
    r = cliente.get("/jobs", headers=H(d["t1"]))
    assert r.status_code == 401
    assert r.json()["detail"] == "tu acceso a Gimnasio Norte fue dado de baja"
    assert cliente.get(f"/videos/{d['s1']}/stream?t={t}").status_code == 401


def test_completar_cuentas_dos_veces_no_duplica_nada(cliente, entorno, registrado):
    from boxtwin_api.acceso import completar_cuentas
    from boxtwin_api.modelos import Cuenta, Miembro
    from sqlalchemy import func, select

    subir(cliente, registrado)
    with entorno["db"].hacer_sesion() as db:
        a = completar_cuentas(db)
        db.commit()
        b = completar_cuentas(db)
        db.commit()
        assert a["cuentas"] == 0 and b == {"cuentas": 0, "sesiones": 0, "boxeadores": 0,
                                           "duraciones": 0}
        assert db.scalar(select(func.count(Cuenta.id))) == 1
        assert db.scalar(select(func.count(Miembro.id))) == 1


def test_un_usuario_creado_a_mano_entra_con_una_cuenta_entrenador(cliente, entorno):
    from boxtwin_api.modelos import Usuario
    from boxtwin_api.seguridad import emitir_token, hashear

    with entorno["db"].hacer_sesion() as db:
        u = Usuario(email="mano@usal.edu.ar", hash_clave=hashear("sparring2026"))
        db.add(u)
        db.commit()
        uid = u.id
    c = cliente.get("/cuenta", headers=H(emitir_token(uid, entorno["cfg"].secreto))).json()
    assert c["version"] == "entrenador" and c["rol"] == "titular"
    assert c["nombre"] == "mano@usal.edu.ar"
    assert c["funciones"]["horas_mes"] == 15 and c["funciones"]["rival"] is True


def test_la_migracion_rellena_una_base_anterior_a_las_cuentas(tmp_path, monkeypatch):
    """Una base con usuarios, sesiones y peleadores sin cuenta queda migrada al arrancar."""
    import importlib
    import json
    import sqlite3
    import sys

    base = tmp_path / "vieja.db"
    datos = tmp_path / "datos"
    con = sqlite3.connect(base)
    con.executescript("""
        CREATE TABLE usuarios (id VARCHAR(32) PRIMARY KEY, email VARCHAR(320), hash_clave VARCHAR(255), creado DATETIME);
        CREATE TABLE boxeadores (id VARCHAR(32) PRIMARY KEY, usuario_id VARCHAR(32), nombre VARCHAR(120),
            guardia VARCHAR(16), notas TEXT, categoria VARCHAR(40), altura_m FLOAT, envergadura_m FLOAT,
            distancia VARCHAR(8), estilo VARCHAR(16), creado DATETIME, CONSTRAINT uq_boxeador_nombre UNIQUE (usuario_id, nombre));
        CREATE TABLE sesiones (id VARCHAR(32) PRIMARY KEY, usuario_id VARCHAR(32), nombre VARCHAR(255),
            estado VARCHAR(32), error TEXT, video_nombre VARCHAR(255), duracion_s FLOAT, fps FLOAT,
            round_s FLOAT, descanso_s FLOAT, semilla_a INTEGER, semilla_b INTEGER, boxeador_a_id VARCHAR(32),
            boxeador_b_id VARCHAR(32), contexto VARCHAR(16), practica VARCHAR(24), rounds INTEGER,
            consentimiento_en DATETIME, consentimiento_por VARCHAR(320), creada DATETIME, actualizada DATETIME);
        INSERT INTO usuarios VALUES ('u1', 'lucas@usal.edu.ar', 'x', '2026-10-01 10:00:00');
        INSERT INTO boxeadores (id, usuario_id, nombre, creado) VALUES ('b1', 'u1', 'Tomás', '2026-10-01 10:00:00');
        INSERT INTO sesiones (id, usuario_id, nombre, estado, video_nombre, descanso_s, creada, actualizada)
            VALUES ('s1', 'u1', 'viejo', 'listo', 'v.mp4', 60, '2026-10-02 10:00:00', '2026-10-02 10:00:00');
    """)
    con.commit()
    con.close()
    (datos / "sesiones" / "s1").mkdir(parents=True)
    (datos / "sesiones" / "s1" / "sesion.json").write_text(
        json.dumps({"video": {"duracion_s": 205.2, "fps": 29.97}}))

    monkeypatch.setenv("BOXTWIN_DB", f"sqlite:///{base}")
    monkeypatch.setenv("BOXTWIN_DATOS", str(datos))
    import boxtwin_api.config as config

    importlib.reload(config)
    import boxtwin_api.db as db_mod

    importlib.reload(db_mod)
    importlib.reload(sys.modules["boxtwin_api.acceso"])
    db_mod.crear_tablas()
    db_mod.crear_tablas()   # dos veces: idempotente

    con = sqlite3.connect(base)
    cuentas = con.execute("SELECT id, nombre, version FROM cuentas").fetchall()
    assert len(cuentas) == 1 and cuentas[0][1:] == ("lucas@usal.edu.ar", "entrenador")
    cid = cuentas[0][0]
    assert con.execute("SELECT cuenta_id, rol FROM miembros").fetchall() == [(cid, "titular")]
    assert con.execute("SELECT cuenta_id, duracion_s, fps FROM sesiones").fetchall() == [(cid, 205.2, 29.97)]
    assert con.execute("SELECT cuenta_id FROM boxeadores").fetchall() == [(cid,)]
    indices = {r[1] for r in con.execute("PRAGMA index_list(boxeadores)")}
    assert "uq_boxeador_cuenta_nombre" in indices
    con.close()


def test_el_nombre_de_peleador_es_unico_por_cuenta_y_no_global(cliente, dos_cuentas):
    d = dos_cuentas
    # El fixture ya creo "Tomás" en las dos cuentas: entre cuentas se repite.
    r = cliente.post("/boxeadores", json={"nombre": "Tomás"}, headers=H(d["t1"]))
    assert r.status_code == 409


def test_en_gimnasio_un_segundo_entrenador_no_repite_el_nombre(cliente, entorno, dos_cuentas):
    from boxtwin_api.modelos import Miembro
    from sqlalchemy import select

    with entorno["db"].hacer_sesion() as db:
        cid = db.scalars(select(Miembro)).first().cuenta_id
    _, tx = crear_miembro(entorno, "x@norte.ar", cid, rol="entrenador")
    assert cliente.post("/boxeadores", json={"nombre": "Tomás"}, headers=H(tx)).status_code == 409


def test_get_cuenta_de_una_cuenta_entrenador_es_la_de_hoy(cliente, registrado):
    c = cliente.get("/cuenta").json()
    assert c["version"] == "entrenador" and c["demo"] is False
    assert c["boxeador_propio"] is None and c["perfil_demo"] is None
    assert c["horas"]["incluidas"] == 15
    assert c["usuario"]["email"] == "lucas@usal.edu.ar"
