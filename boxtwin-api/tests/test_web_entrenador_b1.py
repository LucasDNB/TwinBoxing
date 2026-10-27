"""
B1 de la web entrenador: contexto de la sesion, rounds y consentimiento
(prototipo/specs/05-api-y-datos.md).

El consentimiento es una contramedida comprometida en el Cap 6 (R-09): sin la casilla no se
sube, y eso tiene que valer tambien para un cliente que no sea la web (CA-02.2).
"""

from __future__ import annotations

import sqlite3

from conftest import subir_video


def _subir_sin_consentimiento(cliente, **datos):
    return cliente.post(
        "/videos",
        files={"archivo": ("spar.mp4", b"bytes", "video/mp4")},
        data={"nombre": "x", **datos},
    )


def test_sin_consentimiento_no_se_sube(cliente, registrado, entorno):
    r = _subir_sin_consentimiento(cliente)
    assert r.status_code == 422
    assert r.json()["detail"] == "falta el consentimiento de las personas filmadas"
    assert cliente.get("/jobs").json() == [], "no queda una sesion creada"
    assert not (entorno["cfg"].datos / "sesiones").exists() or not any(
        (entorno["cfg"].datos / "sesiones").iterdir()
    ), "ni un archivo en disco"


def test_consentimiento_distinto_de_true_tampoco(cliente, registrado):
    assert _subir_sin_consentimiento(cliente, consentimiento="false").status_code == 422
    assert _subir_sin_consentimiento(cliente, consentimiento="si").status_code == 422


def test_el_consentimiento_guarda_quien_y_cuando(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        assert s.consentimiento_por == "lucas@usal.edu.ar"
        assert s.consentimiento_en is not None
    assert cliente.get(f"/jobs/{sid}").json()["consentimiento"] is True


def test_por_defecto_es_sparring_con_rounds_vacios(cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    d = cliente.get(f"/jobs/{sid}").json()
    assert d["contexto"] == "sparring"
    assert d["practica"] is None
    assert d["rounds"]["cantidad"] is None
    assert d["boxeadores"] == {"A": None, "B": None}


def test_contexto_rounds_y_practica_llegan_a_la_lista(cliente, registrado):
    subir_video(cliente, contexto="sombra", practica="combinaciones", rounds="3",
                round_s="180", descanso_s="60")
    d = cliente.get("/jobs").json()[0]
    assert d["contexto"] == "sombra"
    assert d["practica"] == "combinaciones"
    assert d["rounds"] == {"round_s": 180.0, "descanso_s": 60.0, "cantidad": 3}


def test_practica_solo_con_sombra(cliente, registrado):
    r = subir_video(cliente, contexto="sparring", practica="defensa")
    assert r.status_code == 422


def test_valores_fuera_de_la_lista_se_rechazan(cliente, registrado):
    assert subir_video(cliente, contexto="pelea").status_code == 422
    assert subir_video(cliente, contexto="sombra", practica="yoga").status_code == 422
    assert subir_video(cliente, rounds="0").status_code == 422
    assert subir_video(cliente, rounds="tres").status_code == 422


def test_la_lista_trae_los_nombres_de_los_peleadores(cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    a = cliente.post("/boxeadores", json={"nombre": "Ana Paz"}).json()
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": a["id"]})
    d = cliente.get("/jobs").json()[0]
    assert d["boxeadores"] == {"A": {"id": a["id"], "nombre": "Ana Paz"}, "B": None}


def test_una_base_vieja_recibe_las_columnas_nuevas_y_sus_sesiones_se_leen(tmp_path, monkeypatch):
    # Una base creada antes de B1: la tabla sesiones sin contexto ni consentimiento.
    ruta = tmp_path / "vieja.db"
    con = sqlite3.connect(ruta)
    con.executescript("""
        CREATE TABLE usuarios (id VARCHAR(32) PRIMARY KEY, email VARCHAR(320), hash_clave VARCHAR(255), creado DATETIME);
        CREATE TABLE boxeadores (id VARCHAR(32) PRIMARY KEY, usuario_id VARCHAR(32), nombre VARCHAR(120), guardia VARCHAR(16), notas TEXT, creado DATETIME);
        CREATE TABLE sesiones (id VARCHAR(32) PRIMARY KEY, usuario_id VARCHAR(32), nombre VARCHAR(255),
            estado VARCHAR(32), error TEXT, video_nombre VARCHAR(255), duracion_s FLOAT, fps FLOAT,
            round_s FLOAT, descanso_s FLOAT, semilla_a INTEGER, semilla_b INTEGER,
            boxeador_a_id VARCHAR(32), boxeador_b_id VARCHAR(32), creada DATETIME, actualizada DATETIME);
        INSERT INTO usuarios VALUES ('u1', 'a@b.com', 'x', '2026-09-22 10:00:00');
        INSERT INTO sesiones (id, usuario_id, nombre, estado, video_nombre, descanso_s, creada, actualizada)
            VALUES ('s1', 'u1', 'vieja', 'listo', 'v.mp4', 60, '2026-09-22 10:00:00', '2026-09-22 10:00:00');
    """)
    con.commit()
    con.close()

    monkeypatch.setenv("BOXTWIN_DB", f"sqlite:///{ruta}")
    import importlib

    import boxtwin_api.config as config
    import boxtwin_api.db as db_mod

    importlib.reload(config)
    importlib.reload(db_mod)
    db_mod.crear_tablas()

    from boxtwin_api.esquemas import sesion_a_dict
    from boxtwin_api.modelos import Sesion

    with db_mod.hacer_sesion() as db:
        d = sesion_a_dict(db.get(Sesion, "s1"))
    assert d["contexto"] == "sparring", "una sesion sin contexto se lee como sparring"
    assert d["consentimiento"] is False, "no se inventa un consentimiento que nadie declaro"


def test_una_subida_cortada_a_mitad_no_deja_sesion(cliente, registrado, entorno):
    # CA-02.3 del lado de la API: cancelar en la web es xhr.abort(), que deja al servidor con
    # un multipart sin terminar. FastAPI recibe el cuerpo entero antes de llamar a la ruta, y
    # la sesion se crea recien adentro de la ruta: un cuerpo cortado no llega a crear nada.
    borde = "----boxtwin"
    cuerpo = (
        f"--{borde}\r\nContent-Disposition: form-data; name=\"consentimiento\"\r\n\r\ntrue\r\n"
        f"--{borde}\r\nContent-Disposition: form-data; name=\"archivo\"; filename=\"spar.mp4\"\r\n"
        "Content-Type: video/mp4\r\n\r\n" + "x" * 4000
    ).encode()   # sin el borde de cierre: la subida se corto al 40%
    r = cliente.post("/videos", content=cuerpo,
                     headers={"Content-Type": f"multipart/form-data; boundary={borde}"})
    assert r.status_code in (400, 422)
    assert cliente.get("/jobs").json() == []
