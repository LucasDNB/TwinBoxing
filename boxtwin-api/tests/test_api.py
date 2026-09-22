"""
La API: cuentas, subida, estado, siembra y Fight-Card.

El bloque que mas importa es el de aislamiento. RNF3 -los videos y resultados de un usuario
solo son visibles para ese usuario- no se verifica leyendo el codigo: se verifica pidiendo
cada recurso con el token del otro y viendo que no llegue.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video


# -- cuentas ----------------------------------------------------------------


def test_registro_y_login(cliente):
    r = cliente.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    assert r.status_code == 200
    assert r.json()["usuario"] == "a@b.com"

    r = cliente.post("/auth/login", json={"email": "a@b.com", "clave": "clavelarga1"})
    assert r.status_code == 200 and r.json()["token"]


def test_no_se_repite_un_email(cliente):
    cliente.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    r = cliente.post("/auth/registro", json={"email": "A@b.com", "clave": "otraclave1"})
    assert r.status_code == 409, "el email no distingue mayusculas"


def test_la_clave_equivocada_y_el_email_inexistente_dicen_lo_mismo(cliente):
    cliente.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    r1 = cliente.post("/auth/login", json={"email": "a@b.com", "clave": "equivocada1"})
    r2 = cliente.post("/auth/login", json={"email": "z@b.com", "clave": "equivocada1"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"], (
        "distinguirlos le dice a cualquiera cuales emails tienen cuenta"
    )


def test_la_clave_no_se_guarda_en_claro(cliente, entorno):
    cliente.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    from boxtwin_api.modelos import Usuario
    from sqlalchemy import select

    with entorno["db"].hacer_sesion() as db:
        u = db.scalar(select(Usuario))
    assert "clavelarga1" not in u.hash_clave
    assert u.hash_clave.startswith("scrypt$")


def test_sin_token_no_se_entra(cliente):
    assert cliente.post("/videos", files={"archivo": ("a.mp4", b"x", "video/mp4")}).status_code == 401
    assert cliente.get("/jobs").status_code == 401


def test_un_token_firmado_por_otro_no_sirve(cliente, registrado):
    from boxtwin_api.seguridad import emitir_token

    cliente.headers["Authorization"] = f"Bearer {emitir_token('abc', 'otro-secreto')}"
    assert cliente.get("/jobs").status_code == 401


# -- subida -----------------------------------------------------------------


def test_subir_devuelve_el_id_sin_esperar(cliente, registrado):
    # RF1. Si esto esperara el procesamiento, el celular del usuario mantendria abierta una
    # conexion durante media hora.
    r = subir_video(cliente, round_s=180.0)
    assert r.status_code == 202
    assert r.json()["estado"] == "en_cola"
    assert r.json()["job_id"]


def test_la_subida_deja_el_trabajo_encolado_en_el_mismo_commit(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    from boxtwin_api.modelos import Trabajo
    from sqlalchemy import select

    with entorno["db"].hacer_sesion() as db:
        t = db.scalars(select(Trabajo).where(Trabajo.sesion_id == sid)).all()
    assert [x.etapa for x in t] == ["procesar"]
    assert t[0].estado == "en_cola"


def test_un_formato_que_no_es_video_se_rechaza(cliente, registrado):
    r = cliente.post("/videos", files={"archivo": ("apunte.pdf", b"x", "application/pdf")})
    assert r.status_code == 415


def test_el_nombre_del_archivo_no_elige_rutas(cliente, registrado, entorno):
    # Un nombre con ../ no puede escribir afuera del directorio de la sesion.
    r = subir_video(cliente, nombre="../../afuera.mp4")
    assert r.status_code == 202
    sid = r.json()["job_id"]
    videos = list((entorno["cfg"].dir_sesion(sid) / "videos").iterdir())
    assert len(videos) == 1
    assert ".." not in videos[0].name


def test_un_round_de_cero_no_se_acepta(cliente, registrado):
    assert subir_video(cliente, round_s=0).status_code == 422


# -- aislamiento entre usuarios (RNF3) --------------------------------------


@pytest.fixture
def sesion_ajena(cliente, registrado):
    """Una sesion de otro usuario, y el cliente vuelve al primero."""
    sid = subir_video(cliente).json()["job_id"]
    propio = cliente.headers["Authorization"]
    r = cliente.post("/auth/registro", json={"email": "otro@b.com", "clave": "otraclave1"})
    cliente.headers["Authorization"] = f"Bearer {r.json()['token']}"
    yield sid
    cliente.headers["Authorization"] = propio


def test_otro_usuario_no_ve_la_sesion(cliente, sesion_ajena):
    for ruta in (f"/jobs/{sesion_ajena}", f"/fightcards/{sesion_ajena}",
                 f"/videos/{sesion_ajena}/stream",
                 f"/jobs/{sesion_ajena}/candidatos/3"):
        assert cliente.get(ruta).status_code == 404, ruta


def test_otro_usuario_no_puede_sembrar_ni_corregir(cliente, sesion_ajena):
    assert cliente.post(f"/jobs/{sesion_ajena}/siembra",
                        json={"track_a": 1, "track_b": 2}).status_code == 404
    assert cliente.patch(f"/fightcards/{sesion_ajena}/golpes/A-izq-10",
                         json={"tipo": "jab"}).status_code == 404


def test_el_listado_es_solo_del_usuario(cliente, sesion_ajena):
    assert cliente.get("/jobs").json() == []


def test_una_sesion_ajena_devuelve_404_y_no_403(cliente, sesion_ajena):
    # 403 confirmaria que el id existe, que es informacion que no hay por que dar.
    assert cliente.get(f"/jobs/{sesion_ajena}").json()["detail"] == "no hay una sesion con ese id"


# -- estado y siembra -------------------------------------------------------


def _dejar_en_espera_siembra(cfg, sid, candidatos=(3, 7)):
    d = cfg.dir_sesion(sid)
    (d / "candidatos").mkdir(parents=True, exist_ok=True)
    for t in candidatos:
        (d / "candidatos" / f"track_{t}.jpg").write_bytes(b"\xff\xd8\xff jpeg de mentira")
    (d / "sesion.json").write_text(json.dumps({
        "estado": "espera_siembra",
        "candidatos": [{"track": t, "cuadro": 100 * i, "recorte": f"candidatos/track_{t}.jpg",
                        "fraccion_guante": 0.8, "alto": 0.9, "cuadros_con_color": 40,
                        "coexiste_con": candidatos[1 - i], "cuadros_de_coexistencia": 40}
                       for i, t in enumerate(candidatos)],
        "identidad": {"pareja_sugerida": list(candidatos)},
        "avisos": [], "etapas": [{"etapa": "preproceso", "segundos": 12.0}],
    }))


def test_el_estado_trae_los_candidatos_con_su_recorte(cliente, registrado, entorno):
    # RF3: dos candidatos con un recorte de cada uno.
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)

    d = cliente.get(f"/jobs/{sid}").json()
    assert d["estado"] == "espera_siembra"
    assert [c["track"] for c in d["candidatos"]] == [3, 7]
    assert d["candidatos"][0]["recorte_url"] == f"/jobs/{sid}/candidatos/3"
    assert d["pareja_sugerida"] == [3, 7]

    r = cliente.get(d["candidatos"][0]["recorte_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_la_siembra_encola_la_segunda_etapa(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)
    cliente.get(f"/jobs/{sid}")     # sincroniza el estado desde disco

    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 7, "track_b": 3})
    assert r.status_code == 202

    d = cliente.get(f"/jobs/{sid}").json()
    assert d["siembra"] == {"track_a": 7, "track_b": 3}
    assert [t["etapa"] for t in d["trabajos"]] == ["procesar", "completar"]


def test_no_se_siembra_antes_de_que_haya_candidatos(cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 1, "track_b": 2})
    assert r.status_code == 409


def test_las_dos_semillas_no_pueden_ser_el_mismo_track(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)
    cliente.get(f"/jobs/{sid}")
    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3, "track_b": 3})
    assert r.status_code == 422


def test_se_puede_volver_a_sembrar_una_sesion_lista(cliente, registrado, entorno):
    # Si el entrenador ve que A y B salieron al reves, corregirlo cuesta minutos y no
    # obliga a volver a subir el video.
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)
    cliente.get(f"/jobs/{sid}")
    cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 3, "track_b": 7})

    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        s.estado = "listo"
        db.commit()
    r = cliente.post(f"/jobs/{sid}/siembra", json={"track_a": 7, "track_b": 3})
    assert r.status_code == 202


# -- quien puede crearse una cuenta -----------------------------------------
#
# Es lo primero que importa cuando esto sale por un tunel: una instancia con el registro
# abierto es una GPU ajena gratis para cualquiera que tenga la URL, y la GPU es una sola.


def _recargar(monkeypatch, **entorno):
    """Levanta la app de nuevo con otra configuracion de registro."""
    import importlib
    import sys

    for k, v in entorno.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import boxtwin_api.config as config

    importlib.reload(config)
    for nombre in ("boxtwin_api.db", "boxtwin_api.cola", "boxtwin_api.app"):
        importlib.reload(sys.modules[nombre])
    import boxtwin_api.app as app_mod
    from fastapi.testclient import TestClient

    app_mod.crear_tablas()
    return TestClient(app_mod.app)


def test_sin_configurar_nada_el_registro_esta_cerrado(entorno, monkeypatch):
    # El default seguro. Si alguien despliega y se olvida de la variable, no se abre solo.
    c = _recargar(monkeypatch, BOXTWIN_INVITACION=None)
    r = c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    assert r.status_code == 403
    assert "BOXTWIN_INVITACION" in r.json()["detail"], "y el error dice como abrirlo"


def test_con_codigo_hace_falta_el_codigo(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_INVITACION="gimnasio-2026")
    sin = c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    assert sin.status_code == 403

    mal = c.post("/auth/registro",
                 json={"email": "a@b.com", "clave": "clavelarga1", "invitacion": "otro"})
    assert mal.status_code == 403

    bien = c.post("/auth/registro",
                  json={"email": "a@b.com", "clave": "clavelarga1",
                        "invitacion": "gimnasio-2026"})
    assert bien.status_code == 200


def test_cerrar_el_registro_no_deja_afuera_al_que_ya_tiene_cuenta(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_INVITACION="abierto")
    c.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})

    cerrada = _recargar(monkeypatch, BOXTWIN_INVITACION=None)
    r = cerrada.post("/auth/login", json={"email": "a@b.com", "clave": "clavelarga1"})
    assert r.status_code == 200


def test_salud_declara_lo_que_hay_que_mirar_antes_de_abrir_el_tunel(cliente):
    d = cliente.get("/salud").json()
    assert d["registro"] == "abierto"
    assert d["secreto_efimero"] is False
    assert d["frontend"] is False, "en los tests no hay frontend construido"


# -- el frontend servido por la misma API -----------------------------------


def test_la_api_sirve_el_frontend_si_esta_construido(entorno, monkeypatch, tmp_path):
    # Un solo origen: con el frontend en otro puerto harian falta dos tuneles y CORS.
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>BoxTwin</title>")
    (dist / "assets" / "app.js").write_text("console.log('hola')")

    c = _recargar(monkeypatch, BOXTWIN_WEB=str(dist), BOXTWIN_INVITACION="abierto")
    r = c.get("/")
    assert r.status_code == 200 and "BoxTwin" in r.text
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/salud").json()["frontend"] is True


def test_el_frontend_no_tapa_las_rutas_de_la_api(entorno, monkeypatch, tmp_path):
    # El mount matchea todo, asi que si quedara antes que las rutas, /salud devolveria el
    # index.html y la app quedaria muerta sin un solo error.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>BoxTwin</title>")

    c = _recargar(monkeypatch, BOXTWIN_WEB=str(dist), BOXTWIN_INVITACION="abierto")
    assert c.get("/salud").json()["ok"] is True
    assert c.post("/auth/login",
                  json={"email": "x@y.com", "clave": "clavelarga1"}).status_code == 401


def test_la_documentacion_interactiva_esta_apagada_por_default(cliente):
    # No filtra datos, pero publica la lista de endpoints a cualquiera que tenga la URL.
    # Detras de un tunel eso es superficie que nadie pidio.
    for ruta in ("/docs", "/redoc", "/openapi.json"):
        assert cliente.get(ruta).status_code == 404, ruta
    assert cliente.get("/salud").json()["docs"] is False


def test_pero_se_puede_prender_para_desarrollar(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_DOCS="1", BOXTWIN_INVITACION="abierto")
    assert c.get("/docs").status_code == 200
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/salud").json()["docs"] is True


# -- el recorte, que lo pide un <img> ---------------------------------------


def test_el_recorte_entra_con_ticket_porque_un_img_no_manda_headers(
    cliente, registrado, entorno
):
    # El bug del 22-09: la pantalla de siembra mostraba seis iconos rotos. El navegador
    # pedia la imagen sin token, la API contestaba 401, y el paso donde el usuario tiene
    # que mirar dos fotos y elegir quedaba inusable.
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)

    tk = cliente.get(f"/videos/{sid}/ticket").json()["ticket"]
    r = cliente.get(f"/jobs/{sid}/candidatos/3?t={tk}", headers={"Authorization": ""})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_sin_ticket_ni_header_el_recorte_no_sale(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)
    assert cliente.get(f"/jobs/{sid}/candidatos/3",
                       headers={"Authorization": ""}).status_code == 401


def test_un_ticket_de_otra_sesion_no_abre_este_recorte(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    otra = subir_video(cliente, nombre="otra.mp4").json()["job_id"]
    _dejar_en_espera_siembra(entorno["cfg"], sid)
    tk = cliente.get(f"/videos/{otra}/ticket").json()["ticket"]
    assert cliente.get(f"/jobs/{sid}/candidatos/3?t={tk}",
                       headers={"Authorization": ""}).status_code == 401


# -- el avance de la etapa que esta corriendo -------------------------------


def test_el_estado_trae_el_avance_de_la_etapa_en_curso(cliente, registrado, entorno):
    # sesion.json se escribe recien cuando la etapa termina. Sobre 30 minutos de video la
    # pose son 45 minutos, y sin esto la pantalla de espera es una rueda quieta.
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "progreso.json").write_text(json.dumps({
        "etapa": "pose", "hecho": 1400, "total": 3741,
        "fraccion": 0.3742, "desde": 0, "segundos": 110.0,
    }))
    p = cliente.get(f"/jobs/{sid}").json()["progreso"]
    assert p["etapa"] == "pose"
    assert p["hecho"] == 1400 and p["total"] == 3741


def test_sin_etapa_corriendo_no_hay_avance(cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    assert cliente.get(f"/jobs/{sid}").json()["progreso"] is None


# -- columnas nuevas sobre una base que ya existe ----------------------------


def test_una_columna_nueva_se_agrega_a_una_tabla_que_ya_tiene_filas(tmp_path, monkeypatch):
    """
    create_all crea tablas nuevas pero NO altera las existentes, asi que agregar una columna
    al modelo dejaba una instalacion andando rota al primer reinicio: el codigo la pide y la
    base no la tiene, y el sintoma aparece recien al abrir una sesion vieja.

    Se simula la base anterior creandola entera y sacandole las dos columnas nuevas, que es
    exactamente el estado del despliegue antes de este cambio.
    """
    import sqlalchemy as sa

    from boxtwin_api import db as mod_db

    ruta = tmp_path / "vieja.db"
    motor = sa.create_engine(f"sqlite:///{ruta}")
    mod_db.Base.metadata.create_all(motor)
    nuevas = {"boxeador_a_id", "boxeador_b_id"}
    viejas = [c.name for c in mod_db.Base.metadata.tables["sesiones"].columns
              if c.name not in nuevas]
    with motor.begin() as con:
        con.execute(sa.text("DROP TABLE boxeadores"))
        # SQLite no deja borrar una columna que participa de una clave foranea, asi que la
        # tabla se rehace con las columnas de antes.
        con.execute(sa.text("ALTER TABLE sesiones RENAME TO sesiones_vieja"))
        con.execute(sa.text(
            f"CREATE TABLE sesiones AS SELECT {', '.join(viejas)} FROM sesiones_vieja WHERE 0"
        ))
        con.execute(sa.text("DROP TABLE sesiones_vieja"))
        con.execute(sa.text(
            "INSERT INTO usuarios (id, email, hash_clave, creado) "
            "VALUES ('u1', 'a@b.com', 'x', '2026-01-01')"
        ))
        con.execute(sa.text(
            "INSERT INTO sesiones (id, usuario_id, nombre, estado, video_nombre, "
            "descanso_s, creada, actualizada) "
            "VALUES ('s1', 'u1', 'spar.mp4', 'listo', 'spar.mp4', 60, "
            "'2026-01-01', '2026-01-01')"
        ))
    motor.dispose()

    monkeypatch.setattr(mod_db, "engine", sa.create_engine(f"sqlite:///{ruta}"))
    agregadas = mod_db._columnas_faltantes()
    assert sorted(agregadas) == ["sesiones.boxeador_a_id", "sesiones.boxeador_b_id"]

    with mod_db.engine.begin() as con:
        cols = {r[1] for r in con.execute(sa.text("PRAGMA table_info(sesiones)"))}
        # La sesion que ya estaba sigue ahi: se agregan columnas, no se recrea la tabla.
        assert con.execute(sa.text("SELECT count(*) FROM sesiones")).scalar() == 1
    assert "boxeador_a_id" in cols and "boxeador_b_id" in cols


def test_no_se_repite_si_ya_estan(tmp_path, monkeypatch):
    import sqlalchemy as sa

    from boxtwin_api import db as mod_db

    ruta = tmp_path / "nueva.db"
    monkeypatch.setattr(mod_db, "engine", sa.create_engine(f"sqlite:///{ruta}"))
    mod_db.Base.metadata.create_all(mod_db.engine)
    assert mod_db._columnas_faltantes() == []


def test_una_columna_obligatoria_sin_default_no_se_inventa(tmp_path, monkeypatch):
    """
    Agregarla pediria un valor para las filas que ya estan, y ese valor no lo puede elegir
    un programa. Se para y se dice, en vez de poner algo plausible.
    """
    import sqlalchemy as sa

    from boxtwin_api import db as mod_db

    ruta = tmp_path / "corta.db"
    motor = sa.create_engine(f"sqlite:///{ruta}")
    with motor.begin() as con:
        con.execute(sa.text("CREATE TABLE sesiones (id TEXT PRIMARY KEY)"))
    motor.dispose()
    monkeypatch.setattr(mod_db, "engine", sa.create_engine(f"sqlite:///{ruta}"))
    with pytest.raises(RuntimeError, match="a mano"):
        mod_db._columnas_faltantes()


# -- el frontend no puede tapar a la API ------------------------------------


def test_el_montaje_del_frontend_va_ultimo(entorno, monkeypatch, tmp_path):
    """
    El mount en "/" matchea TODO, asi que cualquier ruta declarada despues queda tapada.

    Y el sintoma es enganoso: una ruta tapada no da 404 sino 405, porque el que contesta es
    StaticFiles y solo acepta GET y HEAD. Eso se lee como un bug del metodo y no como una
    ruta que no existe, y costo dos rondas de diagnostico el 22-09.
    """
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>BoxTwin</title>")
    c = _recargar(monkeypatch, BOXTWIN_WEB=str(dist), BOXTWIN_INVITACION="abierto")

    from starlette.routing import Mount

    import boxtwin_api.app as app_mod

    rutas = app_mod.app.router.routes
    montajes = [i for i, r in enumerate(rutas) if isinstance(r, Mount) and r.path == ""]
    assert montajes, "el frontend tiene que estar montado"
    assert montajes[0] == len(rutas) - 1, (
        "el montaje del frontend tiene que ser la ULTIMA ruta: lo que venga despues "
        "queda tapado y contesta 405"
    )
    # Y la comprobacion de verdad, por HTTP: un POST a una ruta de la API no puede caer
    # en el frontend.
    assert c.post("/auth/login",
                  json={"email": "x@y.com", "clave": "clavelarga1"}).status_code == 401


def test_salud_dice_cuando_arranco_el_proceso(cliente):
    # Un servicio que quedo corriendo codigo viejo no se distingue de uno al dia sin esto.
    d = cliente.get("/salud").json()
    assert d["arrancado"], "hace falta para saber si el proceso es de antes del deploy"
    assert "commit" in d
