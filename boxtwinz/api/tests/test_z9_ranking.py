"""Z9: aptitud para el ranking (spec 6.4) y perfiles publicos."""

import pytest
from sqlalchemy import select

from boxtwinz_api.config import cfg
from boxtwinz_api.models import Perfil, Sesion, Usuario
from boxtwinz_api.services.ingest import es_apta_ranking
from conftest import golpes_cada, registrar, sesion_lista


def _publico(c):
    p1 = c.get("/api/perfiles").json()[0]
    c.patch(f"/api/perfiles/{p1['id']}", json={"publico": True})
    return p1["id"]


def _apta(db, **cambios):
    u = Usuario(email="x@x.ar", hash="", nombre="x", consentimiento_uso_at=None, consentimiento_uso_version="v")
    p = Perfil(usuario_id="u", ranura=1, nombre="x", estilo="propio", publico=True)
    s = Sesion(usuario_id="u", perfil_id="p", modalidad="sombra", video_ruta="", video_duracion_s=180.0, calidad_pose=0.70)
    for k, v in cambios.items():
        setattr({"u": u, "p": p, "s": s}[k[0]], k[2:], v)
    return es_apta_ranking(s, p, u)


def test_aptitud_borde_exacto(db):
    assert _apta(db)                                   # 180 s y 0,70 justos
    assert not _apta(db, s_video_duracion_s=179.9)
    assert not _apta(db, s_calidad_pose=0.699)
    assert not _apta(db, s_calidad_pose=None)
    assert not _apta(db, p_publico=False)
    assert not _apta(db, p_ranura=2)
    assert not _apta(db, p_fuera_de_ranking=True)
    assert not _apta(db, u_suspendido=True)


def test_aptitud_umbral_sale_de_config(db, monkeypatch):
    monkeypatch.setattr(cfg, "calidad_pose_min", 0.5)
    assert _apta(db, s_calidad_pose=0.6)


def test_ranking_ordena_y_da_posicion_propia(nuevo_cliente, db):
    a, b, c = nuevo_cliente(), nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    registrar(c, email="c@x.ar")
    _publico(a)
    _publico(b)                                         # c queda privado
    sesion_lista(db, "a@x.ar", golpes=golpes_cada(2.0, 240))     # 30 por minuto
    sesion_lista(db, "b@x.ar", golpes=golpes_cada(1.0, 240))     # 60 por minuto
    sesion_lista(db, "c@x.ar", golpes=golpes_cada(0.5, 240))     # privado, no entra
    sesion_lista(db, "a@x.ar", golpes=golpes_cada(0.2, 100), duracion=100)  # corta, no apta
    r = a.get("/api/ranking", params={"modalidad": "sombra", "metrica": "volumen"}).json()
    assert [x["nombre"] for x in r["top"]] == ["b", "a"]
    assert r["propia"]["posicion"] == 2 and r["propia"]["valor"] == pytest.approx(30.0)
    assert r["pie"] == "Ranking de actividad: cuenta golpes detectados, no golpes que conectaron"
    assert c.get("/api/ranking", params={"modalidad": "sombra"}).json()["propia"] is None


def test_ranking_valida_parametros(cliente):
    registrar(cliente)
    assert cliente.get("/api/ranking", params={"modalidad": "kick"}).status_code == 422
    assert cliente.get("/api/ranking", params={"modalidad": "sombra", "metrica": "conectados"}).status_code == 422


def test_volverse_privado_saca_del_ranking_al_momento(cliente, db):
    registrar(cliente)
    pid = _publico(cliente)
    sesion_lista(db, "ana@boxtwinz.ar", golpes=golpes_cada(1.0, 240))
    assert cliente.get("/api/ranking", params={"modalidad": "sombra"}).json()["total"] == 1
    cliente.patch(f"/api/perfiles/{pid}", json={"publico": False})
    assert cliente.get("/api/ranking", params={"modalidad": "sombra"}).json()["total"] == 0


def test_publicos_lista_y_detalle_sin_cajones(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    pid = _publico(a)
    lista = b.get("/api/publicos", params={"q": "a"}).json()
    assert [x["perfil_id"] for x in lista] == [pid]
    d = b.get(f"/api/publicos/{pid}").json()
    assert set(d["stats"]) == {"sombra", "bolsa", "sparring"}
    assert "perfiles" not in d and "cajones" not in d
    p2 = a.get("/api/perfiles").json()[1]["id"]
    assert b.get(f"/api/publicos/{p2}").status_code == 404     # un cajon nunca es publico


def test_perfil_privado_no_se_ve(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]["id"]
    assert b.get(f"/api/publicos/{p1}").status_code == 404
