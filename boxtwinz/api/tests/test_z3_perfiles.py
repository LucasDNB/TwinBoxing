"""Z3: cuatro ranuras al registrarse, seed de arquetipos y permisos free/premium."""

from boxtwinz_api.services.archetypes import curva_sintetica, leer_seed
from conftest import registrar


def test_cuatro_ranuras_al_registrarse(cliente):
    registrar(cliente, guardia="zurda")
    ps = cliente.get("/api/perfiles").json()
    assert [p["ranura"] for p in ps] == [1, 2, 3, 4]
    assert ps[0]["estilo"] == "propio" and ps[0]["guardia"] == "zurda"
    assert [p["estilo"] for p in ps[1:]] == ["fajador", "estilista", "hibrido"]
    assert all(p["rotulo"] == "Perfil simulado" and not p["customizado"] for p in ps[1:])


def test_free_no_edita_cajones(cliente):
    registrar(cliente)
    p2 = cliente.get("/api/perfiles").json()[1]
    r = cliente.patch(f"/api/perfiles/{p2['id']}", json={"nombre": "Mi sparring"})
    assert r.status_code == 403 and r.json()["detail"]["codigo"] == "plan_requerido"


def test_premium_edita_cajon(cliente):
    registrar(cliente)
    cliente.post("/api/yo/premium")
    p2 = cliente.get("/api/perfiles").json()[1]
    r = cliente.patch(f"/api/perfiles/{p2['id']}", json={"nombre": "Tito", "estilo": "estilista", "customizado": True})
    assert r.status_code == 200
    assert r.json()["customizado"] and r.json()["rotulo"] is None and r.json()["estilo"] == "estilista"


def test_ranura_1_edita_guardia_y_publico_pero_no_estilo(cliente):
    registrar(cliente)
    p1 = cliente.get("/api/perfiles").json()[0]
    assert cliente.patch(f"/api/perfiles/{p1['id']}", json={"publico": True, "guardia": "zurda"}).json()["publico"]
    assert cliente.patch(f"/api/perfiles/{p1['id']}", json={"estilo": "fajador"}).status_code == 422


def test_perfil_ajeno_404(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]
    assert b.patch(f"/api/perfiles/{p1['id']}", json={"nombre": "x"}).status_code == 404


def test_arquetipos_free_ve_base_premium_ve_variantes(cliente):
    registrar(cliente)
    assert {a["clave"] for a in cliente.get("/api/arquetipos").json()} == {"fajador", "estilista", "hibrido"}
    cliente.post("/api/yo/premium")
    assert len(cliente.get("/api/arquetipos").json()) == 6


def test_custom_solo_premium(cliente):
    registrar(cliente)
    a = cliente.get("/api/arquetipos").json()[0]
    assert cliente.post("/api/arquetipos/custom", json={"arquetipo_id": a["id"], "ranura": 2}).status_code == 403
    cliente.post("/api/yo/premium")
    r = cliente.post("/api/arquetipos/custom", json={"arquetipo_id": a["id"], "ranura": 3,
                                                     "medias": {"sparring": {"volumen": 30}}})
    assert r.status_code == 201
    assert r.json()["arquetipo"]["metricas"]["sparring"]["medias"]["volumen"] == 30
    assert r.json()["perfil"]["arquetipo"]["id"] == r.json()["arquetipo"]["id"]
    # Otro usuario no ve el custom.


def test_seed_valores_de_la_spec():
    s = {a["clave"]: a for a in leer_seed()}
    assert s["fajador"]["volumen"] == 42 and s["estilista_contra"]["sosten_ritmo"] == 0.97
    assert sum(1 for a in s.values() if a["solo_premium"]) == 3


def test_curva_sintetica_promedia_el_volumen():
    c = curva_sintetica(42, 0.85)
    assert len(c) == 18
    assert abs(sum(c) / 3 - 42) < 0.05          # 180 s = 3 min
    assert abs(c[10] / c[0] - 0.85) < 1e-3       # el resto a sosten del primer minuto


def test_sombra_y_bolsa_con_mas_volumen(cliente):
    registrar(cliente)
    fa = next(a for a in cliente.get("/api/arquetipos").json() if a["clave"] == "fajador")
    assert abs(fa["metricas"]["sombra"]["medias"]["volumen"] - 46.2) < 1e-6
    assert fa["metricas"]["sparring"]["medias"]["volumen"] == 42
