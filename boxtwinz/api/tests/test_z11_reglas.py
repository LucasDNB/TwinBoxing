"""Z11: cada disparador de la seccion 8, la rotacion de frases y el rival fantasma."""

import zlib

import pytest

from boxtwinz_api.services import rules
from boxtwinz_api.services.rules import FRASES, PIE, SIN_ALERTAS, Entrada, consejos, indice_frase
from conftest import golpes_cada, registrar, sesion_lista


def claves(e):
    return [c["clave"] for c in consejos(e)["consejos"]]


def test_sin_disparos():
    r = consejos(Entrada("s", golpes=50, pct_mano_adelantada=0.5, sosten_ritmo=1.0, pct_golpes_en_combinacion=0.5))
    assert r["consejos"] == [{"clave": "sin_alertas", "prioridad": 99, "texto": SIN_ALERTAS}]
    assert r["pie"] == PIE


def test_balance_manos_umbral_y_minimo():
    assert claves(Entrada("s", 30, pct_mano_adelantada=0.71)) == ["balance_manos"]
    assert claves(Entrada("s", 30, pct_mano_adelantada=0.70)) == ["sin_alertas"]   # > 0,70 estricto
    assert claves(Entrada("s", 29, pct_mano_adelantada=0.90)) == ["sin_alertas"]   # menos de 30 golpes


def test_balance_manos_zurdo_invierte():
    orto = consejos(Entrada("s", 40, pct_mano_adelantada=0.8))["consejos"][0]["texto"]
    zurdo = consejos(Entrada("s", 40, pct_mano_adelantada=0.8, guardia="zurda"))["consejos"][0]["texto"]
    assert orto.startswith("Estás trabajando mucho más la izquierda que la derecha")
    assert zurdo.startswith("Estás trabajando mucho más la derecha que la izquierda")


def test_caida_ritmo():
    assert claves(Entrada("s", 5, sosten_ritmo=0.79)) == ["caida_ritmo"]
    assert claves(Entrada("s", 5, sosten_ritmo=0.80)) == ["sin_alertas"]


def test_exposicion_lado():
    e = Entrada("s", 0, lado_izquierdo=7, lado_derecho=13, estilo_rival="fajador")   # 65% derecha, 20 golpes
    r = consejos(e)["consejos"][0]
    assert r["clave"] == "exposicion_lado"
    assert r["texto"] == "Contra los fajadores te entran por la derecha. Cerrá ese lado y salí por el costado."
    assert claves(Entrada("s", 0, lado_izquierdo=7, lado_derecho=12, estilo_rival="fajador")) == ["sin_alertas"]  # 19
    assert claves(Entrada("s", 0, lado_izquierdo=8, lado_derecho=12, estilo_rival="fajador")) == ["sin_alertas"]  # 60%
    assert claves(Entrada("s", 0, lado_izquierdo=0, lado_derecho=30, estilo_rival=None)) == ["sin_alertas"]       # sin estilo
    assert "híbridos" in consejos(Entrada("s", 0, lado_izquierdo=20, lado_derecho=0, estilo_rival="hibrido"))["consejos"][0]["texto"]


def test_golpes_sueltos():
    assert claves(Entrada("s", 30, pct_golpes_en_combinacion=0.29)) == ["golpes_sueltos"]
    assert claves(Entrada("s", 29, pct_golpes_en_combinacion=0.10)) == ["sin_alertas"]


def test_volumen_bajo_vs_rival():
    assert claves(Entrada("s", 0, volumen=23.9, volumen_rival=30)) == ["volumen_bajo_vs_rival"]
    assert claves(Entrada("s", 0, volumen=24.0, volumen_rival=30)) == ["sin_alertas"]


def test_maximo_tres_por_prioridad():
    e = Entrada("s", 40, pct_mano_adelantada=0.9, sosten_ritmo=0.5, pct_golpes_en_combinacion=0.1,
                lado_izquierdo=0, lado_derecho=30, estilo_rival="estilista", volumen=10, volumen_rival=40)
    assert claves(e) == ["exposicion_lado", "volumen_bajo_vs_rival", "caida_ritmo"]


def test_rotacion_determinista(monkeypatch):
    monkeypatch.setitem(FRASES, "caida_ritmo", ["f0", "f1", "f2"])
    for sid in ("a", "b", "sesion-123"):
        esperado = f"f{zlib.crc32(sid.encode()) % 3}"
        assert consejos(Entrada(sid, 0, sosten_ritmo=0.1))["consejos"][0]["texto"] == esperado
    assert indice_frase("x", 3) == indice_frase("x", 3)


def test_umbral_sale_de_config(monkeypatch):
    monkeypatch.setattr(rules.cfg, "umbral_caida_ritmo", 0.5)
    assert claves(Entrada("s", 0, sosten_ritmo=0.6)) == ["sin_alertas"]


def test_rival_arquetipo(cliente, db):
    registrar(cliente)
    p1 = cliente.get("/api/perfiles").json()[0]["id"]
    fa = next(a for a in cliente.get("/api/arquetipos").json() if a["clave"] == "fajador")
    sesion_lista(db, "ana@boxtwinz.ar", "sombra", golpes_cada(3.0, 240))     # 20 por minuto contra 46,2
    r = cliente.get("/api/rival", params={"perfil_id": p1, "modalidad": "sombra", "rival_tipo": "arquetipo",
                                          "rival_id": fa["id"]}).json()
    assert r["rival"]["rotulo"] == "Perfil simulado" and r["rival"]["metricas"]["volumen"] == pytest.approx(46.2)
    assert r["exposicion"]["simulado"]
    assert "volumen_bajo_vs_rival" in [c["clave"] for c in r["consejos"]["consejos"]]
    assert "exposicion_lado" not in [c["clave"] for c in r["consejos"]["consejos"]]   # simulado no dispara


def test_rival_variante_premium_y_cajon(cliente, db):
    registrar(cliente)
    p1 = cliente.get("/api/perfiles").json()[0]["id"]
    cliente.post("/api/yo/premium")
    var = next(a for a in cliente.get("/api/arquetipos").json() if a["clave"] == "fajador_volumen")
    cajon = cliente.get("/api/perfiles").json()[1]
    cliente.patch(f"/api/perfiles/{cajon['id']}", json={"customizado": True, "estilo": "fajador"})
    # El rival tira 25 derechas: entran por el lado izquierdo del usuario.
    sesion_lista(db, "ana@boxtwinz.ar", "sparring", golpes_cada(1.0, 240) + [("rival", 1 + i * 5.0, "derecha") for i in range(25)],
                 rival_perfil_id=cajon["id"])
    r = cliente.get("/api/rival", params={"perfil_id": p1, "modalidad": "sparring", "rival_tipo": "arquetipo",
                                          "rival_id": var["id"]}).json()
    assert r["exposicion"] == {**r["exposicion"], "lado_izquierdo_usuario": 25, "simulado": False}
    assert r["consejos"]["consejos"][0]["texto"].startswith("Contra los fajadores te entran por la izquierda")
    rc = cliente.get("/api/rival", params={"perfil_id": p1, "modalidad": "sparring", "rival_tipo": "cajon",
                                           "rival_id": cajon["id"]})
    assert rc.status_code == 200 and rc.json()["rival"]["metricas"]["volumen"] == pytest.approx(25 / 4)


def test_rival_permisos(nuevo_cliente):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    yb = registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]["id"]
    cajon = a.get("/api/perfiles").json()[1]["id"]
    base = {"perfil_id": p1, "modalidad": "sombra"}
    assert a.get("/api/rival", params={**base, "rival_tipo": "cajon", "rival_id": cajon}).status_code == 403
    assert a.get("/api/rival", params={**base, "rival_tipo": "amigo", "rival_id": yb["id"]}).status_code == 403
    pb = b.get("/api/perfiles").json()[0]["id"]
    assert a.get("/api/rival", params={**base, "rival_tipo": "publico", "rival_id": pb}).status_code == 404
    a.post("/api/yo/premium")
    ajeno = b.get("/api/perfiles").json()[1]["id"]
    assert a.get("/api/rival", params={**base, "rival_tipo": "cajon", "rival_id": ajeno}).status_code == 404
    var = next(x for x in b.get("/api/arquetipos").json() if x["clave"] == "fajador")
    assert b.get("/api/rival", params={"perfil_id": pb, "modalidad": "sombra", "rival_tipo": "arquetipo",
                                       "rival_id": var["id"]}).status_code == 200
