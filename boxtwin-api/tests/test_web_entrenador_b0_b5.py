"""
B0 y B5a de la web entrenador (prototipo/specs/05-api-y-datos.md).

B0: la API sirve la exportacion estatica de Next con sus cabeceras, y ninguna pagina tapa una
ruta de la API ni al reves. Starlette prueba primero las rutas y despues el montaje, asi que
lo que se verifica es que cada prefijo caiga del lado correcto.

B5a: GET /producto, que es de donde la web saca las banderas. La web nunca decide sola si un
bloque condicionado se muestra.
"""

from __future__ import annotations

import pytest
from test_api import _recargar

# Las paginas de 02 §4, como las deja `next build` con trailingSlash.
PAGINAS = ["ingresar", "sesiones", "sesiones/nueva", "sesion", "sesion/identificar",
           "peleadores", "peleador", "rival"]


@pytest.fixture
def out(tmp_path):
    raiz = tmp_path / "out"
    (raiz / "_next" / "static" / "chunks").mkdir(parents=True)
    (raiz / "index.html").write_text("<!doctype html><title>Panel</title>")
    (raiz / "404.html").write_text("<!doctype html><title>No existe</title>")
    for p in PAGINAS:
        (raiz / p).mkdir(parents=True, exist_ok=True)
        (raiz / p / "index.html").write_text(f"<!doctype html><title>{p}</title>")
    (raiz / "_next" / "static" / "chunks" / "app-abc123.js").write_text("console.log(1)")
    return raiz


@pytest.fixture
def con_web(entorno, monkeypatch, out):
    return _recargar(monkeypatch, BOXTWIN_WEB=str(out), BOXTWIN_INVITACION="abierto")


# -- B0 -----------------------------------------------------------------------


@pytest.mark.parametrize("pagina", PAGINAS)
def test_cada_pagina_responde_html(con_web, pagina):
    r = con_web.get(f"/{pagina}/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert f"<title>{pagina}</title>" in r.text


def test_las_paginas_llevan_csp_y_no_se_cachean(con_web):
    r = con_web.get("/sesiones/")
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "media-src 'self' blob:" in csp
    assert "font-src 'self'" in csp
    assert r.headers["cache-control"] == "no-cache"


def test_lo_que_tiene_hash_se_cachea_para_siempre(con_web):
    r = con_web.get("/_next/static/chunks/app-abc123.js")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_una_ruta_desconocida_devuelve_el_404_de_la_web(con_web):
    r = con_web.get("/no/existe/")
    assert r.status_code == 404
    assert "No existe" in r.text


def test_con_la_web_montada_boxeadores_sigue_siendo_json(con_web):
    con_web.post("/auth/registro", json={"email": "a@b.com", "clave": "clavelarga1"})
    t = con_web.post("/auth/login", json={"email": "a@b.com", "clave": "clavelarga1"}).json()
    con_web.headers["Authorization"] = f"Bearer {t['token']}"
    r = con_web.get("/boxeadores")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == []


def test_sesiones_es_pagina_y_put_de_boxeadores_sigue_llegando_a_la_api(con_web):
    # La pagina /sesiones/ y la ruta PUT /sesiones/{id}/boxeadores conviven porque el metodo
    # y la profundidad son distintos.
    assert "<title>sesiones</title>" in con_web.get("/sesiones/").text
    r = con_web.put("/sesiones/no-existe/boxeadores", json={})
    assert r.status_code == 401, "lo atiende la API (pide token), no el montaje estatico"


def test_producto_no_queda_tapado_por_la_web(con_web):
    r = con_web.get("/producto")
    assert r.headers["content-type"].startswith("application/json")


# -- B5a ----------------------------------------------------------------------


def test_producto_sin_autenticacion_con_lo_que_la_web_necesita(cliente):
    d = cliente.get("/producto").json()
    assert d["registro"] == "abierto"
    assert d["formatos"] == [".mp4", ".mov", ".m4v"]
    assert d["max_bytes"] > 0 and d["max_minutos"] == 30
    f = d["factor_procesamiento"]
    assert (f["min"], f["mediana"], f["max"]) == (1.134, 1.447, 2.624)
    assert d["dominio"] == {
        "combinacion": {"patron": "cualquiera", "separacion": "inicio_a_inicio", "max_s": 0.5,
                        "min_golpes": 2},
        "perfil_preliminar_sesiones": 5,
    }


def test_sin_medir_los_criterios_estan_pendientes_y_las_banderas_apagadas(cliente):
    d = cliente.get("/producto").json()
    assert d["criterios"]["C2"] == {"estado": "pendiente", "fecha": None}
    assert d["criterios"]["C3"] == {"estado": "pendiente", "fecha": None}
    assert d["banderas"] == {"comparacion_rounds": False, "guardia": False,
                             "sesiones_individuales": False, "postura_2d": False,
                             # la altura es la excepcion: va por decision de producto (14-10)
                             "objetivo": True, "exposicion": False, "avatar_3d": False}


def test_las_banderas_salen_del_estado_de_cada_criterio(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_C2="aprobado", BOXTWIN_C2_FECHA="2026-10-20",
                  BOXTWIN_C3="rechazado", BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["criterios"]["C2"] == {"estado": "aprobado", "fecha": "2026-10-20"}
    assert d["banderas"]["comparacion_rounds"] is True
    assert d["criterios"]["C3"]["estado"] == "rechazado"
    assert d["banderas"]["guardia"] is False, "rechazado no enciende nada"


def test_un_estado_mal_escrito_cuenta_como_pendiente(entorno, monkeypatch):
    c = _recargar(monkeypatch, BOXTWIN_C3="aprobadoo", BOXTWIN_INVITACION="abierto")
    d = c.get("/producto").json()
    assert d["criterios"]["C3"]["estado"] == "pendiente"
    assert d["banderas"]["guardia"] is False
