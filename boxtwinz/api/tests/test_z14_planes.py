"""Z14 y CZ3: reglas de plan, 402 y 403, en un solo lugar."""

import pytest

from boxtwinz_api.config import cfg
from conftest import registrar


@pytest.mark.parametrize("metodo,ruta,cuerpo", [
    ("patch", "/api/perfiles/{p2}", {"nombre": "x"}),
    ("post", "/api/arquetipos/custom", {"arquetipo_id": "{arq}", "ranura": 2}),
    ("get", "/api/rival?perfil_id={p1}&modalidad=sombra&rival_tipo=cajon&rival_id={p2}", None),
])
def test_403_plan_requerido_en_free(cliente, metodo, ruta, cuerpo):
    registrar(cliente)
    ps = cliente.get("/api/perfiles").json()
    arq = cliente.get("/api/arquetipos").json()[0]["id"]
    fmt = {"p1": ps[0]["id"], "p2": ps[1]["id"], "arq": arq}
    kw = {"json": {k: (v.format(**fmt) if isinstance(v, str) else v) for k, v in cuerpo.items()}} if cuerpo else {}
    r = getattr(cliente, metodo)(ruta.format(**fmt), **kw)
    assert r.status_code == 403 and r.json()["detail"]["codigo"] == "plan_requerido"


def test_free_no_ve_variantes_premium(cliente):
    registrar(cliente)
    assert not any(a["solo_premium"] for a in cliente.get("/api/arquetipos").json())


def test_cupo_por_plan(cliente, monkeypatch):
    monkeypatch.setattr(cfg, "cupo_minutos", {"free": 60.0, "premium": 300.0})
    registrar(cliente)
    assert cliente.get("/api/cupo").json()["limite_min"] == 60
    cliente.post("/api/yo/premium")
    assert cliente.get("/api/cupo").json()["limite_min"] == 300


def test_cajon_en_siembra_ignorado_en_free(cliente, db):
    """La spec dice que el cajon en free se ignora, no que falle."""
    from boxtwinz_api.routers.sesiones import _cajon_valido
    from sqlalchemy import select
    from boxtwinz_api.models import Usuario

    registrar(cliente)
    p2 = cliente.get("/api/perfiles").json()[1]["id"]
    u = db.scalar(select(Usuario))
    assert _cajon_valido(db, u, p2) is None
