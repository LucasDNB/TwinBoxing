"""Fixtures: base sqlite por test, datos en tmp, cliente HTTP y fabricas de usuarios."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from boxtwinz_api import db as dbmod
from boxtwinz_api.config import cfg


@pytest.fixture(autouse=True)
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "datos", tmp_path / "datos")
    monkeypatch.setattr(cfg, "lock_gpu", tmp_path / "gpu.lock")
    monkeypatch.setattr(cfg, "admins", ("admin@boxtwinz.ar",))
    monkeypatch.setattr(cfg, "secreto", "secreto-de-test")
    dbmod.configurar(f"sqlite:///{tmp_path / 'z.db'}")
    dbmod.crear_tablas()
    yield
    dbmod.engine.dispose()


@pytest.fixture
def db():
    with dbmod.hacer_sesion() as s:
        yield s


@pytest.fixture
def app():
    from boxtwinz_api.main import app as a

    return a


def registrar(cliente, email="ana@boxtwinz.ar", guardia="ortodoxa", reent=False, **extra):
    r = cliente.post("/api/auth/registro", json={
        "email": email, "clave": "una-clave-larga", "nombre": email.split("@")[0],
        "guardia": guardia, "consentimiento_uso": True, "consentimiento_reentrenamiento": reent, **extra})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def cliente(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def nuevo_cliente(app):
    """Varios clientes, cada uno con su cookie: para amigos, ranking, denuncias."""
    abiertos = []

    def fabrica():
        c = TestClient(app)
        c.__enter__()
        abiertos.append(c)
        return c

    yield fabrica
    for c in abiertos:
        c.__exit__(None, None, None)


def sesion_lista(db, email, modalidad="sombra", golpes=(), duracion=240.0, calidad=0.9, rival_perfil_id=None, ranura=1):
    """Una sesion lista con golpes, sin pasar por el worker: para ranking, rival y admin."""
    from sqlalchemy import select

    from boxtwinz_api.models import Golpe, Perfil, Sesion, Usuario
    from boxtwinz_api.services.ingest import es_apta_ranking, recalcular_metricas

    u = db.scalar(select(Usuario).where(Usuario.email == email))
    p = db.scalar(select(Perfil).where(Perfil.usuario_id == u.id, Perfil.ranura == ranura))
    s = Sesion(usuario_id=u.id, perfil_id=p.id, modalidad=modalidad, video_ruta="/no/existe.mp4", estado="lista",
               video_duracion_s=duracion, calidad_pose=calidad, rival_perfil_id=rival_perfil_id)
    db.add(s)
    db.flush()
    for quien, t, mano in golpes:
        db.add(Golpe(sesion_id=s.id, quien=quien, mano=mano, t_inicio=t, t_fin=t + 0.2, confianza=0.9))
    db.flush()
    recalcular_metricas(db, s)
    s.apta_ranking = es_apta_ranking(s, p, u)
    db.commit()
    return s


def golpes_cada(paso, duracion, quien="usuario", mano="izquierda"):
    n = int(duracion / paso)
    return [(quien, round(i * paso, 3), mano if i % 2 else ("derecha" if mano == "izquierda" else "izquierda")) for i in range(n)]
