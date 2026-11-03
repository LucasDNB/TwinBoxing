"""Z5: subir, sembrar y ver el resultado por la API, con el motor falso. Y los cupos."""

import shutil
import subprocess

import pytest
from sqlalchemy import select

from boxtwinz_api import worker
from boxtwinz_api.cola import fallar, reclamar, terminar
from boxtwinz_api.config import cfg
from boxtwinz_api.models import Correccion
from conftest import registrar
from motor_falso import MotorFalso

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")


@pytest.fixture(scope="session")
def video_corto(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("v") / "corto.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=3",
                    "-pix_fmt", "yuv420p", str(ruta)], check=True)
    return ruta


def _subir(cliente, video, modalidad="sparring", **extra):
    p1 = cliente.get("/api/perfiles").json()[0]
    with open(video, "rb") as f:
        return cliente.post("/api/sesiones", data={"modalidad": modalidad, "perfil_id": p1["id"], **extra},
                            files={"video": ("corto.mp4", f, "video/mp4")})


def _worker(db, motor):
    t = reclamar(db, "test")
    try:
        estado = worker.ejecutar_trabajo(db, t, correr_motor=motor)
    except Exception as e:  # noqa: BLE001
        db.rollback()
        fallar(db, t, str(e))
        return None
    terminar(db, t, estado_sesion=estado)
    return estado


def test_sparring_subir_sembrar_resultado_valorar(cliente, db, video_corto):
    registrar(cliente)
    r = _subir(cliente, video_corto, rounds_declarados="1")
    assert r.status_code == 202, r.text
    s = r.json()
    assert s["estado"] == "subida" and abs(s["duracion_s"] - 3.0) < 0.2
    assert r.json()["cupo"]["usados_min"] == pytest.approx(s["duracion_s"] / 60)

    motor = MotorFalso(cfg.cmd_boxtwin, golpes_a=[(0.2, "izq"), (0.5, "der"), (2.0, "izq")],
                       golpes_b=[(1.0, "izq")], duracion=3.0)
    assert _worker(db, motor) == "esperando_siembra"
    c = cliente.get(f"/api/sesiones/{s['id']}/candidatos").json()
    assert [x["track"] for x in c["candidatos"]] == [3, 7] and c["cajones"] == []
    assert cliente.get(c["candidatos"][0]["recorte_url"]).status_code == 200

    assert cliente.post(f"/api/sesiones/{s['id']}/siembra", json={"usuario": 3, "rival": 99}).status_code == 422
    assert cliente.post(f"/api/sesiones/{s['id']}/siembra", json={"usuario": 3, "rival": 7}).status_code == 202
    assert _worker(db, motor) == "lista"

    res = cliente.get(f"/api/sesiones/{s['id']}/resultado").json()
    assert res["valoracion_obligatoria"]
    assert [g["quien"] for g in res["golpes"]] == ["usuario", "usuario", "rival", "usuario"]
    u = res["metricas"]["usuario"]["total"]
    assert u["golpes"] == 3 and u["iniciativa"] == 0.75
    assert res["exposicion"]["lado_derecho_usuario"] == 1
    assert res["consejos"]["pie"].startswith("Los consejos salen de los golpes detectados")
    assert res["rotulos"]["golpes"] == "golpes detectados"

    assert cliente.post(f"/api/sesiones/{s['id']}/valoracion", json={"estrellas": 4}).status_code == 201
    assert not cliente.get(f"/api/sesiones/{s['id']}/resultado").json()["valoracion_obligatoria"]

    g = res["golpes"][0]["id"]
    assert cliente.post(f"/api/sesiones/{s['id']}/correcciones", json={"tipo": "otra_mano", "golpe_id": g}).status_code == 201
    assert cliente.post(f"/api/sesiones/{s['id']}/correcciones", json={"tipo": "falto_golpe"}).status_code == 422
    assert db.scalar(select(Correccion)).tipo == "otra_mano"


def test_video_por_rangos(cliente, video_corto):
    registrar(cliente)
    sid = _subir(cliente, video_corto, "sombra").json()["id"]
    r = cliente.get(f"/api/sesiones/{sid}/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-range"].endswith(f"/{video_corto.stat().st_size}")


def test_cupo_agotado_402(cliente, video_corto, monkeypatch):
    monkeypatch.setattr(cfg, "cupo_minutos", {"free": 0.01, "premium": 300.0})
    registrar(cliente)
    r = _subir(cliente, video_corto, "bolsa")
    assert r.status_code == 402 and r.json()["detail"]["codigo"] == "cupo_agotado"
    assert not list((cfg.datos / "videos").glob("*"))


def test_free_no_sube_con_cajon(cliente, video_corto):
    registrar(cliente)
    p2 = cliente.get("/api/perfiles").json()[1]
    with open(video_corto, "rb") as f:
        r = cliente.post("/api/sesiones", data={"modalidad": "sparring", "perfil_id": p2["id"]},
                         files={"video": ("c.mp4", f, "video/mp4")})
    assert r.status_code == 403 and r.json()["detail"]["codigo"] == "plan_requerido"


def test_cajon_premium_recibe_el_rival(cliente, db, video_corto):
    registrar(cliente)
    cliente.post("/api/yo/premium")
    p2 = cliente.get("/api/perfiles").json()[1]
    cliente.patch(f"/api/perfiles/{p2['id']}", json={"customizado": True, "nombre": "Tito"})
    sid = _subir(cliente, video_corto).json()["id"]
    motor = MotorFalso(cfg.cmd_boxtwin, golpes_a=[(0.2, "izq")], golpes_b=[(1.0, "der")], duracion=3.0)
    _worker(db, motor)
    assert cliente.get(f"/api/sesiones/{sid}/candidatos").json()["cajones"][0]["id"] == p2["id"]
    cliente.post(f"/api/sesiones/{sid}/siembra", json={"usuario": 3, "rival": 7, "rival_cajon_perfil_id": p2["id"]})
    _worker(db, motor)
    assert cliente.get(f"/api/sesiones/{sid}").json()["rival_perfil_id"] == p2["id"]
    st = cliente.get(f"/api/perfiles/{p2['id']}/stats", params={"modalidad": "sparring"}).json()
    assert st["sesiones"] == 1 and not st["simulado"] and st["promedio"]["volumen"] == pytest.approx(20.0)


def test_comparar_misma_modalidad(cliente, db, video_corto):
    registrar(cliente)
    a = _subir(cliente, video_corto, "sombra").json()["id"]
    b = _subir(cliente, video_corto, "bolsa").json()["id"]
    assert cliente.get("/api/comparar", params={"a": a, "b": b}).status_code == 422


def test_sesion_ajena_404(nuevo_cliente, video_corto):
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    sid = _subir(a, video_corto, "sombra").json()["id"]
    assert b.get(f"/api/sesiones/{sid}").status_code == 404
    assert b.get(f"/api/sesiones/{sid}/video").status_code == 404
