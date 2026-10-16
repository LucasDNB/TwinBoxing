"""
Tipo de golpe v2 en la API (docs/specs/2026-10-16-fightcard-tipo-v2, 02-diseno.md seccion 4).

CA3: el worker arma los comandos correctos con cada valor de BOXTWIN_CLASIFICADOR, y sin la
variable hace lo de antes. CA4: la guardia sale del boxeador (o se dice desconocida) y
cambiarla, o cambiar la asignacion, encola `clasificar` solo con v2 encendida y la sesion
lista. Ademas: con v2 se encadena la clasificacion aunque no haya BOXTWIN_CMD_CLASIFICADOR, y
/salud dice que metodo esta activo.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video
from sqlalchemy import select


@pytest.fixture
def con_sesion(entorno, cliente, registrado):
    return subir_video(cliente).json()["job_id"]


def _trabajo(entorno, sid, etapa="clasificar"):
    from boxtwin_api.modelos import Sesion

    db_mod, cola = entorno["db"], entorno["cola"]
    db = db_mod.hacer_sesion().__enter__()
    t = cola.encolar(db, sid, etapa)
    db.commit()
    return db, t, db.get(Sesion, sid)


def _en_cola(entorno, sid):
    from boxtwin_api.modelos import Trabajo

    with entorno["db"].hacer_sesion() as db:
        return [t.etapa for t in db.scalars(select(Trabajo).where(
            Trabajo.sesion_id == sid, Trabajo.estado == "en_cola")).all()]


def _listo(entorno, sid):
    from boxtwin_api.modelos import Sesion, Trabajo

    with entorno["db"].hacer_sesion() as db:
        db.get(Sesion, sid).estado = "listo"
        for t in db.scalars(select(Trabajo).where(Trabajo.sesion_id == sid)).all():
            t.estado = "listo"
        db.commit()


# -- CA3: comandos --------------------------------------------------------------


def test_ca3_sin_variable_es_poseconv3d(monkeypatch):
    from boxtwin_api.config import Config

    monkeypatch.delenv("BOXTWIN_CLASIFICADOR", raising=False)
    assert Config().clasificador == "poseconv3d"
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR", "geometrico")
    assert Config().clasificador == "geometrico"
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR", "geometric0")
    assert Config().clasificador == "poseconv3d"


def test_ca3_poseconv3d_arma_lo_de_antes(entorno, con_sesion, monkeypatch):
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "cmd_clasificador", ["python", "clasificar_sesion.py"])
    db, t, s = _trabajo(entorno, con_sesion)
    d = str(cfg.dir_sesion(con_sesion))
    esperado = [
        ["python", "clasificar_sesion.py", d, "--config", str(cfg.config_clasificador),
         "--checkpoint", str(cfg.checkpoint_clasificador)],
        ["echo", "tipos", d],
    ]
    assert worker.comando_de(t, s) == esperado
    monkeypatch.setattr(cfg, "clasificador", "poseconv3d")
    assert worker.comando_de(t, s) == esperado
    db.close()


def test_ca3_geometrico_corre_en_el_entorno_del_anotador(entorno, con_sesion, monkeypatch):
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    monkeypatch.setattr(cfg, "cmd_clasificador", [])
    db, t, s = _trabajo(entorno, con_sesion)
    d = str(cfg.dir_sesion(con_sesion))
    assert worker.comando_de(t, s) == [
        ["echo", "clasificar", d, "--modelo", str(cfg.modelo_clasificador_geo),
         "--guardia-a", "desconocida", "--guardia-b", "desconocida"],
        ["echo", "tipos", d],
    ]
    db.close()


@pytest.mark.parametrize("contexto", ["bolsa", "sombra"])
def test_bolsa_y_sombra_usan_el_modelo_sin_rival(entorno, cliente, registrado, monkeypatch, contexto):
    """16-10: una sola persona no tiene rival; va el modelo de E4a y no hay guardia B."""
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    db, t, s = _trabajo(entorno, sid)
    assert s.contexto == contexto
    d = str(cfg.dir_sesion(sid))
    assert worker.comando_de(t, s)[0] == [
        "echo", "clasificar", d, "--modelo", str(cfg.modelo_clasificador_geo_individual),
        "--guardia-a", "desconocida", "--guardia-b", "desconocida"]
    assert cfg.modelo_clasificador_geo_individual != cfg.modelo_clasificador_geo
    db.close()


@pytest.mark.parametrize("contexto", ["sparring", "combate"])
def test_sparring_y_combate_siguen_con_el_modelo_con_rival(entorno, cliente, registrado, monkeypatch, contexto):
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    db, t, s = _trabajo(entorno, sid)
    assert worker.comando_de(t, s)[0][4] == str(cfg.modelo_clasificador_geo)
    db.close()


def test_modelo_individual_por_variable(monkeypatch, tmp_path):
    from boxtwin_api.config import Config

    monkeypatch.setenv("BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL", str(tmp_path / "m"))
    assert Config().modelo_clasificador_geo_individual == tmp_path / "m"


# -- CA4: guardia y reclasificacion ---------------------------------------------


def _boxeador(cliente, nombre, guardia=None):
    r = cliente.post("/boxeadores", json={"nombre": nombre, "guardia": guardia})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_ca4_la_guardia_sale_del_boxeador(entorno, cliente, con_sesion, monkeypatch):
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    a = _boxeador(cliente, "Mono", "zurda")
    b = _boxeador(cliente, "Sin guardia")
    assert cliente.put(f"/sesiones/{con_sesion}/boxeadores",
                       json={"boxeador_a": a, "boxeador_b": b}).status_code == 200
    db, t, s = _trabajo(entorno, con_sesion)
    cmd = worker.comando_de(t, s)[0]
    assert cmd[cmd.index("--guardia-a") + 1] == "zurda"
    assert cmd[cmd.index("--guardia-b") + 1] == "desconocida"
    db.close()


def test_ca4_cambiar_la_guardia_reclasifica_solo_con_v2_y_sesion_lista(entorno, cliente, con_sesion, monkeypatch):
    cfg = entorno["cfg"]
    a = _boxeador(cliente, "Mono", "ortodoxa")
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": a, "boxeador_b": None})
    _listo(entorno, con_sesion)
    # v1: cambiar la guardia no encola nada
    r = cliente.patch(f"/boxeadores/{a}", json={"nombre": "Mono", "guardia": "zurda"})
    assert r.status_code == 200, r.text
    assert _en_cola(entorno, con_sesion) == []
    # v2: encola clasificar una vez, aunque se cambie dos veces antes de que corra
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    cliente.patch(f"/boxeadores/{a}", json={"nombre": "Mono", "guardia": "ortodoxa"})
    cliente.patch(f"/boxeadores/{a}", json={"nombre": "Mono", "guardia": "zurda"})
    assert _en_cola(entorno, con_sesion) == ["clasificar"]
    # sin cambio de guardia no se encola
    _listo(entorno, con_sesion)
    cliente.patch(f"/boxeadores/{a}", json={"nombre": "Mono Rodriguez", "guardia": "zurda"})
    assert _en_cola(entorno, con_sesion) == []


def test_ca4_una_sesion_que_no_esta_lista_no_se_reclasifica(entorno, cliente, con_sesion, monkeypatch):
    monkeypatch.setattr(entorno["cfg"], "clasificador", "geometrico")
    a = _boxeador(cliente, "Mono", "ortodoxa")
    antes = _en_cola(entorno, con_sesion)
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": a, "boxeador_b": None})
    assert _en_cola(entorno, con_sesion) == antes          # la subida dejo su procesar, nada mas
    assert "clasificar" not in antes


def test_ca4_cambiar_la_asignacion_reclasifica(entorno, cliente, con_sesion, monkeypatch):
    a = _boxeador(cliente, "Mono", "zurda")
    _listo(entorno, con_sesion)
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": a, "boxeador_b": None})
    assert _en_cola(entorno, con_sesion) == []             # v1
    monkeypatch.setattr(entorno["cfg"], "clasificador", "geometrico")
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": a, "boxeador_b": None})
    assert _en_cola(entorno, con_sesion) == []             # misma asignacion: nada que recalcular
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": None, "boxeador_b": a})
    assert _en_cola(entorno, con_sesion) == ["clasificar"]


# -- encadenamiento y salud -----------------------------------------------------


@pytest.mark.parametrize("metodo,cmd,esperado", [
    ("geometrico", [], "clasificar"),
    ("poseconv3d", [], "render"),
    ("poseconv3d", ["python", "x.py"], "clasificar"),
])
def test_despues_de_completar_se_encadena(entorno, con_sesion, monkeypatch, metodo, cmd, esperado):
    from boxtwin_api.modelos import Trabajo

    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", metodo)
    monkeypatch.setattr(cfg, "cmd_clasificador", cmd)
    with entorno["db"].hacer_sesion() as db:
        for t in db.scalars(select(Trabajo).where(Trabajo.sesion_id == con_sesion)).all():
            t.estado = "listo"
        entorno["cola"].encolar(db, con_sesion, "completar", track_a=1, track_b=2)
        db.commit()
    d = cfg.dir_sesion(con_sesion)
    d.mkdir(parents=True, exist_ok=True)

    def falso_run(c, **kw):
        (d / "sesion.json").write_text(json.dumps({"estado": "listo"}))

        class R:
            returncode, stdout, stderr = 0, "", ""

        return R()

    monkeypatch.setattr(worker.subprocess, "run", falso_run)
    assert worker.correr(una_vuelta=True, worker="w") == 1
    assert _en_cola(entorno, con_sesion) == [esperado]


def test_salud_dice_el_metodo(cliente, entorno, monkeypatch):
    assert cliente.get("/salud").json()["clasificador"] == "poseconv3d"
    monkeypatch.setattr(entorno["cfg"], "clasificador", "geometrico")
    assert cliente.get("/salud").json()["clasificador"] == "geometrico"


def test_salud_dice_si_esta_el_modelo_individual(cliente, entorno, monkeypatch, tmp_path):
    cfg = entorno["cfg"]
    assert cliente.get("/salud").json()["modelo_individual"] is None      # con v1 no aplica
    assert cliente.get("/salud").json()["sesiones_individuales"] is False
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    monkeypatch.setattr(cfg, "modelo_clasificador_geo_individual", tmp_path / "sin_rival")
    assert cliente.get("/salud").json()["modelo_individual"] is False
    (tmp_path / "sin_rival").mkdir()
    (tmp_path / "sin_rival" / "meta.json").write_text("{}")
    assert cliente.get("/salud").json()["modelo_individual"] is True
