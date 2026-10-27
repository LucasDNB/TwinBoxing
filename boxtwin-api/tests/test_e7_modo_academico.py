"""
Modo academico E7 en la API (docs/specs/2026-10-25-e7-modo-academico, M4).

Apagado por defecto: solo BOXTWIN_CLASIFICADOR=e7 lo enciende. Con e7, la etapa clasificar
corre tres comandos (elevar con HMR2 en boxtwin_4dhumans, clasificar con --pose3d, tipos); sin
BOXTWIN_CMD_HMR2 falla claro; bolsa y sombra siguen con E4a porque E7 necesita al rival. La
reclasificacion por guardia y el encadenamiento valen igual que con geometrico.
"""

from __future__ import annotations

import pytest
from test_tipo_v2 import _boxeador, _en_cola, _listo, _trabajo, con_sesion  # noqa: F401
from conftest import subir_video

HMR2 = ["/env/4dh/bin/python", "/repo/boxtwin-annotator/tools/elevar_hmr2_sesion.py"]


def test_solo_e7_enciende_el_modo(monkeypatch):
    from boxtwin_api.config import Config

    monkeypatch.delenv("BOXTWIN_CLASIFICADOR", raising=False)
    assert Config().clasificador == "poseconv3d"
    for valor, esperado in (("e7", "e7"), (" E7 ", "e7"), ("geometrico", "geometrico"), ("e8", "poseconv3d")):
        monkeypatch.setenv("BOXTWIN_CLASIFICADOR", valor)
        assert Config().clasificador == esperado


def test_rutas_y_comando_por_variable(monkeypatch, tmp_path):
    from boxtwin_api.config import Config

    monkeypatch.delenv("BOXTWIN_CMD_HMR2", raising=False)
    assert Config().cmd_hmr2 == []
    monkeypatch.setenv("BOXTWIN_CMD_HMR2", " ".join(HMR2))
    monkeypatch.setenv("BOXTWIN_MODELO_CLASIFICADOR_E7", str(tmp_path / "e7"))
    monkeypatch.delenv("BOXTWIN_CLASIFICADOR", raising=False)
    c = Config()
    assert c.cmd_hmr2 == HMR2 and c.modelo_clasificador_e7 == tmp_path / "e7"
    assert not c.tipo_v2                                   # el comando solo no enciende nada
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR", "e7")
    assert Config().tipo_v2 and Config().hay_clasificador


def test_e7_arma_los_tres_comandos(entorno, con_sesion, monkeypatch):  # noqa: F811
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "e7")
    monkeypatch.setattr(cfg, "cmd_hmr2", HMR2)
    db, t, s = _trabajo(entorno, con_sesion)
    d = str(cfg.dir_sesion(con_sesion))
    assert worker.comando_de(t, s) == [
        [*HMR2, d],
        ["echo", "clasificar", d, "--modelo", str(cfg.modelo_clasificador_e7),
         "--guardia-a", "desconocida", "--guardia-b", "desconocida",
         "--pose3d", str(cfg.dir_sesion(con_sesion) / "pose3d_hmr2.npz")],
        ["echo", "tipos", d],
    ]
    db.close()


def test_e7_sin_comando_hmr2_falla_claro(entorno, con_sesion, monkeypatch):  # noqa: F811
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "e7")
    monkeypatch.setattr(cfg, "cmd_hmr2", [])
    db, t, s = _trabajo(entorno, con_sesion)
    with pytest.raises(RuntimeError, match="BOXTWIN_CMD_HMR2"):
        worker.comando_de(t, s)
    db.close()


@pytest.mark.parametrize("contexto", ["bolsa", "sombra"])
def test_e7_en_bolsa_y_sombra_usa_e4a(entorno, cliente, registrado, monkeypatch, contexto):
    cfg, worker = entorno["cfg"], entorno["worker"]
    monkeypatch.setattr(cfg, "clasificador", "e7")
    monkeypatch.setattr(cfg, "cmd_hmr2", HMR2)
    sid = subir_video(cliente, contexto=contexto).json()["job_id"]
    db, t, s = _trabajo(entorno, sid)
    cmds = worker.comando_de(t, s)
    assert len(cmds) == 2 and cmds[0][4] == str(cfg.modelo_clasificador_geo_individual)
    assert "--pose3d" not in cmds[0]
    db.close()


def test_e7_reclasifica_al_cambiar_la_guardia(entorno, cliente, con_sesion, monkeypatch):  # noqa: F811
    monkeypatch.setattr(entorno["cfg"], "clasificador", "e7")
    a = _boxeador(cliente, "Mono", "ortodoxa")
    cliente.put(f"/sesiones/{con_sesion}/boxeadores", json={"boxeador_a": a, "boxeador_b": None})
    _listo(entorno, con_sesion)
    cliente.patch(f"/boxeadores/{a}", json={"nombre": "Mono", "guardia": "zurda"})
    assert _en_cola(entorno, con_sesion) == ["clasificar"]


def test_salud_dice_e7(cliente, entorno, monkeypatch, tmp_path):
    cfg = entorno["cfg"]
    monkeypatch.setattr(cfg, "clasificador", "e7")
    monkeypatch.setattr(cfg, "modelo_clasificador_e7", tmp_path / "e7")
    r = cliente.get("/salud").json()
    assert r["clasificador"] == "e7" and r["modelo_e7"] is False
    (tmp_path / "e7").mkdir()
    (tmp_path / "e7" / "meta.json").write_text("{}")
    assert cliente.get("/salud").json()["modelo_e7"] is True
    monkeypatch.setattr(cfg, "clasificador", "geometrico")
    assert cliente.get("/salud").json()["modelo_e7"] is None
