"""Z1: el esqueleto responde y la configuracion lee el toml."""

from boxtwinz_api.config import cargar


def test_salud_responde(cliente):
    r = cliente.get("/api/salud")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["servicio"] == "boxtwinz"
    assert set(d["cola"]) == {"en_cola", "tomados", "fallidos", "listos"}


def test_provisionales_salen_del_toml(tmp_path):
    t = tmp_path / "boxtwinz.toml"
    t.write_text('[provisional]\ntendencia_n = 7\ncalidad_pose_min = 0.6\n'
                 '[provisional.cupo_minutos]\nfree = 30\n[motor]\ncmd = "/x/boxtwin-annotator"\n')
    c = cargar(t, entorno={})
    assert c.tendencia_n == 7 and c.calidad_pose_min == 0.6
    assert c.cupo_minutos == {"free": 30.0, "premium": 300.0}
    assert c.cmd_boxtwin == ["/x/boxtwin-annotator"]


def test_entorno_pisa_el_toml(tmp_path):
    t = tmp_path / "boxtwinz.toml"
    t.write_text('[servidor]\ndatos = "/tmp/a"\n')
    c = cargar(t, entorno={"BOXTWINZ_DATOS": "/tmp/b", "BOXTWINZ_ADMINS": "X@y.com, z@y.com"})
    assert str(c.datos) == "/tmp/b"
    assert c.admins == ("x@y.com", "z@y.com")


def test_ejemplo_de_config_se_lee():
    from pathlib import Path

    c = cargar(Path(__file__).parents[2] / "boxtwinz.toml.example", entorno={})
    assert c.cupo_minutos["premium"] == 300.0
