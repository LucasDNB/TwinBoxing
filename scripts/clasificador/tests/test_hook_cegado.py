import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[3] / ".claude" / "hooks" / "solo_tiras.py"


def _cargar(monkeypatch, trabajo):
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR", str(trabajo))
    spec = importlib.util.spec_from_file_location("solo_tiras", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_deja_solo_tiras_lote_y_respuestas(monkeypatch, tmp_path):
    h = _cargar(monkeypatch, tmp_path)
    L = tmp_path / "lotes" / "piloto"
    ok = lambda t, e: h.permitido(t, e)[0]  # noqa: E731
    assert ok("Read", {"file_path": str(L / "tiras" / "T0001.png")})
    assert ok("Read", {"file_path": str(L / "lote.jsonl")})
    assert ok("Glob", {"path": str(L / "tiras"), "pattern": "*.png"})
    assert ok("Write", {"file_path": str(L / "respuestas" / "lote_01.jsonl")})
    assert ok("SubagentHandback", {"result": "25 respuestas en respuestas/lote_01.jsonl"})
    # lo que tiene que quedar cerrado
    assert not ok("Read", {"file_path": str(L / "clave" / "clave.csv")})
    assert not ok("Read", {"file_path": str(L / "tiras" / ".." / "clave" / "clave.csv")})
    assert not ok("Read", {"file_path": str(tmp_path / "lotes" / "otro" / "resultado.json")})
    assert not ok("Read", {"file_path": "/home/lucasb/Proyectos/TwinBoxing/anotacion/annotations/Sparring.annot.json"})
    assert not ok("Glob", {"pattern": "**/clave.csv"})
    assert not ok("Glob", {"path": str(L), "pattern": "*/*.csv"})
    assert not ok("Glob", {"path": str(L / "tiras"), "pattern": "../clave/*"})
    assert not ok("Write", {"file_path": str(L / "clave" / "clave.csv")})
    assert not ok("Write", {"file_path": str(L / "respuestas" / "x.py")})
    for t in ("Bash", "Grep", "Edit", "WebFetch"):
        assert not ok(t, {})


def test_symlink_no_abre_la_clave(monkeypatch, tmp_path):
    h = _cargar(monkeypatch, tmp_path)
    L = tmp_path / "lotes" / "piloto"
    (L / "tiras").mkdir(parents=True)
    (L / "clave").mkdir()
    (L / "clave" / "clave.csv").write_text("x")
    (L / "tiras" / "T9999.png").symlink_to(L / "clave" / "clave.csv")
    assert not h.permitido("Read", {"file_path": str(L / "tiras" / "T9999.png")})[0]


def test_codigo_de_salida(tmp_path):
    env = {"BOXTWIN_CLASIFICADOR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    pedir = lambda d: subprocess.run([sys.executable, str(HOOK)], input=json.dumps(d), text=True,  # noqa: E731
                                     capture_output=True, env=env)
    r = pedir({"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "lotes/p/clave/clave.csv")}})
    assert r.returncode == 2 and "cegado" in r.stderr
    r = pedir({"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "lotes/p/tiras/T0001.png")}})
    assert r.returncode == 0
    r = pedir({"tool_name": "SubagentHandback", "tool_input": {"result": "listo"}})
    assert r.returncode == 0
