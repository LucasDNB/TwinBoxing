"""
T9 y T10 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6).

Ensayo de despliegue/produccion.sh fuera de la estacion, con las rutas por entorno, como el del
26-10 (A en 8100, B en 8101, C en 8102), servidos por un /salud falso. Los puertos se eligen
libres al arrancar: el 28-10 los 8100 a 8102 seguian ocupados por los uvicorn del ensayo del
26-10. Sin sudo, sin tocar /etc ni la copia principal: los checkouts son repos git temporales y tailscale es un script.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest despliegue/tests -q
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parents[1]
SH = AQUI / "produccion.sh"
PESOS = b"pesos de yolo26 de mentira"


def _libres(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    puertos = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return puertos


class _Servidor:
    def __init__(self, puerto: int, salud: dict, producto: dict | None = None):
        cuerpo = {"/salud": salud, "/producto": producto or {"demo": {"activo": False}}}

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                d = cuerpo.get(self.path)
                self.send_response(200 if d is not None else 404)
                self.end_headers()
                self.wfile.write(json.dumps(d or {}).encode())

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", puerto), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def cerrar(self):
        self.srv.shutdown()
        self.srv.server_close()


def _repo(ruta: Path) -> str:
    ruta.mkdir(parents=True)
    g = ["git", "-C", str(ruta), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(ruta)], check=True)
    (ruta / "x").write_text(str(ruta))
    subprocess.run(g + ["add", "x"], check=True)
    subprocess.run(g + ["commit", "-qm", "x"], check=True)
    return subprocess.run(["git", "-C", str(ruta), "rev-parse", "--short=7", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def estacion(tmp_path):
    pa, pb, pc = _libres(3)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    ts = bin_ / "tailscale"
    ts.write_text("#!/bin/sh\ncat \"$TS_STATUS\" 2>/dev/null\n")
    ts.chmod(0o755)
    (tmp_path / "ts.txt").write_text(f"https://x.ts.net:8443 (tailnet only)\n|-- / proxy http://127.0.0.1:{pb}\n")
    commits = {k: _repo(tmp_path / k) for k in ("A", "B", "C")}
    modelos = tmp_path / "gemelo" / "modelos"
    modelos.mkdir(parents=True)
    (modelos / "yolo26l-pose.pt").write_bytes(PESOS)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "TS_STATUS": str(tmp_path / "ts.txt"),
           "REPO": str(tmp_path / "A"), "REPO_B": str(tmp_path / "B"), "REPO_C": str(tmp_path / "C"),
           "DIR_B": str(tmp_path / "academico"), "DIR_C": str(tmp_path / "gemelo"),
           "ENV_A": str(tmp_path / "a.env"), "ENV_B": str(tmp_path / "b.env"),
           "PUERTO_A": str(pa), "PUERTO_B": str(pb), "PUERTO_C": str(pc)}
    (tmp_path / "a.env").write_text("BOXTWIN_SECRETO=aaa\nBOXTWIN_INVITACION=ia\n")
    (tmp_path / "b.env").write_text("BOXTWIN_SECRETO=bbb\nBOXTWIN_INVITACION=ib\n")
    servidores = []

    puertos = {"a": pa, "b": pb, "c": pc}

    def levantar(cual, salud, producto=None):
        servidores.append(_Servidor(puertos[cual], salud, producto))

    yield {"tmp": tmp_path, "env": env, "commits": commits, "levantar": levantar, "puertos": puertos}
    for s in servidores:
        s.cerrar()


def _sh(estacion, *args):
    return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=estacion["env"])


def _salud(commit, **kw):
    return {"ok": True, "commit": commit, "secreto_efimero": False, "registro": "invitacion",
            "frontend": True, "clasificador": "geometrico", "modelo_individual": True,
            "modelo_e7": None, **kw}


def _salud_c(estacion, **kw):
    sha = hashlib.sha256(PESOS).hexdigest()[:12]
    return _salud(estacion["commits"]["C"], modelo_pose={"archivo": "yolo26l-pose.pt", "sha256": sha}, **kw)


# -- verificar a y b: lo mismo que antes ----------------------------------------------


def test_verificar_a_y_b_como_antes(estacion):
    c = estacion["commits"]
    estacion["levantar"]("a", _salud(c["A"]))
    estacion["levantar"]("b", _salud(c["B"], clasificador="e7", modelo_e7=True))
    ra, rb = _sh(estacion, "verificar", "a"), _sh(estacion, "verificar", "b")
    assert ra.returncode == 0, ra.stderr
    assert rb.returncode == 0, rb.stderr
    assert "ok   commit " + c["A"] in ra.stdout and "version de prueba apagada" in ra.stdout
    assert "ok   clasificador e7" in rb.stdout and "ok   modelo_e7" in rb.stdout


def test_verificar_a_sigue_fallando_con_e7(estacion):
    estacion["levantar"]("a", _salud(estacion["commits"]["A"], clasificador="e7"))
    assert _sh(estacion, "verificar", "a").returncode != 0


# -- verificar c (CG2) -------------------------------------------------------------------


def test_verificar_c_pasa_con_lo_de_03_8(estacion):
    estacion["levantar"]("c", _salud_c(estacion), {"demo": {"activo": True}})
    r = _sh(estacion, "verificar", "c")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "modelo_pose yolo26l-pose.pt" in r.stdout
    assert "AVISO" in r.stdout and "esperado" in r.stdout          # la demo en C no falla


@pytest.mark.parametrize("cambio", [
    {"modelo_pose": {"archivo": "yolov8l-pose.pt", "sha256": "0" * 12}},
    {"modelo_pose": {"archivo": "yolo26l-pose.pt", "sha256": "0" * 12}},
    {"modelo_pose": None},
    {"clasificador": "e7"},
    {"modelo_individual": False},
    {"secreto_efimero": True},
    {"frontend": False},
    {"commit": "1234567"},
])
def test_verificar_c_falla_si_algo_no_es_lo_esperado(estacion, cambio):
    estacion["levantar"]("c", {**_salud_c(estacion), **cambio})
    assert _sh(estacion, "verificar", "c").returncode != 0


# -- previo c (rigen CG3 y la regla 2 de convivencia) ----------------------------------------


def test_previo_c_pasa_con_8002_libre_b_apagada_y_a_arriba(estacion):
    estacion["levantar"]("a", _salud(estacion["commits"]["A"]))
    r = _sh(estacion, "previo", "c")
    assert r.returncode == 0, r.stderr + r.stdout
    assert f"nada en :{estacion['puertos']['c']}" in r.stdout and "B apagada" in r.stdout


def test_previo_c_falla_con_b_encendida(estacion):
    estacion["levantar"]("a", _salud(estacion["commits"]["A"]))
    estacion["levantar"]("b", _salud(estacion["commits"]["B"], clasificador="e7"))
    assert _sh(estacion, "previo", "c").returncode != 0


def test_previo_c_falla_si_tailscale_publica_8002(estacion):
    estacion["levantar"]("a", _salud(estacion["commits"]["A"]))
    pc = estacion["puertos"]["c"]
    (estacion["tmp"] / "ts.txt").write_text(f"https://x.ts.net:9000\n|-- / proxy http://127.0.0.1:{pc}\n")
    assert _sh(estacion, "previo", "c").returncode != 0


def test_previo_sin_argumento_sigue_siendo_el_de_b(estacion):
    estacion["levantar"]("a", _salud(estacion["commits"]["A"]))
    (estacion["tmp"] / "ts.txt").write_text("")
    r = _sh(estacion, "previo")
    assert r.returncode == 0, r.stderr
    assert f"nada en :{estacion['puertos']['b']}" in r.stdout


# -- worktree-gemelo y env-gemelo ------------------------------------------------------------


def test_worktree_gemelo_desacoplado_y_no_se_pisa(estacion):
    env = estacion["env"]
    commit = estacion["commits"]["A"]
    destino = estacion["tmp"] / "TwinBoxing-gemelo"
    env["REPO_C"] = str(destino)
    r = _sh(estacion, "worktree-gemelo", commit)
    assert r.returncode == 0, r.stderr
    cab = subprocess.run(["git", "-C", str(destino), "symbolic-ref", "-q", "HEAD"], capture_output=True)
    assert cab.returncode != 0                                     # desacoplado, no en una rama
    assert _sh(estacion, "worktree-gemelo", commit).returncode != 0


def test_env_gemelo_600_con_secretos_nuevos_y_las_unidades(estacion):
    r = _sh(estacion, "env-gemelo")
    assert r.returncode == 0, r.stderr
    destino = estacion["tmp"] / "gemelo" / "boxtwin-gemelo.env"
    assert oct(destino.stat().st_mode & 0o777) == "0o600"
    vals = dict(l.split("=", 1) for l in destino.read_text().splitlines() if "=" in l and not l.startswith("#"))
    assert vals["BOXTWIN_SECRETO"] not in ("aaa", "bbb") and vals["BOXTWIN_INVITACION"] not in ("ia", "ib")
    assert "<nuev" not in destino.read_text()
    assert vals["BOXTWIN_MODELO_POSE"].endswith("yolo26l-pose.pt")
    assert sorted(p.name for p in (estacion["tmp"] / "gemelo" / "systemd").iterdir()) == [
        "boxtwin-gemelo-api.service", "boxtwin-gemelo-worker.service"]
    assert _sh(estacion, "env-gemelo").returncode != 0              # no se pisan secretos


def test_huella_b_por_entorno(estacion):
    env = estacion["env"]
    (estacion["tmp"] / "academico" / "datos").mkdir(parents=True)
    import sqlite3
    con = sqlite3.connect(estacion["tmp"] / "academico" / "datos" / "boxtwin.db")
    con.execute("create table sesiones (id text)")
    con.commit()
    con.close()
    r = _sh(estacion, "huella", "b")
    assert r.returncode == 0, r.stderr
    assert estacion["commits"]["B"] in subprocess.run(
        ["git", "-C", env["REPO_B"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    assert "## HEAD del worktree de B" in r.stdout and "apagada" in r.stdout


# -- T10: las unidades de C son las de B con cinco cosas cambiadas -----------------------------


@pytest.mark.parametrize("parte", ["api", "worker"])
def test_unidades_de_c_solo_difieren_de_las_de_b_en_lo_declarado(parte):
    b = (AQUI / "academico" / "systemd" / f"boxtwin-academico-{parte}.service").read_text()
    c = (AQUI / "gemelo" / "systemd" / f"boxtwin-gemelo-{parte}.service").read_text()
    sin_com = lambda t: [l for l in t.splitlines() if not l.startswith("#")]
    nb, nc = sin_com(b), sin_com(c)
    assert len(nb) == len(nc)
    distintas = [(x, y) for x, y in zip(nb, nc) if x != y]
    claves = {x.split("=", 1)[0].strip() for x, _ in distintas}
    assert claves <= {"Description", "WorkingDirectory", "EnvironmentFile", "After",
                      "--host 127.0.0.1 --port 8001 --proxy-headers"}
    for x, y in distintas:
        assert y == (x.replace("BoxTwin academica (E7, modo academico)", "BoxTwin gemelo C (YOLO26l-pose, local)")
                      .replace("TwinBoxing-academico", "TwinBoxing-gemelo")
                      .replace("boxtwin-academico", "boxtwin-gemelo").replace("--port 8001", "--port 8002"))
    assert "Restart=on-failure" in c
    if parte == "api":
        assert "--host 127.0.0.1 --port 8002" in c
