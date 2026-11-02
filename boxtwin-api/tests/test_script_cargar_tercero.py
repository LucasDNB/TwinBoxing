"""cargar_tercero.py (gemelo-tercero, 2-11): la sesion queda como la crea la ruta, con el origen."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cargar_tercero", Path(__file__).resolve().parents[1] / "scripts" / "cargar_tercero.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _args(video, *extra):
    return m.armar_parser().parse_args(["--email", "lucas@usal.edu.ar", "--video", str(video),
                                        *extra])


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "fuente" / "Pacquiao vs Margarito.mp4"
    v.parent.mkdir()
    v.write_bytes(b"no es un video de verdad")
    return v


def _cargar(entorno, args):
    with entorno["db"].hacer_sesion() as db:
        return m.cargar(db, entorno["cfg"], args)


def test_defaults_de_la_spec(video):
    a = _args(video)
    assert (a.contexto, a.rounds, a.round_s, a.descanso_s) == ("combate", 12, 180.0, 0.0)
    assert (a.guardia_a, a.guardia_b) == ("zurda", "ortodoxa")
    assert m.validar(a) == []


@pytest.mark.parametrize("extra, error", [
    (["--contexto", "pelea"], "contexto"),
    (["--guardia-a", "southpaw"], "guardia_a"),
    (["--seudonimo-b", "Tercero Z"], "iguales"),
    (["--rounds", "0"], "rounds"),
    (["--origen", "x" * 321], "320"),
])
def test_validar_rechaza(video, extra, error):
    assert any(error in e for e in m.validar(_args(video, *extra)))


def test_carga_sesion_peleadores_video_y_trabajo(entorno, registrado, video):
    from sqlalchemy import select

    from boxtwin_api.modelos import Boxeador, Sesion, Trabajo

    out = _cargar(entorno, _args(video))
    assert out["sesion"]["nueva"] and out["sesion"]["video"] in ("enlace", "copia")
    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, out["sesion"]["id"])
        assert s.contexto == "combate" and s.rounds == 12 and s.descanso_s == 0.0
        assert s.estado == "en_cola"
        # El origen va donde iria el consentimiento: nada dice que alguien consintio.
        assert s.consentimiento_por == m.ORIGEN and "sin consentimiento" in s.consentimiento_por
        a, b = db.get(Boxeador, s.boxeador_a_id), db.get(Boxeador, s.boxeador_b_id)
        assert (a.nombre, a.guardia) == ("Tercero Z", "zurda")
        assert (b.nombre, b.guardia) == ("Tercero O", "ortodoxa")
        assert a.cuenta_id == s.cuenta_id and "tercero" in a.notas
        trabajos = db.scalars(select(Trabajo).where(Trabajo.sesion_id == s.id)).all()
        assert [(t.etapa, t.parametros) for t in trabajos] == [
            ("procesar", {"round_s": 180.0, "descanso_s": 0.0})]
        destino = entorno["cfg"].dir_sesion(s.id) / "videos" / "Pacquiao_vs_Margarito.mp4"
        assert destino.read_bytes() == video.read_bytes()


def test_la_api_la_muestra_como_cualquier_sesion(entorno, cliente, registrado, video):
    out = _cargar(entorno, _args(video))
    r = cliente.get("/jobs")
    assert r.status_code == 200 and out["sesion"]["id"] in r.text
    j = cliente.get(f"/jobs/{out['sesion']['id']}")
    assert j.status_code == 200 and j.json()["estado"] == "en_cola"
    pb = cliente.get("/boxeadores")
    assert pb.status_code == 200 and "Tercero Z" in pb.text


def test_reanudable_no_duplica(entorno, registrado, video):
    from sqlalchemy import func, select

    from boxtwin_api.modelos import Boxeador, Sesion

    primero = _cargar(entorno, _args(video))
    segundo = _cargar(entorno, _args(video))
    assert segundo["sesion"] == {"id": primero["sesion"]["id"], "estado": "en_cola",
                                 "nueva": False}
    with entorno["db"].hacer_sesion() as db:
        assert db.scalar(select(func.count()).select_from(Sesion)) == 1
        assert db.scalar(select(func.count()).select_from(Boxeador)) == 2


def test_ensayo_no_escribe(entorno, registrado, video):
    from sqlalchemy import func, select

    from boxtwin_api.modelos import Boxeador, Sesion, Trabajo

    out = _cargar(entorno, _args(video, "--ensayo"))
    assert out["ensayo"] and out["sesion"]["nueva"]
    with entorno["db"].hacer_sesion() as db:
        for modelo in (Sesion, Boxeador, Trabajo):
            assert db.scalar(select(func.count()).select_from(modelo)) == 0
    assert not (entorno["cfg"].datos / "sesiones").exists() or \
        not any((entorno["cfg"].datos / "sesiones").iterdir())


def test_usuario_inexistente(entorno, video):
    with pytest.raises(SystemExit):
        _cargar(entorno, _args(video))
