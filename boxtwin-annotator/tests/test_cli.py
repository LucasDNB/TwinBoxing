"""
CLI: parseo y deduccion del directorio de proyecto.

No se ejercita el preproceso completo aca. Ese camino necesita GPU, pesos y un video, y
tenerlo como test unitario haria que la suite dejara de correr en cualquier maquina sin
CUDA. Se valida a mano corriendo el comando sobre un clip corto.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boxtwin.cli import build_parser, infer_project_dir, main
from boxtwin.version import __version__


# -- parser ----------------------------------------------------------------


def test_exige_subcomando(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_preprocess_defaults() -> None:
    a = build_parser().parse_args(["preprocess", "videos/spar.mp4"])
    assert a.comando == "preprocess"
    assert a.video == Path("videos/spar.mp4")
    assert a.imgsz == 640
    assert a.device == "0"
    assert a.half is False
    assert a.restart is False
    assert a.no_proxy is False
    assert a.project is None


def test_preprocess_acepta_overrides() -> None:
    a = build_parser().parse_args(
        ["preprocess", "v.mp4", "--imgsz", "960", "--device", "cpu", "--half",
         "--no-proxy", "--restart", "--shard-frames", "500", "--project", "/tmp/p"]
    )
    assert (a.imgsz, a.device, a.half) == (960, "cpu", True)
    assert (a.no_proxy, a.restart, a.shard_frames) == (True, True, 500)
    assert a.project == Path("/tmp/p")


def test_probe_defaults() -> None:
    a = build_parser().parse_args(["probe", "v.mp4"])
    assert a.comando == "probe"
    assert a.count is False
    assert build_parser().parse_args(["probe", "v.mp4", "--count"]).count is True


def test_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert __version__ in capsys.readouterr().out


# -- directorio de proyecto ------------------------------------------------


def test_infiere_proyecto_desde_videos(tmp_path: Path) -> None:
    """El layout del proyecto pone los videos en videos/, asi que el padre es el proyecto."""
    videos = tmp_path / "proyecto" / "videos"
    videos.mkdir(parents=True)
    v = videos / "spar.mp4"
    v.touch()
    assert infer_project_dir(v) == (tmp_path / "proyecto").resolve()


def test_video_suelto_usa_su_directorio(tmp_path: Path) -> None:
    v = tmp_path / "spar.mp4"
    v.touch()
    assert infer_project_dir(v) == tmp_path.resolve()


# -- manejo de errores -----------------------------------------------------


def test_error_sale_por_stderr_con_codigo_1(capsys: pytest.CaptureFixture) -> None:
    """Un video inexistente no tiene que dejar un traceback en pantalla."""
    codigo = main(["probe", "/no/existe/nunca.mp4"])
    salida = capsys.readouterr()
    assert codigo == 1
    assert salida.err.startswith("error: ")
    assert "Traceback" not in salida.err
