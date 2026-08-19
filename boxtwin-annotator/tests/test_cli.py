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


# -- el preproceso no puede destruir una anotacion --------------------------


def test_project_paths_no_sigue_symlinks(tmp_path) -> None:
    """
    Un proyecto de prueba con el video enlazado al original tiene que quedarse en el de
    prueba. Con resolve() apuntaba al original y las escrituras caian sobre la anotacion de
    verdad; paso dos veces armando pruebas aisladas.
    """
    import os

    from boxtwin.core.project import project_paths

    (tmp_path / "real" / "videos").mkdir(parents=True)
    (tmp_path / "real" / "videos" / "v.mp4").write_bytes(b"x")
    (tmp_path / "copia" / "videos").mkdir(parents=True)
    os.symlink(tmp_path / "real" / "videos" / "v.mp4", tmp_path / "copia" / "videos" / "v.mp4")

    assert project_paths(tmp_path / "copia" / "videos" / "v.mp4").project == tmp_path / "copia"


def test_preprocess_se_niega_a_pisar_una_anotacion(tmp_path) -> None:
    """
    Rehacer el cache renumera los track_id y las asignaciones pasan a apuntar a otra persona.
    No se ve en el overlay, solo en el export, asi que tiene que fallar antes y no avisar
    despues.
    """
    import json

    import pytest

    from boxtwin.preprocess.runner import AnotacionEnRiesgoError, _anotacion_en_riesgo

    (tmp_path / "annotations").mkdir()
    (tmp_path / "cache").mkdir()
    annot = tmp_path / "annotations" / "v.annot.json"

    # Sin anotacion no hay riesgo.
    assert _anotacion_en_riesgo(tmp_path, "v") is None

    # Una anotacion vacia tampoco: no hay trabajo que perder.
    annot.write_text(json.dumps({"events": [], "identity": {"assignments": []}}))
    assert _anotacion_en_riesgo(tmp_path, "v") is None

    # Con asignaciones, si.
    annot.write_text(json.dumps({"events": [], "identity": {"assignments": [{"id": "a"}]}}))
    assert _anotacion_en_riesgo(tmp_path, "v") == (0, 1)

    # Con eventos, tambien.
    annot.write_text(json.dumps({"events": [{"id": "e"}], "identity": {"assignments": []}}))
    assert _anotacion_en_riesgo(tmp_path, "v") == (1, 0)

    # Un archivo ilegible es trabajo igual: no se pisa por no poder contarlo.
    annot.write_text("{ roto")
    assert _anotacion_en_riesgo(tmp_path, "v") == (-1, -1)
    assert AnotacionEnRiesgoError is not None
