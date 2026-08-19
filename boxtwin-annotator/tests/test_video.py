"""
Sondeo de metadatos y reconciliacion de fps.

El video de prueba se genera con ffmpeg en el momento: 25 cuadros a 25 fps exactos. Asi
los tests no dependen de ningun archivo del repo y siguen valiendo en otra maquina.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from boxtwin.core.types import FpsSource
from boxtwin.core.video import (
    FPS_TOLERANCE,
    ProbeError,
    VideoProbe,
    count_frames_by_decoding,
    probe,
    reconcile_fps,
    sha256_file,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="hacen falta ffmpeg y ffprobe",
)


@pytest.fixture(scope="module")
def video_prueba(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destino = tmp_path_factory.mktemp("video") / "prueba.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=1:size=64x48:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destino)],
        check=True,
    )
    return destino


def _probe_falso(**kw) -> VideoProbe:
    base = VideoProbe(
        path=Path("falso.mp4"), width=1920, height=1080, codec="h264",
        fps_rational=Fraction(30000, 1001), fps_declared=29.97,
        duration_s=100.0, nb_frames_declared=None,
    )
    return replace(base, **kw)


# -- probe -----------------------------------------------------------------


def test_probe_lee_lo_que_declara(video_prueba: Path) -> None:
    info = probe(video_prueba)
    assert (info.width, info.height) == (64, 48)
    assert info.codec == "h264"
    assert info.fps_rational == Fraction(25, 1)
    assert info.duration_s == pytest.approx(1.0, abs=0.05)


def test_probe_falla_claro_si_no_existe(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="no existe"):
        probe(tmp_path / "no_existe.mp4")


def test_probe_falla_si_no_hay_stream_de_video(tmp_path: Path) -> None:
    basura = tmp_path / "basura.mp4"
    basura.write_bytes(b"esto no es un video")
    with pytest.raises(ProbeError):
        probe(basura)


def test_conteo_por_decodificacion(video_prueba: Path) -> None:
    assert count_frames_by_decoding(video_prueba) == 25


# -- reconciliacion de fps -------------------------------------------------


def test_fps_concordante_usa_el_racional_exacto() -> None:
    """
    30000/1001 es exacto y 29,97 es una aproximacion. Sobre un video de una hora la
    diferencia entre los dos son mas de tres cuadros de deriva.
    """
    info = _probe_falso(duration_s=100.0)
    real = round(100.0 * 30000 / 1001)  # 2997
    v = reconcile_fps(info, real)
    assert v.source is FpsSource.CONTAINER_VERIFIED
    assert v.fps == float(Fraction(30000, 1001))
    assert not v.suspected_vfr


def test_fps_discordante_cae_al_medido() -> None:
    """Si el contenedor declara 30 y hay la mitad de cuadros, el contenedor miente."""
    info = _probe_falso(fps_rational=Fraction(30, 1), duration_s=100.0)
    v = reconcile_fps(info, 1500)
    assert v.source is FpsSource.MEASURED
    assert v.fps == pytest.approx(15.0)
    assert v.suspected_vfr


def test_tolerancia_es_el_borde_exacto() -> None:
    info = _probe_falso(fps_rational=Fraction(30, 1), duration_s=100.0)
    # Justo dentro de la tolerancia.
    adentro = round(3000 * (1 + FPS_TOLERANCE * 0.9))
    assert reconcile_fps(info, adentro).source is FpsSource.CONTAINER_VERIFIED
    # Justo afuera.
    afuera = round(3000 * (1 + FPS_TOLERANCE * 2))
    assert reconcile_fps(info, afuera).source is FpsSource.MEASURED


def test_reconciliar_exige_frames_positivos() -> None:
    with pytest.raises(ValueError):
        reconcile_fps(_probe_falso(), 0)


def test_nb_frames_ausente_no_rompe(video_prueba: Path) -> None:
    """
    Los nueve videos del proyecto salen con nb_frames en N/A. Que el campo falte tiene que
    ser un caso normal, no una excepcion.
    """
    info = probe(video_prueba)
    assert info.nb_frames_declared is None or isinstance(info.nb_frames_declared, int)
    v = reconcile_fps(info, 25)
    assert v.total_frames == 25


# -- hash ------------------------------------------------------------------


def test_sha256_es_estable_y_sensible(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 10_000)
    b.write_bytes(b"x" * 9_999 + b"y")
    assert sha256_file(a) == sha256_file(a)
    assert sha256_file(a) != sha256_file(b)
    assert len(sha256_file(a)) == 64


# -- deteccion de cortes de plano ------------------------------------------


def test_detectar_cortes_encuentra_los_pegados(tmp_path) -> None:
    """
    Control con la respuesta conocida: tres clips distintos pegados tienen dos cortes. Un
    umbral sin control no informa nada, que es la leccion que dejo el detector de placas de
    BoxingVI marcando 528 falsos positivos.
    """
    import shutil
    import subprocess

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("sin ffmpeg")

    from boxtwin.preprocess.cuts import detectar_cortes

    # Fuentes con textura y no colores planos: un color solido es un caso patologico para
    # la deteccion de escena y no se parece en nada a video real. Con colores planos este
    # mismo control encontraba 1 de 2 cortes.
    partes = []
    for i, fuente in enumerate(("testsrc", "smptebars", "rgbtestsrc")):
        f = tmp_path / f"p{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"{fuente}=s=160x120:d=1:r=10", "-pix_fmt", "yuv420p", str(f)],
            check=True,
        )
        partes.append(f)
    lista = tmp_path / "l.txt"
    lista.write_text("".join(f"file '{p}'\n" for p in partes))
    pegado = tmp_path / "pegado.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lista), "-c", "copy", str(pegado)],
        check=True,
    )

    cortes = detectar_cortes(pegado, fps=10.0)
    assert len(cortes) == 2, cortes
    # Caen donde empieza cada clip nuevo, con tolerancia de un cuadro por el redondeo.
    assert all(abs(c - esperado) <= 1 for c, esperado in zip(cortes, (10, 20)))


def test_una_sola_toma_no_tiene_cortes(tmp_path) -> None:
    import shutil
    import subprocess

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("sin ffmpeg")

    from boxtwin.preprocess.cuts import detectar_cortes

    f = tmp_path / "uno.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=s=160x120:d=3:r=10", "-pix_fmt", "yuv420p", str(f)],
        check=True,
    )
    assert detectar_cortes(f, fps=10.0) == []


def test_una_transicion_no_produce_cuatro_cortes(tmp_path, monkeypatch) -> None:
    """
    Una disolvencia mantiene el puntaje de escena alto varios cuadros y el filtro dispara en
    cada uno. Sobre 20 s de la pelea salieron cortes en 195, 197, 199 y 201: es una sola
    transicion, y reiniciar el tracker cuatro veces seguidas no aporta nada.
    """
    from boxtwin.preprocess import cuts

    salida = "pts_time:3.25\npts_time:3.283\npts_time:3.316\npts_time:3.35\npts_time:20.0\n"

    class Fake:
        stderr = salida

    monkeypatch.setattr(cuts.subprocess, "run", lambda *a, **k: Fake())
    assert cuts.detectar_cortes(tmp_path / "x.mp4", fps=60.0) == [195, 1200]


def test_una_disolvencia_larga_colapsa_a_un_solo_corte(tmp_path, monkeypatch) -> None:
    """Se compara contra el ultimo guardado: si no, los cortes se encadenarian de a pares."""
    from boxtwin.preprocess import cuts

    tiempos = [3.0 + i * 0.05 for i in range(10)]   # medio segundo disparando cada 3 cuadros

    class Fake:
        stderr = "".join(f"pts_time:{t:.3f}\n" for t in tiempos)

    monkeypatch.setattr(cuts.subprocess, "run", lambda *a, **k: Fake())
    assert cuts.detectar_cortes(tmp_path / "x.mp4", fps=60.0) == [180]
