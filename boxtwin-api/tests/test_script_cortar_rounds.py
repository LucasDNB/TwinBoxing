"""cortar_rounds.py (gemelo-tercero, 2-11): tiempos, validacion de cortes y comandos."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cortar_rounds", Path(__file__).resolve().parents[1] / "scripts" / "cortar_rounds.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


@pytest.mark.parametrize("texto, seg", [("41", 41.0), ("4:43", 283.0), ("1:02:05.5", 3725.5),
                                        ("0:00", 0.0)])
def test_a_segundos(texto, seg):
    assert m.a_segundos(texto) == seg


@pytest.mark.parametrize("texto", ["", "1::2", "1:2:3:4", "a:10"])
def test_a_segundos_rechaza(texto):
    with pytest.raises(ValueError):
        m.a_segundos(texto)


def test_planear_doce_rounds_con_descansos_variables():
    inicios = [40 + i * 245 for i in range(12)]   # descansos de 65 s
    c = m.planear(inicios, 180, duracion_s=3000)
    assert len(c) == 12 and all(x["dur_s"] == 180 for x in c)
    assert c[0]["inicio_s"] == 40 and c[-1]["round"] == 12


def test_planear_nocaut_y_ultimo_recortado_al_video():
    c = m.planear([0, 240, 480], 180, duracion_s=600, fin_ultimo=555)
    assert c[-1]["dur_s"] == 75
    c = m.planear([0, 240], 180, duracion_s=400)
    assert c[-1]["dur_s"] == 160


@pytest.mark.parametrize("inicios, dur, error", [
    ([0, 100], 1000, "despues del inicio"),
    ([0, 240, 200], 1000, "despues del inicio"),
    ([0, 1100], 1000, "dura"),
    ([0, 900, 1200], 1000, "dura"),
    ([], 1000, "al menos"),
])
def test_planear_rechaza(inicios, dur, error):
    with pytest.raises(ValueError, match=error):
        m.planear(inicios, 180, duracion_s=dur)


def test_comandos_reencodean_y_unen_sin_reencodear(tmp_path):
    cmd = m.comando_corte(Path("v.mp4"), {"inicio_s": 40.0, "dur_s": 180.0}, tmp_path / "r.mp4")
    assert cmd[cmd.index("-ss") + 1] == "40.000" and cmd.index("-ss") < cmd.index("-i")
    assert "libx264" in cmd and cmd[cmd.index("-t") + 1] == "180.000"
    u = m.comando_union(tmp_path / "lista.txt", tmp_path / "out.mp4")
    assert u[u.index("-c") + 1] == "copy"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="sin ffmpeg")
def test_de_punta_a_punta_con_un_video_sintetico(tmp_path):
    video = tmp_path / "trans.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=20:size=160x120:rate=30", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    salida = tmp_path / "netos.mp4"
    assert m.main([str(video), "--inicios", "0:02", "0:10", "--round-s", "5",
                   "-o", str(salida)]) == 0
    assert abs(m.duracion(salida) - 10.0) < 0.2
    meta = salida.with_suffix(".cortes.json").read_text()
    assert '"sha256_original"' in meta
    # Reanudable: la segunda corrida no vuelve a cortar.
    antes = (salida.with_suffix(".rounds") / "round01.mp4").stat().st_mtime
    assert m.main([str(video), "--inicios", "0:02", "0:10", "--round-s", "5",
                   "-o", str(salida)]) == 0
    assert (salida.with_suffix(".rounds") / "round01.mp4").stat().st_mtime == antes
