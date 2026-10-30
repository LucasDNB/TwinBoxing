"""T10: mejoras.sh en ensayo (MEJORAS_ENSAYO=1): imprime los comandos de cada etapa, cada una
detras de su candado, y no corre nada."""

import os
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / "mejoras.sh"
ETAPAS = [f"S{i}" for i in range(3, 16)]


def _correr(tmp_path, *args, **env):
    e = {**os.environ, "BOXTWIN_MEJORAS": str(tmp_path / "mejoras"),
         "BOXTWIN_GEMELO": str(tmp_path / "gemelo"), "MEJORAS_ENSAYO": "1", **env}
    return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=e)


def test_ensayo_imprime_cada_etapa_detras_de_su_candado(tmp_path):
    r = _correr(tmp_path, *ETAPAS)
    assert r.returncode == 0, r.stderr
    lineas = r.stdout.splitlines()
    assert [l.split()[2] for l in lineas if l.startswith("+ candado")] == ETAPAS
    for e in ETAPAS:
        i = next(i for i, l in enumerate(lineas) if l.startswith(f"+ candado {e} "))
        siguiente = lineas[i + 1] if i + 1 < len(lineas) else ""
        assert siguiente.startswith("+ ") and not siguiente.startswith("+ candado"), e


def test_ensayo_no_corre_nada(tmp_path):
    r = _correr(tmp_path, *ETAPAS)
    assert r.returncode == 0
    assert not (tmp_path / "mejoras").exists()


def test_los_comandos_de_cada_mejora(tmp_path):
    out = _correr(tmp_path, *ETAPAS).stdout
    assert "kreuso.py" in out and "calibrar_umbrales.py" in out
    assert "--umbrales" in out and "BOXTWIN_UMBRALES=" in out
    assert "--relleno" in out and "BOXTWIN_RELLENO=1" in out
    assert "--guantes" in out and "guantes_por_cuadro.py" in out
    assert "medir_e11.py desarrollo" in out and "medir_e11.py confirmacion" in out
    for exp in ("e8b", "e10", "e11", "e12"):
        assert f"lectura.py {exp}" in out


def test_nada_escribe_en_e8(tmp_path):
    out = _correr(tmp_path, *ETAPAS).stdout
    e8 = str(tmp_path / "gemelo" / "e8")
    for l in out.splitlines():
        for marca in ("--out ", "-o "):
            if marca in l:
                assert not l.split(marca)[1].split()[0].startswith(e8), l
        if " cp " in l:
            assert not l.split()[-1].startswith(e8), l


def test_una_etapa_hecha_no_se_repite(tmp_path):
    (tmp_path / "mejoras" / "etapas").mkdir(parents=True)
    (tmp_path / "mejoras" / "etapas" / "S4.hecho").write_text("")
    out = _correr(tmp_path, "S4", "S5").stdout
    assert "S4 ya hecha" in out and "+ candado S4 " not in out and "+ candado S5 " in out


def test_sin_etapas_o_desconocida_es_error(tmp_path):
    assert _correr(tmp_path).returncode == 2
    assert _correr(tmp_path, "S99").returncode == 2


def test_s11_sin_videos_fuera_de_ensayo_falla(tmp_path):
    (tmp_path / "mejoras" / "etapas").mkdir(parents=True)
    r = _correr(tmp_path, "S11", MEJORAS_ENSAYO="0", PY="/bin/true")
    assert r.returncode != 0 and "MEJORAS_E11_VIDEOS" in r.stderr


def test_golpes_de_e8_se_borra_antes_de_rearmarlo(tmp_path):
    lineas = _correr(tmp_path, "S3").stdout.splitlines()
    i_rm = next(i for i, l in enumerate(lineas) if l.startswith("+ rm -f") and "golpes.npz" in l)
    i_f2 = next(i for i, l in enumerate(lineas) if "features2d.py" in l)
    i_cp = next(i for i, l in enumerate(lineas) if l.startswith("+ cp "))
    assert i_cp < i_rm < i_f2
