"""T8: e8.sh en modo ensayo (E8_ENSAYO=1): el orden de las etapas, el candado antes de cada
una, --out en cada export y que una etapa hecha no se repite."""

import os
import subprocess
from pathlib import Path

E8 = Path(__file__).resolve().parents[1] / "e8.sh"
ETAPAS = ["S7", "S8", "S9", "S10", "S11", "S12", "S13"]


def _correr(gemelo, *args, **env):
    e = {**os.environ, "BOXTWIN_GEMELO": str(gemelo), "E8_ENSAYO": "1", **env}
    r = subprocess.run(["bash", str(E8), *args], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_cada_etapa_va_detras_de_su_candado_y_en_orden(gemelo):
    lineas = _correr(gemelo)
    candados = [l.split()[2] for l in lineas if l.startswith("+ candado")]
    assert candados == ETAPAS
    pos = {e: next(i for i, l in enumerate(lineas) if l.startswith(f"+ candado {e} ")) for e in ETAPAS}
    assert [pos[e] for e in ETAPAS] == sorted(pos.values())


def test_cada_export_lleva_out_bajo_el_gemelo(gemelo):
    exports = [l for l in _correr(gemelo) if " export " in l]
    assert len(exports) == 14                                  # siete fuentes por dos condiciones
    for l in exports:
        out = l.split("--out ")[1].split()[0]
        assert out.startswith(str(gemelo / "e8"))


def test_las_dos_condiciones_con_su_raiz_de_fuentes(gemelo):
    lineas = _correr(gemelo)
    raices = {l.split("BOXTWIN_RAIZ_FUENTES=")[1].split()[0] for l in lineas if "datos.py" in l}
    assert str(gemelo / "raiz-y26") in raices and len(raices) == 2


def test_una_etapa_hecha_no_se_repite(gemelo):
    (gemelo / "e8" / "etapas").mkdir(parents=True)
    (gemelo / "e8" / "etapas" / "S7.hecho").write_text("")
    lineas = _correr(gemelo)
    assert not any(l.startswith("+ candado S7 ") for l in lineas)
    assert any("S7 ya hecha" in l for l in lineas)


def test_una_sola_etapa(gemelo):
    lineas = _correr(gemelo, "S12")
    assert [l.split()[2] for l in lineas if l.startswith("+ candado")] == ["S12"]


def test_rama_d_h2_recachea_v8_sin_reid_y_lo_pasa_por_el_puente(gemelo):
    lineas = _correr(gemelo, E8_DH2="1")
    recaches = [l for l in lineas if "recache.py" in l]
    assert len(recaches) == 2 and all("botsort-sin-reid.yaml" in l for l in recaches)
    assert any("yolov8l-pose.pt" in l for l in recaches)
    assert sum("puente_identidad.py" in l and "--k1" not in l for l in lineas) == 2
    assert any("medir_trackeo.py" in l and "--raiz-base" in l for l in lineas)
