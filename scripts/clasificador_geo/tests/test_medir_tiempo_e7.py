"""medir_tiempo_e7.py (M6): mismos comandos que el worker con e7 y copia sin tocar la sesion."""

import medir_tiempo_e7 as mt


def test_e7_eleva_clasifica_con_pose3d_y_aplica_tipos(tmp_path):
    c = mt.comandos(tmp_path)
    assert len(c["E4"]) == 2 and len(c["E7"]) == 3
    assert c["E7"][0][1].endswith("boxtwin-annotator/tools/elevar_hmr2_sesion.py")
    assert "boxtwin_4dhumans" in c["E7"][0][0]
    cl = c["E7"][1]
    assert cl[cl.index("--modelo") + 1].endswith("clasificador_e7_v1")
    assert cl[cl.index("--pose3d") + 1] == str(tmp_path / "pose3d_hmr2.npz")
    assert c["E4"][0][c["E4"][0].index("--modelo") + 1].endswith("clasificador_geo_v1")
    assert c["E4"][1][-2:] == c["E7"][2][-2:] == ["tipos", str(tmp_path)]


def test_copiar_solo_los_archivos_de_la_sesion(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    for n in ("segmentos.json", "poses.npz", "sesion.json", "fightcard.json", "tipos.json"):
        (s / n).write_text(n)
    d = mt.copiar(s, tmp_path / "d")
    assert sorted(p.name for p in d.iterdir()) == ["fightcard.json", "poses.npz", "segmentos.json", "sesion.json"]
    assert (s / "tipos.json").read_text() == "tipos.json"
