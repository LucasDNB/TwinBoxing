"""
T5 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6).

BOXTWIN_RAIZ_FUENTES cambia la raiz de los proyectos de anotacion. Sin la variable,
`ruta_annot` es la de hoy: el repo.
"""

import importlib


def test_sin_la_variable_ruta_annot_igual_que_hoy(monkeypatch):
    monkeypatch.delenv("BOXTWIN_RAIZ_FUENTES", raising=False)
    import comun
    importlib.reload(comun)
    assert comun.RAIZ_FUENTES == comun.REPO
    assert comun.ruta_annot("01-sparring") == (
        comun.REPO / "anotacion-spar-01" / "annotations" / "01-sparring.annot.json")


def test_con_la_variable_apunta_a_la_raiz_dada(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXTWIN_RAIZ_FUENTES", str(tmp_path))
    import comun
    importlib.reload(comun)
    try:
        assert comun.ruta_annot("pacquiao_margarito") == (
            tmp_path / "anotacion-pacquiao" / "annotations" / "pacquiao_margarito.annot.json")
    finally:
        monkeypatch.delenv("BOXTWIN_RAIZ_FUENTES")
        importlib.reload(comun)


def test_la_variable_expande_el_home(monkeypatch):
    monkeypatch.setenv("BOXTWIN_RAIZ_FUENTES", "~/boxtwin-gemelo/raiz-y26")
    import comun
    importlib.reload(comun)
    try:
        assert "~" not in str(comun.RAIZ_FUENTES)
    finally:
        monkeypatch.delenv("BOXTWIN_RAIZ_FUENTES")
        importlib.reload(comun)
