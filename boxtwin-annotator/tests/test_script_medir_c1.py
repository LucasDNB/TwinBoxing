"""Las cuentas de C1 (pre-registro del 20-10): siembra simulada, puntaje sin cruce y resumen."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "medir_c1", Path(__file__).resolve().parents[1] / "scripts" / "medir_c1.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

A, B, IGN = "fighter_A", "fighter_B", "ignore"
VERDAD = {1: A, 2: B, 3: A, 4: B, 5: A}


def _asig(track, rol, ini=0, fin=10):
    return SimpleNamespace(track_id=track, role=rol, start_frame=ini, end_frame_excl=fin)


def test_verdad_es_el_rol_mayoritario_por_cuadros_y_ignora_lo_que_no_es_peleador():
    v, c = m.verdad_de([_asig(1, A, 0, 30), _asig(1, B, 30, 40), _asig(9, IGN),
                        _asig(2, IGN, 0, 50), _asig(2, B, 50, 55)])
    assert v == {1: A, 2: B}
    assert c == {1: {A: 30, B: 10}, 2: {A: 0, B: 5}}


def test_verdad_cuenta_los_cuadros_reales_del_track_en_el_cache():
    import numpy as np

    frames = {1: np.array([0, 1, 2, 35, 36])}
    v, c = m.verdad_de([_asig(1, A, 0, 30), _asig(1, B, 30, 40)], frames.__getitem__)
    assert v == {1: A} and c == {1: {A: 3, B: 2}}


def test_un_track_sin_cuadros_en_su_rango_queda_afuera():
    import numpy as np

    v, c = m.verdad_de([_asig(1, A, 0, 30), _asig(2, B, 50, 60)],
                       {1: np.array([5]), 2: np.array([5])}.__getitem__)
    assert v == {1: A} and 2 not in c


def test_par_sugerido_de_dos_peleadores_se_confirma_orientado():
    s = m.sembrar((2, 1), [2, 1, 7], VERDAD)
    assert s == {"semillas": (1, 2), "caso": "par_sugerido"}


def test_par_sugerido_con_un_arbitro_se_busca_en_la_lista_en_orden():
    s = m.sembrar((1, 9), [9, 3, 1, 4, 2], VERDAD)
    assert s == {"semillas": (3, 4), "caso": "lista"}


def test_sin_par_sugerido_tambien_se_busca_en_la_lista():
    assert m.sembrar(None, [4, 5], VERDAD)["caso"] == "lista_sin_par_sugerido"


def test_dos_peleadores_del_mismo_lado_no_alcanzan_como_par():
    s = m.sembrar((1, 3), [1, 3], VERDAD)
    assert s == {"semillas": None, "caso": "sin_siembra_posible"}


def test_puntaje_no_prueba_la_correspondencia_cruzada():
    roles = {1: B, 2: A, 3: A, 4: IGN}
    cuadros = {1: {A: 10, B: 0}, 2: {A: 0, B: 10}, 3: {A: 30, B: 0},
               4: {A: 0, B: 5}, 5: {A: 5, B: 0}}
    p = m.puntuar(roles, VERDAD, cuadros)
    assert (p["evaluables"], p["aciertos"], p["invertidos"]) == (3, 1, 2)
    assert (p["descartados"], p["sin_decidir"]) == (1, 1)
    assert (p["cuadros_peleador"], p["cuadros_bien"]) == (60, 30)


def test_un_track_mixto_suma_solo_los_cuadros_de_su_rol():
    p = m.puntuar({1: A}, {1: A}, {1: {A: 30, B: 10}})
    assert (p["aciertos"], p["cuadros_bien"], p["cuadros_peleador"]) == (1, 30, 40)
    assert p["tracks_mixtos"] == 1


def _fuente(nombre, k, n, total, caso="par_sugerido", cb=0, ct=0):
    return {"fuente": nombre, "peleadores_en_la_verdad": total, "cuadros_peleador": ct,
            "siembra": {"caso": caso},
            "puntaje": None if caso == "sin_siembra_posible" else
            {"aciertos": k, "evaluables": n, "cuadros_bien": cb}}


def test_resumen_agrega_y_aplica_el_umbral():
    fs = [_fuente(f"f{i}", 19, 20, 25, cb=90, ct=100) for i in range(6)]
    r = m.resumir(fs)
    assert r["primaria"] == 0.95 and r["pasa"] and r["completo"]
    assert r["cobertura_tracks"] == round(114 / 150, 4)
    assert r["cobertura_cuadros"] == 0.9
    lo, hi = r["wilson95"]
    assert lo < 0.95 < hi


def test_una_fuente_sin_siembra_posible_no_pasa_aunque_el_resto_si():
    fs = [_fuente(f"f{i}", 20, 20, 20) for i in range(5)]
    fs.append(_fuente("f5", 0, 0, 12, caso="sin_siembra_posible"))
    r = m.resumir(fs)
    assert r["primaria"] == 1.0 and not r["pasa"]
    assert r["sin_siembra_posible"] == ["f5"]


def test_debajo_del_umbral_no_pasa():
    r = m.resumir([_fuente(f"f{i}", 18, 20, 20) for i in range(6)])
    assert r["primaria"] == 0.9 and not r["pasa"]


@pytest.mark.parametrize("ruta,a,b", [
    ("anotacion-spar-01/annotations/01-sparring.annot.json", 7, 7),
    ("anotacion-sparring-3/annotations/sparring-3-rounds.annot.json", 34, 34),
])
def test_verdad_sobre_las_anotaciones_del_repo(ruta, a, b):
    from boxtwin.core.annotations import load

    p = Path(__file__).resolve().parents[2] / ruta
    if not p.is_file():
        pytest.skip("anotacion no disponible")
    doc, _ = load(p)
    v, _ = m.verdad_de(doc.identity.assignments)
    assert sum(r == A for r in v.values()) == a
    assert sum(r == B for r in v.values()) == b


def test_medir_fuente_de_punta_a_punta_con_evidencia_simulada(tmp_path, monkeypatch):
    """El pegamento contra las piezas reales: proponer, elegir y la anotacion del repo."""
    import numpy as np

    import boxtwin.core.identidad_auto as ia
    import boxtwin.core.posecache as pc
    from boxtwin.core.annotations import load
    from boxtwin.core.types import TrackRole

    ruta = Path(__file__).resolve().parents[2] / "anotacion-spar-01/annotations/01-sparring.annot.json"
    if not ruta.is_file():
        pytest.skip("anotacion no disponible")
    proyecto = tmp_path / "anotacion-spar-01"
    (proyecto / "annotations").mkdir(parents=True)
    (proyecto / "cache").mkdir()
    (proyecto / "videos").mkdir()
    (proyecto / "annotations" / "01-sparring.annot.json").write_text(ruta.read_text())
    (proyecto / "cache" / "01-sparring.pose.npz").write_bytes(b"")
    video = proyecto / "videos" / "01-sparring.mp4"
    video.write_bytes(b"")
    salida = tmp_path / "c1"
    salida.mkdir()
    (salida / "01-sparring.evidencia.json").write_text("{}")

    doc, _ = load(ruta)
    verdad, _ = m.verdad_de(doc.identity.assignments)
    a = sorted(t for t, r in verdad.items() if r == A)
    b = sorted(t for t, r in verdad.items() if r == B)

    class Cache:
        def frames_of_track(self, t):
            return np.arange(0, 10**6)

    def proponer_falso(ev, cfg, total, fps, semillas=None):
        roles = {t: TrackRole.A for t in a} | {t: TrackRole.B for t in b}
        if semillas is None:   # sin siembra: el ultimo de cada lado queda sin decidir
            roles.pop(a[-1]); roles.pop(b[-1])
        prop = ia.Propuesta(roles=roles, rangos={})
        prop.diagnostico["nucleo_tracks"] = a + b + [999]
        return prop

    def elegir_falso(ev, nucleo, **kw):
        cands = [SimpleNamespace(track=t) for t in [999, b[0], a[0]]]
        return (999, a[0]), cands

    import boxtwin.mvp.candidatos as cand
    monkeypatch.setattr(ia, "cargar_evidencia", lambda p: {})
    monkeypatch.setattr(ia, "proponer", proponer_falso)
    monkeypatch.setattr(cand, "elegir", elegir_falso)
    monkeypatch.setattr(pc.PoseCache, "open", classmethod(lambda cls, p: Cache()))

    res = m.medir_fuente(video, tmp_path / "guantes.pt", salida, "cpu")
    assert res["siembra"] == {"caso": "lista", "semillas": [a[0], b[0]]}
    assert res["puntaje"]["aciertos"] == res["puntaje"]["evaluables"] == len(verdad)
    assert res["sin_siembra"]["evaluables"] == len(verdad) - 2
    assert res["peleadores_en_la_verdad"] == 14
