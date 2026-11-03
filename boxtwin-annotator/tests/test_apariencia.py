"""
IE7 (docs/specs/2026-11-02-identidad-estable/02-ie7-apariencia.md): verificador de apariencia
con guante, remera y pantalon. Todo sintetico.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxtwin.core import apariencia as ap
from boxtwin.core.identidad_auto import (
    ConfigIdentidadAuto, EvidenciaTrack, cargar_evidencia, guardar_evidencia,
)
from boxtwin.mvp.revision import Tramo, cola_de_revision

CFG = ConfigIdentidadAuto()
FPS = 25.0
NARANJA = (10.0, 200.0, 150.0)          # el mismo guante para los dos: no separa
NARANJA2 = (12.0, 205.0, 145.0)
BLANCA = (-1.0, 20.0, 230.0)            # remera de A
NEGRA = (-1.0, 30.0, 25.0)              # remera de B
AZUL = (110.0, 180.0, 120.0)            # pantalon de A
VERDE = (60.0, 180.0, 120.0)            # pantalon de B


def _kp(hombro_y=100, cadera_y=200, rodilla_y=300, x0=50, x1=110):
    kp = np.zeros((17, 2), np.float32)
    for i, x in ((5, x0), (6, x1), (11, x0 + 5), (12, x1 - 5), (13, x0 + 5), (14, x1 - 5)):
        kp[i, 0] = x
    kp[[5, 6], 1] = hombro_y
    kp[[11, 12], 1] = cadera_y
    kp[[13, 14], 1] = rodilla_y
    return kp


def _ev(tid, frames, guante, remera, pantalon, aislado=True):
    frames = list(frames)
    return EvidenciaTrack(tid, frames[0], frames[-1], 0.9, len(frames), len(frames),
                          [(f, guante) for f in frames], frames, [(f, 0.0) for f in frames],
                          ropa=[(f, remera, pantalon, aislado) for f in frames])


def _semillas():
    return {1: _ev(1, range(0, 400, 5), NARANJA, BLANCA, AZUL),
            2: _ev(2, range(0, 400, 5), NARANJA, NEGRA, VERDE)}


# -- medir --------------------------------------------------------------------


def test_las_zonas_salen_de_la_pose_y_se_niegan_sin_puntaje():
    kp, sc = _kp(), np.ones(17)
    assert ap.zona_remera(kp, sc) == (50, 100, 110, 200)
    assert ap.zona_pantalon(kp, sc) == (55, 200, 105, 240)
    sc[11] = 0.1
    assert ap.zona_remera(kp, sc) is None and ap.zona_pantalon(kp, sc) is None


def test_medir_ropa_lee_el_color_de_cada_zona():
    import cv2

    img = np.zeros((400, 200, 3), np.uint8)
    img[100:200] = (0, 0, 220)              # remera roja (BGR)
    img[200:300] = (220, 0, 0)              # pantalon azul
    remera, pantalon = ap.medir_ropa(img, _kp(), np.ones(17))
    tono_rojo = cv2.cvtColor(np.uint8([[[0, 0, 220]]]), cv2.COLOR_BGR2HSV)[0, 0, 0]
    tono_azul = cv2.cvtColor(np.uint8([[[220, 0, 0]]]), cv2.COLOR_BGR2HSV)[0, 0, 0]
    assert remera[0] == pytest.approx(tono_rojo, abs=1)
    assert pantalon[0] == pytest.approx(tono_azul, abs=1)


# -- modelo -------------------------------------------------------------------


def test_la_senal_que_separa_pesa_y_la_que_no_pesa_cero():
    m = ap.ajustar(_semillas(), (1, 2))
    assert m.confiabilidad["remera"] == 1.0 and m.confiabilidad["pantalon"] == 1.0
    assert m.pesos["remera"] == pytest.approx(np.log(0.99 / 0.01))
    # el mismo naranja en los dos: el guante no da signo y queda afuera del modelo
    assert "guante" not in m.pesos
    assert m.util


def test_sin_ropa_en_la_evidencia_el_modelo_es_solo_guante_y_no_inventa():
    ev = _semillas()
    for e in ev.values():
        e.ropa = []
    m = ap.ajustar(ev, (1, 2))
    assert m.pesos == {} and not m.util


def test_la_confiabilidad_es_por_bloques_de_tiempo():
    # en el ultimo bloque las semillas se invierten (un track de semilla mixto): ese bloque
    # falla entero y el acierto queda en 3 de 4, no en 1
    ma = [(f, NEGRA if f >= 300 else BLANCA) for f in range(0, 400, 5)]
    mb = [(f, BLANCA if f >= 300 else NEGRA) for f in range(0, 400, 5)]
    assert ap.confiabilidad(ma, mb) == pytest.approx(0.75)


# -- sospecha -----------------------------------------------------------------


def test_un_tramo_con_la_ropa_del_otro_contradice():
    m = ap.ajustar(_semillas(), (1, 2))
    e = _ev(7, range(0, 200, 5), NARANJA, NEGRA, VERDE)          # viste como B
    assert ap.sospecha(m, e, "A", 0, 200, CFG) == "apariencia_contradice"
    assert ap.sospecha(m, e, "B", 0, 200, CFG) is None


def test_un_tramo_que_cambia_de_ropa_en_el_medio_es_mezcla():
    m = ap.ajustar(_semillas(), (1, 2))
    a = _ev(7, range(0, 300, 5), NARANJA, BLANCA, AZUL)
    b = _ev(7, range(300, 400, 5), NARANJA, NEGRA, VERDE)
    a.ropa += b.ropa
    a.colores += b.colores
    assert ap.sospecha(m, a, "A", 0, 400, CFG) == "apariencia_mezcla"


def test_en_contacto_la_ropa_tambien_vota():
    m = ap.ajustar(_semillas(), (1, 2))
    e = _ev(7, range(0, 200, 5), NARANJA, NEGRA, VERDE, aislado=False)
    e.colores = []                       # en contacto no hay guante aislado
    assert ap.sospecha(m, e, "A", 0, 200, CFG) == "apariencia_contradice"


def test_sin_modelo_no_hay_sospecha():
    e = _ev(7, range(0, 200, 5), NARANJA, NEGRA, VERDE)
    assert ap.sospecha(None, e, "A", 0, 200, CFG) is None


# -- evidencia y cola ---------------------------------------------------------


def test_la_ropa_viaja_en_la_evidencia_y_la_vieja_se_sigue_leyendo(tmp_path):
    import json

    ev = _semillas()
    ev[1].ropa[0] = (0, None, AZUL, False)
    guardar_evidencia(ev, tmp_path / "e.json")
    leida = cargar_evidencia(tmp_path / "e.json")
    assert leida[1].ropa[0] == (0, None, AZUL, False)
    assert leida[2].ropa[3] == (15, NEGRA, VERDE, True)
    d = json.loads((tmp_path / "e.json").read_text())
    for t in d["tracks"]:
        del t["ropa"]
    (tmp_path / "vieja.json").write_text(json.dumps(d))
    assert cargar_evidencia(tmp_path / "vieja.json")[1].ropa == []


def test_la_cola_suma_el_motivo_solo_con_el_modelo():
    ev = _semillas()
    ev[7] = _ev(7, range(500, 700, 5), NARANJA, NEGRA, VERDE)
    auto = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(7, 500, 700, "A")]
    perfiles = {"fighter_A": NARANJA, "fighter_B": NARANJA2}        # el guante no separa
    sin = cola_de_revision(auto, auto, ev, perfiles, CFG, FPS, todos=False)
    assert sin == []
    con = cola_de_revision(auto, auto, ev, perfiles, CFG, FPS, todos=False,
                           apariencia=ap.ajustar(ev, (1, 2)))
    assert [(i["track"], i["motivo"], i["tipo"]) for i in con] == [
        (7, "apariencia_contradice", "sospechoso")]
