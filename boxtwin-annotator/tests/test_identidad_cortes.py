"""
T6 (mejoras post-E8, E11 §2): cortar los tracks mixtos donde cambia el color, antes de
`particionar`. Apagado por defecto: con `cortar_mixtos=False` la propuesta es la de antes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from boxtwin.core.identidad_auto import (
    AsignarIdentidadAuto, ConfigIdentidadAuto, EvidenciaTrack, cargar_evidencia, proponer,
    segmentar,
)
from boxtwin.core.types import TrackRole

DATOS = Path(__file__).parent / "datos"
ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)
VIOLETA = (160.0, 200.0, 120.0)   # cerca del rojo: con el rojo no separa
CFG = ConfigIdentidadAuto()
CON = replace(CFG, cortar_mixtos=True)


def _track(tid, tramos, x=300.0):
    cols = [(f, c) for fs, c in tramos for f in fs]
    frames = [f for f, _ in cols]
    return EvidenciaTrack(tid, frames[0], frames[-1], 0.9, len(frames), len(frames),
                          cols, frames, [(f, x) for f in frames])


def _semillas(color_a=ROJO, color_b=AZUL):
    return {1: _track(1, [(range(0, 200, 5), color_a)], 100.0),
            2: _track(2, [(range(0, 200, 5), color_b)], 500.0)}


class _Dets:
    def __init__(self, ids, cajas):
        self.track_id = np.asarray(ids)
        self.bbox = np.asarray(cajas, float)

    def __len__(self):
        return len(self.track_id)

    def index_of_track(self, t):
        w = np.where(self.track_id == t)[0]
        return int(w[0]) if len(w) else None


class _Cache:
    """El track 3 se superpone con el 9 en el cuadro 447: ahi se confunde el tracker."""

    def detections(self, f):
        otro = 100.0 - (60.0 if f == 447 else 0.0)
        return _Dets([3, 9, 7], [(200, 0, 300, 400), (otro + 150, 0, otro + 250, 400),
                                 (220, 300, 260, 340)])   # el 7 es chico: no cuenta


def test_un_corte_en_el_cuadro_de_iou_maximo():
    ev = {**_semillas(), 3: _track(3, [(range(300, 450, 5), ROJO), (range(450, 600, 5), AZUL)])}
    segs, diag = segmentar(ev, CON, (1, 2), cache=_Cache())
    partes = sorted((s.inicio, s.fin) for s in segs.values() if s.track_id == 3)
    assert partes == [(300, 447), (447, 596)]
    assert diag["tracks_partidos"] == 1
    (c,) = diag["cortes"]
    assert c["track"] == 3 and c["cuadro"] == 447 and c["medianas"][0] > 0 > c["medianas"][1]


def test_sin_cache_el_corte_va_al_punto_medio():
    ev = {**_semillas(), 3: _track(3, [(range(300, 450, 5), ROJO), (range(450, 600, 5), AZUL)])}
    segs, _ = segmentar(ev, CON, (1, 2))
    assert sorted(s.inicio for s in segs.values() if s.track_id == 3) == [300, 448]


def test_los_segmentos_reparten_la_evidencia():
    ev = {**_semillas(), 3: _track(3, [(range(300, 450, 5), ROJO), (range(450, 600, 5), AZUL)])}
    segs, _ = segmentar(ev, CON, (1, 2))
    partes = [s for s in segs.values() if s.track_id == 3]
    assert sum(len(s.evidencia.colores) for s in partes) == 60
    assert sum(s.evidencia.recortes for s in partes) == ev[3].recortes
    for s in partes:
        assert all(s.inicio <= f < s.fin for f, _ in s.evidencia.colores)
        assert s.evidencia.primer_frame == s.inicio and s.evidencia.ultimo_frame == s.fin - 1


def test_semillas_que_no_separan_no_cortan_nada():
    ev = {**_semillas(ROJO, VIOLETA),
          3: _track(3, [(range(300, 450, 5), ROJO), (range(450, 600, 5), VIOLETA)])}
    segs, diag = segmentar(ev, CON, (1, 2))
    assert set(segs) == set(ev) and diag["tracks_partidos"] == 0
    assert "separacion" in diag["motivo"]


def test_menos_de_cinco_muestras_de_un_lado_no_corta():
    # el minimo es por lado del corte (m <= k <= n - m), no por color
    ev = {**_semillas(), 3: _track(3, [(range(300, 320, 5), ROJO), (range(320, 340, 5), AZUL)])}
    segs, diag = segmentar(ev, CON, (1, 2))
    assert diag["tracks_partidos"] == 0
    ev[3] = _track(3, [(range(300, 325, 5), ROJO), (range(325, 350, 5), AZUL)])   # 5 justas
    _, diag = segmentar(ev, CON, (1, 2))
    assert diag["tracks_partidos"] == 1


def test_un_track_de_un_solo_color_no_se_corta():
    ev = {**_semillas(), 3: _track(3, [(range(300, 600, 5), ROJO)])}
    segs, diag = segmentar(ev, CON, (1, 2))
    assert segs[3].evidencia is ev[3] and diag["tracks_partidos"] == 0


def test_dos_cortes_como_maximo():
    # azul 8, rojo 40, azul 40, rojo 8: el primer corte va al medio y las dos mitades piden
    # otro; sin el tope serian tres
    tramos = [(range(300, 340, 5), AZUL), (range(340, 540, 5), ROJO),
              (range(540, 740, 5), AZUL), (range(740, 780, 5), ROJO)]
    ev = {**_semillas(), 3: _track(3, tramos)}
    segs, diag = segmentar(ev, CON, (1, 2))
    assert len([s for s in segs.values() if s.track_id == 3]) == 3
    assert len(diag["cortes"]) == 2


def test_una_semilla_mixta_queda_en_el_segmento_del_recorte():
    ev = {1: _track(1, [(range(0, 200, 5), ROJO), (range(200, 400, 5), AZUL)], 100.0),
          2: _track(2, [(range(0, 400, 5), AZUL)], 500.0)}
    # el perfil de A sale de la semilla entera, antes de partir: rojo y azul mezclados
    segs, diag = segmentar(ev, CON, (1, 2), cuadros_semilla={1: 50, 2: 50})
    assert diag["semillas"][0] != 1 and segs[diag["semillas"][0]].inicio == 0


def test_apagado_da_la_propuesta_de_antes():
    ev = cargar_evidencia(DATOS / "evidencia_mixta.json")
    hoy = json.loads((DATOS / "propuesta_mixta_hoy.json").read_text())
    for nombre, sem in (("sin_semillas", None), ("con_semillas", (1, 2))):
        p = proponer(ev, CFG, 800, 30.0, semillas=sem)
        assert {str(t): r.value for t, r in sorted(p.roles.items())} == hoy[nombre]["roles"]
        assert {str(t): list(r) for t, r in sorted(p.rangos.items())} == hoy[nombre]["rangos"]
        assert sorted(p.sin_asignar) == hoy[nombre]["sin_asignar"]
        assert p.tracks == {}


def test_con_corte_el_track_mixto_recibe_dos_roles():
    ev = cargar_evidencia(DATOS / "evidencia_mixta.json")
    p = proponer(ev, CON, 800, 30.0, semillas=(1, 2))
    partes = {s: p.tracks[s] for s in p.roles if p.tracks.get(s) == 3}
    roles = {p.rangos[s][0]: p.roles[s] for s in partes}
    assert roles == {300: TrackRole.A, 448: TrackRole.B}
    assert p.diagnostico["tracks_partidos"] == 1
    assert 3 not in p.sin_asignar


def test_sin_semillas_no_se_corta():
    ev = cargar_evidencia(DATOS / "evidencia_mixta.json")
    p = proponer(ev, CON, 800, 30.0)
    assert p.tracks == {} and p.diagnostico["tracks_partidos"] == 0


def test_el_comando_escribe_un_assignment_por_segmento(doc_min):
    ev = cargar_evidencia(DATOS / "evidencia_mixta.json")
    p = proponer(ev, CON, 800, 30.0, semillas=(1, 2))
    AsignarIdentidadAuto(p).do(doc_min)
    del3 = sorted((a.start_frame, a.end_frame_excl, a.role) for a in doc_min.identity.assignments
                  if a.track_id == 3)
    assert del3 == [(300, 448, TrackRole.A), (448, 596, TrackRole.B)]
    assert all(a.track_id >= 0 for a in doc_min.identity.assignments)
