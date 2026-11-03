"""
IE1 (docs/specs/2026-11-02-identidad-estable): tramos, correcciones por rango y la cola de
revision que tambien ve lo asignado.
"""

from __future__ import annotations

from dataclasses import replace

from boxtwin.core.identidad_auto import ConfigIdentidadAuto, EvidenciaTrack, proponer
from boxtwin.core.types import TrackRole
from boxtwin.mvp.revision import (
    Correccion, Tramo, a_propuesta, aplicar_correcciones, cola_de_revision, conflictos,
    correcciones_de_sesion, degradar_conflictos, extension_de, tramos_de_propuesta,
)
from boxtwin.mvp.tramos import aplicar_manuales

ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)
CFG = ConfigIdentidadAuto()
PERFILES = {"fighter_A": ROJO, "fighter_B": AZUL}
FPS = 25.0


def _track(tid, tramos, x=300.0):
    cols = [(f, c) for fs, c in tramos for f in fs]
    frames = [f for f, _ in cols]
    return EvidenciaTrack(tid, frames[0], frames[-1], 0.9, len(frames), len(frames),
                          cols, frames, [(f, x) for f in frames])


def _ev():
    return {1: _track(1, [(range(0, 400, 5), ROJO)], 100.0),
            2: _track(2, [(range(0, 400, 5), AZUL)], 500.0)}


# -- correcciones -----------------------------------------------------------


def test_una_correccion_parcial_parte_el_tramo_y_deja_los_costados():
    auto = [Tramo(7, 100, 400, "A")]
    out = aplicar_correcciones(auto, [Correccion(7, 200, 300, "B")])
    assert [(t.desde, t.hasta, t.rol, t.origen) for t in out] == [
        (100, 200, "A", "auto"), (200, 300, "B", "manual"), (300, 400, "A", "auto")]


def test_partir_un_track_mixto_son_dos_correcciones():
    auto = [Tramo(7, 100, 400, "A")]
    out = aplicar_correcciones(auto, [Correccion(7, 100, 250, "A"), Correccion(7, 250, 400, "B")])
    assert [(t.desde, t.hasta, t.rol) for t in out] == [(100, 250, "A"), (250, 400, "B")]
    assert all(t.origen == "manual" for t in out)


def test_la_correccion_se_recorta_a_la_extension_del_track_y_la_ultima_manda():
    auto = [Tramo(7, 100, 400, "A"), Tramo(8, 0, 50, "")]
    out = aplicar_correcciones(auto, [Correccion(7, 0, 9999, "B"), Correccion(7, 300, 9999, "A"),
                                      Correccion(99, 0, 10, "A")])     # el 99 no es de la sesion
    assert [(t.track, t.desde, t.hasta, t.rol) for t in out] == [
        (7, 100, 300, "B"), (7, 300, 400, "A"), (8, 0, 50, "")]


def test_correcciones_de_sesion_lee_lo_viejo_y_lo_nuevo_en_ese_orden():
    ext = {7: (100, 400), 8: (0, 50)}
    ident = {"asignaciones_manuales": {"7": "A", "55": "B"},
             "correcciones": [{"track": 7, "desde": 10.0, "hasta": None, "rol": "B"}]}
    cs = correcciones_de_sesion(ident, ext, FPS)
    assert cs == [Correccion(7, 100, 400, "A"), Correccion(7, 250, 400, "B")]


# -- sin correcciones, nada cambia ------------------------------------------


def test_sin_correcciones_la_propuesta_vuelve_identica():
    ev = {**_ev(), 3: _track(3, [(range(0, 100, 5), ROJO)], 120.0),
          4: _track(4, [(range(0, 10, 5), AZUL)], 480.0)}
    prop = proponer(ev, CFG, 400, FPS, semillas=(1, 2))
    tramos = tramos_de_propuesta(prop, ev, CFG)
    p2 = a_propuesta(aplicar_correcciones(tramos, []), prop)
    antes = sorted((t, *prop.rangos[t], r) for t, r in prop.roles.items())
    despues = sorted((p2.tracks[k], *p2.rangos[k], r) for k, r in p2.roles.items())
    assert antes == despues
    assert sorted(p2.tracks[k] for k in p2.sin_asignar) == sorted(prop.sin_asignar)


def test_la_asignacion_vieja_por_track_da_lo_mismo_que_aplicar_manuales():
    ev = {**_ev(), 3: _track(3, [(range(0, 15, 5), AZUL)], 120.0)}
    prop = proponer(ev, CFG, 400, FPS, semillas=(1, 2))
    tramos = tramos_de_propuesta(prop, ev, CFG)
    cs = correcciones_de_sesion({"asignaciones_manuales": {"3": "B"}}, extension_de(tramos), FPS)
    p2 = a_propuesta(aplicar_correcciones(tramos, cs), prop)
    aplicar_manuales(prop, {3: "B"})
    assert sorted((p2.tracks[k], r) for k, r in p2.roles.items()) == sorted(prop.roles.items())


# -- sospechas --------------------------------------------------------------


def test_dos_tramos_del_mismo_rol_en_el_mismo_cuadro_son_un_conflicto():
    ev = {**_ev(), 3: _track(3, [(range(100, 200, 5), ROJO)])}
    tramos = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(3, 100, 200, "A")]
    assert conflictos(tramos, ev, CFG) == [(0, 2)]
    out, n = degradar_conflictos(tramos, ev, CFG)
    assert n == 2 and [t.rol for t in out] == ["", "B", ""]


def test_lo_que_decidio_una_persona_no_se_degrada():
    ev = {**_ev(), 3: _track(3, [(range(100, 200, 5), ROJO)])}
    tramos = [Tramo(1, 0, 400, "A", origen="manual"), Tramo(2, 0, 400, "B"),
              Tramo(3, 100, 200, "A")]
    out, n = degradar_conflictos(tramos, ev, CFG)
    assert n == 1 and [t.rol for t in out] == ["A", "B", ""]


def test_la_cola_ve_los_asignados_sospechosos_y_los_ordena():
    ev = {**_ev(),
          3: _track(3, [(range(100, 200, 5), ROJO)]),                      # conflicto con 1
          4: _track(4, [(range(500, 700, 5), AZUL)]),                       # dice B, esta en A
          5: _track(5, [(range(1100, 1200, 5), ROJO), (range(1200, 1300, 5), AZUL)]),  # mezcla
          6: _track(6, [(range(800, 900, 5), ROJO)]),                       # bien
          8: _track(8, [(range(900, 1000, 5), AZUL)])}                      # sin asignar
    auto = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(3, 100, 200, "A"),
            Tramo(4, 500, 700, "A"), Tramo(5, 1100, 1300, "A"), Tramo(6, 800, 900, "A"),
            Tramo(8, 900, 1000, "", motivo="sin_coexistencia")]
    cola = cola_de_revision(auto, auto, ev, PERFILES, CFG, FPS)
    orden = [(c["track"], c["motivo"], c["tipo"]) for c in cola]
    assert orden[:5] == [(1, "conflicto", "sospechoso"), (3, "conflicto", "sospechoso"),
                         (4, "color_contradice", "sospechoso"),
                         (5, "posible_mezcla", "sospechoso"),
                         (8, "sin_coexistencia", "sin_asignar")]
    # los asignados sin sospecha van al final y no piden revision
    resto = {c["track"]: c for c in cola[5:]}
    assert set(resto) == {2, 6} and not any(c["revision_pendiente"] for c in resto.values())
    assert cola_de_revision(auto, auto, ev, PERFILES, CFG, FPS, todos=False)[-1]["track"] == 8


def test_un_mixto_fuerte_sin_cortar_tambien_sale_como_sospechoso():
    ev = {**_ev(), 5: _track(5, [(range(500, 600, 5), ROJO), (range(600, 650, 5), AZUL)])}
    auto = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(5, 500, 650, "A")]
    (item,) = [c for c in cola_de_revision(auto, auto, ev, PERFILES, CFG, FPS) if c["track"] == 5]
    assert item["motivo"] == "posible_mezcla" and item["revision_pendiente"]


def test_sin_perfiles_que_separen_no_se_sospecha_por_color():
    ev = {**_ev(), 4: _track(4, [(range(500, 700, 5), AZUL)])}
    auto = [Tramo(4, 500, 700, "A")]
    cola = cola_de_revision(auto, auto, ev, {"fighter_A": ROJO, "fighter_B": ROJO}, CFG, FPS)
    assert cola[0]["tipo"] == "asignado"


def test_la_decision_de_la_persona_queda_a_la_vista_y_baja_la_pendiente():
    ev = {**_ev(), 5: _track(5, [(range(500, 600, 5), ROJO), (range(600, 700, 5), AZUL)])}
    auto = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(5, 500, 700, "A")]
    corr = aplicar_correcciones(auto, [Correccion(5, 500, 600, "A"), Correccion(5, 600, 700, "B")])
    (item,) = [c for c in cola_de_revision(auto, corr, ev, PERFILES, CFG, FPS) if c["track"] == 5]
    assert item["clave"] == "5" and item["asignado"] is None
    assert [p["rol"] for p in item["partes"]] == ["A", "B"]
    assert item["partes"][0]["t_hasta"] == 24.0 and not item["revision_pendiente"]


def test_estricta_muestra_el_conflicto_como_sin_asignar():
    ev = {**_ev(), 3: _track(3, [(range(100, 200, 5), ROJO)])}
    auto = [Tramo(1, 0, 400, "A"), Tramo(2, 0, 400, "B"), Tramo(3, 100, 200, "A")]
    cola = cola_de_revision(auto, auto, ev, PERFILES, CFG, FPS, estricta=True)
    c3 = next(c for c in cola if c["track"] == 3)
    assert c3["tipo"] == "sin_asignar" and c3["rol_actual"] is None


def test_con_cortes_la_clave_dice_el_pedazo():
    ev = {**_ev(), 3: _track(3, [(range(500, 700, 5), ROJO), (range(700, 900, 5), AZUL)], 300.0)}
    prop = proponer(ev, replace(CFG, cortar_mixtos=True), 1000, FPS, semillas=(1, 2))
    tramos = tramos_de_propuesta(prop, ev, CFG)
    del3 = [t for t in tramos if t.track == 3]
    assert len(del3) == 2
    claves = {c["clave"] for c in cola_de_revision(tramos, tramos, ev, PERFILES, CFG, FPS)
              if c["track"] == 3}
    assert claves == {f"3-{del3[0].desde}", f"3-{del3[1].desde}"}
    assert all(r is not TrackRole.IGNORE for r in prop.roles.values())
