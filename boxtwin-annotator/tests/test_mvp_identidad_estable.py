"""
IE2 (docs/specs/2026-11-02-identidad-estable): `completar` con correcciones por rango, el
corte de tracks mixtos y el modo estricto, sobre la sesion de prueba de B3.

La sesion: A es el track 1 (primera mitad) y el 3 (segunda mitad, sin color); B es el 2,
todo el video.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from boxtwin.core.identidad_auto import (  # noqa: E402
    EvidenciaTrack, cargar_evidencia, guardar_evidencia,
)
from boxtwin.mvp.orquesta import completar  # noqa: E402
from boxtwin.mvp.sesion import Sesion  # noqa: E402
from tests.test_mvp_tramos import AZUL, FPS, MITAD, ROJO, T, detector, sesion  # noqa: E402,F401


def _completar(s, det, **kw):
    return completar(s, semilla_a=1, semilla_b=2, detector=det, umbral=0.8, device="cpu", **kw)


def _corregir(s, correcciones):
    ses = Sesion.cargar(s)
    ses.identidad["correcciones"] = correcciones
    ses.guardar()


def _tramos(s, track):
    return [t for t in Sesion.cargar(s).identidad["tramos"] if t["track"] == track]


def test_partir_un_tramo_a_mano_asigna_cada_mitad(sesion, detector):
    _completar(sesion, detector)
    corte = 300 / FPS
    _corregir(sesion, [{"track": 3, "desde": MITAD / FPS, "hasta": corte, "rol": "A"},
                       {"track": 3, "desde": corte, "hasta": None, "rol": "ninguno"}])
    fc = _completar(sesion, detector, manual=True)
    assert fc["identidad"]["cobertura_A"] == pytest.approx(0.75, abs=0.02)
    (t3,) = _tramos(sesion, 3)
    assert [p["rol"] for p in t3["partes"]] == ["A", "ninguno"]
    assert t3["asignado"] is None and not t3["revision_pendiente"]
    assert fc["identidad"]["tramos_asignados_manual"] == 1


def test_un_track_ya_asignado_tambien_se_corrige(sesion, detector):
    """Antes del 2-11 un asignado no se podia tocar: el error quedaba en la Fight-Card."""
    antes = _completar(sesion, detector)
    assert antes["identidad"]["cobertura_B"] == pytest.approx(1.0, abs=0.02)
    _corregir(sesion, [{"track": 2, "desde": 0.0, "hasta": MITAD / FPS, "rol": "ninguno"}])
    fc = _completar(sesion, detector, manual=True)
    assert fc["identidad"]["cobertura_B"] == pytest.approx(0.5, abs=0.02)


def test_un_conflicto_se_muestra_y_en_estricta_le_saca_el_rol_a_lo_automatico(sesion, detector):
    _completar(sesion, detector)
    # La persona dice que el 2 era A en la primera mitad: ahora coexisten dos A (el 1, auto).
    _corregir(sesion, [{"track": 2, "desde": 0.0, "hasta": MITAD / FPS, "rol": "A"}])
    fc = _completar(sesion, detector, manual=True)
    (t1,) = _tramos(sesion, 1)
    assert (t1["motivo"], t1["tipo"], t1["rol_actual"]) == ("conflicto", "sospechoso", "A")
    assert fc["identidad"]["revision_pendiente"] >= 1
    assert fc["identidad"]["cobertura_A"] == pytest.approx(0.5, abs=0.02)   # A del 1 y del 2

    fc = _completar(sesion, detector, manual=True, estricta=True)
    (t1,) = _tramos(sesion, 1)
    assert (t1["motivo"], t1["tipo"], t1["rol_actual"]) == ("conflicto", "sin_asignar", None)
    # el 1 perdio el rol; lo que decidio la persona sobre el 2 se queda
    assert fc["identidad"]["cobertura_A"] == pytest.approx(0.5, abs=0.02)
    assert fc["identidad"]["identidad_estricta"] is True


def test_sin_banderas_ni_rangos_la_identidad_es_la_de_antes(sesion, detector):
    fc = _completar(sesion, detector)
    ident = fc["identidad"]
    assert (ident["tramos_total"], ident["tramos_asignados_auto"]) == (3, 2)
    assert ident["corte_de_mixtos"] is False and ident["revision_pendiente"] == 0
    assert ident["cobertura_A"] == pytest.approx(0.5, abs=0.01)
    # el resultado de la web muestra solo esto: los sin asignar que nadie decidio
    assert ident["tramos_sin_asignar"] == sum(
        1 for t in Sesion.cargar(sesion).identidad["tramos"] if t["tipo"] == "sin_asignar" and t["revision_pendiente"])


def test_el_corte_de_mixtos_parte_el_track_que_cambia_de_color(sesion, detector):
    ev = cargar_evidencia(sesion / "evidencia.json")
    todo = list(range(0, T, 5))
    # El 2 es B hasta el 300 y despues el tracker se pasa a A (guante rojo).
    ev[2] = EvidenciaTrack(2, 0, T - 1, 0.61, len(todo), len(todo),
                           [(f, AZUL if f < 300 else ROJO) for f in todo], todo,
                           [(f, 800.0) for f in todo])
    guardar_evidencia(ev, sesion / "evidencia.json")

    _completar(sesion, detector)
    assert len(_tramos(sesion, 2)) == 1
    # sin cortar, el track 2 entero es B y el color de su final lo contradice: se sospecha
    assert _tramos(sesion, 2)[0]["motivo"] == "posible_mezcla"

    fc = _completar(sesion, detector, cortar_mixtos=True)
    partes = _tramos(sesion, 2)
    assert len(partes) == 2
    desdes = sorted(p["desde"] for p in partes)
    assert desdes[0] == 0 and 295 <= desdes[1] <= 300
    # ninguno de los dos pedazos es el track entero: los dos llevan el cuadro en la clave
    assert {p["clave"] for p in partes} == {f"2-{d}" for d in desdes}
    assert fc["identidad"]["corte_de_mixtos"] is True


def test_la_apariencia_suma_a_la_revision_sin_cambiar_la_asignacion(sesion, detector):
    """IE7: con --apariencia, la ropa del otro peleador manda el tramo a la cola; sin ella, no."""
    ev = cargar_evidencia(sesion / "evidencia.json")
    blanca, negra = (-1.0, 20.0, 230.0), (-1.0, 30.0, 25.0)
    azul_p, verde_p = (110.0, 180.0, 120.0), (60.0, 180.0, 120.0)
    for t, e in ev.items():
        frames = [f for f, _ in e.colores]
        e.ropa = [(f, blanca, azul_p, True) if t == 1 else (f, negra, verde_p, True)
                  for f in frames]
    # el 3 (A por coexistencia) tiene guante rojo como A pero remera y pantalon de B
    todo = list(range(MITAD, T, 5))
    ev[3] = EvidenciaTrack(3, MITAD, T - 1, 0.6, len(todo), len(todo),
                           [(f, ROJO) for f in todo], todo, [(f, 200.0) for f in todo],
                           ropa=[(f, negra, verde_p, False) for f in todo])
    guardar_evidencia(ev, sesion / "evidencia.json")

    antes = _completar(sesion, detector)
    assert antes["identidad"]["identidad_apariencia"] is None
    rol_3 = [t["rol_actual"] for t in _tramos(sesion, 3)]
    assert [t["motivo"] for t in _tramos(sesion, 3)] == [None]

    fc = _completar(sesion, detector, apariencia=True)
    assert [t["motivo"] for t in _tramos(sesion, 3)] == ["apariencia_contradice"]
    assert [t["rol_actual"] for t in _tramos(sesion, 3)] == rol_3 == ["A"]
    assert fc["identidad"]["identidad_apariencia"]["pesos"]["remera"] > 0
    # la asignacion es la misma: solo cambia que se pregunta
    assert fc["identidad"]["cobertura_A"] == antes["identidad"]["cobertura_A"]
    assert fc["identidad"]["revision_pendiente"] == antes["identidad"]["revision_pendiente"] + 1
