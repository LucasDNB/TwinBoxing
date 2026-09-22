"""
La Fight-Card.

La mitad de estos tests no son sobre lo que el documento dice sino sobre lo que NO dice, y
esa es la parte que importa: un sistema monocular no establece contacto fisico, asi que
conectados, puntuacion y veredicto no pueden aparecer ni por accidente. El resto fija que
el conteo nunca viaje sin su recall al lado.
"""

from __future__ import annotations

import pytest

from boxtwin.mvp.fightcard import (
    RECALL_DETECTOR,
    VERSION_FIGHTCARD,
    construir,
    id_de_golpe,
)
from boxtwin.mvp.sesion import Sesion

VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": 400.0, "total_frames": 12000}


class _Golpe:
    def __init__(self, peleador, brazo, t, dur=0.25, score=0.9, fps=30.0):
        self.peleador, self.brazo = peleador, brazo
        self.t_inicio, self.t_fin = t, t + dur
        self.inicio, self.fin = int(t * fps), int((t + dur) * fps)
        self.score = score


@pytest.fixture
def sesion(tmp_path):
    return Sesion.nueva(tmp_path, VIDEO, round_s=180.0, descanso_s=20.0)


# -- lo que no dice ---------------------------------------------------------


def test_no_dice_conectados_ni_puntuacion_ni_veredicto(sesion):
    fc = construir(sesion, [_Golpe("A", "left", 10.0)])
    texto = repr(fc).lower()
    for prohibido in ("conectad", "puntuacion", "veredicto", "ganador", "knockdown"):
        # Aparecen SOLO adentro de la lista que declara que no estan.
        fuera = repr({k: v for k, v in fc.items() if k != "no_incluye"}).lower()
        assert prohibido not in fuera, f"la Fight-Card no puede hablar de {prohibido}"
    assert "monocular" in texto, "y tiene que decir por que"


def test_el_conteo_viaja_con_su_recall(sesion):
    fc = construir(sesion, [_Golpe("A", "left", 10.0)])
    assert fc["detector"]["recall_medido"] == RECALL_DETECTOR
    assert "por debajo" in fc["lectura"]["advertencia"]


def test_el_tipo_sale_rotulado_como_estimacion(sesion):
    fc = construir(sesion, [_Golpe("A", "left", 10.0)])
    assert fc["clasificador"]["estimado"] is True


# -- conteo -----------------------------------------------------------------


def test_cuenta_por_peleador_y_por_brazo(sesion):
    golpes = [
        _Golpe("A", "left", 10.0), _Golpe("A", "left", 12.0), _Golpe("A", "right", 14.0),
        _Golpe("B", "right", 16.0),
    ]
    fc = construir(sesion, golpes)
    assert fc["peleadores"]["A"]["total"] == {"total": 3, "izq": 2, "der": 1}
    assert fc["peleadores"]["B"]["total"] == {"total": 1, "izq": 0, "der": 1}


def test_cada_golpe_enlaza_a_su_instante(sesion):
    # RF7. Sin esto la Fight-Card es un numero que hay que creer; con esto, el entrenador
    # verifica cada evento.
    fc = construir(sesion, [_Golpe("A", "left", 10.0)])
    g = fc["peleadores"]["A"]["golpes"][0]
    assert g["t_inicio"] == 10.0
    assert g["cuadro_inicio"] == 300


def test_el_conteo_por_round_usa_la_ventana_real(sesion):
    # El segundo round arranca a los 200 s (180 + 20 de descanso) y el video dura 400, asi
    # que mide 180. Un golpe en el descanso no cae en ningun round.
    golpes = [_Golpe("A", "left", 5.0), _Golpe("A", "left", 190.0),
              _Golpe("A", "left", 250.0)]
    fc = construir(sesion, golpes)
    rondas = fc["peleadores"]["A"]["por_round"]
    assert [r["total"] for r in rondas] == [1, 1]
    assert rondas[0]["por_minuto"] == pytest.approx(60 / 180, abs=0.01)


def test_sin_rounds_declarados_no_hay_agregado_por_round(tmp_path):
    ses = Sesion.nueva(tmp_path, VIDEO, round_s=None, descanso_s=0.0)
    fc = construir(ses, [_Golpe("A", "left", 10.0)])
    assert fc["peleadores"]["A"]["por_round"] == []
    assert fc["video"]["rounds"] == []


def test_la_caida_entre_rounds_se_calcula_una_sola_vez(sesion):
    # F4 pide la caida entre rounds. Va resuelta en el documento y no en la pantalla, si no
    # cada vista inventa la suya.
    golpes = [_Golpe("A", "left", 10.0 + i) for i in range(10)]
    golpes += [_Golpe("A", "left", 210.0 + i) for i in range(4)]
    fc = construir(sesion, golpes)
    caida = fc["lectura"]["caida_entre_rounds"]["A"]
    assert caida["variacion"] < 0, "tiro menos en el segundo round"


# -- identidad --------------------------------------------------------------


def test_la_cobertura_de_identidad_viaja_en_el_documento(sesion):
    # RF5: cuando el sistema se abstiene, el usuario tiene que ver cuanto tiempo quedo sin
    # mirar. Un conteo sobre el 40% del video no es el mismo dato que uno sobre el 95%.
    fc = construir(sesion, [], identidad={"cobertura_A": 0.91, "cobertura_B": 0.88,
                                          "sin_asignar": 0.07})
    assert fc["identidad"]["sin_asignar"] == 0.07


# -- tipos ------------------------------------------------------------------


def test_el_tipo_se_agrega_despues_sin_reprocesar(sesion):
    # RF12: reclasificar una sesion vieja con un checkpoint nuevo no puede pedir repetir
    # pose ni deteccion. Por eso el tipo entra por afuera y el id del golpe es estable.
    g = _Golpe("A", "left", 10.0)
    gid = id_de_golpe("A", "left", g.inicio)
    fc = construir(sesion, [g], tipos={gid: {"tipo": "cross", "confianza": 0.71,
                                             "crudo": "cross"}})
    ev = fc["peleadores"]["A"]["golpes"][0]
    assert (ev["tipo"], ev["confianza_tipo"]) == ("cross", 0.71)


def test_sin_clasificador_el_tipo_es_nulo_y_no_una_adivinanza(sesion):
    fc = construir(sesion, [_Golpe("A", "left", 10.0)])
    assert fc["peleadores"]["A"]["golpes"][0]["tipo"] is None


def test_el_id_de_un_golpe_no_depende_del_orden_de_la_lista():
    assert id_de_golpe("A", "left", 300) == id_de_golpe("A", "left", 300)
    assert id_de_golpe("A", "left", 300) != id_de_golpe("A", "right", 300)


def test_la_nomenclatura_de_pantalla_es_rioplatense(sesion):
    fc = construir(sesion, [])
    assert fc["nomenclatura"]["cross"] == "gancho"


def test_la_version_del_contrato_esta_declarada(sesion):
    assert construir(sesion, [])["version"] == VERSION_FIGHTCARD
