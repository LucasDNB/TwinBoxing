"""
El estado de una sesion y las ventanas de round.

Lo que se fija aca es lo que la API lee mientras el worker escribe, y el denominador de
"golpes por minuto", que si se toma nominal miente sobre el ultimo round.
"""

from __future__ import annotations

import json

import pytest

from boxtwin.mvp.sesion import ESTADOS, Sesion, ventanas_de_round

VIDEO = {"nombre": "spar.mp4", "fps": 30.0, "duracion_s": 600.0, "total_frames": 18000}


# -- ventanas de round ------------------------------------------------------


def test_tres_rounds_de_tres_minutos_con_un_minuto_de_descanso():
    v = ventanas_de_round(duracion_s=660.0, round_s=180.0, descanso_s=60.0)
    assert [r["round"] for r in v] == [1, 2, 3]
    assert (v[0]["inicio_s"], v[0]["fin_s"]) == (0.0, 180.0)
    assert (v[1]["inicio_s"], v[1]["fin_s"]) == (240.0, 420.0), "arranca tras el descanso"


def test_el_ultimo_round_se_recorta_a_lo_que_dura_el_video():
    # Si la sesion corto a mitad de round, darle el largo nominal le pone al denominador de
    # golpes por minuto un tiempo que no existio y el round parece mas flojo de lo que fue.
    v = ventanas_de_round(duracion_s=300.0, round_s=180.0, descanso_s=60.0)
    assert len(v) == 2
    assert v[1]["inicio_s"] == 240.0
    assert v[1]["fin_s"] == 300.0, "un minuto de round, no tres"


def test_un_video_que_termina_en_el_descanso_no_agrega_un_round_vacio():
    v = ventanas_de_round(duracion_s=200.0, round_s=180.0, descanso_s=60.0)
    assert len(v) == 1


def test_sin_rounds_declarados_no_se_inventan():
    assert ventanas_de_round(600.0, None, 60.0) == []
    assert ventanas_de_round(600.0, 0.0, 60.0) == []


def test_sin_descanso_los_rounds_son_contiguos():
    v = ventanas_de_round(300.0, 100.0, 0.0)
    assert [(r["inicio_s"], r["fin_s"]) for r in v] == [
        (0.0, 100.0), (100.0, 200.0), (200.0, 300.0)
    ]


# -- estado -----------------------------------------------------------------


def test_una_sesion_nueva_arranca_procesando(tmp_path):
    s = Sesion.nueva(tmp_path, VIDEO, round_s=180.0, descanso_s=60.0)
    assert s.estado == "procesando"
    assert len(s.ventanas) == 3


def test_la_sesion_sobrevive_la_ida_y_vuelta_a_disco(tmp_path):
    s = Sesion.nueva(tmp_path, VIDEO, round_s=180.0, descanso_s=60.0)
    s.estado = "espera_siembra"
    s.candidatos = [{"track": 3}, {"track": 7}]
    s.anotar_etapa("preproceso", 12.5, cuadros=18000)
    s.guardar()

    v = Sesion.cargar(tmp_path)
    assert v.estado == "espera_siembra"
    assert [c["track"] for c in v.candidatos] == [3, 7]
    assert v.etapas[0]["etapa"] == "preproceso"


def test_un_estado_que_no_existe_no_se_escribe(tmp_path):
    s = Sesion.nueva(tmp_path, VIDEO, None, 60.0)
    s.estado = "casi_listo"
    with pytest.raises(ValueError, match="estado desconocido"):
        s.guardar()


def test_el_archivo_nunca_queda_a_medias(tmp_path):
    # La API lee este archivo mientras el worker lo escribe. Un json truncado es un estado
    # que no existe, y el que lo lee no tiene forma de distinguirlo de uno corrupto.
    s = Sesion.nueva(tmp_path, VIDEO, None, 60.0)
    s.guardar()
    assert json.loads(Sesion.ruta_de(tmp_path).read_text())["estado"] == "procesando"
    assert not list(tmp_path.glob("*.tmp"))


def test_cargar_una_sesion_que_no_existe_dice_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="procesar"):
        Sesion.cargar(tmp_path)


# -- tiempo -----------------------------------------------------------------


def test_el_tiempo_de_maquina_no_cuenta_la_espera_humana(tmp_path):
    # RNF1 es sobre el procesamiento y no sobre el reloj de pared: entre las dos etapas la
    # sesion puede quedar horas esperando que alguien mire dos recortes.
    s = Sesion.nueva(tmp_path, VIDEO, None, 60.0)
    s.anotar_etapa("preproceso", 300.0)
    s.anotar_etapa("detector", 120.0)
    assert s.segundos_de_maquina == 420.0
    assert s.factor_tiempo_real == 0.7


def test_sin_duracion_no_hay_factor(tmp_path):
    s = Sesion.nueva(tmp_path, {"nombre": "x.mp4"}, None, 60.0)
    s.anotar_etapa("preproceso", 10.0)
    assert s.factor_tiempo_real is None


def test_los_estados_son_los_del_flujo():
    assert ESTADOS.index("espera_siembra") < ESTADOS.index("listo")
