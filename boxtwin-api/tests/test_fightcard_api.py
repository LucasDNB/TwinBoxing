"""
Servir la Fight-Card: lectura, export, correccion y streaming con rangos.

El rango HTTP tiene su propio bloque porque es lo que hace posible la funcion que convierte
un conteo dudoso en una herramienta de revision: tocar un evento y que el video salte ahi.
Sin 206, el navegador baja el archivo entero antes de poder posicionarse.
"""

from __future__ import annotations

import json

import pytest
from conftest import subir_video

FIGHTCARD = {
    "version": "0.1",
    "video": {"nombre": "spar.mp4", "duracion_s": 400.0, "fps": 30.0,
              "rounds": [{"round": 1, "inicio_s": 0.0, "fin_s": 180.0}]},
    "identidad": {"cobertura_A": 0.95, "cobertura_B": 0.93, "sin_asignar": 0.04},
    "detector": {"checkpoint": "ens.pt", "umbral": 0.8,
                 "precision_medida": 0.885, "recall_medido": 0.484},
    "clasificador": {"checkpoint": "", "exactitud_familia_fuente_no_vista": None,
                     "estimado": True},
    "peleadores": {
        "A": {
            "golpes": [
                {"id": "A-izq-300", "t_inicio": 10.0, "t_fin": 10.3, "cuadro_inicio": 300,
                 "cuadro_fin": 309, "brazo": "izq", "score": 0.91, "tipo": "jab",
                 "confianza_tipo": 0.62, "tipo_clasificador": "jab", "corregido": None},
            ],
            "por_round": [{"round": 1, "inicio_s": 0.0, "fin_s": 180.0, "total": 1,
                           "izq": 1, "der": 0, "por_minuto": 0.33}],
            "total": {"total": 1, "izq": 1, "der": 0},
            "guardia": [],
        },
        "B": {"golpes": [], "por_round": [], "total": {"total": 0, "izq": 0, "der": 0},
              "guardia": []},
    },
    "nomenclatura": {"jab": "jab", "cross": "gancho", "hook": "hook", "uppercut": "uppercut"},
    "no_incluye": ["golpes conectados: un sistema monocular no establece contacto fisico"],
    "avisos": [],
    "lectura": {"volumen_relativo": {"A": 1, "B": 0, "cociente_A_sobre_B": None},
                "caida_entre_rounds": {},
                "advertencia": "el conteo es de golpes DETECTADOS y esta por debajo del real"},
}


@pytest.fixture
def lista(cliente, registrado, entorno):
    """Una sesion con Fight-Card en disco."""
    sid = subir_video(cliente).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "fightcard.json").write_text(json.dumps(FIGHTCARD))
    (d / "sesion.json").write_text(json.dumps({"estado": "listo", "candidatos": []}))
    return sid


# -- lectura ----------------------------------------------------------------


def test_la_fightcard_se_sirve_entera(cliente, lista):
    fc = cliente.get(f"/fightcards/{lista}").json()
    assert fc["detector"]["recall_medido"] == 0.484
    assert fc["peleadores"]["A"]["total"]["total"] == 1


def test_antes_de_estar_lista_se_dice_que_no_esta(cliente, registrado):
    sid = subir_video(cliente).json()["job_id"]
    r = cliente.get(f"/fightcards/{sid}")
    assert r.status_code == 409


# -- export -----------------------------------------------------------------


def test_el_csv_tiene_una_fila_por_golpe(cliente, lista):
    r = cliente.get(f"/fightcards/{lista}/export?formato=csv")
    assert r.status_code == 200
    filas = [f for f in r.text.strip().splitlines() if f]
    assert len(filas) == 2, "cabecera y un golpe"
    assert filas[1].startswith("A,izq,10.0")
    assert "attachment" in r.headers["content-disposition"]


def test_el_csv_marca_a_que_round_cae_cada_golpe(cliente, lista):
    r = cliente.get(f"/fightcards/{lista}/export?formato=csv")
    assert r.text.strip().splitlines()[1].endswith(",1")


def test_la_pagina_imprimible_lleva_el_recall_arriba(cliente, lista):
    # Un conteo impreso circula solo, fuera de la pantalla que lo explica. Por eso el
    # margen tiene que ir en el documento y no al lado.
    r = cliente.get(f"/fightcards/{lista}/export?formato=pdf")
    assert r.status_code == 200
    assert "0.484" in r.text
    assert "no golpes lanzados" in r.text
    assert "monocular" in r.text


def test_un_formato_que_no_existe_se_rechaza(cliente, lista):
    assert cliente.get(f"/fightcards/{lista}/export?formato=docx").status_code == 422


# -- correccion -------------------------------------------------------------


def test_corregir_el_tipo_guarda_la_etiqueta_sin_pisar_la_original(cliente, lista, entorno):
    pytest.importorskip("boxtwin.mvp.orquesta")
    r = cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "hook"})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["tipo"] == "hook"
    assert g["corregido"]["tipo_original"] == "jab"

    # Queda en el jsonl, que es el dataset, y en la base, que es lo que se consulta.
    linea = json.loads(
        (entorno["cfg"].dir_sesion(lista) / "correcciones.jsonl").read_text().splitlines()[0]
    )
    assert linea["tipo_original"] == "jab" and linea["por"] == "lucas@usal.edu.ar"

    from boxtwin_api.modelos import Correccion
    from sqlalchemy import select

    with entorno["db"].hacer_sesion() as db:
        c = db.scalars(select(Correccion)).all()
    assert len(c) == 1 and c[0].tipo == "hook"


def test_no_se_corrige_a_un_tipo_que_no_existe(cliente, lista):
    r = cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "volea"})
    assert r.status_code == 422


def test_no_se_corrige_un_golpe_que_no_esta(cliente, lista):
    pytest.importorskip("boxtwin.mvp.orquesta")
    r = cliente.patch(f"/fightcards/{lista}/golpes/A-izq-99999", json={"tipo": "jab"})
    assert r.status_code == 422


# -- video ------------------------------------------------------------------


def test_el_video_se_sirve_por_rangos(cliente, lista):
    r = cliente.get(f"/videos/{lista}/stream", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 10
    assert r.headers["content-range"].startswith("bytes 0-9/")


def test_sin_rango_se_sirve_entero(cliente, lista):
    r = cliente.get(f"/videos/{lista}/stream")
    assert r.status_code == 200
    assert r.content.startswith(b"no es un video")


def test_un_rango_que_empieza_pasado_el_final_es_416(cliente, lista):
    r = cliente.get(f"/videos/{lista}/stream", headers={"Range": "bytes=999999-"})
    assert r.status_code == 416


def test_un_rango_de_los_ultimos_bytes(cliente, lista):
    r = cliente.get(f"/videos/{lista}/stream", headers={"Range": "bytes=-5"})
    assert r.status_code == 206
    assert len(r.content) == 5
