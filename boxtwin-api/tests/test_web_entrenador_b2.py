"""
B2 de la web entrenador: los candidatos de siembra llegan con color, tramo y altura relativa
(prototipo/specs/05-api-y-datos.md). La API no calcula nada: pasa lo que escribio el
anotador en sesion.json.
"""

from __future__ import annotations

import json

from conftest import subir_video


def _espera_con_b2(cfg, sid):
    d = cfg.dir_sesion(sid)
    (d / "candidatos").mkdir(parents=True, exist_ok=True)
    (d / "candidatos" / "track_7.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "sesion.json").write_text(json.dumps({
        "estado": "espera_siembra",
        "candidatos": [
            {"track": 7, "cuadro": 120, "recorte": "candidatos/track_7.jpg", "alto": 0.9,
             "fraccion_guante": 0.9, "cuadros_con_color": 40, "coexiste_con": 9,
             "cuadros_de_coexistencia": 30, "avisos": [],
             "color_guante": {"hex": "#2F74C8", "nombre": "azules"},
             "t_desde": 4.0, "t_hasta": 178.2, "alto_relativo": 0.97, "sugerido": True},
            {"track": 9, "cuadro": 120, "recorte": None, "alto": 0.93,
             "fraccion_guante": 0.4, "cuadros_con_color": 0, "coexiste_con": 7,
             "cuadros_de_coexistencia": 30, "avisos": ["no se pudo leer el cuadro 120"],
             "color_guante": None, "t_desde": 4.0, "t_hasta": 60.0, "alto_relativo": 1.0,
             "sugerido": True},
        ],
        "identidad": {"pareja_sugerida": [7, 9]},
        "avisos": [], "etapas": [],
    }))


def test_los_campos_de_b2_llegan_a_la_web(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    _espera_con_b2(entorno["cfg"], sid)
    d = cliente.get(f"/jobs/{sid}").json()
    c7, c9 = d["candidatos"]
    assert c7["color_guante"] == {"hex": "#2F74C8", "nombre": "azules"}
    assert (c7["t_desde"], c7["t_hasta"], c7["alto_relativo"], c7["sugerido"]) == (4.0, 178.2, 0.97, True)
    assert c7["recorte_url"] == f"/jobs/{sid}/candidatos/7"


def test_un_track_sin_color_llega_con_color_nulo(cliente, registrado, entorno):
    sid = subir_video(cliente).json()["job_id"]
    _espera_con_b2(entorno["cfg"], sid)
    c9 = cliente.get(f"/jobs/{sid}").json()["candidatos"][1]
    assert c9["color_guante"] is None
    assert c9["recorte_url"] is None, "sin recorte no se inventa una URL"
