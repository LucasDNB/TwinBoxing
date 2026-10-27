"""
B4 de la web entrenador: "no fue un golpe" (prototipo/specs/05-api-y-datos.md), y el bloque
`sesion` que GET /fightcards/{id} agrega al responder (§4).
"""

from __future__ import annotations

import csv
import io
import json

from test_fightcard_api import lista  # noqa: F401  (fixture)


def test_no_golpe_se_acepta_y_no_cambia_el_total(cliente, lista):  # noqa: F811
    r = cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "no_golpe"})
    assert r.status_code == 200, r.text
    assert r.json()["descartado"] is True
    fc = cliente.get(f"/fightcards/{lista}").json()
    a = fc["peleadores"]["A"]
    assert a["descartados"] == 1
    assert a["total"]["total"] == 1, "detectados no cambia"
    assert len(a["golpes"]) == 1


def test_no_golpe_y_despues_jab(cliente, lista):  # noqa: F811
    cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "no_golpe"})
    cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "jab"})
    g = cliente.get(f"/fightcards/{lista}").json()["peleadores"]["A"]["golpes"][0]
    assert g["descartado"] is False and g["tipo"] == "jab"


def test_la_correccion_no_golpe_queda_como_etiqueta(cliente, lista, entorno):  # noqa: F811
    cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "no_golpe"})
    from boxtwin_api.modelos import Correccion

    with entorno["db"].hacer_sesion() as db:
        c = db.query(Correccion).one()
    assert c.tipo == "no_golpe" and c.tipo_original == "jab"


def test_el_csv_suma_la_columna_descartado(cliente, lista):  # noqa: F811
    cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "no_golpe"})
    filas = list(csv.DictReader(io.StringIO(cliente.get(f"/fightcards/{lista}/export?formato=csv").text)))
    assert filas[0]["descartado"] == "si"
    assert filas[0]["corregido_a"] == "no_golpe"


def test_un_tipo_inventado_sigue_rechazado(cliente, lista):  # noqa: F811
    assert cliente.patch(f"/fightcards/{lista}/golpes/A-izq-300", json={"tipo": "volea"}).status_code == 422


def test_la_fightcard_trae_el_bloque_sesion_sin_guardarlo(cliente, lista, entorno):  # noqa: F811
    b = cliente.post("/boxeadores", json={"nombre": "Ana Paz", "guardia": "zurda"}).json()
    cliente.put(f"/sesiones/{lista}/boxeadores", json={"boxeador_a": b["id"]})
    fc = cliente.get(f"/fightcards/{lista}").json()
    assert fc["sesion"]["contexto"] == "sparring"
    assert fc["sesion"]["boxeadores"]["A"]["nombre"] == "Ana Paz"
    assert fc["sesion"]["boxeadores"]["A"]["guardia"] == "zurda"
    assert fc["sesion"]["boxeadores"]["B"] is None
    en_disco = json.loads((entorno["cfg"].dir_sesion(lista) / "fightcard.json").read_text())
    assert "sesion" not in en_disco, "sale de la base: no se escribe en el archivo"
