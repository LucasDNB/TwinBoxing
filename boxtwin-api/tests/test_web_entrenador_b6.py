"""
B6 de la web entrenador: categoria, altura y envergadura del peleador
(prototipo/specs/05-api-y-datos.md).
"""

from __future__ import annotations


def test_alta_con_los_datos_nuevos(cliente, registrado):
    r = cliente.post("/boxeadores", json={"nombre": "Ana Paz", "guardia": "zurda", "categoria": "Welter",
                                          "altura_m": 1.72, "envergadura_m": 1.78})
    assert r.status_code == 201, r.text
    b = r.json()
    assert (b["categoria"], b["altura_m"], b["envergadura_m"]) == ("Welter", 1.72, 1.78)
    assert b["sesiones"] == 0 and b["ultima_sesion"] is None


def test_todo_es_opcional(cliente, registrado):
    b = cliente.post("/boxeadores", json={"nombre": "Solo nombre"}).json()
    assert b["categoria"] is None and b["altura_m"] is None and b["envergadura_m"] is None


def test_un_error_de_unidad_se_rechaza(cliente, registrado):
    assert cliente.post("/boxeadores", json={"nombre": "X", "altura_m": 175}).status_code == 422
    assert cliente.post("/boxeadores", json={"nombre": "X", "envergadura_m": 2.6}).status_code == 422
    assert cliente.post("/boxeadores", json={"nombre": "X", "altura_m": 1.1}).status_code == 422


def test_editar_los_datos_nuevos(cliente, registrado):
    b = cliente.post("/boxeadores", json={"nombre": "Ana"}).json()
    r = cliente.patch(f"/boxeadores/{b['id']}", json={"nombre": "Ana", "categoria": "Liviano", "altura_m": 1.65})
    assert r.status_code == 200
    filas = cliente.get("/boxeadores").json()
    assert filas[0]["categoria"] == "Liviano" and filas[0]["altura_m"] == 1.65


def test_la_fightcard_lleva_la_categoria_del_peleador(cliente, registrado, entorno):
    import json

    from conftest import subir_video

    sid = subir_video(cliente).json()["job_id"]
    (entorno["cfg"].dir_sesion(sid) / "fightcard.json").write_text(json.dumps({"peleadores": {}}))
    b = cliente.post("/boxeadores", json={"nombre": "Ana", "categoria": "Welter"}).json()
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": b["id"]})
    assert cliente.get(f"/fightcards/{sid}").json()["sesion"]["boxeadores"]["A"]["categoria"] == "Welter"
