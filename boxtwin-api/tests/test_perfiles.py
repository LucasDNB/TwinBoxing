"""
Boxeadores con nombre y sus perfiles.

Lo que se fija aca es que el perfil no invente: que ignore una sesion incompleta en vez de
sumarle ceros, que avise cuando compara sesiones medidas con detectores distintos, y que la
mezcla de golpes viaje siempre con su reserva.
"""

from __future__ import annotations

import json

import pytest

from conftest import subir_video


def _fightcard(golpes_a=3, golpes_b=2, duracion=120.0, tipos=True, detector="det-v1"):
    def golpe(i, brazo, tipo):
        return {
            "id": f"x-{i}", "t_inicio": float(i), "t_fin": float(i) + 0.2,
            "cuadro_inicio": i * 30, "cuadro_fin": i * 30 + 6, "brazo": brazo,
            "score": 0.9, "tipo": tipo if tipos else None,
            "confianza_tipo": 0.8 if tipos else None,
            "tipo_clasificador": tipo if tipos else None, "corregido": None,
        }

    return {
        "version": "0.1",
        "video": {"duracion_s": duracion},
        "detector": {"checkpoint": detector},
        "clasificador": {"checkpoint": "clf-v1" if tipos else "",
                         "exactitud_familia_fuente_no_vista": 0.746 if tipos else None},
        "peleadores": {
            "A": {"golpes": [golpe(i, "izq" if i % 2 else "der", "jab") for i in range(golpes_a)],
                  "por_round": [], "total": {}, "guardia": []},
            "B": {"golpes": [golpe(i, "der", "cross") for i in range(golpes_b)],
                  "por_round": [], "total": {}, "guardia": []},
        },
    }


@pytest.fixture
def boxeador(cliente, registrado):
    return cliente.post("/boxeadores", json={"nombre": "Martin"}).json()


def _sesion_con_fc(cliente, entorno, fc, nombre="spar.mp4"):
    sid = subir_video(cliente, nombre=nombre).json()["job_id"]
    d = entorno["cfg"].dir_sesion(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "fightcard.json").write_text(json.dumps(fc))
    return sid


# -- alta ------------------------------------------------------------------


def test_se_crea_y_se_lista(cliente, registrado):
    r = cliente.post("/boxeadores", json={"nombre": "Juan", "guardia": "zurda"})
    assert r.status_code == 201
    assert r.json()["nombre"] == "Juan" and r.json()["guardia"] == "zurda"
    assert [b["nombre"] for b in cliente.get("/boxeadores").json()] == ["Juan"]


def test_el_nombre_es_unico_por_usuario_y_no_global(cliente, registrado):
    cliente.post("/boxeadores", json={"nombre": "Juan"})
    assert cliente.post("/boxeadores", json={"nombre": "Juan"}).status_code == 409
    # Otro usuario puede tener un Juan propio: los nombres son de quien los carga.
    r = cliente.post("/auth/registro", json={"email": "otro@x.com", "clave": "sparring2026"})
    cliente.headers["Authorization"] = f"Bearer {r.json()['token']}"
    assert cliente.post("/boxeadores", json={"nombre": "Juan"}).status_code == 201


def test_el_boxeador_de_otro_usuario_no_se_ve(cliente, registrado, boxeador):
    r = cliente.post("/auth/registro", json={"email": "otro@x.com", "clave": "sparring2026"})
    cliente.headers["Authorization"] = f"Bearer {r.json()['token']}"
    assert cliente.get("/boxeadores").json() == []
    assert cliente.get(f"/boxeadores/{boxeador['id']}/perfil").status_code == 404


# -- asignacion ------------------------------------------------------------


def test_se_asigna_un_lado_de_la_sesion(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    r = cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    assert r.status_code == 200
    assert r.json()["boxeador_a"] == boxeador["id"]
    assert r.json()["boxeador_b"] is None


def test_se_puede_asignar_uno_solo(cliente, registrado, entorno, boxeador):
    # Sparring contra alguien que no esta cargado es el caso normal; obligar a nombrar a los
    # dos convertiria una anotacion util en un tramite.
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    assert cliente.put(f"/sesiones/{sid}/boxeadores",
                       json={"boxeador_b": boxeador["id"]}).status_code == 200


def test_el_mismo_boxeador_no_puede_ser_los_dos_lados(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    r = cliente.put(f"/sesiones/{sid}/boxeadores",
                    json={"boxeador_a": boxeador["id"], "boxeador_b": boxeador["id"]})
    assert r.status_code == 422


def test_asignar_un_boxeador_ajeno_es_404(cliente, registrado, entorno):
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    r = cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": "noexiste"})
    assert r.status_code == 404


def test_reasignar_no_toca_la_fightcard(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    antes = (entorno["cfg"].dir_sesion(sid) / "fightcard.json").read_text()
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_b": boxeador["id"]})
    assert (entorno["cfg"].dir_sesion(sid) / "fightcard.json").read_text() == antes


# -- perfil ----------------------------------------------------------------


def test_el_perfil_junta_las_sesiones_del_boxeador(cliente, registrado, entorno, boxeador):
    for n in ("uno.mp4", "dos.mp4"):
        sid = _sesion_con_fc(cliente, entorno, _fightcard(golpes_a=4), nombre=n)
        cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["totales"]["sesiones"] == 2
    assert p["totales"]["golpes"] == 8
    assert len(p["evolucion"]) == 2


def test_toma_el_lado_que_corresponde_y_no_el_otro(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard(golpes_a=7, golpes_b=2))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_b": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["totales"]["golpes"] == 2, "es el peleador B, no el A"


def test_una_sesion_sin_fightcard_se_ignora_y_no_suma_cero(cliente, registrado, entorno, boxeador):
    # Un cero se lee como "no tiro golpes" y no como "no se midio". Son cosas distintas.
    sid = subir_video(cliente, nombre="rota.mp4").json()["job_id"]
    entorno["cfg"].dir_sesion(sid).mkdir(parents=True, exist_ok=True)
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["totales"]["sesiones"] == 0


def test_el_ritmo_sale_por_minuto_y_no_solo_el_total(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard(golpes_a=6, duracion=120.0))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["sesiones"][0]["golpes_por_minuto"] == 3.0


def test_la_mezcla_viaja_con_su_reserva(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard())
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["totales"]["mezcla"]["jab"] == 3
    assert any("no generaliza" in a for a in p["avisos"])


def test_los_golpes_sin_tipo_se_cuentan_aparte(cliente, registrado, entorno, boxeador):
    sid = _sesion_con_fc(cliente, entorno, _fightcard(golpes_a=5, tipos=False))
    cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert p["totales"]["golpes"] == 5
    assert p["totales"]["sin_clasificar"] == 5
    assert p["totales"]["mezcla"] == {}


def test_avisa_cuando_las_sesiones_se_midieron_con_detectores_distintos(
    cliente, registrado, entorno, boxeador
):
    # Comparar dos sesiones medidas con detectores distintos puede atribuirle al boxeador
    # una mejora que es del software.
    for n, det in (("uno.mp4", "det-v1"), ("dos.mp4", "det-v2")):
        sid = _sesion_con_fc(cliente, entorno, _fightcard(detector=det), nombre=n)
        cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    assert any("detectores" in a for a in p["avisos"])


def test_la_evolucion_va_de_la_mas_vieja_a_la_mas_nueva(cliente, registrado, entorno, boxeador):
    for n in ("uno.mp4", "dos.mp4", "tres.mp4"):
        sid = _sesion_con_fc(cliente, entorno, _fightcard(), nombre=n)
        cliente.put(f"/sesiones/{sid}/boxeadores", json={"boxeador_a": boxeador["id"]})
    p = cliente.get(f"/boxeadores/{boxeador['id']}/perfil").json()
    fechas = [e["fecha"] for e in p["evolucion"]]
    assert fechas == sorted(fechas), "al reves, una mejora se lee como lo contrario"
