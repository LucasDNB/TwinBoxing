"""Z13: disparadores de capacidad con series sinteticas (CZ6) y el panel de admin."""

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from boxtwinz_api.models import NotificacionAdmin, Perfil, Usuario, UsoGpuDiario
from boxtwinz_api.routers.admin import percentil
from boxtwinz_api.services.capacity import cerrar_dia, disparadores, racha, sumar_segundos
from conftest import golpes_cada, registrar, sesion_lista

HOY = date(2026, 11, 20)


def serie(valores):
    """valores[0] es el dia mas viejo y valores[-1] es HOY."""
    return {HOY - timedelta(days=len(valores) - 1 - i): v for i, v in enumerate(valores)}


def test_70_15_cruza_el_dia_15_y_no_antes():
    assert disparadores(serie([0.71] * 14), HOY) == {"capacidad_70_15": False, "capacidad_90_5": False}
    assert disparadores(serie([0.71] * 15), HOY)["capacidad_70_15"]


def test_70_15_un_dia_en_el_umbral_corta_la_racha():
    s = serie([0.71] * 7 + [0.70] + [0.71] * 7)          # 0,70 no es > 0,70
    assert racha(s, HOY, 0.70) == 7 and not disparadores(s, HOY)["capacidad_70_15"]


def test_una_vez_por_episodio():
    assert not disparadores(serie([0.75] * 16), HOY)["capacidad_70_15"]   # dia 16 de la racha: ya avisado


def test_90_5():
    assert disparadores(serie([0.5] * 10 + [0.95] * 5), HOY) == {"capacidad_70_15": False, "capacidad_90_5": True}
    assert not disparadores(serie([0.95] * 4), HOY)["capacidad_90_5"]


def test_dia_sin_dato_corta():
    s = serie([0.95] * 5)
    del s[HOY - timedelta(days=2)]
    assert racha(s, HOY, 0.9) == 2


def test_cerrar_dia_notifica_una_sola_vez(db):
    for i in range(15):
        db.add(UsoGpuDiario(fecha=HOY - timedelta(days=14 - i), segundos_ocupados=0.8 * 86400,
                            utilizacion=None if i == 14 else 0.8))
    db.flush()
    cerrar_dia(db, HOY)
    cerrar_dia(db, HOY)                                   # un reinicio del worker
    db.commit()
    ns = db.scalars(select(NotificacionAdmin)).all()
    assert [n.tipo for n in ns] == ["capacidad_70_15"]
    assert ns[0].payload["texto"] == "Evaluar pase a servidor externo"


def test_sumar_segundos_parte_en_medianoche(db):
    local = datetime(2026, 11, 3, 23, 59, 0).astimezone()
    sumar_segundos(db, local, local + timedelta(minutes=2))
    db.commit()
    filas = {f.fecha: f.segundos_ocupados for f in db.scalars(select(UsoGpuDiario))}
    assert filas == {date(2026, 11, 3): pytest.approx(60), date(2026, 11, 4): pytest.approx(60)}


def test_percentil():
    assert percentil([1, 2, 3, 4], 0.5) == 2.5 and percentil([], 0.5) is None and percentil([7], 0.95) == 7


def _admin(nuevo_cliente):
    a = nuevo_cliente()
    registrar(a, email="admin@boxtwinz.ar")
    return a


def test_panel_solo_admin(cliente):
    registrar(cliente)
    for b in ("resumen", "capacidad", "rendimiento", "calidad", "ingresos", "denuncias", "notificaciones"):
        assert cliente.get(f"/api/admin/{b}").status_code == 403
    assert cliente.get("/api/admin/exportar/capacidad.csv").status_code == 403


def test_bloques_y_csv(nuevo_cliente, db):
    adm = _admin(nuevo_cliente)
    u = nuevo_cliente()
    registrar(u, reent=True)
    s = sesion_lista(db, "ana@boxtwinz.ar", "bolsa", golpes_cada(1.0, 200), duracion=200)
    s.segundos_gpu = 400.0
    db.commit()
    u.post(f"/api/sesiones/{s.id}/valoracion", json={"estrellas": 4})
    r = adm.get("/api/admin/rendimiento").json()
    bolsa = next(x for x in r["tablas"]["gpu_sobre_video_por_modalidad"] if x["modalidad"] == "bolsa")
    assert bolsa["mediana"] == pytest.approx(2.0)
    c = adm.get("/api/admin/calidad").json()
    assert c["sesiones_con_consentimiento_reentrenamiento"] == 1
    assert next(x for x in c["tablas"]["valoracion_por_modalidad"] if x["modalidad"] == "bolsa")["promedio"] == 4
    cap = adm.get("/api/admin/capacidad").json()
    assert cap["proyeccion"]["horas_video_por_hora_gpu"] == pytest.approx(0.5)
    assert cap["proyeccion"]["horas_video_mes_70"] == pytest.approx(0.5 * 720 * 0.7)
    assert len(cap["tablas"]["utilizacion_diaria"]) == 30
    assert adm.get("/api/admin/resumen").json()["tablas"]["usuarios_por_plan"][0]["usuarios"] == 2
    csv = adm.get("/api/admin/exportar/rendimiento.csv")
    assert csv.status_code == 200 and "text/csv" in csv.headers["content-type"]
    assert "# gpu_sobre_video_por_modalidad" in csv.text and "bolsa,1,2.0,2.0" in csv.text
    assert adm.get("/api/admin/exportar/nada.csv").status_code == 404


def test_moderacion_tanda_y_acciones(nuevo_cliente, db):
    adm = _admin(nuevo_cliente)
    a, b = nuevo_cliente(), nuevo_cliente()
    registrar(a, email="a@x.ar")
    registrar(b, email="b@x.ar")
    p1 = a.get("/api/perfiles").json()[0]["id"]
    a.patch(f"/api/perfiles/{p1}", json={"publico": True})
    d1 = b.post("/api/denuncias", json={"perfil_id": p1, "motivo": "video_ajeno"}).json()["id"]
    d2 = b.post("/api/denuncias", json={"perfil_id": p1, "motivo": "no_es_la_persona"}).json()["id"]
    assert adm.get("/api/admin/notificaciones").json()["no_leidas"] == 2
    t = adm.post("/api/admin/tandas").json()
    assert t["denuncias"] == 2
    assert adm.post("/api/admin/tandas").status_code == 409
    assert adm.post(f"/api/admin/denuncias/{d1}/accion", json={"accion": "sacar_ranking"}).json()["perfil"]["fuera_de_ranking"]
    assert adm.post(f"/api/admin/denuncias/{d1}/accion", json={"accion": "desestimar"}).status_code == 409
    adm.post(f"/api/admin/denuncias/{d2}/accion", json={"accion": "suspender", "nota": "video de otro gimnasio"})
    tanda = adm.get("/api/admin/denuncias").json()["tandas"][0]
    assert tanda["cantidad_revisada"] == 2 and tanda["fin"] is not None
    db.expire_all()
    assert db.scalar(select(Usuario).where(Usuario.email == "a@x.ar")).suspendido
    assert a.get("/api/yo").status_code == 403
    n = adm.get("/api/admin/notificaciones").json()["notificaciones"][0]
    assert adm.post(f"/api/admin/notificaciones/{n['id']}/leida").json()["leida"]
