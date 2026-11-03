"""Z2: registro con los dos checks, ingreso, roles, plan simulado y baja."""

from sqlalchemy import select

from boxtwinz_api.models import Perfil, Sesion, Usuario, Valoracion
from boxtwinz_api.seguridad import emitir_jwt, leer_jwt
from conftest import registrar


def test_registro_sin_consentimiento_de_uso_falla(cliente):
    r = cliente.post("/api/auth/registro", json={
        "email": "a@b.ar", "clave": "una-clave-larga", "nombre": "A", "guardia": "ortodoxa"})
    assert r.status_code == 422
    assert r.json()["detail"]["codigo"] == "consentimiento_requerido"


def test_registro_guarda_consentimientos_y_pone_cookie(cliente, db):
    yo = registrar(cliente, reent=True)
    assert yo["plan"] == "free" and yo["rol"] == "usuario" and yo["consentimiento_reentrenamiento"]
    assert "boxtwinz_sesion" in cliente.cookies
    u = db.scalar(select(Usuario))
    assert u.consentimiento_uso_version and u.consentimiento_uso_ip and u.consentimiento_reent_version
    assert u.hash.startswith("scrypt$")


def test_cookie_httponly_samesite(cliente):
    r = cliente.post("/api/auth/registro", json={
        "email": "c@b.ar", "clave": "una-clave-larga", "nombre": "C", "guardia": "zurda",
        "consentimiento_uso": True})
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "samesite=lax" in sc and "max-age=2592000" in sc


def test_email_repetido(cliente):
    registrar(cliente)
    r = cliente.post("/api/auth/registro", json={
        "email": "ANA@boxtwinz.ar", "clave": "una-clave-larga", "nombre": "x", "guardia": "ortodoxa",
        "consentimiento_uso": True})
    assert r.status_code == 409


def test_ingreso_y_salida(cliente):
    registrar(cliente)
    cliente.post("/api/auth/salida")
    assert cliente.get("/api/yo").status_code == 401
    assert cliente.post("/api/auth/ingreso", json={"email": "ana@boxtwinz.ar", "clave": "mala-clave"}).status_code == 401
    r = cliente.post("/api/auth/ingreso", json={"email": "ana@boxtwinz.ar", "clave": "una-clave-larga"})
    assert r.status_code == 200
    assert cliente.get("/api/yo").json()["email"] == "ana@boxtwinz.ar"


def test_admin_por_lista(cliente):
    assert registrar(cliente, email="admin@boxtwinz.ar")["rol"] == "admin"


def test_patch_yo_guardia_y_revocar(cliente):
    registrar(cliente, reent=True)
    r = cliente.patch("/api/yo", json={"guardia": "zurda", "consentimiento_reentrenamiento": False})
    assert r.json()["guardia"] == "zurda" and not r.json()["consentimiento_reentrenamiento"]


def test_premium_simulado(cliente):
    registrar(cliente)
    r = cliente.post("/api/yo/premium")
    assert r.json()["plan"] == "premium"
    assert r.json()["aviso"] == "Pago simulado — versión de demostración"


def test_baja_borra_y_anonimiza_valoraciones_usadas(cliente, db):
    registrar(cliente)
    u = db.scalar(select(Usuario))
    p = db.scalar(select(Perfil).where(Perfil.ranura == 1))
    s1 = Sesion(usuario_id=u.id, perfil_id=p.id, modalidad="sombra", video_ruta="/no/existe.mp4", estado="lista")
    s2 = Sesion(usuario_id=u.id, perfil_id=p.id, modalidad="bolsa", video_ruta="/no/existe.mp4", estado="lista")
    db.add_all([s1, s2])
    db.flush()
    db.add_all([Valoracion(sesion_id=s1.id, estrellas=4, comentario="bien", modalidad="sombra", usada_en_modelo=True),
                Valoracion(sesion_id=s2.id, estrellas=2, modalidad="bolsa")])
    db.commit()
    assert cliente.delete("/api/yo").status_code == 204
    db.expire_all()
    assert db.scalar(select(Usuario)) is None
    assert db.scalars(select(Perfil)).all() == [] and db.scalars(select(Sesion)).all() == []
    vs = db.scalars(select(Valoracion)).all()
    assert len(vs) == 1 and vs[0].sesion_id is None and vs[0].comentario is None and vs[0].estrellas == 4


def test_jwt_vence_y_rechaza_firma_ajena():
    t = emitir_jwt("u1", "s", dias=30, ahora=1000)
    assert leer_jwt(t, "s", ahora=1000 + 29 * 86400) == "u1"
    assert leer_jwt(t, "s", ahora=1000 + 31 * 86400) is None
    assert leer_jwt(t, "otro", ahora=1000) is None


def test_suspendido_no_entra(cliente, db):
    registrar(cliente)
    u = db.scalar(select(Usuario))
    u.suspendido = True
    db.commit()
    assert cliente.get("/api/yo").status_code == 403


def test_impresion_solo_en_free(cliente, db):
    from boxtwinz_api.models import EventoPublicidad

    registrar(cliente)
    assert cliente.post("/api/publicidad/impresion", json={"ubicacion": "lista_sesiones"}).json()["registrada"]
    cliente.post("/api/yo/premium")
    assert not cliente.post("/api/publicidad/impresion", json={"ubicacion": "lista_sesiones"}).json()["registrada"]
    assert len(db.scalars(select(EventoPublicidad)).all()) == 1
