"""
B29 de las versiones: ingreso de prueba y cuentas de prueba (versiones-26-10, 04 B29).

I3 de 05 §2. Lo que no se negocia: los botones no alcanzan una cuenta real (RNF-D1), la
bandera viene apagada (RNF-D2), el cupo bloquea siempre (RNF-D3) y reiniciar deja las cuentas
reales identicas fila por fila y carpeta por carpeta (RF-D3).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import select

from test_versiones_b19 import CUERPOS, OTROS, _rutas_con_id
from versiones_util import H, crear_cuenta, crear_miembro, lista_con_fightcard, poner


@pytest.fixture
def demo_encendida(monkeypatch):
    monkeypatch.setenv("BOXTWIN_DEMO", "1")


def sesion_real(cliente, entorno, token, contexto="sparring") -> str:
    """Una sesion real con su carpeta como la deja el worker: video, cache y JSON con rutas."""
    sid = lista_con_fightcard(cliente, entorno, token, contexto=contexto)
    poner(entorno, sid, duracion_s=180.0, fps=30.0)
    d = entorno["cfg"].dir_sesion(sid)
    (d / "cache").mkdir(exist_ok=True)
    (d / "cache" / "spar.pose.npz").write_bytes(b"pose" * 100)
    (d / "cache" / "spar.meta.json").write_text(json.dumps({"path": str(d / "videos/spar.mp4")}))
    (d / "procesado.mp4").write_bytes(b"procesado")
    (d / "sesion.json").write_text(json.dumps({
        "estado": "listo", "candidatos": [], "video": {"ruta": str(d / "videos/spar.mp4"),
                                                      "duracion_s": 180.0}}))
    return sid


@pytest.fixture
def real(cliente, entorno):
    """La cuenta real que la version de prueba no tiene que tocar nunca."""
    cid = crear_cuenta(entorno, "entrenador", "lucas@usal.edu.ar")
    uid, t = crear_miembro(entorno, "lucas@usal.edu.ar", cid)
    s = {ctx: sesion_real(cliente, entorno, t, ctx) for ctx in ("bolsa", "sombra", "sparring")}
    b1 = cliente.post("/boxeadores", json={"nombre": "Tomás Ríos"}, headers=H(t)).json()["id"]
    b2 = cliente.post("/boxeadores", json={"nombre": "Nico Ferreyra"}, headers=H(t)).json()["id"]
    cliente.put(f"/sesiones/{s['sparring']}/boxeadores",
                json={"boxeador_a": b1, "boxeador_b": b2}, headers=H(t))
    return {"cuenta": cid, "usuario": uid, "token": t, "sesiones": s, "boxeadores": [b1, b2]}


def yaml_de(real) -> dict:
    s = real["sesiones"]
    return {
        "individual": {"cuenta": "Individual de prueba",
                       "peleador": {"nombre": "Boxeador de prueba", "guardia": "ortodoxa"},
                       "sesiones": [s["bolsa"], s["sombra"]]},
        "entrenador": {"cuenta": "Entrenador de prueba",
                       "peleadores": [{"nombre": "Atleta 1", "guardia": "ortodoxa"},
                                      {"nombre": "Atleta 2", "guardia": "zurda"}],
                       "sesiones": [{"id": s["sparring"], "a": "Atleta 1", "b": "Atleta 2"}]},
        "gimnasio": {"cuenta": "Gimnasio de prueba",
                     "miembros": {"gimnasio_titular": "Titular de prueba",
                                  "gimnasio_entrenador": "Entrenador de prueba"},
                     "peleadores": [{"nombre": "Atleta 3", "entrenador": "gimnasio_entrenador"},
                                    {"nombre": "Atleta 4"}],
                     "sesiones": [{"id": s["sparring"], "a": "Atleta 3", "b": "Atleta 4",
                                   "subida_por": "gimnasio_entrenador"}]},
    }


def sembrar(entorno, real):
    from boxtwin_api.demo import sembrar as _sembrar

    with entorno["db"].hacer_sesion() as db:
        return _sembrar(db, yaml_de(real))


def entrar(cliente, perfil):
    return cliente.post("/auth/demo", json={"perfil": perfil})


# ---------------------------------------------------------------------------
# Bandera apagada
# ---------------------------------------------------------------------------


def test_I3_apagada_la_ruta_no_existe_y_producto_no_lista_perfiles(cliente, entorno, real):
    sembrar(entorno, real)
    assert entrar(cliente, "individual").status_code == 404
    assert cliente.get("/producto").json()["demo"] == {"activo": False}


def test_los_usuarios_de_prueba_no_entran_por_login(cliente, entorno, real):
    from boxtwin_api.modelos import Usuario

    sembrar(entorno, real)
    with entorno["db"].hacer_sesion() as db:
        u = db.scalars(select(Usuario).where(Usuario.email == "individual@prueba.boxtwin")).one()
        assert u.hash_clave.startswith("scrypt$")
    r = cliente.post("/auth/login", json={"email": "individual@prueba.boxtwin", "clave": "x" * 8})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Bandera encendida
# ---------------------------------------------------------------------------


def test_I3_producto_anuncia_los_cuatro_perfiles_y_las_versiones(demo_encendida, cliente):
    d = cliente.get("/producto").json()
    assert [p["perfil"] for p in d["demo"]["perfiles"]] == [
        "individual", "entrenador", "gimnasio_titular", "gimnasio_entrenador"]
    assert d["demo"]["activo"] is True and d["demo"]["subir"] is True
    assert d["versiones"] == {"individual": {"horas_mes": 4}, "entrenador": {"horas_mes": 15},
                              "gimnasio": {"horas_mes": 50}}
    assert all("email" not in p and "id" not in p for p in d["demo"]["perfiles"])


def test_I3_sin_sembrar_dice_que_falta_y_un_perfil_que_no_existe_da_422(demo_encendida, cliente):
    r = entrar(cliente, "individual")
    assert r.status_code == 409 and "demo sembrar" in r.json()["detail"]
    assert entrar(cliente, "federacion").status_code == 422
    assert cliente.post("/auth/demo", json={"perfil": "individual", "email": "lucas@usal.edu.ar"}
                        ).status_code == 409   # el email se ignora: sigue sin sembrar


@pytest.mark.parametrize("perfil,version,rol", [
    ("individual", "individual", "titular"), ("entrenador", "entrenador", "titular"),
    ("gimnasio_titular", "gimnasio", "titular"), ("gimnasio_entrenador", "gimnasio", "entrenador"),
])
def test_I3_cada_perfil_entra_a_su_cuenta_y_no_ve_la_real(demo_encendida, cliente, entorno, real,
                                                         perfil, version, rol):
    sembrar(entorno, real)
    r = entrar(cliente, perfil)
    assert r.status_code == 200, r.text
    t = r.json()["token"]
    assert r.json()["perfil"] == perfil
    c = cliente.get("/cuenta", headers=H(t)).json()
    assert (c["version"], c["rol"], c["demo"]) == (version, rol, True)
    assert c["usuario"]["email"] is None and c["perfil_demo"]["perfil"] == perfil
    # Con ese token, la cuenta real da 404 en todas las rutas de I1.
    for sid_real in real["sesiones"].values():
        for ruta, params in _rutas_con_id(entorno["app"].app):
            url = ruta.path
            for p in params:
                valor = {"sesion_id": sid_real, "boxeador_id": real["boxeadores"][0]}.get(
                    p, OTROS.get(p, "x"))
                url = url.replace("{" + p + "}", valor)
            for metodo in ruta.methods - {"HEAD"}:
                resp = cliente.request(metodo, url, json=CUERPOS.get((metodo, ruta.path)),
                                       headers=H(t))
                assert resp.status_code == 404, f"{perfil}: {metodo} {ruta.path} dio {resp.status_code}"
    reales = set(real["sesiones"].values())
    assert not reales & {s["id"] for s in cliente.get("/jobs", headers=H(t)).json()}


def test_I3_el_token_de_prueba_dura_8_horas(demo_encendida, cliente, entorno, real):
    import base64
    import time

    sembrar(entorno, real)
    t = entrar(cliente, "entrenador").json()["token"]
    cuerpo = t.split(".")[0]
    exp = json.loads(base64.urlsafe_b64decode(cuerpo + "=" * (-len(cuerpo) % 4)))["exp"]
    assert 7.9 * 3600 < exp - time.time() <= 8 * 3600


def test_I3_si_alguien_saca_la_marca_demo_la_ruta_deja_de_emitir(demo_encendida, cliente, entorno, real):
    from boxtwin_api.modelos import Cuenta

    sembrar(entorno, real)
    with entorno["db"].hacer_sesion() as db:
        for c in db.scalars(select(Cuenta).where(Cuenta.demo.is_(True))).all():
            c.demo = False
        db.commit()
    r = entrar(cliente, "gimnasio_titular")
    assert r.status_code == 500 and "token" not in r.json()


def test_I3_un_perfil_no_disponible_da_409_en_construccion(demo_encendida, cliente, entorno, real,
                                                          monkeypatch):
    import dataclasses

    import boxtwin_api.planes as planes

    sembrar(entorno, real)
    p = planes.PERFILES_DEMO["gimnasio_entrenador"]
    monkeypatch.setitem(planes.PERFILES_DEMO, "gimnasio_entrenador",
                        dataclasses.replace(p, disponible=False))
    r = entrar(cliente, "gimnasio_entrenador")
    assert r.status_code == 409 and r.json()["detail"] == "en construcción"
    perfiles = {x["perfil"]: x for x in cliente.get("/producto").json()["demo"]["perfiles"]}
    assert perfiles["gimnasio_entrenador"]["disponible"] is False


def test_I3_en_la_cuenta_de_prueba_el_cupo_bloquea_aunque_diga_informar(demo_encendida, cliente,
                                                                       entorno, real):
    from versiones_util import subir

    sembrar(entorno, real)
    assert entorno["cfg"].cupo == "informar"
    t = entrar(cliente, "individual").json()["token"]
    h = cliente.get("/cuenta", headers=H(t)).json()["horas"]
    assert h["modo"] == "bloquear" and h["usadas"] == 0.1   # bolsa + sombra, 3 min cada una
    r = subir(cliente, t, duracion_declarada_s=str(4 * 3600), contexto="bolsa")
    assert r.status_code == 409
    assert subir(cliente, t, duracion_declarada_s="120", contexto="bolsa").status_code == 202


def test_I3_con_demo_subir_apagada_la_subida_da_403(demo_encendida, cliente, entorno, real, monkeypatch):
    from versiones_util import subir

    sembrar(entorno, real)
    monkeypatch.setattr(entorno["cfg"], "demo_subir", False)
    t = entrar(cliente, "entrenador").json()["token"]
    r = subir(cliente, t)
    assert r.status_code == 403
    assert r.json()["detail"] == "la subida está apagada en la versión de prueba"
    assert cliente.get("/producto").json()["demo"]["subir"] is False
    # La cuenta real sigue subiendo.
    assert subir(cliente, real["token"]).status_code == 202


# ---------------------------------------------------------------------------
# sembrar y reiniciar
# ---------------------------------------------------------------------------


def test_sembrar_copia_con_ids_nuevos_enlaza_lo_pesado_y_reescribe_las_rutas(cliente, entorno, real):
    from boxtwin_api.modelos import Boxeador, Cuenta, Sesion

    res = sembrar(entorno, real)
    assert res["cuentas"] == 3 and res["usuarios"] == 4 and res["sesiones"] == 4
    assert res["consentimientos"] == 2
    with entorno["db"].hacer_sesion() as db:
        ind = db.scalars(select(Cuenta).where(Cuenta.version == "individual")).one()
        copias = db.scalars(select(Sesion).where(Sesion.cuenta_id == ind.id)).all()
        assert len(copias) == 2 and not {s.id for s in copias} & set(real["sesiones"].values())
        assert all(s.boxeador_a_id == ind.boxeador_propio_id and s.boxeador_b_id is None
                   for s in copias)
        assert db.get(Boxeador, ind.boxeador_propio_id).nombre == "Boxeador de prueba"
        copia = copias[0]
    origen = entorno["cfg"].dir_sesion(real["sesiones"]["bolsa"]
                                       if copia.contexto == "bolsa" else real["sesiones"]["sombra"])
    destino = entorno["cfg"].dir_sesion(copia.id)
    assert (destino / "videos" / "spar.mp4").stat().st_ino == (origen / "videos" / "spar.mp4").stat().st_ino
    assert (destino / "cache" / "spar.pose.npz").stat().st_nlink == 2
    # Lo que render y completar reescriben se copia: un enlace truncaria el de la real.
    assert (destino / "procesado.mp4").stat().st_ino != (origen / "procesado.mp4").stat().st_ino
    sj = (destino / "sesion.json").read_text()
    assert str(destino) in sj and str(origen) not in sj
    assert str(origen) not in (destino / "cache" / "spar.meta.json").read_text()
    # La Fight-Card no se reprocesa: es la misma.
    assert (destino / "fightcard.json").read_text() == (origen / "fightcard.json").read_text()


def test_sembrar_dos_veces_no_hace_nada_la_segunda(cliente, entorno, real):
    sembrar(entorno, real)
    assert sembrar(entorno, real) == {"ya_estaban": True}


def test_sembrar_con_una_sesion_que_no_existe_no_deja_nada(cliente, entorno, real):
    from boxtwin_api.demo import ErrorDemo, sembrar as _sembrar
    from boxtwin_api.modelos import Cuenta

    y = yaml_de(real)
    y["gimnasio"]["sesiones"] = [{"id": "no-existe", "a": "Atleta 3"}]
    with entorno["db"].hacer_sesion() as db, pytest.raises(ErrorDemo):
        _sembrar(db, y)
    with entorno["db"].hacer_sesion() as db:
        assert db.scalars(select(Cuenta).where(Cuenta.demo.is_(True))).all() == []
    assert len(list((entorno["cfg"].datos / "sesiones").iterdir())) == 3


def _foto(entorno, real) -> dict:
    """Las filas de la cuenta real y el hash de cada archivo de sus carpetas."""
    import sqlite3

    ruta = entorno["cfg"].base_de_datos.replace("sqlite:///", "")
    con = sqlite3.connect(ruta)
    filas = {}
    for tabla, col in (("sesiones", "cuenta_id"), ("boxeadores", "cuenta_id"),
                       ("miembros", "cuenta_id"), ("cuentas", "id")):
        filas[tabla] = sorted(con.execute(
            f"SELECT * FROM {tabla} WHERE {col} = ?", (real["cuenta"],)).fetchall())
    filas["usuarios"] = con.execute("SELECT * FROM usuarios WHERE id = ?", (real["usuario"],)).fetchall()
    filas["trabajos"] = sorted(con.execute(
        "SELECT * FROM trabajos WHERE sesion_id IN (%s)" % ",".join("?" * 3),
        list(real["sesiones"].values())).fetchall())
    con.close()
    archivos = {}
    for sid in real["sesiones"].values():
        for f in sorted(entorno["cfg"].dir_sesion(sid).rglob("*")):
            if f.is_file():
                archivos[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {"filas": filas, "archivos": archivos}


def test_RF_D3_reiniciar_deja_las_reales_identicas_fila_por_fila_y_carpeta_por_carpeta(
        demo_encendida, cliente, entorno, real):
    from boxtwin_api.demo import reiniciar
    from versiones_util import subir

    sembrar(entorno, real)
    antes = _foto(entorno, real)
    # Un visitante usa las cuentas de prueba: sube, corrige, carga un atleta.
    t = entrar(cliente, "gimnasio_titular").json()["token"]
    assert subir(cliente, t, duracion_declarada_s="60").status_code == 202
    assert cliente.post("/boxeadores", json={"nombre": "Atleta 9"}, headers=H(t)).status_code == 201
    n_demo = len(cliente.get("/jobs", headers=H(t)).json())
    assert n_demo == 2
    with entorno["db"].hacer_sesion() as db:
        res = reiniciar(db, yaml_de(real))
    assert res["borradas"]["sesiones"] == 5 and res["sesiones"] == 4
    assert _foto(entorno, real) == antes
    t = entrar(cliente, "gimnasio_titular").json()["token"]
    nombres = [b["nombre"] for b in cliente.get("/boxeadores", headers=H(t)).json()]
    assert nombres == ["Atleta 3", "Atleta 4"]
    assert len(cliente.get("/jobs", headers=H(t)).json()) == 1
    # Las carpetas de la siembra anterior ya no estan; solo quedan las reales y las nuevas.
    assert len(list((entorno["cfg"].datos / "sesiones").iterdir())) == 3 + 4


def test_reiniciar_no_borra_nada_si_una_cuenta_real_depende_de_lo_de_prueba(cliente, entorno, real):
    from boxtwin_api.demo import ErrorDemo, reiniciar
    from boxtwin_api.modelos import Boxeador, Cuenta, Sesion

    sembrar(entorno, real)
    with entorno["db"].hacer_sesion() as db:
        gim = db.scalars(select(Cuenta).where(Cuenta.version == "gimnasio")).one()
        atleta = db.scalars(select(Boxeador).where(Boxeador.cuenta_id == gim.id)).first()
        # A mano, una sesion real apunta a un atleta de prueba: reiniciar no puede borrarlo.
        db.get(Sesion, real["sesiones"]["sparring"]).boxeador_b_id = atleta.id
        db.commit()
        n = len(db.scalars(select(Sesion)).all())
    with entorno["db"].hacer_sesion() as db, pytest.raises(ErrorDemo):
        reiniciar(db, yaml_de(real))
    with entorno["db"].hacer_sesion() as db:
        assert len(db.scalars(select(Sesion)).all()) == n


def test_reiniciar_con_un_yaml_roto_no_borra_nada(cliente, entorno, real):
    from boxtwin_api.demo import ErrorDemo, reiniciar
    from boxtwin_api.modelos import Cuenta

    sembrar(entorno, real)
    with entorno["db"].hacer_sesion() as db, pytest.raises(ErrorDemo):
        reiniciar(db, {"individual": {}})
    with entorno["db"].hacer_sesion() as db:
        assert len(db.scalars(select(Cuenta).where(Cuenta.demo.is_(True))).all()) == 3


def test_la_linea_de_comando_siembra_desde_un_yaml(cliente, entorno, real, tmp_path, capsys):
    import yaml

    from boxtwin_api import demo

    archivo = tmp_path / "demo.yaml"
    archivo.write_text(yaml.safe_dump(yaml_de(real), allow_unicode=True))
    assert demo.main(["sembrar", "--yaml", str(archivo)]) == 0
    assert demo.main(["sembrar", "--yaml", str(archivo)]) == 0   # ya estaban: no falla
    assert demo.main(["reiniciar", "--yaml", str(tmp_path / "no-esta.yaml")]) == 2


def test_el_ejemplo_versionado_tiene_la_forma_que_lee_demo(tmp_path):
    from pathlib import Path

    from boxtwin_api.demo import cargar

    ejemplo = Path(__file__).resolve().parents[2] / "despliegue" / "demo.yaml.ejemplo"
    d = cargar(ejemplo)
    assert set(d["gimnasio"]["miembros"]) == {"gimnasio_titular", "gimnasio_entrenador"}
