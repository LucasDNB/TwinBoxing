"""
Ayudas de las pruebas de las versiones (prototipo/specs/versiones-26-10, 05 §2).

Las cuentas se crean directo en la base: el registro por version (B21) y las invitaciones
(B23) quedaron postergados por la version de prueba, asi que no hay ruta que las cree. Lo
que se prueba es lo que pasa DESPUES, que es donde estan las fugas posibles.
"""

from __future__ import annotations

import json

FC_MINIMA = {
    "version": "0.3",
    "video": {"rounds": [{"round": 1, "inicio_s": 0, "fin_s": 180}], "duracion_s": 180},
    "detector": {"recall_medido": 0.484},
    "peleadores": {"A": {"total": {"total": 3}, "golpes": []},
                   "B": {"total": {"total": 1}, "golpes": []}},
}


def rutas(app) -> list:
    """
    Todas las APIRoute de la app, tambien las de routers incluidos. FastAPI 0.141 ya no las
    aplana en app.routes (quedan adentro de un _IncludedRouter), y una prueba que enumera
    app.routes a secas se salteaba las de cuentas.py sin avisar. Ninguna lleva prefijo.
    """
    from fastapi.routing import APIRoute

    out = []
    for r in app.routes:
        if isinstance(r, APIRoute):
            out.append(r)
        elif hasattr(r, "original_router"):
            out.extend(rutas(r.original_router))
    return out


def H(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def crear_cuenta(entorno, version="entrenador", nombre="Cuenta", demo=False) -> str:
    from boxtwin_api.modelos import Cuenta

    with entorno["db"].hacer_sesion() as db:
        c = Cuenta(nombre=nombre, version=version, demo=demo)
        db.add(c)
        db.commit()
        return c.id


def crear_miembro(entorno, email, cuenta_id, rol="titular", nombre=None) -> tuple[str, str]:
    """Un usuario miembro de esa cuenta. Devuelve (usuario_id, token)."""
    from boxtwin_api.modelos import Miembro, Usuario
    from boxtwin_api.seguridad import emitir_token, hashear

    with entorno["db"].hacer_sesion() as db:
        u = Usuario(email=email, hash_clave=hashear("sparring2026"), nombre=nombre)
        db.add(u)
        db.flush()
        db.add(Miembro(cuenta_id=cuenta_id, usuario_id=u.id, rol=rol))
        db.commit()
        return u.id, emitir_token(u.id, entorno["cfg"].secreto)


def peleador_propio(entorno, cuenta_id, autor_id, nombre="Lucas") -> str:
    from boxtwin_api.modelos import Boxeador, Cuenta

    with entorno["db"].hacer_sesion() as db:
        b = Boxeador(usuario_id=autor_id, cuenta_id=cuenta_id, nombre=nombre)
        db.add(b)
        db.flush()
        db.get(Cuenta, cuenta_id).boxeador_propio_id = b.id
        db.commit()
        return b.id


def subir(cliente, token, nombre="spar.mp4", **datos):
    return cliente.post(
        "/videos",
        files={"archivo": (nombre, b"no es un video de verdad", "video/mp4")},
        data={"nombre": "sparring del martes", "consentimiento": "true", **datos},
        headers=H(token),
    )


def poner(entorno, sid, **campos) -> None:
    from boxtwin_api.modelos import Sesion

    with entorno["db"].hacer_sesion() as db:
        s = db.get(Sesion, sid)
        for k, v in campos.items():
            setattr(s, k, v)
        db.commit()


def lista_con_fightcard(cliente, entorno, token, fc=None, **datos) -> str:
    sid = subir(cliente, token, **datos).json()["job_id"]
    poner(entorno, sid, estado="listo")
    d = entorno["cfg"].dir_sesion(sid)
    (d / "fightcard.json").write_text(json.dumps(fc or FC_MINIMA))
    (d / "sesion.json").write_text(json.dumps({"estado": "listo", "candidatos": []}))
    return sid
