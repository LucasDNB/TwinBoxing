"""
BoxTwin API - Claves y tokens, con la biblioteca estandar.

POR QUE SIN DEPENDENCIAS
  passlib y python-jose son dos dependencias mas para hacer dos cosas que la biblioteca
  estandar hace bien: scrypt para la clave y hmac para firmar el token. Menos superficie
  que instalar en la imagen del worker y menos que justificar.

  scrypt y no un sha con sal: un hash rapido es exactamente lo que no se quiere para una
  clave. Los parametros son los que recomienda la documentacion de CPython, y viajan
  adentro del hash para poder subirlos despues sin invalidar las claves existentes.

QUE HACE
  Hashea y verifica claves, y emite y valida tokens firmados con vencimiento.

USO
  from boxtwin_api.seguridad import hashear, verificar, emitir_token, leer_token
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

__all__ = ["emitir_token", "hashear", "leer_token", "verificar"]

_N, _R, _P = 2**14, 8, 1


def hashear(clave: str) -> str:
    sal = secrets.token_bytes(16)
    h = hashlib.scrypt(clave.encode(), salt=sal, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${sal.hex()}${h.hex()}"


def verificar(clave: str, guardado: str) -> bool:
    try:
        algo, n, r, p, sal, h = guardado.split("$")
        if algo != "scrypt":
            return False
        calculado = hashlib.scrypt(
            clave.encode(), salt=bytes.fromhex(sal),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(h)),
        )
    except (ValueError, TypeError):
        return False
    # Comparacion en tiempo constante: comparar con == filtra por cuanto tarda en fallar.
    return hmac.compare_digest(calculado, bytes.fromhex(h))


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _de_b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def emitir_token(usuario_id: str, secreto: str, horas: int = 72) -> str:
    cuerpo = _b64(
        json.dumps({"u": usuario_id, "exp": int(time.time()) + horas * 3600}).encode()
    )
    firma = _b64(hmac.new(secreto.encode(), cuerpo.encode(), hashlib.sha256).digest())
    return f"{cuerpo}.{firma}"


def leer_token(token: str, secreto: str) -> str | None:
    """El id del usuario, o None si el token esta vencido, roto o no lo firmamos nosotros."""
    try:
        cuerpo, firma = token.split(".")
    except ValueError:
        return None
    esperada = _b64(hmac.new(secreto.encode(), cuerpo.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(firma, esperada):
        return None
    try:
        d = json.loads(_de_b64(cuerpo))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(d.get("exp", 0)) < time.time():
        return None
    return str(d.get("u")) or None
