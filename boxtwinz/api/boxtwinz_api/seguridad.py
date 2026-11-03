"""
BoxTwinZ API - Hash de claves y JWT.

POR QUE EXISTE
  La spec pide argon2id y JWT. twinboxing_env no trae argon2-cffi ni PyJWT y la instalacion
  de dependencias nuevas quedo sin aprobar (desvio D1), asi que se hace con la biblioteca
  estandar, como boxtwin-api: scrypt para la clave y un JWT HS256 firmado con hmac.

  El hash lleva el algoritmo en el prefijo. Pasar a argon2id es agregar una rama en
  `verificar` y rehashear al ingresar, sin migracion.

QUE HACE
  hashear/verificar una clave; emitir/leer un JWT HS256 estandar (header.payload.firma en
  base64url), con exp en segundos epoch.

USO
  from boxtwinz_api.seguridad import hashear, verificar, emitir_jwt, leer_jwt
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

__all__ = ["hashear", "verificar", "emitir_jwt", "leer_jwt"]

_N, _R, _P = 2**14, 8, 1


def hashear(clave: str) -> str:
    sal = os.urandom(16)
    h = hashlib.scrypt(clave.encode(), salt=sal, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${sal.hex()}${h.hex()}"


def verificar(clave: str, guardado: str) -> bool:
    try:
        algo, n, r, p, sal, h = guardado.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(clave.encode(), salt=bytes.fromhex(sal), n=int(n), r=int(r), p=int(p), dklen=32)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(calc, bytes.fromhex(h))


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _de_b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


_HEADER = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())


def emitir_jwt(sub: str, secreto: str, dias: int = 30, ahora: float | None = None) -> str:
    t = int(ahora if ahora is not None else time.time())
    cuerpo = _b64(json.dumps({"sub": sub, "iat": t, "exp": t + dias * 86400}, separators=(",", ":")).encode())
    firma = _b64(hmac.new(secreto.encode(), f"{_HEADER}.{cuerpo}".encode(), hashlib.sha256).digest())
    return f"{_HEADER}.{cuerpo}.{firma}"


def leer_jwt(token: str, secreto: str, ahora: float | None = None) -> str | None:
    """El `sub` si el token es valido y no vencio; None en cualquier otro caso."""
    try:
        h, c, f = token.split(".")
        if h != _HEADER:
            return None
        esperada = _b64(hmac.new(secreto.encode(), f"{h}.{c}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(f, esperada):
            return None
        datos = json.loads(_de_b64(c))
    except (ValueError, TypeError):
        return None
    if int(datos.get("exp", 0)) < (ahora if ahora is not None else time.time()):
        return None
    return datos.get("sub")
