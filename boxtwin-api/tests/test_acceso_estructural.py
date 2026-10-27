"""
Prueba estructural del control de acceso (prototipo/specs/versiones-26-10, 05 §3, RNF-V2).

Es la que impide que una tarea futura vuelva a filtrar por usuario sin que nadie lo note. No
mira lo que devuelve una ruta sino como esta escrito el codigo:

1. Ninguna comparacion de `usuario_id` con == o != en los modulos que deciden que se ve. Las
   asignaciones (`usuario_id=` al crear algo) se permiten: dicen quien hizo algo.
2. Toda ruta con un identificador en el camino depende de acceso_actual o del lector de
   tickets que verifica la cuenta.
3. Toda emision de token fuera de /auth/login y /auth/registro pasa antes por la
   verificacion de `cuenta.demo`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from versiones_util import rutas

SRC = Path(__file__).resolve().parents[1] / "src" / "boxtwin_api"
MODULOS = ("app.py", "panel.py", "perfil.py", "rival.py", "cuentas.py")
EXCEPCIONES = ("/auth/", "/producto", "/salud", "/invitaciones/{token}")


def _nombre(n) -> str | None:
    if isinstance(n, ast.Attribute):
        return n.attr
    if isinstance(n, ast.Name):
        return n.id
    return None


def test_1_nadie_compara_usuario_id():
    hallados = []
    for nombre in MODULOS:
        arbol = ast.parse((SRC / nombre).read_text(), filename=nombre)
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in nodo.ops):
                continue
            lados = [nodo.left, *nodo.comparators]
            if any(_nombre(x) == "usuario_id" for x in lados):
                hallados.append(f"{nombre}:{nodo.lineno}")
    assert not hallados, f"comparaciones de usuario_id (tienen que ir por cuenta): {hallados}"


def _llamadas(dependant) -> set[str]:
    nombres = set()
    for d in dependant.dependencies:
        if d.call is not None:
            nombres.add(getattr(d.call, "__name__", ""))
        nombres |= _llamadas(d)
    return nombres


def test_2_toda_ruta_con_id_pasa_por_la_cuenta(entorno):
    sin = []
    for r in rutas(entorno["app"].app):
        if not re.search(r"{\w+}", r.path):
            continue
        if any(r.path.startswith(e) or r.path == e for e in EXCEPCIONES):
            continue
        if not _llamadas(r.dependant) & {"acceso_actual", "acceso_por_ticket"}:
            sin.append(f"{sorted(r.methods)} {r.path}")
    assert not sin, f"rutas con identificador que no resuelven la cuenta: {sin}"


def test_2_tambien_las_listas_pasan_por_la_cuenta(entorno):
    """Las rutas sin id que devuelven datos tambien: /jobs, /boxeadores, /panel, /cuenta."""
    for ruta in ("/jobs", "/boxeadores", "/panel", "/cuenta", "/rival"):
        r = next(x for x in rutas(entorno["app"].app) if x.path == ruta)
        assert "acceso_actual" in _llamadas(r.dependant), ruta


def test_3_toda_emision_de_token_fuera_del_login_verifica_la_cuenta_de_prueba():
    malas = []
    for nombre in ("app.py", "demo.py"):
        arbol = ast.parse((SRC / nombre).read_text(), filename=nombre)
        for fn in ast.walk(arbol):
            if not isinstance(fn, ast.FunctionDef):
                continue
            emisiones = [n.lineno for n in ast.walk(fn)
                         if isinstance(n, ast.Call) and _nombre(n.func) == "emitir_token"]
            if not emisiones or fn.name in ("registro", "login"):
                continue
            # Una lectura de `.demo` dentro de una condicion, antes de la primera emision.
            chequeos = [n.lineno for n in ast.walk(fn)
                        if isinstance(n, (ast.If, ast.IfExp, ast.Assert))
                        and any(isinstance(x, ast.Attribute) and x.attr == "demo"
                                for x in ast.walk(n.test))]
            if not chequeos or min(chequeos) > min(emisiones):
                malas.append(f"{nombre}:{fn.name}")
    assert not malas, f"emiten token sin verificar cuenta.demo antes: {malas}"


def test_la_prueba_3_encuentra_la_ruta_de_prueba():
    """Que la regla 3 no pase por vacia: la emision de prueba existe y esta verificada."""
    arbol = ast.parse((SRC / "app.py").read_text())
    nombres = {fn.name for fn in ast.walk(arbol) if isinstance(fn, ast.FunctionDef)
               and any(isinstance(n, ast.Call) and _nombre(n.func) == "emitir_token"
                       for n in ast.walk(fn))}
    assert nombres == {"registro", "login", "_token_de_prueba"}
