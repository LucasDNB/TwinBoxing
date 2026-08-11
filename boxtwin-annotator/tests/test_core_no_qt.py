"""
La regla de dependencias: core no sabe que existe la GUI.

Sin este test la separacion se erosiona sola. Basta un import de conveniencia para mostrar
un color o un QRect y de golpe el pipeline de export deja de correr en un servidor sin
display. El test importa todo core con las librerias pesadas bloqueadas y falla si alguna
se cuela.
"""

from __future__ import annotations

import builtins
import importlib
import pkgutil
import sys

import pytest

PROHIBIDAS = ("PySide6", "torch", "ultralytics", "cv2")

MODULOS_CORE = [
    "boxtwin.core",
    "boxtwin.core.types",
    "boxtwin.core.constants",
    "boxtwin.core.schema",
    "boxtwin.core.validation",
    "boxtwin.core.annotations",
    "boxtwin.core.migrations",
    "boxtwin.core.video",
    "boxtwin.core.posecache",
    "boxtwin.core.identity",
    "boxtwin.core.gloves",
    "boxtwin.core.undo",
    "boxtwin.core.metrics",
    "boxtwin.core.interpolation",
    "boxtwin.core.identity_ops",
    "boxtwin.core.project",
    "boxtwin.core.export",
]


def test_los_modulos_listados_son_todos_los_de_core() -> None:
    """Si se agrega un modulo a core y no se lista aca, el test dejaria de cubrirlo."""
    import boxtwin.core as core

    encontrados = {"boxtwin.core"} | {
        f"boxtwin.core.{m.name}" for m in pkgutil.iter_modules(core.__path__)
    }
    assert encontrados == set(MODULOS_CORE)


def test_core_importa_sin_las_librerias_pesadas(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _import_vigilado(name, globals=None, locals=None, fromlist=(), level=0):
        raiz = name.split(".")[0]
        if raiz in PROHIBIDAS:
            raise AssertionError(
                f"core intento importar {name!r}; core/ no puede depender de la GUI ni "
                "del stack de inferencia"
            )
        return real_import(name, globals, locals, fromlist, level)

    # Se sacan de sys.modules los modulos de core para forzar la reimportacion real.
    for nombre in list(sys.modules):
        if nombre.startswith("boxtwin"):
            monkeypatch.delitem(sys.modules, nombre, raising=False)

    monkeypatch.setattr(builtins, "__import__", _import_vigilado)
    for nombre in MODULOS_CORE:
        importlib.import_module(nombre)


def test_core_no_arrastra_las_librerias_pesadas_a_sys_modules() -> None:
    """
    Complementa al anterior: aunque no las importe directo, podria hacerlo via otro
    modulo. Si alguna aparece cargada despues de importar solo core, hay una cadena.
    """
    import subprocess

    codigo = (
        "import sys, boxtwin.core, boxtwin.core.annotations, boxtwin.core.validation;"
        f"prohibidas={PROHIBIDAS!r};"
        "coladas=[m for m in prohibidas if m in sys.modules];"
        "print(','.join(coladas))"
    )
    salida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )
    assert salida.stdout.strip() == ""
