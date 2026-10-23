"""
Pruebas de scripts/boxmind_prueba. Corren sin GPU, sin torch, sin videos, sin el clon de
BoxMind y sin BoxingWeb: todo lo que necesita datos reales se reemplaza por datos sinteticos.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/boxmind_prueba/tests -q
"""

import importlib
import sys
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI))


@pytest.fixture
def trabajo(tmp_path, monkeypatch):
    """Trabajo de BM y de E4 temporales; recarga los modulos para que los tomen."""
    monkeypatch.setenv("BOXTWIN_BM", str(tmp_path / "bm"))
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR_GEO", str(tmp_path / "geo"))
    import bm            # agrega scripts/clasificador_geo al path
    import geo
    importlib.reload(geo)
    importlib.reload(bm)
    for nombre in ("boxingweb", "entrenar", "evaluar_bm", "bm0"):
        if nombre in sys.modules:
            importlib.reload(sys.modules[nombre])
    return tmp_path
