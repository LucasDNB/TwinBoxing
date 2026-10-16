"""
Pruebas de scripts/clasificador_geo. Corren sin GPU, sin videos y sin las anotaciones reales:
todo lo que necesita datos reales se reemplaza por datos sinteticos chicos.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/clasificador_geo/tests -q
"""

import importlib
import sys
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI))


@pytest.fixture
def trabajo(tmp_path, monkeypatch):
    """Directorio de trabajo temporal; recarga los modulos para que lo tomen."""
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR_GEO", str(tmp_path))
    import geo
    importlib.reload(geo)
    for nombre in ("datos", "features2d", "entrenar", "evaluar_geo", "regla_c8", "rol_por_brazo",
                   "features3d", "elevar3d"):
        if nombre in sys.modules:
            importlib.reload(sys.modules[nombre])
    return tmp_path
