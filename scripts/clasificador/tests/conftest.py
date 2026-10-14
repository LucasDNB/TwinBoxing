"""
Pruebas de scripts/clasificador. Corren sin GPU, sin videos de la estacion y sin boxtwin:
todo lo que necesita datos reales se reemplaza por datos sinteticos chicos.

  cd scripts/clasificador && python -m pytest tests -q
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
    monkeypatch.setenv("BOXTWIN_CLASIFICADOR", str(tmp_path))
    import comun
    importlib.reload(comun)
    for nombre in ("armar_folds", "evaluar", "comparar_etiquetas", "extraer_tiras", "huellas"):
        if nombre in sys.modules:
            importlib.reload(sys.modules[nombre])
    return tmp_path
