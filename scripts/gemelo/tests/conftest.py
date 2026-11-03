"""
Pruebas de scripts/gemelo (docs/specs/2026-10-27-gemelo-yolo26). Corren sin GPU, sin videos y
sin las anotaciones reales: los caches y las anotaciones son sinteticos y chicos.

  ~/miniforge3/envs/twinboxing_env/bin/python -m pytest scripts/gemelo/tests -q
"""

import importlib
import sys
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI))
# boxtwin del mismo checkout, como el PYTHONPATH de C
for paquete in ("boxtwin-annotator/src",):
    ruta = AQUI.parents[1] / paquete
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))


@pytest.fixture
def gemelo(tmp_path, monkeypatch):
    """~/boxtwin-gemelo temporal; recarga los modulos para que lo tomen."""
    raiz = tmp_path / "boxtwin-gemelo"
    raiz.mkdir()
    monkeypatch.setenv("BOXTWIN_GEMELO", str(raiz))
    import guardas
    importlib.reload(guardas)
    for nombre in ("puente_identidad", "medir_trackeo", "recache"):
        if nombre in sys.modules:
            importlib.reload(sys.modules[nombre])
    return raiz
