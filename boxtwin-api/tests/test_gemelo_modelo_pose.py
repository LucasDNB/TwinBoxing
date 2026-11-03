"""
T2 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6).

/salud dice que checkpoint de pose usa el worker: el nombre y los 12 primeros del sha256 de
`cfg.modelo_pose`, calculados una vez al arrancar. Es lo que mira CG2 para saber que C corre
yolo26l-pose y no el de A.
"""

from __future__ import annotations

import hashlib
import importlib
import sys


def test_archivo_temporal_da_nombre_y_hash(tmp_path):
    from boxtwin_api.app import _modelo_pose

    ruta = tmp_path / "yolo26l-pose.pt"
    ruta.write_bytes(b"pesos de mentira")
    esperado = hashlib.sha256(b"pesos de mentira").hexdigest()[:12]
    assert _modelo_pose(ruta) == {"archivo": "yolo26l-pose.pt", "sha256": esperado}


def test_archivo_inexistente_da_null_sin_excepcion(tmp_path):
    from boxtwin_api.app import _modelo_pose

    assert _modelo_pose(tmp_path / "no-esta.pt") is None


def test_salud_lo_trae_calculado_al_arrancar(entorno, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    ruta = tmp_path / "yolo26l-pose.pt"
    ruta.write_bytes(b"otros pesos")
    monkeypatch.setenv("BOXTWIN_MODELO_POSE", str(ruta))
    import boxtwin_api.config as config
    importlib.reload(config)
    app_mod = importlib.reload(sys.modules["boxtwin_api.app"])
    with TestClient(app_mod.app) as c:
        mp = c.get("/salud").json()["modelo_pose"]
        assert mp == {"archivo": "yolo26l-pose.pt",
                      "sha256": hashlib.sha256(b"otros pesos").hexdigest()[:12]}
        # Una vez al arrancar: cambiar el archivo despues no cambia lo que se informa.
        ruta.write_bytes(b"cambiado")
        assert c.get("/salud").json()["modelo_pose"] == mp
