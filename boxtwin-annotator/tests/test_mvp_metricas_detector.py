"""
T3 del gemelo YOLO26 (docs/specs/2026-10-27-gemelo-yolo26, 01-percepcion.md §6).

La Fight-Card de C muestra la precision y el recall que E8 midio para SU detector, no los de A
(CG5). Salen de `<checkpoint>.metricas.json` al lado del checkpoint; sin ese archivo quedan
0,885 y 0,484 como hoy. Ademas lleva un bloque `pose` con el modelo y el sha256 que escribio
el preproceso en el meta.json de la sesion. En los dos casos cumple el contrato 0.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

pytest.importorskip("torch")

from boxtwin.mvp.orquesta import completar  # noqa: E402

from .test_mvp_orquesta import detector, sesion  # noqa: E402,F401

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "prototipo/contrato/fightcard-0.3.schema.json").read_text()
)
SHA_POSE = "ab" * 32


def _sin_correcciones(fc):
    return {k: v for k, v in fc.items() if k != "_correcciones"}


def test_sin_archivo_de_metricas_quedan_las_de_hoy(sesion, detector):  # noqa: F811
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    assert fc["detector"]["precision_medida"] == 0.885
    assert fc["detector"]["recall_medido"] == 0.484
    jsonschema.validate(_sin_correcciones(fc), SCHEMA)


def test_con_archivo_de_metricas_la_fightcard_muestra_esos_numeros(sesion, detector):  # noqa: F811
    metricas = {"precision_medida": 0.871, "recall_medido": 0.512,
                "medido_sobre": "E8, seis folds comparables, ensamble de cinco semillas",
                "otra_cosa": "no viaja"}
    detector.with_suffix(".metricas.json").write_text(json.dumps(metricas))
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    det = fc["detector"]
    assert det["precision_medida"] == 0.871
    assert det["recall_medido"] == 0.512
    assert det["medido_sobre"].startswith("E8")
    assert "otra_cosa" not in det
    jsonschema.validate(_sin_correcciones(fc), SCHEMA)


def test_el_nombre_del_archivo_de_metricas_saca_solo_el_pt(tmp_path):
    from boxtwin.mvp.orquesta import _metricas_medidas

    ruta = tmp_path / "detector_7fuentes_y26.ens.pt"
    (tmp_path / "detector_7fuentes_y26.ens.metricas.json").write_text(
        json.dumps({"precision_medida": 0.9, "recall_medido": 0.5}))
    assert _metricas_medidas(ruta) == {"precision_medida": 0.9, "recall_medido": 0.5}
    assert _metricas_medidas(tmp_path / "otro.pt") == {}


def test_el_bloque_pose_sale_del_meta_de_la_sesion(sesion, detector):  # noqa: F811
    meta_path = sesion / "cache" / "spar.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["inference"] = {"model_path": "/m/yolo26l-pose.pt", "model_sha256": SHA_POSE}
    meta_path.write_text(json.dumps(meta))
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    assert fc["detector"]["pose"] == {"modelo": "yolo26l-pose.pt", "sha256": SHA_POSE}
    jsonschema.validate(_sin_correcciones(fc), SCHEMA)


def test_sin_inference_en_el_meta_el_bloque_pose_va_en_nulo(sesion, detector):  # noqa: F811
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    assert fc["detector"]["pose"] == {"modelo": None, "sha256": None}
