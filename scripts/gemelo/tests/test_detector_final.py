"""S17: detector_final.py (03-instancia-c.md §5): metricas de E8, meta con procedencia y control."""

import json

import numpy as np
import pytest


def _veredictos(tmp_path):
    v = {"P12a": {"y26": {"precision_080": 0.81, "recall_080": 0.52, "f1_080": 0.633,
                          "folds": ["a", "b", "c", "d", "e", "f"]}}}
    p = tmp_path / "veredictos.json"
    p.write_text(json.dumps(v))
    return p


def test_las_metricas_salen_de_p12a_en_080(gemelo, tmp_path):
    import detector_final as df
    m = df.metricas_de_e8(_veredictos(tmp_path))
    assert m["precision_medida"] == 0.81 and m["recall_medido"] == 0.52
    assert "E8" in m["medido_sobre"] and "0,80" in m["medido_sobre"] and "seis folds" in m["medido_sobre"]


def test_sin_p12a_se_niega(gemelo, tmp_path):
    import detector_final as df
    p = tmp_path / "v.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        df.metricas_de_e8(p)


def test_meta_con_commit_sha_de_cada_tensor_y_comando(gemelo, tmp_path):
    import detector_final as df
    a, b = tmp_path / "x.det.npz", tmp_path / "y.det.npz"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    m = df.meta([a, b], ["entrenar"], semillas=[42, 1], val="todas")
    assert set(m["tensores_sha256"]) == {"x.det.npz", "y.det.npz"} and len(m["tensores_sha256"]["x.det.npz"]) == 64
    assert m["comando"] == "entrenar" and m["semillas"] == [42, 1] and m["val"] == "todas"
    assert "commit" in m


def test_comparar_probabilidades(gemelo):
    import detector_final as df
    a = [np.array([0.1, 0.5]), np.array([0.9, 0.2])]
    r = df.comparar(a, [x + 0.001 for x in a])
    assert r["max_abs"] == pytest.approx(0.001) and r["carriles"] == 2
    assert df.comparar(a, a)["iguales"] is True


def test_el_destino_va_bajo_el_gemelo(gemelo, tmp_path):
    import detector_final as df
    import guardas
    with pytest.raises(guardas.GuardaError):
        df.exigir_destino(tmp_path / "afuera" / "d.ens.pt")
    assert df.exigir_destino(gemelo / "modelos" / "d.ens.pt").name == "d.ens.pt"
