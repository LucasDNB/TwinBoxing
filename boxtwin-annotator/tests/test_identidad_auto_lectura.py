"""analizar lee exactamente los cuadros 0, paso, 2*paso... y en orden (C5, 20-10)."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxtwin.core.identidad_auto import ConfigIdentidadAuto, analizar

N = 60


@pytest.fixture
def video(tmp_path):
    # Cada cuadro es de un gris que codifica su numero. MJPG es intra-cuadro: el gris vuelve
    # casi exacto, sin depender de que el contenedor sepa saltar.
    ruta = tmp_path / "numerado.avi"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 64))
    for f in range(N):
        w.write(np.full((64, 64, 3), 4 * f, np.uint8))
    w.release()
    return ruta


class _Cache:
    """Un track que esta en todos los cuadros."""

    def __len__(self):
        return N

    def detections(self, f):
        return _Dets()


class _Dets(SimpleNamespace):
    def __init__(self):
        super().__init__(bbox=[[0, 0, 40, 60]], track_id=[1],
                         keypoints=[np.zeros((17, 2))], kp_score=[np.zeros(17)])

    def __len__(self):
        return 1


class _Detector:
    def __init__(self):
        self.vistos = []

    def detectar_con_color(self, frame, caja):
        self.vistos.append(int(round(float(frame.mean()) / 4)))
        return []


@pytest.mark.parametrize("paso", [1, 5, 7])
def test_analizar_lee_los_cuadros_del_paso_y_ningun_otro(video, paso):
    det = _Detector()
    cfg = ConfigIdentidadAuto(paso=paso)
    ev = analizar(_Cache(), video, det, cfg, alto_imagen=64, fps=30.0)
    assert det.vistos == list(range(0, N, paso))
    assert ev[1].primer_frame == 0 and ev[1].ultimo_frame == (N - 1) // paso * paso


def test_analizar_corta_si_el_video_es_mas_corto_que_la_cache(video):
    class Larga(_Cache):
        def __len__(self):
            return N + 40

    det = _Detector()
    analizar(Larga(), video, det, ConfigIdentidadAuto(paso=5), alto_imagen=64, fps=30.0)
    assert det.vistos == list(range(0, N, 5))
