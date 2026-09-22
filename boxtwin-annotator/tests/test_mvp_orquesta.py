"""
La etapa 2 entera, sobre una sesion sintetica en disco.

Es el unico test que recorre el camino completo -identidad sembrada, series por peleador,
detector congelado, guardia, Fight-Card- y por eso es el que atrapa las roturas de
cableado, que son las que ningun test de unidad ve. El detector es una TCN sin entrenar:
lo que se prueba es que los datos lleguen enteros de una punta a la otra, no que acierte.

La pose se sintetiza desde el cache y no desde un video, asi que no hace falta material
propio ni GPU para correrlo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxtwin.core.identidad_auto import EvidenciaTrack, guardar_evidencia  # noqa: E402
from boxtwin.core.posecache import FrameStatus, PoseArrays, write_pose_cache  # noqa: E402
from boxtwin.core.schema import PoseRef, VideoInfo, new_document  # noqa: E402
from boxtwin.core.types import FpsSource, KeypointFormat  # noqa: E402
from boxtwin.mvp.orquesta import completar  # noqa: E402
from boxtwin.mvp.sesion import Sesion  # noqa: E402

T = 400
FPS = 30.0
ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)
SHA = "0" * 64


def _pose_de(cx: float, t: int, pega: bool) -> tuple[np.ndarray, np.ndarray]:
    """Un cuerpo de pie; con `pega`, la muneca izquierda sale disparada al frente."""
    kp = np.zeros((17, 2), np.float32)
    kp[0] = (cx, 100)                       # nariz
    kp[5], kp[6] = (cx - 40, 160), (cx + 40, 160)
    kp[7], kp[8] = (cx - 60, 220), (cx + 60, 220)
    kp[9] = (cx - 70 + (120 if pega else 0), 240)
    kp[10] = (cx + 70, 240)
    kp[11], kp[12] = (cx - 30, 300), (cx + 30, 300)
    kp[13], kp[14] = (cx - 30, 400), (cx + 30, 400)
    kp[15], kp[16] = (cx - 30, 500), (cx + 30, 500)
    return kp, np.full(17, 0.9, np.float32)


@pytest.fixture
def sesion(tmp_path):
    """Una sesion en espera_siembra: dos tracks que coexisten todo el video."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "spar.mp4").write_bytes(b"no es un video, nadie lo abre aca")

    kps, scs, ids, cajas = [], [], [], []
    for f in range(T):
        for tid, cx in ((1, 300.0), (2, 800.0)):
            pega = tid == 1 and 100 <= f < 110
            kp, sc = _pose_de(cx, f, pega)
            kps.append(kp)
            scs.append(sc)
            ids.append(tid)
            cajas.append([cx - 80, 80, cx + 80, 520])
    n = len(ids)
    arrays = PoseArrays(
        frame_index=np.arange(0, 2 * T + 1, 2, dtype=np.int64),
        frame_status=np.full(T, FrameStatus.OK, np.uint8),
        track_id=np.asarray(ids, np.int32),
        bbox=np.asarray(cajas, np.float32),
        det_conf=np.full(n, 0.9, np.float32),
        keypoints=np.stack(kps).astype(np.float32),
        kp_score=np.stack(scs).astype(np.float32),
    )
    meta = {
        "video": {
            "name": "spar.mp4", "sha256": SHA, "size_bytes": 33,
            "mtime": datetime.now(timezone.utc).isoformat(), "fps": FPS,
            "fps_source": "measured", "width": 1280, "height": 720,
            "total_frames": T, "duration_s": T / FPS, "codec": "h264",
        },
        "keypoint_format": "coco17",
    }
    (tmp_path / "cache").mkdir()
    write_pose_cache(tmp_path / "cache" / "spar.pose.npz", arrays, meta)
    (tmp_path / "cache" / "spar.meta.json").write_text(json.dumps(meta))

    frames = list(range(0, T, 5))
    ev = {
        1: EvidenciaTrack(1, 0, T - 1, 0.61, len(frames), len(frames),
                          [(f, ROJO) for f in frames], frames,
                          [(f, 300.0) for f in frames]),
        2: EvidenciaTrack(2, 0, T - 1, 0.61, len(frames), len(frames),
                          [(f, AZUL) for f in frames], frames,
                          [(f, 800.0) for f in frames]),
    }
    guardar_evidencia(ev, tmp_path / "evidencia.json")

    s = Sesion.nueva(
        tmp_path,
        video={"nombre": "spar.mp4", "ruta": str(tmp_path / "videos" / "spar.mp4"),
               "sha256": SHA, "fps": FPS, "total_frames": T, "duracion_s": T / FPS,
               "ancho": 1280, "alto": 720},
        round_s=5.0, descanso_s=1.0,
    )
    s.estado = "espera_siembra"
    s.guardar()
    return tmp_path


@pytest.fixture
def detector(tmp_path):
    """Un ensamble de una TCN sin entrenar. Alcanza para probar el cableado."""
    from boxtwin_detector.entrenamiento import Config, Estandarizador
    from boxtwin_detector.ensamble import Ensamble, guardar
    from boxtwin_detector.features import N_FEATURES
    from boxtwin_detector.modelo import TCN

    cfg = Config()
    m = TCN(n_features=N_FEATURES, canales=cfg.canales, dropout=cfg.dropout)
    est = Estandarizador(media=np.zeros(N_FEATURES, np.float32),
                         desvio=np.ones(N_FEATURES, np.float32))
    ruta = tmp_path / "detector.pt"
    guardar(Ensamble([m], est, cfg, [42]), ruta)
    return ruta


# -- el camino completo -----------------------------------------------------


def test_de_la_siembra_a_la_fightcard(sesion, detector):
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector,
                   umbral=0.8, device="cpu")

    assert (sesion / "fightcard.json").is_file()
    assert (sesion / "segmentos.json").is_file()
    assert (sesion / "poses.npz").is_file(), "el clasificador corre en el otro entorno"
    assert Sesion.cargar(sesion).estado == "listo"
    assert fc["version"] == "0.1"
    assert set(fc["peleadores"]) == {"A", "B"}


def test_la_semilla_decide_quien_es_A(sesion, detector):
    fc = completar(sesion, semilla_a=2, semilla_b=1, detector=detector, device="cpu")
    doc = json.loads((sesion / "annotations" / "spar.annot.json").read_text())
    de_a = {a["track_id"] for a in doc["identity"]["assignments"]
            if a["role"] in ("A", "fighter_A")}
    assert de_a == {2}
    assert fc["identidad"]["cobertura_A"] > 0.9


def test_la_cobertura_sale_medida_y_no_supuesta(sesion, detector):
    fc = completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    ident = fc["identidad"]
    assert ident["cobertura_A"] == pytest.approx(1.0)
    assert ident["cobertura_B"] == pytest.approx(1.0)
    assert ident["sin_asignar"] == 0.0


def test_las_poses_guardadas_son_las_del_peleador_y_no_las_del_track(sesion, detector):
    # El error que este test atrapa: guardar los keypoints en el orden del tracker. El
    # clasificador correria sobre el cuerpo equivocado y nadie lo veria en el JSON.
    completar(sesion, semilla_a=2, semilla_b=1, detector=detector, device="cpu")
    d = np.load(sesion / "poses.npz")
    kp = d["keypoints"]
    assert kp.shape == (2, T, 17, 2)
    # La semilla A fue el track 2, que esta en x=800.
    assert kp[0, 0, 0, 0] == pytest.approx(800.0)
    assert kp[1, 0, 0, 0] == pytest.approx(300.0)


def test_los_segmentos_quedan_para_reclasificar_sin_reprocesar(sesion, detector):
    # RF12: publicar un checkpoint nuevo no puede obligar a repetir pose ni deteccion.
    completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    d = json.loads((sesion / "segmentos.json").read_text())
    assert d["poses"] == "poses.npz"
    assert d["detector"]["umbral"] == 0.8
    assert isinstance(d["segmentos"], list)


def test_el_tiempo_por_etapa_queda_registrado(sesion, detector):
    # RNF1 es un criterio de aceptacion y medirlo despues a mano mezcla el procesamiento
    # con la espera humana entre las dos etapas.
    completar(sesion, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
    etapas = {e["etapa"] for e in Sesion.cargar(sesion).etapas}
    assert {"identidad", "series", "detector", "guardia"} <= etapas


def test_dos_semillas_iguales_no_se_aceptan(sesion, detector):
    with pytest.raises(ValueError, match="mismo track"):
        completar(sesion, semilla_a=1, semilla_b=1, detector=detector, device="cpu")


def test_una_sesion_sin_procesar_no_se_puede_completar(tmp_path, detector):
    with pytest.raises(FileNotFoundError):
        completar(tmp_path, semilla_a=1, semilla_b=2, detector=detector, device="cpu")
