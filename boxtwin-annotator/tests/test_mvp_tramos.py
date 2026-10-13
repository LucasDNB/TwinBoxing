"""
B3 de la web entrenador: tramos sin asignar y asignacion manual
(prototipo/specs/05-api-y-datos.md).

La sesion de prueba tiene al peleador A partido en dos tracks: el 1 en la primera mitad y el
3 en la segunda. El 3 no tiene cuadros con color y no coexiste con nadie que lo ate, asi que
la identidad lo deja sin asignar, que es exactamente el caso que la persona resuelve en la
web. Asignarlo a A tiene que subir la cobertura de A y habilitar la serie de A en ese tramo,
que es por donde entran sus golpes al detector.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("torch")

from boxtwin.core.identidad_auto import EvidenciaTrack, guardar_evidencia  # noqa: E402
from boxtwin.core.posecache import FrameStatus, PoseArrays, write_pose_cache  # noqa: E402
from boxtwin.mvp.orquesta import completar  # noqa: E402
from boxtwin.mvp.sesion import Sesion  # noqa: E402

T = 400
MITAD = 200
FPS = 30.0
ROJO = (175.0, 200.0, 120.0)
AZUL = (105.0, 200.0, 120.0)
SHA = "0" * 64


def _pose(cx: float) -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros((17, 2), np.float32)
    kp[0] = (cx, 100)
    kp[5], kp[6] = (cx - 40, 160), (cx + 40, 160)
    kp[7], kp[8] = (cx - 60, 220), (cx + 60, 220)
    kp[9], kp[10] = (cx - 70, 240), (cx + 70, 240)
    kp[11], kp[12] = (cx - 30, 300), (cx + 30, 300)
    kp[13], kp[14] = (cx - 30, 400), (cx + 30, 400)
    kp[15], kp[16] = (cx - 30, 500), (cx + 30, 500)
    return kp, np.full(17, 0.9, np.float32)


@pytest.fixture
def sesion(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "spar.mp4").write_bytes(b"no es un video")
    kps, scs, ids, cajas = [], [], [], []
    for f in range(T):
        a = 1 if f < MITAD else 3            # A cambia de track a mitad del video
        for tid, cx in ((a, 300.0), (2, 800.0)):
            kp, sc = _pose(cx)
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
    meta = {"video": {"name": "spar.mp4", "sha256": SHA, "size_bytes": 14,
                      "mtime": datetime.now(timezone.utc).isoformat(), "fps": FPS,
                      "fps_source": "measured", "width": 1280, "height": 720,
                      "total_frames": T, "duration_s": T / FPS, "codec": "h264"},
            "keypoint_format": "coco17"}
    (tmp_path / "cache").mkdir()
    write_pose_cache(tmp_path / "cache" / "spar.pose.npz", arrays, meta)
    (tmp_path / "cache" / "spar.meta.json").write_text(json.dumps(meta))

    primera = list(range(0, MITAD, 5))
    todo = list(range(0, T, 5))
    segunda = list(range(MITAD, T, 5))
    ev = {
        1: EvidenciaTrack(1, 0, MITAD - 1, 0.61, len(primera), len(primera),
                          [(f, ROJO) for f in primera], primera, [(f, 300.0) for f in primera]),
        2: EvidenciaTrack(2, 0, T - 1, 0.61, len(todo), len(todo),
                          [(f, AZUL) for f in todo], todo, [(f, 800.0) for f in todo]),
        # Sin cuadros con color: guante tapado. No hay con que votar ni con que atarlo.
        3: EvidenciaTrack(3, MITAD, T - 1, 0.61, len(segunda), len(segunda),
                          [], segunda, [(f, 300.0) for f in segunda]),
    }
    guardar_evidencia(ev, tmp_path / "evidencia.json")
    s = Sesion.nueva(tmp_path, video={"nombre": "spar.mp4", "ruta": str(tmp_path / "videos" / "spar.mp4"),
                                      "sha256": SHA, "fps": FPS, "total_frames": T,
                                      "duracion_s": T / FPS, "ancho": 1280, "alto": 720},
                     round_s=None, descanso_s=0.0)
    s.estado = "espera_siembra"
    s.guardar()
    return tmp_path


@pytest.fixture
def detector(tmp_path):
    from boxtwin_detector.entrenamiento import Config, Estandarizador
    from boxtwin_detector.ensamble import Ensamble, guardar
    from boxtwin_detector.features import N_FEATURES
    from boxtwin_detector.modelo import TCN

    cfg = Config()
    m = TCN(n_features=N_FEATURES, canales=cfg.canales, dropout=cfg.dropout)
    est = Estandarizador(media=np.zeros(N_FEATURES, np.float32), desvio=np.ones(N_FEATURES, np.float32))
    ruta = tmp_path / "detector.pt"
    guardar(Ensamble([m], est, cfg, [42]), ruta)
    return ruta


def _completar(sesion, detector, manual=False):
    return completar(sesion, semilla_a=1, semilla_b=2, detector=detector, umbral=0.8,
                     device="cpu", manual=manual)


def _manuales(sesion, d):
    s = Sesion.cargar(sesion)
    s.identidad["asignaciones_manuales"] = d
    s.guardar()


def test_el_tramo_sin_asignar_se_lista_con_su_motivo(sesion, detector):
    fc = _completar(sesion, detector)
    tramos = Sesion.cargar(sesion).identidad["tramos"]
    assert [t["track"] for t in tramos] == [3]
    t = tramos[0]
    assert t["motivo"] == "color_insuficiente"
    assert (t["t_desde"], t["t_hasta"]) == (round(MITAD / FPS, 2), round(T / FPS, 2))
    assert t["asignado"] is None
    assert t["recorte"] is None, "el video de prueba no se puede abrir: sin recorte, sin romper"
    ident = fc["identidad"]
    assert (ident["tramos_total"], ident["tramos_asignados_auto"], ident["tramos_asignados_manual"]) == (3, 2, 0)
    assert ident["cobertura_A"] == pytest.approx(0.5, abs=0.01)


def test_asignar_el_tramo_a_A_sube_la_cobertura_y_habilita_la_serie(sesion, detector):
    _completar(sesion, detector)
    _manuales(sesion, {"3": "A"})
    fc = _completar(sesion, detector, manual=True)
    assert fc["identidad"]["cobertura_A"] == pytest.approx(1.0, abs=0.01)
    assert fc["identidad"]["tramos_asignados_manual"] == 1
    valid = np.load(sesion / "poses.npz")["valid"]
    assert valid[0, MITAD:].all(), "los golpes de A en ese tramo ya pueden entrar al detector"
    tramos = Sesion.cargar(sesion).identidad["tramos"]
    assert tramos[0]["asignado"] == "A", "el tramo sigue en la lista con su asignacion"


def test_asignar_ninguno_no_cambia_los_conteos(sesion, detector):
    antes = _completar(sesion, detector)
    _manuales(sesion, {"3": "ninguno"})
    despues = _completar(sesion, detector, manual=True)
    for p in ("A", "B"):
        assert despues["peleadores"][p]["total"] == antes["peleadores"][p]["total"]
    assert despues["identidad"]["cobertura_A"] == antes["identidad"]["cobertura_A"]
    assert despues["identidad"]["tramos_asignados_manual"] == 0


def test_el_completar_manual_no_corre_la_pose(sesion, detector):
    _completar(sesion, detector)
    _manuales(sesion, {"3": "A"})
    _completar(sesion, detector, manual=True)
    etapas = [e["etapa"] for e in Sesion.cargar(sesion).etapas]
    assert "preproceso" not in etapas and "evidencia" not in etapas
    ultima_identidad = [e for e in Sesion.cargar(sesion).etapas if e["etapa"] == "identidad"][-1]
    assert ultima_identidad["manual"] is True


def test_sin_manual_las_asignaciones_guardadas_no_se_aplican(sesion, detector):
    _completar(sesion, detector)
    _manuales(sesion, {"3": "A"})
    fc = _completar(sesion, detector, manual=False)
    assert fc["identidad"]["tramos_asignados_manual"] == 0


# -- motivo_de, sin correr nada ------------------------------------------------------


def _prop(**diag):
    from boxtwin.core.identidad_auto import Propuesta

    return Propuesta(roles={}, rangos={9: (0, 90)}, sin_asignar=[9], diagnostico=diag)


def _ev(recortes=40, colores=10):
    return {9: EvidenciaTrack(9, 0, 89, 0.6, recortes, recortes, [(f, ROJO) for f in range(colores)], [], [])}


def test_motivos():
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto
    from boxtwin.mvp.tramos import motivo_de

    cfg = ConfigIdentidadAuto()
    assert motivo_de(9, _ev(), _prop(votos={9: (3, 3)}, separacion_de_perfiles=1.2), cfg) == "abstencion"
    assert motivo_de(9, _ev(recortes=4), _prop(separacion_de_perfiles=1.2), cfg) == "corto"
    assert motivo_de(9, _ev(), _prop(separacion_de_perfiles=0.3), cfg) == "color_insuficiente"
    assert motivo_de(9, _ev(colores=2), _prop(separacion_de_perfiles=1.2), cfg) == "color_insuficiente"
    assert motivo_de(9, _ev(), _prop(separacion_de_perfiles=1.2), cfg) == "sin_coexistencia"


def test_solo_tramos_de_mas_de_un_segundo():
    from boxtwin.core.identidad_auto import ConfigIdentidadAuto, Propuesta
    from boxtwin.mvp.tramos import listar_tramos

    prop = Propuesta(roles={}, rangos={1: (0, 30), 2: (0, 31), 3: (100, 400)}, sin_asignar=[1, 2, 3])
    ev = {t: EvidenciaTrack(t, 0, 1, 0.6, 40, 40, [], [], []) for t in (1, 2, 3)}
    tramos = listar_tramos(ev, prop, ConfigIdentidadAuto(), fps=30.0)
    assert [t["track"] for t in tramos] == [3, 2], "el de 1 s justo queda afuera; el mas largo primero"


def test_un_rol_manual_desconocido_no_se_acepta():
    from boxtwin.mvp.tramos import aplicar_manuales

    with pytest.raises(ValueError, match="rol desconocido"):
        aplicar_manuales(_prop(), {9: "arbitro"})
