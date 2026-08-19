"""
Flujo de anotacion completo, de la tecla al archivo.

No se prueba pintado sino la cadena que produce el dato: marcar, clasificar, confirmar,
guardar, deshacer. Es la parte que, si se rompe, produce un dataset mal etiquetado sin que
nada se vea raro en pantalla.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from boxtwin.core.posecache import N_KEYPOINTS, FrameStatus, PoseArrays, write_pose_cache  # noqa: E402
from boxtwin.core.types import (  # noqa: E402
    Completeness,
    FighterId,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)
from boxtwin.gui.main_window import MainWindow  # noqa: E402
from boxtwin.gui.state import Session  # noqa: E402
from boxtwin.gui.widgets.classify_dialog import ClassifyDialog  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="hace falta ffmpeg")

W, H, N = 64, 48, 50
SHA = "e" * 64


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def proyecto(tmp_path: Path) -> Path:
    (tmp_path / "videos").mkdir()
    (tmp_path / "cache").mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration=2:size={W}x{H}:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12",
         str(tmp_path / "videos" / "mini.mp4")],
        check=True,
    )
    meta = {
        "format_version": 1, "kind": "boxtwin.pose_meta", "keypoint_format": "coco17",
        "video": {
            "path": "videos/mini.mp4", "sha256": SHA, "size_bytes": 1, "width": W, "height": H,
            "mtime": "2026-08-11T10:00:00-03:00", "codec": "h264", "fps": 25.0,
            "fps_declared": 25.0, "fps_source": "container_verified", "duration_s": 2.0,
            "total_frames": N, "total_frames_declared": N,
        },
        "counts": {"n_detections": N}, "resume_seams": [],
    }
    arrays = PoseArrays(
        frame_index=np.arange(N + 1, dtype=np.int64),
        frame_status=np.full(N, FrameStatus.OK, np.uint8),
        track_id=np.ones(N, np.int32),
        bbox=np.tile(np.array([1.0, 1.0, 20.0, 30.0], np.float32), (N, 1)),
        det_conf=np.full(N, 0.9, np.float32),
        keypoints=np.zeros((N, N_KEYPOINTS, 2), np.float32),
        kp_score=np.full((N, N_KEYPOINTS), 0.8, np.float32),
    )
    write_pose_cache(tmp_path / "cache" / "mini.pose.npz", arrays, meta)
    (tmp_path / "cache" / "mini.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "config.yaml").write_text("annotator: tester\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ventana(app: QApplication, proyecto: Path):
    s = Session.open(proyecto / "videos" / "mini.mp4")
    w = MainWindow(s)
    w.resize(800, 600)
    yield w
    w.session.close()


def clasificar(monkeypatch, acciones: list[str]) -> None:
    """Reemplaza el exec del dialogo por una secuencia de acciones, como si fueran teclas."""

    def exec_dirigido(self):
        for a in acciones:
            self._ejecutar(a)
        return QDialog.DialogCode.Accepted if self.resultado() else QDialog.DialogCode.Rejected

    monkeypatch.setattr(ClassifyDialog, "exec", exec_dirigido)


COMPLETO = [
    "event.side_right", "event.type_hook", "event.target_head",
    "event.landed", "event.confirm",
]


# -- flujo -----------------------------------------------------------------


def test_marcar_y_confirmar(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(10)
    ventana._marcar_inicio()
    assert ventana._abierto == 10

    ventana.player.seek(28)
    ventana._marcar_fin()

    doc = ventana.session.doc
    assert len(doc.events) == 1
    ev = doc.events[0]
    assert (ev.start_frame, ev.end_frame) == (10, 28)
    assert (ev.side, ev.punch_type, ev.target) == (Side.RIGHT, PunchType.HOOK, Target.HEAD)
    assert ev.landed is Landed.LANDED
    assert ev.completeness is Completeness.FULL
    assert ev.quality is Quality.CLEAN
    # Al confirmar, el foco vuelve al reproductor en el final del evento.
    assert ventana.player.cursor == 28
    assert ventana._abierto is None


def test_confirmar_guarda_en_el_acto(ventana, monkeypatch) -> None:
    """Un corte no puede costar mas de un evento: se guarda al confirmar, no solo por timer."""
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()

    en_disco = json.loads(ventana.session.paths.annot.read_text(encoding="utf-8"))
    assert len(en_disco["events"]) == 1
    assert ventana._sucio is False


def test_metricas_de_proceso(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()

    doc = ventana.session.doc
    eid = doc.events[0].id
    m = doc.process.event_metrics[eid]
    assert m.annotator == "tester"  # sale de config.yaml
    assert m.session_id == ventana.session_id
    assert m.confirmed_at is not None
    assert m.edits == 0
    assert doc.process.sessions[-1].events_created == 1


def test_cancelar_el_dialogo_no_crea_nada(ventana, monkeypatch) -> None:
    monkeypatch.setattr(
        ClassifyDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    assert ventana.session.doc.events == []
    assert ventana._abierto is None


def test_faltando_campos_no_confirma(ventana, monkeypatch) -> None:
    """Sin lado, tipo o altura el dialogo no deja salir: son obligatorios en el esquema."""
    clasificar(monkeypatch, ["event.side_right", "event.confirm"])
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    assert ventana.session.doc.events == []


def test_fin_sin_inicio_no_hace_nada(ventana) -> None:
    ventana.player.seek(20)
    ventana._marcar_fin()
    assert ventana.session.doc.events == []


def test_fin_anterior_al_inicio_se_rechaza(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(20)
    ventana._marcar_inicio()
    ventana.player.seek(10)
    ventana._marcar_fin()
    assert ventana.session.doc.events == []
    # El evento sigue abierto: el anotador puede seguir buscando el final.
    assert ventana._abierto == 20


def test_cancelar_el_evento_abierto(ventana) -> None:
    ventana.player.seek(10)
    ventana._marcar_inicio()
    ventana._cancelar_abierto()
    assert ventana._abierto is None
    assert ventana.session.doc.events == []


# -- edicion y borrado -----------------------------------------------------


def test_editar_pasa_por_el_historial(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    eid = ventana.session.doc.events[0].id

    ventana._editar_campo(eid, "target", Target.BODY)
    assert ventana.session.doc.event_by_id(eid).target is Target.BODY
    assert ventana.session.doc.process.event_metrics[eid].edits == 1

    ventana._deshacer()
    assert ventana.session.doc.event_by_id(eid).target is Target.HEAD


def test_borrar_y_deshacer(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    eid = ventana.session.doc.events[0].id

    ventana._seleccionado = eid
    ventana._borrar_seleccionado()
    assert ventana.session.doc.events == []

    ventana._deshacer()
    assert [e.id for e in ventana.session.doc.events] == [eid]


def test_los_ids_no_se_reusan_tras_cancelar(ventana, monkeypatch) -> None:
    """
    Cancelar consume el id igual. Reusarlo haria que una referencia externa, por ejemplo la
    de un reanno.json, pueda apuntar a otro evento.
    """
    monkeypatch.setattr(ClassifyDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    primero = ventana.session.doc.counters.event

    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(25)
    ventana._marcar_inicio()
    ventana.player.seek(40)
    ventana._marcar_fin()
    assert ventana.session.doc.counters.event > primero
    assert ventana.session.doc.events[0].id != f"ev_{primero:04d}"


# -- sesion ----------------------------------------------------------------


def test_la_sesion_queda_registrada(ventana) -> None:
    doc = ventana.session.doc
    assert len(doc.process.sessions) == 1
    s = doc.process.sessions[0]
    assert s.annotator == "tester"
    assert s.ended_at is None  # todavia abierta
    assert any(a.id == "tester" for a in doc.process.annotators)


def test_cerrar_totaliza(ventana, monkeypatch) -> None:
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()

    ventana.session.end_session(12_345)
    doc = ventana.session.doc
    assert doc.process.sessions[-1].ended_at is not None
    assert doc.process.sessions[-1].active_ms == 12_345
    assert doc.process.totals.events == 1
    assert doc.process.totals.active_ms == 12_345
    assert doc.process.totals.median_ms_per_event is not None


# -- dialogo ---------------------------------------------------------------


def test_el_dialogo_hereda_la_guardia(ventana, monkeypatch) -> None:
    """
    La guardia sale de la vigente para ese peleador en el inicio del golpe. El rol lead/rear
    se deriva de ahi, nunca al reves.
    """
    from boxtwin.core.types import Guard

    ventana.session.doc.fighters[FighterId.A].guard = Guard.SOUTHPAW
    clasificar(monkeypatch, COMPLETO)
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()

    ev = ventana.session.doc.events[0]
    assert ev.guard is Guard.SOUTHPAW
    assert ev.side is Side.RIGHT
    assert ev.arm_role.value == "lead"  # derecha en zurdo es mano adelantada


def test_el_amague_es_toggle(ventana, monkeypatch) -> None:
    """Marcarlo por error no puede obligar a cancelar el dialogo entero."""
    clasificar(monkeypatch, [
        "event.side_right", "event.type_hook", "event.target_head",
        "event.feint", "event.feint", "event.confirm",
    ])
    ventana.player.seek(5)
    ventana._marcar_inicio()
    ventana.player.seek(20)
    ventana._marcar_fin()
    assert ventana.session.doc.events[0].completeness is Completeness.FULL


def test_remarcar_fronteras_recorta_el_pico(ventana, monkeypatch) -> None:
    """Mover una frontera puede dejar el pico afuera, y el esquema lo rechazaria."""
    from boxtwin.core.metrics import EventTimer
    from datetime import datetime

    dlg = ClassifyDialog(
        ventana.session, ventana.keymap, fighter=FighterId.A,
        start_frame=10, end_frame=40, timer=EventTimer(created_at=datetime.now().astimezone()),
        event_id="ev_test", annotator="tester", session_id="se_0001", parent=ventana,
    )
    dlg._cursor = 35
    dlg._ejecutar("event.mark_peak")
    assert dlg.peak_frame == 35
    dlg._cursor = 20
    dlg._ejecutar("event.mark_end")          # el fin baja a 20, el pico queda afuera
    assert dlg.end_frame == 20
    assert dlg.peak_frame == 20
    dlg.close()
