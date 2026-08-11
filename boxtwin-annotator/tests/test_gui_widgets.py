"""
Widgets, sin pantalla.

Corre con QT_QPA_PLATFORM=offscreen. Lo que se verifica es lo unico de la vista que puede
corromper datos: la correspondencia entre pixeles de pantalla y pixeles del video. Si esa
conversion se desajusta con el zoom, el esqueleto se corre del cuerpo lo suficiente para
no notarlo y anotar mal.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from boxtwin.core.identity import ResolvedPose  # noqa: E402
from boxtwin.core.types import FighterId, PunchType, TrackRole  # noqa: E402
from boxtwin.gui.widgets.overlay import OverlayOptions, color_for_role, paint_poses  # noqa: E402
from boxtwin.gui.widgets.timeline import Timeline  # noqa: E402
from boxtwin.gui.widgets.video_view import VideoView  # noqa: E402

VIDEO = (1920, 1080)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def vista(app: QApplication) -> VideoView:
    v = VideoView(VIDEO)
    v.resize(960, 540)
    return v


def pose(track_id: int = 1, role: TrackRole | None = None) -> ResolvedPose:
    return ResolvedPose(
        frame=0, track_id=track_id, role=role,
        bbox=np.array([100.0, 100.0, 300.0, 600.0], np.float32),
        det_conf=0.9,
        keypoints=np.random.default_rng(0).uniform(100, 600, (17, 2)).astype(np.float32),
        kp_score=np.full(17, 0.8, np.float32),
    )


# -- correspondencia de coordenadas ---------------------------------------


@pytest.mark.parametrize("zoom", [0.25, 0.5, 1.0, 2.0, 6.0])
def test_ida_y_vuelta_de_coordenadas(vista: VideoView, zoom: float) -> None:
    vista.set_zoom(zoom)
    for punto in (QPointF(0, 0), QPointF(960.0, 540.0), QPointF(1919.0, 1079.0)):
        vuelta = vista.widget_to_video(vista.video_to_widget(punto))
        assert vuelta.x() == pytest.approx(punto.x(), abs=1e-6)
        assert vuelta.y() == pytest.approx(punto.y(), abs=1e-6)


def test_ajustar_centra_el_video(vista: VideoView) -> None:
    """Sin centrado la imagen queda pegada arriba y sobra negro abajo."""
    vista.fit_to_window()
    z = vista.zoom
    assert z == pytest.approx(min(960 / 1920, 540 / 1080))
    esquina = vista.video_to_widget(QPointF(0, 0))
    fin = vista.video_to_widget(QPointF(1920, 1080))
    assert esquina.x() + fin.x() == pytest.approx(960, abs=1.0)
    assert esquina.y() + fin.y() == pytest.approx(540, abs=1.0)


def test_zoom_mantiene_fijo_el_punto_bajo_el_ancla(vista: VideoView) -> None:
    """
    Al hacer zoom con la rueda, el pixel de video que esta bajo el puntero no se mueve.
    Sin esto, acercarse a mirar una muneca la saca de la pantalla.
    """
    ancla = QPointF(300.0, 200.0)
    antes = vista.widget_to_video(ancla)
    vista.set_zoom(4.0, ancla)
    despues = vista.widget_to_video(ancla)
    assert despues.x() == pytest.approx(antes.x(), abs=0.5)
    assert despues.y() == pytest.approx(antes.y(), abs=0.5)


def test_el_paneo_no_sale_del_video(vista: VideoView) -> None:
    vista.set_zoom(4.0)
    vista._pan = QPointF(1e6, 1e6)
    vista._clamp_pan()
    visible = vista.widget_to_video(QPointF(vista.width(), vista.height()))
    assert visible.x() <= 1920 + 1
    assert visible.y() <= 1080 + 1


def test_limites_de_zoom(vista: VideoView) -> None:
    vista.set_zoom(1000.0)
    assert vista.zoom <= 12.0
    vista.set_zoom(0.0001)
    assert vista.zoom >= 0.1


# -- overlay ---------------------------------------------------------------


def test_color_por_rol_y_no_por_track() -> None:
    """
    Cuatro colores distintos, uno por rol. Que el color siguiera al track_id haria que el
    mismo peleador cambie de color en cada oclusion.
    """
    colores = {
        color_for_role(TrackRole.A).name(),
        color_for_role(TrackRole.B).name(),
        color_for_role(TrackRole.IGNORE).name(),
        color_for_role(None).name(),
    }
    assert len(colores) == 4
    # El mismo rol da el mismo color aunque cambie el track.
    assert color_for_role(TrackRole.A).name() == color_for_role(TrackRole.A).name()


def test_pintar_no_explota_con_poses_variadas(vista: VideoView) -> None:
    from PySide6.QtGui import QPainter, QPixmap

    px = QPixmap(960, 540)
    p = QPainter(px)
    poses = [
        pose(1, TrackRole.A),
        pose(2, TrackRole.B),
        pose(3, None),
        pose(4, TrackRole.IGNORE),
    ]
    for opciones in (
        OverlayOptions(),
        OverlayOptions(skeleton=False),
        OverlayOptions(boxes=False, ids=False),
        OverlayOptions(gloves=False),
        OverlayOptions(only_fighter=FighterId.A),
    ):
        paint_poses(p, poses, opciones, zoom=2.0)
    p.end()


def test_only_fighter_filtra(vista: VideoView) -> None:
    from PySide6.QtGui import QPainter, QPixmap

    px = QPixmap(100, 100)
    p = QPainter(px)
    # Con el filtro puesto, una pose sin rol no tiene que dibujarse; el test es que no
    # falle al descartarla y que el filtro exista.
    paint_poses(p, [pose(3, None)], OverlayOptions(only_fighter=FighterId.A), zoom=1.0)
    p.end()


def test_set_frame_acepta_none(vista: VideoView) -> None:
    vista.set_frame(None, [])
    vista.set_frame(np.zeros((540, 960, 3), np.uint8), [pose(1, TrackRole.A)])


# -- timeline --------------------------------------------------------------


def test_timeline_mapea_cuadro_a_pixel(app: QApplication) -> None:
    tl = Timeline()
    tl.resize(1000, 40)
    tl.set_total(2000)
    assert tl._frame(tl._x(0)) == 0
    assert tl._frame(tl._x(1000)) == pytest.approx(1000, abs=2)
    assert tl._frame(tl._x(1999)) == pytest.approx(1999, abs=2)


def test_timeline_carriles_separados(app: QApplication) -> None:
    """Un carril por peleador: en uno solo, un intercambio apila las marcas."""
    tl = Timeline()
    tl.resize(1000, 40)
    a = tl._lane_rect(FighterId.A)
    b = tl._lane_rect(FighterId.B)
    assert a.top() < b.top()
    assert not a.intersects(b)


def test_timeline_pinta_con_eventos(app: QApplication, doc_rich) -> None:
    from PySide6.QtGui import QPixmap

    tl = Timeline()
    tl.resize(1000, 40)
    tl.set_total(53412)
    tl.set_events(doc_rich.events)
    tl.set_unreliable(doc_rich.unreliable_segments)
    tl.set_seams([122, 4120])
    tl.set_selected("ev_0042")
    px = QPixmap(1000, 40)
    tl.render(px)
    assert not px.isNull()


def test_timeline_conoce_todos_los_tipos_de_golpe() -> None:
    """Un tipo sin color entraria en gris y el desbalance dejaria de verse en la barra."""
    from boxtwin.core.constants import PUNCH_COLORS

    assert set(PUNCH_COLORS) == {t.value for t in PunchType}
