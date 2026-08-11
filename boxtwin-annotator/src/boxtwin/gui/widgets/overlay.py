"""
BoxTwin - Dibujo del esqueleto sobre el cuadro.

POR QUE EXISTE
  El overlay no es decoracion, es el instrumento con el que se juzga si la pose que va a
  entrar al dataset esta bien. De ahi tres decisiones que no son esteticas.

  El color va por ROL y nunca por track_id. Un track_id cambia en cada oclusion y en cada
  reanudacion del preproceso; si el color lo siguiera, el mismo peleador cambiaria de color
  solo y el anotador perderia justo la senal que necesita para detectar un intercambio de
  identidad.

  Los keypoints de baja confianza se dibujan atenuados en vez de ocultarse. Ocultarlos
  haria que una pose mala se vea como una pose incompleta, y son dos cosas distintas: una
  se corrige marcando el tramo no confiable y la otra no.

  El grosor de lineas se divide por el zoom para que se mantenga constante en pixeles de
  pantalla. Si no, al acercarse a mirar una muneca el esqueleto tapa lo que se quiere ver.

QUE HACE
  Pinta esqueleto, cajas, ids y guantes derivados, en coordenadas del video original.

USO
  paint_poses(painter, poses, OverlayOptions(), zoom=vista.zoom)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from boxtwin.core.constants import (
    COCO17_EDGES,
    GLOVE_EDGES,
    ROLE_COLOR_A,
    ROLE_COLOR_B,
    ROLE_COLOR_IGNORE,
    ROLE_COLOR_UNASSIGNED,
)
from boxtwin.core.gloves import derive_gloves
from boxtwin.core.identity import ResolvedPose
from boxtwin.core.types import FighterId, TrackRole

__all__ = ["OverlayOptions", "paint_poses", "color_for_role"]


@dataclass
class OverlayOptions:
    skeleton: bool = True
    boxes: bool = True
    ids: bool = True
    gloves: bool = True
    only_fighter: FighterId | None = None
    kp_threshold: float = 0.3
    glove_k: float = 0.35
    # Alfa de los keypoints por debajo del umbral. No cero: tienen que verse.
    dim_alpha: int = 70


def color_for_role(role: TrackRole | None) -> QColor:
    if role is TrackRole.A:
        return QColor(*ROLE_COLOR_A)
    if role is TrackRole.B:
        return QColor(*ROLE_COLOR_B)
    if role is TrackRole.IGNORE:
        return QColor(*ROLE_COLOR_IGNORE)
    return QColor(*ROLE_COLOR_UNASSIGNED)


def paint_poses(
    painter: QPainter,
    poses: list[ResolvedPose],
    opciones: OverlayOptions,
    *,
    zoom: float = 1.0,
) -> None:
    """Pinta todas las detecciones del cuadro. El painter ya viene con el transform puesto."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    grosor = max(0.5, 2.0 / zoom)
    radio = max(0.8, 3.0 / zoom)

    for pose in poses:
        if opciones.only_fighter is not None and pose.fighter is not opciones.only_fighter:
            continue

        color = color_for_role(pose.role)
        if pose.shadowed or not pose.reliable:
            # Punteado: la deteccion esta, pero no se va a usar tal cual.
            color = QColor(color)
            color.setAlpha(140)

        if opciones.boxes:
            _pintar_caja(painter, pose, color, grosor)
        if opciones.skeleton:
            _pintar_esqueleto(painter, pose, color, opciones, grosor, radio)
        if opciones.ids:
            _pintar_etiqueta(painter, pose, color, zoom)


def _pintar_caja(painter: QPainter, pose: ResolvedPose, color: QColor, grosor: float) -> None:
    x1, y1, x2, y2 = (float(v) for v in pose.bbox)
    pen = QPen(color, grosor)
    if pose.shadowed or not pose.reliable:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))


def _pintar_esqueleto(
    painter: QPainter,
    pose: ResolvedPose,
    color: QColor,
    opciones: OverlayOptions,
    grosor: float,
    radio: float,
) -> None:
    xy = pose.keypoints
    sc = pose.kp_score
    aristas = list(COCO17_EDGES)

    if opciones.gloves:
        g_xy, g_sc = derive_gloves(xy[None, ...], sc[None, ...], opciones.glove_k)
        xy = np.concatenate([xy, g_xy[0]], axis=0)
        sc = np.concatenate([sc, g_sc[0]], axis=0)
        aristas += list(GLOVE_EDGES)

    umbral = opciones.kp_threshold
    tenue = QColor(color)
    tenue.setAlpha(opciones.dim_alpha)

    for a, b in aristas:
        if a >= len(xy) or b >= len(xy):
            continue
        fuerte = sc[a] >= umbral and sc[b] >= umbral
        painter.setPen(QPen(color if fuerte else tenue, grosor))
        painter.drawLine(
            QPointF(float(xy[a][0]), float(xy[a][1])),
            QPointF(float(xy[b][0]), float(xy[b][1])),
        )

    painter.setPen(Qt.PenStyle.NoPen)
    for i in range(len(xy)):
        fuerte = sc[i] >= umbral
        painter.setBrush(color if fuerte else tenue)
        painter.drawEllipse(QPointF(float(xy[i][0]), float(xy[i][1])), radio, radio)
    painter.setBrush(Qt.BrushStyle.NoBrush)


def _pintar_etiqueta(painter: QPainter, pose: ResolvedPose, color: QColor, zoom: float) -> None:
    """
    Etiqueta con rol y track_id.

    El rol va primero y el id entre parentesis: lo que importa al anotar es de quien es la
    deteccion, el id es dato de diagnostico.
    """
    x1, y1 = float(pose.bbox[0]), float(pose.bbox[1])
    rol = pose.role.value if pose.role else "sin asignar"
    texto = f"{rol} (#{pose.track_id})"
    if pose.manual:
        texto += " manual"
    if pose.shadowed:
        texto += " duplicado"
    if not pose.reliable:
        texto += " no confiable"

    # El tamano de fuente se divide por el zoom, igual que el grosor de linea, para que la
    # etiqueta ocupe siempre lo mismo en pantalla y no tape la pose al acercarse.
    fuente = QFont()
    fuente.setPointSizeF(max(3.0, 11.0 / zoom))
    painter.setFont(fuente)
    painter.setPen(QPen(color, max(0.5, 1.0 / zoom)))
    painter.drawText(QPointF(x1, y1 - 4.0 / zoom), texto)
