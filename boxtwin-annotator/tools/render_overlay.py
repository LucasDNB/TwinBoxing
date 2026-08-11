#!/usr/bin/env python3
"""
BoxTwin - Renderiza el overlay a video, sin interfaz.

POR QUE EXISTE
  Nacio como diagnostico y quedo como herramienta. En agosto de 2026 aparecio un desfasaje
  aparente entre el esqueleto y la persona, y no habia forma de discutirlo: la maquina de
  trabajo es headless, mirar la aplicacion en vivo requiere VNC, y una impresion sobre una
  ventana remota no es evidencia.

  Este script quema el overlay usando EXACTAMENTE el mismo camino de dibujo que la
  aplicacion (_imagen -> VideoView.set_frame -> paintEvent -> paint_poses, con la misma
  transformacion de coordenadas) y estampa dos numeros por cuadro: de que cuadro es la
  imagen y de que cuadro son los keypoints. Con eso, cualquiera puede mirar el archivo y
  decidir sin depender de una sesion remota.

  Veredicto de aquella vez: los dos numeros van siempre iguales, asi que no habia
  desfasaje. Lo que se veia era el cliente web pidiendo cuadros de mas y dejando la imagen
  atrasada respecto del esqueleto, arreglado en la rama feat/boxtwin-annotator-web.

  Sirve tambien para revisar anotaciones de corrido sin abrir la aplicacion y para producir
  figuras del capitulo.

QUE HACE
  Escribe un mp4 con el overlay dibujado sobre cada cuadro del video, opcionalmente
  estampado con los numeros de cuadro y a la velocidad que se pida.

USO
  python tools/render_overlay.py proyecto/videos/spar.mp4 --out /tmp/overlay.mp4
  python tools/render_overlay.py proyecto/videos/spar.mp4 --from 100 --to 400 --fps 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Antes de importar Qt: este script no abre ninguna ventana.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    p = argparse.ArgumentParser(description="Renderiza el overlay de pose a un video.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None, help="por defecto <video>.overlay.mp4")
    p.add_argument("--from", dest="desde", type=int, default=0)
    p.add_argument("--to", dest="hasta", type=int, default=None)
    p.add_argument(
        "--fps", type=float, default=8.0,
        help="fps del archivo de salida. 8 es ~0,25x sobre 30 y es donde se juzga la pose.",
    )
    p.add_argument("--width", type=int, default=960)
    p.add_argument(
        "--no-stamp", action="store_true",
        help="sin los numeros de cuadro. Se estampan por defecto: son la unica forma de "
             "saber, mirando el archivo, si la imagen y los keypoints son del mismo cuadro.",
    )
    args = p.parse_args()

    import cv2
    import numpy as np
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QDockWidget

    from boxtwin.gui.main_window import MainWindow
    from boxtwin.gui.state import Session

    app = QApplication([])
    s = Session.open(args.video)
    w = MainWindow(s)
    # El dock y el timeline se llevan el espacio; para renderizar estorban.
    for d in w.findChildren(QDockWidget):
        d.hide()
    w.timeline.hide()
    w.statusBar().hide()
    alto = round(args.width * s.video_size[1] / s.video_size[0])
    w.resize(args.width, alto)
    app.processEvents()
    w.view.fit_to_window()

    desde = max(0, args.desde)
    hasta = min(s.total_frames, args.hasta if args.hasta is not None else s.total_frames)
    salida = args.out or args.video.with_suffix(".overlay.mp4")

    fuente = QFont()
    fuente.setPointSize(16)
    fuente.setBold(True)
    escritor = None

    def a_bgr(px):
        """QPixmap -> BGR. bytesPerLine trae relleno y hay que recortarlo."""
        qimg = px.toImage().convertToFormat(QImage.Format.Format_RGB888)
        h, ancho, paso = qimg.height(), qimg.width(), qimg.bytesPerLine()
        plano = np.frombuffer(qimg.constBits(), dtype=np.uint8, count=h * paso)
        rgb = plano.reshape(h, paso)[:, : ancho * 3].reshape(h, ancho, 3)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    print(f"renderizando {hasta - desde} cuadros de {args.video.name}")
    try:
        for f in range(desde, hasta):
            img = w._imagen(f)
            poses = s.resolver.resolve_frame(f)
            w.view.set_frame(img, poses)
            px = w.view.grab()

            if not args.no_stamp:
                # Se estampa DESPUES de dibujar. Si formara parte del dibujo podria quedar
                # sincronizado con la imagen por accidente y no probaria nada.
                painter = QPainter(px)
                painter.setFont(fuente)
                painter.fillRect(0, 0, 300, 60, QColor(0, 0, 0, 210))
                painter.setPen(QColor(255, 235, 59))
                painter.drawText(10, 25, f"IMG  {f}")
                painter.setPen(QColor(120, 220, 255))
                painter.drawText(10, 50, f"POSE {f if poses else '-'}   dets {len(poses)}")
                painter.end()

            bgr = a_bgr(px)
            if escritor is None:
                # El tamano sale del cuadro real: el layout puede imponer otro y un
                # desajuste hace que VideoWriter descarte todo en silencio.
                alto_r, ancho_r = bgr.shape[:2]
                escritor = cv2.VideoWriter(
                    str(salida), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ancho_r, alto_r)
                )
                if not escritor.isOpened():
                    raise SystemExit(f"no se pudo abrir {salida} para escribir")
            escritor.write(bgr)

            if (f - desde) % 200 == 0:
                print(f"  {f - desde}/{hasta - desde}", flush=True)
    finally:
        if escritor is not None:
            escritor.release()
        s.close()

    mb = salida.stat().st_size / 2**20
    print(f"listo -> {salida} ({mb:.1f} MB)")
    print("  mp4v no lo abren todos los reproductores. Para h264:")
    print(f"  ffmpeg -i {salida} -c:v libx264 -crf 20 -pix_fmt yuv420p {salida.stem}.h264.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
