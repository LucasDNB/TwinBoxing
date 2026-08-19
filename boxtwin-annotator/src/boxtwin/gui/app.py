"""
BoxTwin - Arranque de la aplicacion.

POR QUE EXISTE
  Separar el arranque de la ventana permite abrir una sesion desde un test o desde un
  notebook sin levantar la interfaz entera, y permite correr la GUI headless con
  QT_QPA_PLATFORM=offscreen, que es como se verifica que el overlay dibuja lo que debe sin
  necesitar pantalla.

QUE HACE
  Crea la QApplication, abre la sesion y muestra la ventana.

USO
  boxtwin-annotator annotate proyecto/videos/spar.mp4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["run", "NoDisplayError", "check_display"]


class NoDisplayError(RuntimeError):
    pass


def check_display() -> None:
    """
    Falla con un mensaje util si no hay sesion grafica.

    Sin esto Qt aborta con un core dump y un mensaje sobre libxcb-cursor que despista: esa
    libreria esta y resuelve bien, lo que falta es el display. Perder media hora
    persiguiendo la pista equivocada es evitable con diez lineas.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return  # el usuario ya eligio plataforma, es asunto suyo
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise NoDisplayError(
        "no hay sesion grafica: DISPLAY y WAYLAND_DISPLAY estan vacias.\n"
        "  Si estas en el escritorio de la maquina, abri una terminal de esa sesion.\n"
        "  Si entraste por ssh, reconectate con 'ssh -X'.\n"
        "  Sin display, se puede servir la ventana por VNC sin instalar nada:\n"
        "      boxtwin-annotator annotate <video> --platform vnc\n"
        "  y conectar un cliente VNC a localhost:5900."
    )


def run(
    video: Path,
    *,
    buffer_mb: int = 512,
    platform: str | None = None,
    vnc_size: str = "1600x1000",
    vnc_port: int = 5900,
    argv: list[str] | None = None,
) -> int:
    es_vnc = platform == "vnc"
    if platform:
        # Tiene que estar en el entorno antes de que se cree la QApplication.
        # El plugin vnc de Qt sirve 1024x768 por defecto y ahi la ventana del anotador
        # queda recortada, asi que el tamano se pasa siempre explicito.
        os.environ["QT_QPA_PLATFORM"] = (
            f"vnc:size={vnc_size}:port={vnc_port}" if es_vnc else platform
        )
    check_display()

    from PySide6.QtWidgets import QApplication

    from boxtwin.gui.main_window import MainWindow
    from boxtwin.gui.state import Session

    app = QApplication.instance() or QApplication(argv or sys.argv[:1])
    if es_vnc:
        import socket

        # El puerto local del tunel se sugiere distinto del remoto a proposito. En Windows,
        # Hyper-V y WSL reservan rangos que suelen incluir el 5900, y ahi ssh falla con
        # "bind: Permission denied", que parece un problema del servidor y no lo es.
        local = vnc_port + 10000
        print(
            f"ventana servida por VNC en el puerto {vnc_port}, pantalla {vnc_size}.\n"
            f"  Desde tu maquina:\n"
            f"    ssh -L {local}:localhost:{vnc_port} "
            f"{os.environ.get('USER', 'usuario')}@{socket.gethostname()}\n"
            f"  y conectar un cliente VNC a localhost:{local}\n"
            f"  (si ese puerto local esta ocupado, cambialo: el tunel no exige que "
            f"coincidan los dos lados)",
            # Sin flush el mensaje queda en el buffer cuando la salida no es una terminal,
            # y es justamente el que explica como conectarse.
            flush=True,
        )

    sesion = Session.open(video, buffer_mb=buffer_mb)
    if not sesion.source.verify_exact_seek():
        sesion.close()
        raise RuntimeError(
            f"{sesion.source.path.name} no permite navegacion exacta por numero de cuadro. "
            "No se puede anotar sobre este archivo: las fronteras temporales saldrian "
            "corridas. Regenerar el proxy con el preproceso."
        )

    ventana = MainWindow(sesion)
    if sesion.migrated:
        ventana.statusBar().showMessage(
            f"anotacion migrada desde la version {sesion.migrated[0]}, backup guardado", 6000
        )
    # La ventana se ajusta a la pantalla disponible en vez de a un tamano fijo. Sobre VNC
    # o en una notebook chica, un 1400x900 hardcodeado deja controles fuera de vista.
    disponible = app.primaryScreen().availableGeometry()
    ventana.resize(
        min(1400, disponible.width()), min(900, disponible.height())
    )
    ventana.show()
    return app.exec()
