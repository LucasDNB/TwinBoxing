"""
BoxTwin API - Configuracion, toda por variables de entorno.

POR QUE EXISTE
  El worker corre adentro de un contenedor con GPU y la API afuera, o los dos adentro, o
  los dos afuera durante el desarrollo. Lo unico que cambia entre esos tres casos son
  rutas y comandos, asi que estan todos en un solo lugar y ninguno cableado.

  El comando de clasificacion es variable de entorno por una razon concreta del plan: si la
  imagen con los dos entornos conda no sale a tiempo, la clasificacion corre afuera del
  contenedor, en el entorno local, leyendo el mismo JSON. Eso tiene que ser un cambio de
  configuracion y no un cambio de codigo.

QUE HACE
  Lee el entorno una vez y expone valores tipados con defaults que sirven para desarrollo.

USO
  from boxtwin_api.config import cfg
  cfg.base_de_datos
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "cfg"]


def _ruta(nombre: str, defecto: str) -> Path:
    return Path(os.environ.get(nombre, defecto)).expanduser()


def _lista(nombre: str, defecto: str) -> list[str]:
    return [x for x in os.environ.get(nombre, defecto).split() if x]


@dataclass
class Config:
    # sqlite alcanza para desarrollo y para los tests. En produccion es PostgreSQL, que es
    # lo que hace falta para SELECT ... FOR UPDATE SKIP LOCKED con mas de un worker.
    base_de_datos: str = field(
        default_factory=lambda: os.environ.get("BOXTWIN_DB", "sqlite:///boxtwin.db")
    )
    datos: Path = field(default_factory=lambda: _ruta("BOXTWIN_DATOS", "./datos"))

    # Un secreto que se genera solo sirve para desarrollo y hay que decirlo: al reiniciar,
    # todas las sesiones abiertas se caen. En produccion se pasa por entorno.
    secreto: str = field(
        default_factory=lambda: os.environ.get("BOXTWIN_SECRETO") or secrets.token_hex(32)
    )
    secreto_efimero: bool = field(
        default_factory=lambda: not os.environ.get("BOXTWIN_SECRETO")
    )
    horas_de_sesion: int = field(
        default_factory=lambda: int(os.environ.get("BOXTWIN_HORAS_SESION", "72"))
    )

    # Limite de subida. 30 minutos de 1080p desde un celular andan por los 3 GB; el limite
    # esta puesto sobre la duracion (RF1) y esto es solo la red de contencion de disco.
    max_bytes: int = field(
        default_factory=lambda: int(os.environ.get("BOXTWIN_MAX_BYTES", str(6 * 1024**3)))
    )
    max_minutos: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_MAX_MINUTOS", "30"))
    )
    formatos: tuple[str, ...] = (".mp4", ".mov", ".m4v")

    # Comandos de cada etapa. Son listas porque van a subprocess sin shell.
    cmd_boxtwin: list[str] = field(
        default_factory=lambda: _lista("BOXTWIN_CMD", "boxtwin-annotator")
    )
    cmd_clasificador: list[str] = field(
        default_factory=lambda: _lista("BOXTWIN_CMD_CLASIFICADOR", "")
    )
    modelo_guantes: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_GUANTES", "modelos/guantes-v2.pt")
    )
    modelo_detector: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_DETECTOR", "modelos/detector.pt")
    )
    config_clasificador: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_CONFIG_CLASIFICADOR", "modelos/poseC3D.py")
    )
    checkpoint_clasificador: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_CHECKPOINT_CLASIFICADOR", "modelos/poseC3D.pth")
    )
    umbral_detector: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_UMBRAL", "0.80"))
    )
    device: str = field(default_factory=lambda: os.environ.get("BOXTWIN_DEVICE", "0"))

    # Cuanto espera el worker entre sondeos cuando la cola esta vacia.
    espera_s: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_ESPERA", "2.0"))
    )
    # Un trabajo tomado por un worker que se murio tiene que volver a la cola. Sin esto, un
    # kill -9 en medio del preproceso deja la sesion colgada para siempre.
    minutos_para_reclamar: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_TIMEOUT_MIN", "180"))
    )
    intentos_maximos: int = field(
        default_factory=lambda: int(os.environ.get("BOXTWIN_INTENTOS", "2"))
    )

    def dir_sesion(self, sesion_id: str) -> Path:
        return self.datos / "sesiones" / sesion_id


cfg = Config()
