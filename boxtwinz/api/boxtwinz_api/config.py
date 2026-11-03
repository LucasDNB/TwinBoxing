"""
BoxTwinZ API - Configuracion: boxtwinz.toml mas variables de entorno.

POR QUE EXISTE
  La spec (0.7) pide que todo valor [PROVISIONAL] viva en configuracion y no en el codigo:
  cupos, umbrales de reglas, N de tendencia, calidad de pose. Lucas los valida despues y
  cambiarlos tiene que ser editar un archivo, no un commit.

  Las rutas y comandos del motor se copian de boxtwin-api (config.py de A) con los mismos
  defaults, para que el worker invoque el motor igual que el de A.

QUE HACE
  Lee BOXTWINZ_CONFIG (o ./boxtwinz.toml si existe), aplica encima las variables BOXTWINZ_*
  y expone un objeto `cfg`. Sin archivo, los defaults son los de la spec.

USO
  from boxtwinz_api.config import cfg
  cfg.cupo_minutos["free"]
"""

from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "cargar", "cfg"]


def _lista(valor: str) -> list[str]:
    return [x for x in valor.split() if x]


def _bandera(valor: str | None, defecto: bool) -> bool:
    if valor is None or not valor.strip():
        return defecto
    return valor.strip().lower() in ("1", "si", "true", "on")


@dataclass
class Config:
    base_de_datos: str = "sqlite:///boxtwinz.db"
    datos: Path = Path("~/boxtwinz-datos").expanduser()
    secreto: str = ""
    secreto_efimero: bool = True
    web: Path | None = None
    admins: tuple[str, ...] = ()
    dias_de_sesion: int = 30
    # Cookie Secure: apagada en desarrollo (http://localhost), encendida detras del tunel.
    cookie_segura: bool = False
    max_bytes: int = 6 * 1024**3
    max_minutos: float = 30.0
    formatos: tuple[str, ...] = (".mp4", ".mov", ".m4v")

    # -- Motor: copiado de boxtwin-api/config.py, mismos nombres de variable con BOXTWINZ_ --
    cmd_boxtwin: list[str] = field(default_factory=lambda: ["boxtwin-annotator"])
    modelo_guantes: Path = Path("modelos/guantes-v2.pt")
    modelo_pose: Path = Path("yolov8l-pose.pt")
    modelo_detector: Path = Path("modelos/detector.pt")
    umbral_detector: float = 0.80
    device: str = "0"
    # Las tres banderas de identidad estable, con los defaults de A al 3-11.
    corte_mixtos: bool = False
    identidad_estricta: bool = False
    identidad_apariencia: bool = True
    espera_s: float = 2.0
    minutos_para_reclamar: float = 180.0
    intentos_maximos: int = 2
    lock_gpu: Path = Path("/tmp/boxtwin-gpu.lock")

    # -- [PROVISIONAL] (spec 6.3, 6.4, 8, 9, 10) -------------------------------------------
    cupo_minutos: dict[str, float] = field(default_factory=lambda: {"free": 60.0, "premium": 300.0})
    tendencia_n: int = 10
    tendencia_banda: float = 0.05
    calidad_pose_min: float = 0.70
    duracion_ranking_s: float = 180.0
    umbral_balance: float = 0.70
    umbral_caida_ritmo: float = 0.80
    umbral_exposicion_lado: float = 0.65
    min_golpes_rival_exposicion: int = 20
    umbral_golpes_sueltos: float = 0.30
    min_golpes_reglas: int = 30
    umbral_volumen_vs_rival: float = 0.80
    gap_combinacion_s: float = 0.5
    factor_volumen_individual: float = 1.10
    capacidad_umbral_alto: float = 0.70
    capacidad_dias_alto: int = 15
    capacidad_umbral_critico: float = 0.90
    capacidad_dias_critico: int = 5

    version_consentimiento_uso: str = "uso-2026-11-03"
    version_consentimiento_reent: str = "reent-2026-11-03"

    def dir_trabajo(self, sesion_id: str) -> Path:
        return self.datos / "trabajos" / sesion_id

    def dir_videos(self) -> Path:
        return self.datos / "videos"


# Las claves de [provisional] que se aceptan en el toml, con su tipo.
_PROVISIONALES = {
    "tendencia_n": int, "tendencia_banda": float, "calidad_pose_min": float,
    "duracion_ranking_s": float, "umbral_balance": float, "umbral_caida_ritmo": float,
    "umbral_exposicion_lado": float, "min_golpes_rival_exposicion": int,
    "umbral_golpes_sueltos": float, "min_golpes_reglas": int,
    "umbral_volumen_vs_rival": float, "gap_combinacion_s": float,
    "factor_volumen_individual": float, "capacidad_umbral_alto": float,
    "capacidad_dias_alto": int, "capacidad_umbral_critico": float,
    "capacidad_dias_critico": int,
}


def cargar(ruta: Path | None = None, entorno: dict | None = None) -> Config:
    """Arma la configuracion. `entorno` existe para los tests; por defecto os.environ."""
    env = os.environ if entorno is None else entorno
    c = Config()
    ruta = ruta or (Path(env["BOXTWINZ_CONFIG"]) if env.get("BOXTWINZ_CONFIG") else Path("boxtwinz.toml"))
    datos: dict = {}
    if ruta.is_file():
        datos = tomllib.loads(ruta.read_text())

    prov = datos.get("provisional", {})
    for k, tipo in _PROVISIONALES.items():
        if k in prov:
            setattr(c, k, tipo(prov[k]))
    if "cupo_minutos" in prov:
        c.cupo_minutos = {**c.cupo_minutos, **{k: float(v) for k, v in prov["cupo_minutos"].items()}}

    motor = datos.get("motor", {})
    serv = datos.get("servidor", {})

    def valor(clave_env: str, seccion: dict, clave: str, defecto):
        if env.get(clave_env):
            return env[clave_env]
        return seccion.get(clave, defecto)

    c.base_de_datos = valor("BOXTWINZ_DB", serv, "base_de_datos", c.base_de_datos)
    c.datos = Path(str(valor("BOXTWINZ_DATOS", serv, "datos", str(c.datos)))).expanduser()
    w = valor("BOXTWINZ_WEB", serv, "web", "")
    c.web = Path(w).expanduser() if w else None
    c.secreto = env.get("BOXTWINZ_SECRETO") or ""
    c.secreto_efimero = not c.secreto
    if c.secreto_efimero:
        # Igual que A: un secreto generado solo sirve para desarrollo. Al reiniciar se caen
        # todas las sesiones abiertas.
        c.secreto = secrets.token_hex(32)
    c.admins = tuple(e.strip().lower() for e in env.get("BOXTWINZ_ADMINS", "").replace(",", " ").split() if e.strip())
    c.cookie_segura = _bandera(env.get("BOXTWINZ_COOKIE_SEGURA"), bool(serv.get("cookie_segura", False)))

    cmd = valor("BOXTWINZ_CMD", motor, "cmd", "boxtwin-annotator")
    c.cmd_boxtwin = _lista(cmd) if isinstance(cmd, str) else list(cmd)
    c.modelo_guantes = Path(str(valor("BOXTWINZ_MODELO_GUANTES", motor, "modelo_guantes", c.modelo_guantes))).expanduser()
    c.modelo_pose = Path(str(valor("BOXTWINZ_MODELO_POSE", motor, "modelo_pose", c.modelo_pose))).expanduser()
    c.modelo_detector = Path(str(valor("BOXTWINZ_MODELO_DETECTOR", motor, "modelo_detector", c.modelo_detector))).expanduser()
    c.umbral_detector = float(valor("BOXTWINZ_UMBRAL", motor, "umbral", c.umbral_detector))
    c.device = str(valor("BOXTWINZ_DEVICE", motor, "device", c.device))
    c.corte_mixtos = _bandera(env.get("BOXTWINZ_CORTE_MIXTOS"), bool(motor.get("corte_mixtos", False)))
    c.identidad_estricta = _bandera(env.get("BOXTWINZ_IDENTIDAD_ESTRICTA"), bool(motor.get("identidad_estricta", False)))
    c.identidad_apariencia = _bandera(env.get("BOXTWINZ_IDENTIDAD_APARIENCIA"), bool(motor.get("identidad_apariencia", True)))
    c.lock_gpu = Path(str(valor("BOXTWINZ_LOCK_GPU", motor, "lock_gpu", c.lock_gpu)))
    c.espera_s = float(valor("BOXTWINZ_ESPERA", motor, "espera_s", c.espera_s))
    c.intentos_maximos = int(valor("BOXTWINZ_INTENTOS", motor, "intentos", c.intentos_maximos))
    return c


cfg = cargar()
