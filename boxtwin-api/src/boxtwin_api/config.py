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


ESTADOS_CRITERIO = ("pendiente", "aprobado", "rechazado")


def _criterio(nombre: str) -> str:
    """El estado de un criterio pre-registrado. Un valor mal escrito cuenta como pendiente."""
    v = os.environ.get(nombre, "pendiente").strip().lower()
    return v if v in ESTADOS_CRITERIO else "pendiente"


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

    # Quien puede crearse una cuenta. El default es CERRADO y es a proposito: apenas esto
    # sale por un tunel, una instancia con el registro abierto es una GPU ajena gratis para
    # cualquiera que tenga la URL. Abrirlo es una decision, no un olvido.
    #
    #   sin setear  -> nadie se registra, y el error dice como habilitarlo
    #   "abierto"   -> cualquiera, que es lo que se quiere en desarrollo
    #   otra cosa   -> es el codigo de invitacion y hay que presentarlo
    invitacion: str = field(
        default_factory=lambda: os.environ.get("BOXTWIN_INVITACION", "")
    )

    # La documentacion interactiva de FastAPI. Apagada por default por el mismo criterio
    # que el registro: no filtra datos -toda ruta con datos pide token- pero publica la
    # lista completa de endpoints a cualquiera que tenga la URL, y detras de un tunel eso
    # es superficie que nadie pidio. En desarrollo sirve, asi que se prende con una
    # variable en vez de desaparecer del codigo.
    docs: bool = field(
        default_factory=lambda: os.environ.get("BOXTWIN_DOCS", "") not in ("", "0", "no")
    )

    # El frontend construido. Servirlo desde la misma API deja UN solo origen, que es lo que
    # hace que alcance con un tunel y que no haga falta CORS en produccion.
    web: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["BOXTWIN_WEB"]).expanduser() if os.environ.get("BOXTWIN_WEB")
            else None
        )
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
    # Modo academico E7 (25-10, docs/specs/2026-10-25-e7-modo-academico): la elevacion con
    # HMR2 corre en boxtwin_4dhumans (tools/elevar_hmr2_sesion.py). Vacio = el modo no corre.
    cmd_hmr2: list[str] = field(
        default_factory=lambda: _lista("BOXTWIN_CMD_HMR2", "")
    )
    modelo_guantes: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_GUANTES", "modelos/guantes-v2.pt")
    )
    # Explicito y no el default de ultralytics: sin esto la ruta se resuelve contra el
    # directorio de trabajo del proceso, que en un servicio de systemd no es el que uno
    # cree. Costo una corrida.
    modelo_pose: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_POSE", "yolov8l-pose.pt")
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
    # Tipo de golpe v2 (docs/specs/2026-10-16-fightcard-tipo-v2): "geometrico" clasifica la
    # familia con el modelo de E4 en el entorno del anotador, sin boxtwin_mmaction. Por
    # defecto "poseconv3d", el comportamiento de antes; lo enciende Lucas. Un valor mal
    # escrito cuenta como el de antes. "e7" (25-10) es el modo academico: E4 mas el 3D de
    # HMR2, con SMPL bajo licencia de investigacion. Nunca para usuarios (enmienda, 2 y 3).
    clasificador: str = field(
        default_factory=lambda: (
            v if (v := os.environ.get("BOXTWIN_CLASIFICADOR", "").strip().lower()) in ("geometrico", "e7")
            else "poseconv3d"
        )
    )
    modelo_clasificador_geo: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_CLASIFICADOR_GEO", "~/boxtwin/modelos/clasificador_geo_v1")
    )
    modelo_clasificador_e7: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_CLASIFICADOR_E7", "~/boxtwin/modelos/clasificador_e7_v1")
    )
    # Bolsa y sombra (16-10): el modelo de E4a, sin las 6 features del rival. Con el de E4 una
    # sesion de una sola persona entraria con el rival en cero todo el golpe.
    modelo_clasificador_geo_individual: Path = field(
        default_factory=lambda: _ruta("BOXTWIN_MODELO_CLASIFICADOR_GEO_INDIVIDUAL",
                                      "~/boxtwin/modelos/clasificador_geo_v1_sin_rival")
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

    # -- Lo que la web lee de GET /producto (05 B5) -------------------------------------
    #
    # El estado de los criterios pre-registrados lo cambia Lucas cuando mide, por entorno y
    # con fecha. La web no decide nada sobre esto: las banderas salen de aca.
    c2: str = field(default_factory=lambda: _criterio("BOXTWIN_C2"))
    c2_fecha: str | None = field(default_factory=lambda: os.environ.get("BOXTWIN_C2_FECHA") or None)
    c3: str = field(default_factory=lambda: _criterio("BOXTWIN_C3"))
    c3_fecha: str | None = field(default_factory=lambda: os.environ.get("BOXTWIN_C3_FECHA") or None)
    # Ampliacion del 13-10 (08 §3): altura del golpe, sesgo del detector por zona, avatar.
    # C8 se midio el 13-10 y no paso (0,726): queda "rechazado" cuando Lucas lo cargue.
    c8: str = field(default_factory=lambda: _criterio("BOXTWIN_C8"))
    c8_fecha: str | None = field(default_factory=lambda: os.environ.get("BOXTWIN_C8_FECHA") or None)
    c9: str = field(default_factory=lambda: _criterio("BOXTWIN_C9"))
    c9_fecha: str | None = field(default_factory=lambda: os.environ.get("BOXTWIN_C9_FECHA") or None)
    c10: str = field(default_factory=lambda: _criterio("BOXTWIN_C10"))
    c10_fecha: str | None = field(default_factory=lambda: os.environ.get("BOXTWIN_C10_FECHA") or None)
    # Altura del golpe por decision de producto (14-10, Lucas): se muestra con la regla
    # medida aunque C8 no haya pasado, rotulada como estimacion y con lo medido al lado, igual
    # que el clasificador de familia. C8 no se toca: sigue diciendo lo que se midio. Con 0 la
    # bandera vuelve a depender solo de C8.
    objetivo_estimado: bool = field(
        default_factory=lambda: os.environ.get("BOXTWIN_OBJETIVO_ESTIMADO", "1") in ("1", "si")
    )
    # Comparacion entre rounds por decision de producto (cambios-14-10, DC-1): se muestra
    # rotulada como sin validar aunque C2 este pendiente. C2 no se toca. Con 0, depende de C2.
    rounds_estimado: bool = field(
        default_factory=lambda: os.environ.get("BOXTWIN_ROUNDS_ESTIMADO", "1") in ("1", "si")
    )
    # B8: se enciende solo despues de la medicion con criterio fijado en el spec.
    sesiones_individuales: bool = field(
        default_factory=lambda: os.environ.get("BOXTWIN_SESIONES_INDIVIDUALES", "") in ("1", "si")
    )
    # Factor de procesamiento medido en la campana de costo del Cap 5 (D2). Reemplaza el
    # "<= 20 min" del diseno, que no salia de ninguna medicion. Se actualiza al medir C5.
    factor_min: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_FACTOR_MIN", "1.134"))
    )
    factor_mediana: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_FACTOR_MEDIANA", "1.447"))
    )
    factor_max: float = field(
        default_factory=lambda: float(os.environ.get("BOXTWIN_FACTOR_MAX", "2.624"))
    )
    # Decisiones de dominio de Lucas. Valores iniciales del spec, a confirmar antes de la
    # prueba con usuarios (07 §3).
    perfil_preliminar_sesiones: int = field(
        default_factory=lambda: int(os.environ.get("BOXTWIN_PERFIL_PRELIMINAR", "5"))
    )

    @property
    def modo_registro(self) -> str:
        if not self.invitacion:
            return "cerrado"
        return "abierto" if self.invitacion == "abierto" else "invitacion"

    @property
    def tipo_v2(self) -> bool:
        """El tipo sale del modelo geometrico (E4, o E7 en modo academico): la guardia cuenta."""
        return self.clasificador in ("geometrico", "e7")

    @property
    def hay_clasificador(self) -> bool:
        """Si la etapa clasificar se puede correr: v2 no necesita BOXTWIN_CMD_CLASIFICADOR."""
        return self.tipo_v2 or bool(self.cmd_clasificador)

    def dir_sesion(self, sesion_id: str) -> Path:
        return self.datos / "sesiones" / sesion_id


cfg = Config()
