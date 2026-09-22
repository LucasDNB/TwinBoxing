"""
BoxTwin - El estado de una sesion de analisis.

POR QUE EXISTE
  El procesamiento se corta en dos por una razon de producto y no de ingenieria: entre la
  pose y el detector hay una pregunta que solo una persona puede contestar, que es cual de
  los dos cuerpos es cual. Medido, la diferencia entre resolver la identidad sola y
  resolverla con esa respuesta es 82,4% contra 99,1% de los tracks.

  Eso obliga a que el trabajo sea reanudable en un punto intermedio y a que ese punto sea
  explicito. Un pipeline de un solo tiro no puede parar a preguntar, y guardar el estado en
  la base de datos de la API ataria el comando de linea a tener una base al lado. El estado
  vive en el directorio de la sesion, en un archivo, y la API lo lee.

  Los tiempos por etapa se registran porque RNF1 -no mas de 2x la duracion del video- es un
  criterio de aceptacion, y medirlo despues a mano sobre el reloj de pared mezcla el
  procesamiento con la espera humana de la siembra, que no es tiempo de maquina.

QUE HACE
  Define los estados por los que pasa una sesion, los lee y los escribe, y calcula las
  ventanas de round a partir de lo que declaro el usuario.

USO
  from boxtwin.mvp.sesion import Sesion
  s = Sesion.nueva(dir_sesion, video_meta, rounds=(180, 60))
  s.estado = "espera_siembra"
  s.guardar()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["ESTADOS", "NOMBRE_ARCHIVO", "Sesion", "ventanas_de_round"]

NOMBRE_ARCHIVO = "sesion.json"
VERSION = "0.1"

# El orden es el del flujo y no alfabetico: cada estado solo puede venir del anterior.
ESTADOS = (
    "procesando",       # pose, tracking y evidencia de identidad
    "espera_siembra",   # hay candidatos y falta que una persona diga cual es cual
    "completando",      # color, detector, indicadores
    "listo",            # hay fightcard.json
    "fallo",
)


def ventanas_de_round(
    duracion_s: float, round_s: float | None, descanso_s: float
) -> list[dict]:
    """
    Las ventanas de round a partir de lo que declaro el usuario.

    No se detectan: el gong no se ve en el video y un detector de campana es un componente
    nuevo con su propia validacion. El usuario sabe cuanto dura su round y decirlo cuesta dos
    campos. Sin rounds declarados la lista sale vacia y la Fight-Card se agrega por sesion.

    El ultimo round se recorta a lo que dure el video: si la sesion corto a mitad de round,
    inventarle el final completo le pone al denominador de golpes por minuto un tiempo que
    no existio.
    """
    if not round_s or round_s <= 0 or duracion_s <= 0:
        return []
    ventanas = []
    t = 0.0
    n = 1
    while t < duracion_s - 1e-9:
        fin = min(t + round_s, duracion_s)
        ventanas.append({"round": n, "inicio_s": round(t, 3), "fin_s": round(fin, 3)})
        t = fin + max(descanso_s, 0.0)
        n += 1
    return ventanas


@dataclass
class Sesion:
    """El estado de una sesion, tal como vive en `sesion.json`."""

    directorio: Path
    estado: str = "procesando"
    version: str = VERSION
    video: dict[str, Any] = field(default_factory=dict)
    rounds: dict[str, Any] = field(default_factory=dict)
    etapas: list[dict] = field(default_factory=list)
    candidatos: list[dict] = field(default_factory=list)
    siembra: dict[str, Any] | None = None
    identidad: dict[str, Any] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    error: str | None = None
    creada: str = ""
    actualizada: str = ""

    # -- io -----------------------------------------------------------------

    @classmethod
    def ruta_de(cls, directorio: Path) -> Path:
        return Path(directorio) / NOMBRE_ARCHIVO

    @classmethod
    def nueva(
        cls, directorio: Path, video: dict, round_s: float | None, descanso_s: float
    ) -> Sesion:
        ahora = datetime.now().astimezone().isoformat()
        return cls(
            directorio=Path(directorio),
            video=video,
            rounds={
                "round_s": round_s,
                "descanso_s": descanso_s,
                "ventanas": ventanas_de_round(
                    float(video.get("duracion_s", 0.0)), round_s, descanso_s
                ),
            },
            creada=ahora,
            actualizada=ahora,
        )

    @classmethod
    def cargar(cls, directorio: Path) -> Sesion:
        ruta = cls.ruta_de(directorio)
        if not ruta.is_file():
            raise FileNotFoundError(
                f"no hay sesion en {directorio}: falta {NOMBRE_ARCHIVO}. "
                "Corre primero 'boxtwin-annotator procesar'"
            )
        d = json.loads(ruta.read_text())
        campos = {f for f in cls.__dataclass_fields__ if f != "directorio"}
        return cls(directorio=Path(directorio), **{k: v for k, v in d.items() if k in campos})

    def guardar(self) -> Path:
        if self.estado not in ESTADOS:
            raise ValueError(f"estado desconocido: {self.estado!r}")
        self.actualizada = datetime.now().astimezone().isoformat()
        d = {k: v for k, v in self.__dict__.items() if k != "directorio"}
        ruta = self.ruta_de(self.directorio)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atomica: la API lee este archivo mientras el worker lo escribe, y un
        # json a medias es un estado que no existe.
        tmp = ruta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        tmp.replace(ruta)
        return ruta

    # -- etapas -------------------------------------------------------------

    def anotar_etapa(self, nombre: str, segundos: float, **extra: Any) -> None:
        self.etapas.append(
            {
                "etapa": nombre,
                "segundos": round(float(segundos), 2),
                "cuando": datetime.now().astimezone().isoformat(),
                **extra,
            }
        )

    @property
    def segundos_de_maquina(self) -> float:
        """Suma de las etapas. No es el reloj de pared: la espera humana no cuenta."""
        return round(sum(float(e.get("segundos", 0.0)) for e in self.etapas), 2)

    @property
    def factor_tiempo_real(self) -> float | None:
        """Cuantas veces la duracion del video costo procesarlo. RNF1 pide 2 o menos."""
        dur = float(self.video.get("duracion_s") or 0.0)
        if dur <= 0:
            return None
        return round(self.segundos_de_maquina / dur, 3)

    @property
    def ventanas(self) -> list[dict]:
        return list(self.rounds.get("ventanas") or [])
