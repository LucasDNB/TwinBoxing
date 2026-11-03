"""
BoxTwinZ - Traduce la salida del motor (Fight-Card 0.3) a golpes izquierda/derecha.

POR QUE EXISTE
  El motor no se edita (spec 0.3): se invoca y se lee lo que deja en disco. La Fight-Card
  trae los golpes por peleador A/B con el brazo anatomico del detector ("izq"/"der"), y
  BoxTwinZ los necesita como usuario/rival e izquierda/derecha, sin tipo de golpe.

  El tipo que la Fight-Card pueda traer se ignora: BoxTwinZ no clasifica (spec 1). La altura
  (objetivo cabeza/cuerpo) la agrega `completar` con la regla de C8 sin pasar por el
  clasificador, asi que la capa opcional de 6.2 se puede leer sin tocar el motor.

QUE HACE
  Lee fightcard.json y sesion.json del directorio de trabajo. Devuelve los golpes con
  identidad resuelta (A = usuario, B = rival, por como siembra el worker), la calidad de pose
  (cobertura de A) y la fraccion de cuadros sin identidad, que es lo que el contrato permite
  saber de los golpes sin identidad (desvio D6).

USO
  from boxtwinz_api.services.engine_adapter import leer_salida
  salida = leer_salida(cfg.dir_trabajo(sesion_id))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["GolpeMotor", "SalidaMotor", "leer_salida", "leer_sesion_json", "candidato_individual"]

_MANO = {"izq": "izquierda", "der": "derecha", "left": "izquierda", "right": "derecha"}
_QUIEN = {"A": "usuario", "B": "rival"}


@dataclass
class GolpeMotor:
    quien: str
    mano: str
    t_inicio: float
    t_fin: float
    confianza: float
    altura_estimada: str | None
    id_motor: str | None


@dataclass
class SalidaMotor:
    golpes: list[GolpeMotor] = field(default_factory=list)
    calidad_pose: float | None = None
    fraccion_sin_identidad: float | None = None
    duracion_s: float | None = None
    fps: float | None = None
    descartados: int = 0


def leer_sesion_json(directorio: Path) -> dict:
    ruta = directorio / "sesion.json"
    if not ruta.is_file():
        return {}
    try:
        return json.loads(ruta.read_text())
    except (OSError, ValueError):
        return {}


def leer_salida(directorio: Path) -> SalidaMotor:
    ruta = directorio / "fightcard.json"
    fc = json.loads(ruta.read_text())
    if str(fc.get("version", "")).split(".")[:2] != ["0", "3"]:
        raise ValueError(f"contrato de Fight-Card no soportado: {fc.get('version')!r}, se espera 0.3")
    out = SalidaMotor()
    video = fc.get("video") or {}
    out.duracion_s = video.get("duracion_s")
    out.fps = video.get("fps")
    ident = fc.get("identidad") or {}
    if isinstance(ident.get("cobertura_A"), (int, float)):
        out.calidad_pose = float(ident["cobertura_A"])
    if isinstance(ident.get("sin_asignar"), (int, float)):
        out.fraccion_sin_identidad = float(ident["sin_asignar"])

    for lado, pel in (fc.get("peleadores") or {}).items():
        quien = _QUIEN.get(lado)
        if quien is None:
            continue
        for g in pel.get("golpes", []):
            mano = _MANO.get(g.get("brazo"))
            if mano is None:
                continue
            # Un golpe que alguien descarto en A no es un golpe detectado.
            if g.get("descartado"):
                out.descartados += 1
                continue
            obj = g.get("objetivo_corregido") or g.get("objetivo")
            out.golpes.append(GolpeMotor(
                quien=quien, mano=mano,
                t_inicio=float(g["t_inicio"]), t_fin=float(g["t_fin"]),
                confianza=float(g.get("score", 0.0)),
                altura_estimada=obj if obj in ("cabeza", "cuerpo") else None,
                id_motor=g.get("id"),
            ))
    out.golpes.sort(key=lambda x: (x.quien, x.t_inicio))
    return out


def candidato_individual(sesion_json: dict) -> int | None:
    """
    La semilla automatica de una sesion de sombra o bolsa (desvio D3).

    El motor pide una semilla A aunque haya una sola persona (proponer_individual). La spec
    dice que en estas modalidades no hay siembra, asi que se elige sola: el candidato que el
    motor marca como sugerido y que mas tiempo esta en cuadro; sin sugeridos, el de mayor
    permanencia. Si en el video hay otra persona mas tiempo que el boxeador, esto se equivoca,
    y por eso bolsa y sombra llevan "Reconocimiento en evaluacion".
    """
    cands = sesion_json.get("candidatos") or []
    if not cands:
        return None

    def permanencia(c: dict) -> float:
        return float(c.get("t_hasta", 0) or 0) - float(c.get("t_desde", 0) or 0)

    sugeridos = [c for c in cands if c.get("sugerido")]
    elegido = max(sugeridos or cands, key=lambda c: (permanencia(c), c.get("alto_relativo") or 0))
    return int(elegido["track"])
