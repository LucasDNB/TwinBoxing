"""
BoxTwinZ - Arquetipos: carga del seed y metricas sinteticas (spec 7).

POR QUE EXISTE
  Las ranuras 2 a 4 de un usuario free son arquetipos fijos que sirven de rival fantasma. Sus
  numeros no salen de ninguna medicion: son [PROVISIONAL] y se muestran siempre con el rotulo
  "Perfil simulado". Vive aparte para que la generacion sea una sola y testeable.

QUE HACE
  Lee seeds/arquetipos.yaml, arma medias y desvios por modalidad (sombra y bolsa con el
  factor de volumen individual de la config) y una curva de ritmo sintetica coherente con el
  sosten: un round de 180 s cuyo primer minuto tiene el volumen v1 y el resto v1 * sosten,
  con el promedio igual al volumen declarado.

USO
  from boxtwinz_api.services.archetypes import cargar_seed, metricas_arquetipo
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from boxtwinz_api.config import cfg
from boxtwinz_api.models import MODALIDADES, Arquetipo

__all__ = ["ruta_seed", "leer_seed", "cargar_seed", "construir", "curva_sintetica",
           "metricas_arquetipo", "BASE"]

BASE = ("fajador", "estilista", "hibrido")
ROTULO = "Perfil simulado"


def ruta_seed() -> Path:
    if os.environ.get("BOXTWINZ_SEED"):
        return Path(os.environ["BOXTWINZ_SEED"])
    return Path(__file__).resolve().parents[3] / "seeds" / "arquetipos.yaml"


def leer_seed(ruta: Path | None = None) -> list[dict]:
    return yaml.safe_load((ruta or ruta_seed()).read_text())["arquetipos"]


def curva_sintetica(volumen: float, sosten: float, duracion_s: float = 180.0) -> list[float]:
    """
    Golpes por bloque de 10 s. Con D = duracion en minutos, el primer minuto a v1 y el resto a
    v1*sosten, y v1 sale de pedir que el promedio sea `volumen`:
        v1 * (1 + sosten * (D - 1)) / D = volumen
    """
    d = duracion_s / 60.0
    v1 = volumen * d / (1 + sosten * (d - 1))
    bloques = int(duracion_s // 10)
    return [round((v1 if i < 6 else v1 * sosten) / 6.0, 3) for i in range(bloques)]


def construir(item: dict, factor_individual: float | None = None) -> dict:
    """Los campos de la tabla `arquetipos` a partir de una entrada del seed."""
    f = cfg.factor_volumen_individual if factor_individual is None else factor_individual
    desvio = float(item.get("desvio", 0.1))
    metricas: dict = {}
    for mod in MODALIDADES:
        k = 1.0 if mod == "sparring" else f
        medias = {
            "volumen": item["volumen"] * k,
            "pct_mano_adelantada": item["pct_mano_adelantada"],
            # Ortodoxa supuesta (spec 6.2): la mano adelantada es la izquierda.
            "pct_izquierda": item["pct_mano_adelantada"],
            "combinaciones_min": item["combinaciones_min"] * k,
            "sosten_ritmo": item["sosten_ritmo"],
        }
        metricas[mod] = {"medias": medias,
                         "desvios": {m: round(v * desvio, 4) for m, v in medias.items()}}
    return {
        "clave": item["clave"], "nombre": item["nombre"], "estilo": item["estilo"],
        "variante": item.get("variante"), "solo_premium": bool(item.get("solo_premium")),
        "metricas": metricas,
        "exposicion": {"izquierda": item["exposicion"]["izquierda"],
                       "derecha": item["exposicion"]["derecha"]},
        "curva_ritmo": curva_sintetica(item["volumen"], item["sosten_ritmo"]),
    }


def cargar_seed(db: Session, ruta: Path | None = None) -> int:
    """Inserta o actualiza los arquetipos del catalogo. Idempotente. No commitea."""
    n = 0
    for item in leer_seed(ruta):
        datos = construir(item)
        a = db.scalar(select(Arquetipo).where(Arquetipo.clave == datos["clave"]))
        if a is None:
            db.add(Arquetipo(**datos))
        else:
            for k, v in datos.items():
                setattr(a, k, v)
        n += 1
    db.flush()
    return n


def metricas_arquetipo(a: Arquetipo, modalidad: str) -> dict:
    """Las medias de una modalidad, con los rotulos que la interfaz tiene que mostrar."""
    m = (a.metricas or {}).get(modalidad) or {}
    return {**(m.get("medias") or {}), "rotulo": ROTULO, "guardia_supuesta": True}
