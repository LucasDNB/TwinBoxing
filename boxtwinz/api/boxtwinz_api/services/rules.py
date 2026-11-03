"""
BoxTwinZ - Consejos por reglas (spec 8).

POR QUE EXISTE
  Los consejos salen de los golpes detectados con reglas fijas, no de un modelo: cada regla
  tiene un disparador, una prioridad y una lista de frases. Las frases aprobadas por Lucas van
  primero; los umbrales son [PROVISIONAL] y estan en boxtwinz.toml.

  La rotacion de frases es hash(sesion_id) % len(frases), como pide la spec, pero con crc32:
  el hash() de Python cambia entre procesos para un str y la misma sesion mostraria consejos
  distintos despues de cada reinicio.

USO
  from boxtwinz_api.services.rules import consejos, Entrada
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from boxtwinz_api.config import cfg

__all__ = ["Entrada", "FRASES", "PRIORIDAD", "SIN_ALERTAS", "PIE", "consejos", "indice_frase"]

FRASES: dict[str, list[str]] = {
    "balance_manos": [
        "Estás trabajando mucho más la {adelantada} que la {atrasada}, tratemos de alternar las manos.",
    ],
    "caida_ritmo": [
        "Se detectó una caída del volumen del golpeo después del primer minuto, tratemos de "
        "mantener el ritmo constante todo el round.",
    ],
    "exposicion_lado": [
        "Contra los {estilo_plural} te entran por la {lado}. Cerrá ese lado y salí por el costado.",
    ],
    "golpes_sueltos": [
        "Tirás de a uno. Soltá las manos de a dos o tres y salí.",
    ],
    # PENDIENTE: frase a escribir por Lucas (spec 8). Mientras tanto, la provisoria.
    "volumen_bajo_vs_rival": [
        "Tu volumen está por debajo del de tu rival en esta modalidad.",
    ],
}
# Menor numero, mas arriba. Se muestran como mucho 3.
PRIORIDAD = {"exposicion_lado": 1, "volumen_bajo_vs_rival": 2, "caida_ritmo": 3,
             "balance_manos": 4, "golpes_sueltos": 5}
SIN_ALERTAS = "Sin alertas en esta sesión. Seguí así."
PIE = "Los consejos salen de los golpes detectados; no son una predicción."
PLURAL = {"fajador": "fajadores", "estilista": "estilistas", "hibrido": "híbridos"}


@dataclass
class Entrada:
    sesion_id: str
    golpes: int
    pct_mano_adelantada: float | None = None
    sosten_ritmo: float | None = None
    pct_golpes_en_combinacion: float | None = None
    guardia: str | None = "ortodoxa"
    # Exposicion: golpes dirigidos del rival por lado del usuario, y el estilo del rival.
    lado_izquierdo: int = 0
    lado_derecho: int = 0
    estilo_rival: str | None = None
    # Solo rival fantasma.
    volumen: float | None = None
    volumen_rival: float | None = None


def indice_frase(sesion_id: str, n: int) -> int:
    return zlib.crc32(sesion_id.encode()) % n if n else 0


def _frase(clave: str, sesion_id: str, **campos) -> dict:
    frases = FRASES[clave]
    texto = frases[indice_frase(sesion_id, len(frases))].format(**campos)
    return {"clave": clave, "prioridad": PRIORIDAD[clave], "texto": texto}


def consejos(e: Entrada, maximo: int = 3) -> dict:
    disparados: list[dict] = []
    if (e.pct_mano_adelantada is not None and e.pct_mano_adelantada > cfg.umbral_balance
            and e.golpes >= cfg.min_golpes_reglas):
        zurdo = e.guardia == "zurda"
        disparados.append(_frase("balance_manos", e.sesion_id,
                                 adelantada="derecha" if zurdo else "izquierda",
                                 atrasada="izquierda" if zurdo else "derecha"))
    if e.sosten_ritmo is not None and e.sosten_ritmo < cfg.umbral_caida_ritmo:
        disparados.append(_frase("caida_ritmo", e.sesion_id))
    tot = e.lado_izquierdo + e.lado_derecho
    if e.estilo_rival in PLURAL and tot >= cfg.min_golpes_rival_exposicion:
        izq, der = e.lado_izquierdo / tot, e.lado_derecho / tot
        if max(izq, der) >= cfg.umbral_exposicion_lado:
            disparados.append(_frase("exposicion_lado", e.sesion_id, estilo_plural=PLURAL[e.estilo_rival],
                                     lado="izquierda" if izq >= der else "derecha"))
    if (e.pct_golpes_en_combinacion is not None and e.pct_golpes_en_combinacion < cfg.umbral_golpes_sueltos
            and e.golpes >= cfg.min_golpes_reglas):
        disparados.append(_frase("golpes_sueltos", e.sesion_id))
    if (e.volumen is not None and e.volumen_rival is not None
            and e.volumen < cfg.umbral_volumen_vs_rival * e.volumen_rival):
        disparados.append(_frase("volumen_bajo_vs_rival", e.sesion_id))
    disparados.sort(key=lambda d: d["prioridad"])
    lista = disparados[:maximo] or [{"clave": "sin_alertas", "prioridad": 99, "texto": SIN_ALERTAS}]
    return {"consejos": lista, "pie": PIE}
