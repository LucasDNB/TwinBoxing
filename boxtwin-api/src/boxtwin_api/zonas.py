"""
BoxTwin API - Zonas de ataque y de exposicion (cambios del 14-10, B16).

POR QUE EXISTE
  El perfil pide la distribucion de golpes por mano y por altura, y el mapa de exposicion a
  partir de los golpes del rival. Las dos cosas son la misma cuenta sobre golpes distintos (los
  propios y los del otro lado de cada sesion), y el Rival Fantasma las cruza zona por zona
  (B18). Vive en un solo modulo para que las tres pantallas cuenten igual.

  Decisiones (prototipo/specs/cambios-14-10/00-README.md):
    DC-2  el lado es la mano con que vino el golpe, no el lado del cuerpo que lo recibe. Las
          dos cosas son medidas (brazo del detector); el espejo al cuerpo seria una suposicion
    DC-3  la exposicion son golpes DIRIGIDOS a la zona. Una camara no ve si entraron
    DC-4  el detector encuentra menos golpes al cuerpo que a la cabeza (nivel 2 de C8, 14-10:
          0,512 contra 0,348). La regla de C9, pre-registrada en ampliacion-13-10/04 §3, dice
          que con mas de 0,10 de diferencia las fracciones se corrigen: cada zona dividida por
          el recall de su altura y renormalizada. Los conteos crudos no se tocan

QUE HACE
  contar_zonas(golpes)        conteo por zona, manos, alturas y sin dato
  corregir(zonas, recall)     fracciones corregidas por el sesgo del detector
  bloque(golpes)              el bloque que devuelve el perfil (ataque o exposicion)

USO
  from boxtwin_api.zonas import bloque
  perfil["ataque"] = bloque(golpes_propios)
"""

from __future__ import annotations

__all__ = ["DIFERENCIA_MAXIMA", "RECALL_ALTURA", "ZONAS", "bloque", "contar_zonas", "corregir",
           "sesgo_detector"]

ZONAS = ("cabeza_mano_izq", "cabeza_mano_der", "cuerpo_mano_izq", "cuerpo_mano_der")

# Nivel 2 de C8, una corrida del 14-10 sobre cuatro fuentes no usadas para calibrar:
# docs/experiments/2026-10-13-cabeza-cuerpo.md §7 (345 de 674 a la cabeza, 46 de 132 al cuerpo).
RECALL_ALTURA = {"cabeza": 0.512, "cuerpo": 0.348}
# La regla de C9 (ampliacion-13-10/04 §3): hasta esta diferencia el mapa va sin corregir.
DIFERENCIA_MAXIMA = 0.10

_MANO = {"izq": "izq", "der": "der", "left": "izq", "right": "der"}


def _hay_que_corregir(recall: dict) -> bool:
    return abs(recall["cabeza"] - recall["cuerpo"]) > DIFERENCIA_MAXIMA


def sesgo_detector(recall: dict | None = None) -> dict:
    recall = recall or RECALL_ALTURA
    return {"recall_cabeza": recall["cabeza"], "recall_cuerpo": recall["cuerpo"],
            "corregido": _hay_que_corregir(recall)}


def contar_zonas(golpes: list[dict]) -> dict:
    """
    Cuenta los golpes no descartados por zona. La altura es la vigente: si el entrenador la
    corrigio, el documento ya la trae corregida. Un golpe sin altura o sin mano va a sin dato
    para las zonas, pero su mano cuenta igual en `manos`, que no depende de la altura.
    """
    zonas = {z: 0 for z in ZONAS}
    manos = {"izq": 0, "der": 0}
    alturas = {"cabeza": 0, "cuerpo": 0, "sin_dato": 0}
    total = 0
    sin_dato = 0
    for g in golpes:
        if g.get("descartado"):
            continue
        total += 1
        mano = _MANO.get(g.get("brazo"))
        if mano:
            manos[mano] += 1
        altura = g.get("objetivo")
        if altura in ("cabeza", "cuerpo"):
            alturas[altura] += 1
        else:
            alturas["sin_dato"] += 1
        if altura in ("cabeza", "cuerpo") and mano:
            zonas[f"{altura}_mano_{mano}"] += 1
        else:
            sin_dato += 1
    return {"zonas": zonas, "manos": manos, "alturas": alturas, "total": total, "sin_dato": sin_dato}


def corregir(zonas: dict[str, int], recall: dict | None = None) -> dict[str, float | None]:
    """Fracciones por zona, divididas por el recall de su altura y renormalizadas (DC-4)."""
    recall = recall or RECALL_ALTURA
    if not _hay_que_corregir(recall):
        pesos = {z: float(n) for z, n in zonas.items()}
    else:
        pesos = {z: n / recall[z.split("_")[0]] for z, n in zonas.items()}
    suma = sum(pesos.values())
    return {z: (round(p / suma, 4) if suma else None) for z, p in pesos.items()}


def bloque(golpes: list[dict], recall: dict | None = None) -> dict:
    """El bloque de zonas del perfil: crudo, fraccion cruda y fraccion corregida por zona."""
    c = contar_zonas(golpes)
    con_dato = sum(c["zonas"].values())
    corregidas = corregir(c["zonas"], recall)
    return {
        "zonas": {
            z: {"n": n, "fraccion": round(n / con_dato, 4) if con_dato else None,
                "fraccion_corregida": corregidas[z]}
            for z, n in c["zonas"].items()
        },
        "manos": c["manos"],
        "alturas": c["alturas"],
        "total": c["total"],
        "sin_dato": c["sin_dato"],
        "con_dato": con_dato,
        "sesgo_detector": sesgo_detector(recall),
    }
