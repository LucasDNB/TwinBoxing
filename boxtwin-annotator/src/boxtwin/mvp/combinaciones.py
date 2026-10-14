"""
BoxTwin - Combinaciones izquierda-derecha-izquierda (B10 de la ampliacion del 13-10).

POR QUE EXISTE
  B7 contaba las cinco secuencias de 2 o 3 golpes mas frecuentes nombradas por TIPO, y el
  tipo sale de un clasificador que no generaliza. El pedido del 13-10 es otro: contar solo
  izquierda-derecha-izquierda con menos de 0,5 s entre golpes. El patron se define por el
  BRAZO, que sale del detector y esta medido, asi que el numero deja de ser estimacion. El
  tipo de cada golpe viaja al lado como dato secundario.

  Las tres decisiones de dominio estan tomadas en el spec (ampliacion-13-10/01 §2) y viajan
  adentro de la regla, para que una Fight-Card vieja diga con que regla se conto:
    DA-1  la separacion se mide de inicio a inicio, la frontera mas precisa del detector
    DA-2  sin solapar: izq-der-izq-der-izq cuenta una, que es lo que cuenta un entrenador
    DA-3  izquierda-derecha-izquierda anatomica para todos, tambien para los zurdos

  Vive en el anotador y la API la importa: la regla existe en un solo lugar (W11 ya saco la
  copia de la web; aca no se hace una copia en la API).

QUE HACE
  ternas(golpes, regla)  las ternas de golpes consecutivos que cumplen el patron
  resumen(golpes, rondas, regla)  el bloque `combinaciones` de un peleador en la Fight-Card

  Un golpe sin tipo NO corta la combinacion (el patron es por brazo). Uno descartado por el
  entrenador (B4) si: ahi la persona dijo que no hubo golpe, asi que no hay terna.

USO
  from boxtwin.mvp.combinaciones import REGLA, ternas
  for g1, g2, g3 in ternas(golpes): ...
"""

from __future__ import annotations

__all__ = ["PATRON", "REGLA", "resumen", "ternas"]

PATRON = ("izq", "der", "izq")

REGLA = {"patron": "izq-der-izq", "separacion": "inicio_a_inicio", "max_s": 0.5}

# El brazo puede venir en el vocabulario de la Fight-Card (izq/der) o en el del detector y
# las anotaciones (left/right). Se normaliza aca y no en cada llamador.
_BRAZO = {"izq": "izq", "der": "der", "left": "izq", "right": "der"}


def _separacion(g1: dict, g2: dict, regla: dict) -> float:
    if regla.get("separacion", "inicio_a_inicio") == "fin_a_inicio":
        return float(g2["t_inicio"]) - float(g1["t_fin"])
    return float(g2["t_inicio"]) - float(g1["t_inicio"])


def ternas(golpes: list[dict], regla: dict | None = None,
           solapadas: bool = False) -> list[tuple[dict, dict, dict]]:
    """
    Las ternas izq-der-izq de una lista de golpes de UN peleador en UNA sesion.

    Se ordena por `t_inicio` y se sacan los descartados antes de recorrer: un descartado en el
    medio deja de ser un golpe y los de los costados quedan consecutivos, pero separados por
    el hueco del descartado, que casi siempre los deja fuera de los 0,5 s. La comparacion es
    estricta ("menos de 0,5 s"): 0,5 justo no es combinacion.

    `solapadas` existe solo para reproducir la tabla del spec; el producto usa la tomada.
    """
    regla = regla or REGLA
    maximo = float(regla.get("max_s", 0.5))
    validos = sorted((g for g in golpes if not g.get("descartado")),
                     key=lambda g: float(g["t_inicio"]))
    salida: list[tuple[dict, dict, dict]] = []
    i = 0
    while i + 2 < len(validos):
        trio = validos[i:i + 3]
        if (tuple(_BRAZO.get(g.get("brazo"), "") for g in trio) == PATRON
                and _separacion(trio[0], trio[1], regla) < maximo
                and _separacion(trio[1], trio[2], regla) < maximo):
            salida.append((trio[0], trio[1], trio[2]))
            i += 1 if solapadas else 3
        else:
            i += 1
    return salida


def _tipo_vigente(g: dict) -> str | None:
    """El tipo que se ve en pantalla: el corregido si hay correccion, si no el del modelo."""
    c = g.get("corregido") or {}
    if c.get("tipo") and c["tipo"] != "no_golpe":
        return c["tipo"]
    return g.get("tipo")


def _round_de(t: float, rondas: list[dict]) -> int | None:
    """
    El round de un instante. Uno que cae en el descanso va al round que acaba de terminar:
    una terna justo despues de la campana es de ese round, y asi la suma por round es igual
    al total (las ventanas excluyen el descanso).
    """
    previas = [r for r in rondas if r["inicio_s"] <= t]
    if previas:
        return previas[-1]["round"]
    return rondas[0]["round"] if rondas else None


def resumen(golpes: list[dict], rondas: list[dict], regla: dict | None = None) -> dict:
    """
    El bloque `combinaciones` de un peleador (contrato 0.3).

    Una terna que cruza el fin de un round cuenta en el round de su primer golpe: es donde el
    entrenador la va a buscar en el video.
    """
    regla = regla or REGLA
    lista = []
    for g1, g2, g3 in ternas(golpes, regla):
        lista.append({
            "golpes": [g1["id"], g2["id"], g3["id"]],
            "t_inicio": round(float(g1["t_inicio"]), 3),
            "t_fin": round(float(g3["t_fin"]), 3),
            "tipos": [_tipo_vigente(g) for g in (g1, g2, g3)],
            "round": _round_de(float(g1["t_inicio"]), rondas),
        })
    return {
        "patron": regla["patron"],
        "separacion": regla["separacion"],
        "max_s": regla["max_s"],
        "lista": lista,
        "total": len(lista),
        "por_round": [
            {"round": r["round"], "total": sum(1 for c in lista if c["round"] == r["round"])}
            for r in rondas
        ],
    }
