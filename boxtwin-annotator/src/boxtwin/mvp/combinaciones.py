"""
BoxTwin - Combinaciones: cualquier secuencia de dos o mas golpes seguidos (14-10).

POR QUE EXISTE
  B7 contaba las cinco secuencias de 2 o 3 golpes mas frecuentes nombradas por TIPO, y el
  tipo sale de un clasificador que no generaliza. El 13-10 se paso a contar solo
  izquierda-derecha-izquierda por BRAZO. El 14-10 Lucas amplio el pedido: es combinacion
  CUALQUIER secuencia de mas de un golpe con menos de 0,5 s entre cada uno, sin importar el
  brazo. El criterio sigue saliendo del detector (tiempo y brazo, medidos), asi que el numero
  sigue siendo medido; el tipo de cada golpe viaja al lado como dato secundario.

  Decisiones que viajan adentro de la regla, para que una Fight-Card vieja diga con que regla
  se conto (las de 2026-10-13 decian patron "izq-der-izq"):
    DA-1  la separacion se mide de inicio a inicio, la frontera mas precisa del detector
    DA-2  una serie larga es UNA combinacion: se toma la cadena maxima, no sus tramos. Cinco
          golpes seguidos son una combinacion de cinco, que es lo que cuenta un entrenador
    DA-3  ya no hay patron de brazos, asi que la guardia (zurdo u ortodoxo) no interviene

  Vive en el anotador y la API la importa: la regla existe en un solo lugar.

QUE HACE
  cadenas(golpes, regla)  las combinaciones, como tuplas de golpes consecutivos
  resumen(golpes, rondas, regla)  el bloque `combinaciones` de un peleador en la Fight-Card

  Un golpe sin tipo NO corta la combinacion (el criterio es por tiempo). Uno descartado por
  el entrenador (B4) si: ahi la persona dijo que no hubo golpe.

USO
  from boxtwin.mvp.combinaciones import REGLA, cadenas
  for combinacion in cadenas(golpes): ...
"""

from __future__ import annotations

__all__ = ["REGLA", "brazos_de", "cadenas", "resumen"]

REGLA = {"patron": "cualquiera", "separacion": "inicio_a_inicio", "max_s": 0.5, "min_golpes": 2}

# El brazo puede venir en el vocabulario de la Fight-Card (izq/der) o en el del detector y
# las anotaciones (left/right). Se normaliza aca y no en cada llamador.
_BRAZO = {"izq": "izq", "der": "der", "left": "izq", "right": "der"}


def _separacion(g1: dict, g2: dict, regla: dict) -> float:
    if regla.get("separacion", "inicio_a_inicio") == "fin_a_inicio":
        return float(g2["t_inicio"]) - float(g1["t_fin"])
    return float(g2["t_inicio"]) - float(g1["t_inicio"])


def cadenas(golpes: list[dict], regla: dict | None = None) -> list[tuple[dict, ...]]:
    """
    Las combinaciones de una lista de golpes de UN peleador en UNA sesion.

    Se ordena por `t_inicio`, se sacan los descartados y se cortan cadenas donde dos golpes
    consecutivos quedan a 0,5 s o mas. Cada cadena de `min_golpes` o mas es una combinacion.
    Un descartado en el medio deja de ser un golpe y los de los costados quedan consecutivos,
    pero separados por el hueco del descartado, que casi siempre los deja fuera de los 0,5 s.
    La comparacion es estricta ("menos de 0,5 s"): 0,5 justo no es combinacion.
    """
    regla = regla or REGLA
    maximo = float(regla.get("max_s", 0.5))
    minimo = int(regla.get("min_golpes", 2))
    validos = sorted((g for g in golpes if not g.get("descartado")),
                     key=lambda g: float(g["t_inicio"]))
    salida: list[tuple[dict, ...]] = []
    actual: list[dict] = []
    for g in validos:
        if actual and _separacion(actual[-1], g, regla) >= maximo:
            if len(actual) >= minimo:
                salida.append(tuple(actual))
            actual = []
        actual.append(g)
    if len(actual) >= minimo:
        salida.append(tuple(actual))
    return salida


def brazos_de(golpes) -> list[str | None]:
    """La secuencia de brazos de una combinacion, en el vocabulario de la Fight-Card."""
    return [_BRAZO.get(g.get("brazo")) for g in golpes]


def _tipo_vigente(g: dict) -> str | None:
    """El tipo que se ve en pantalla: el corregido si hay correccion, si no el del modelo."""
    c = g.get("corregido") or {}
    if c.get("tipo") and c["tipo"] != "no_golpe":
        return c["tipo"]
    return g.get("tipo")


def _round_de(t: float, rondas: list[dict]) -> int | None:
    """
    El round de un instante. Uno que cae en el descanso va al round que acaba de terminar:
    una combinacion justo despues de la campana es de ese round, y asi la suma por round es igual
    al total (las ventanas excluyen el descanso).
    """
    previas = [r for r in rondas if r["inicio_s"] <= t]
    if previas:
        return previas[-1]["round"]
    return rondas[0]["round"] if rondas else None


def resumen(golpes: list[dict], rondas: list[dict], regla: dict | None = None) -> dict:
    """
    El bloque `combinaciones` de un peleador (contrato 0.3).

    Una combinacion que cruza el fin de un round cuenta en el round de su primer golpe: es
    donde el entrenador la va a buscar en el video.
    """
    regla = regla or REGLA
    lista = []
    for c in cadenas(golpes, regla):
        lista.append({
            "golpes": [g["id"] for g in c],
            "t_inicio": round(float(c[0]["t_inicio"]), 3),
            "t_fin": round(float(c[-1]["t_fin"]), 3),
            "brazos": brazos_de(c),
            "tipos": [_tipo_vigente(g) for g in c],
            "round": _round_de(float(c[0]["t_inicio"]), rondas),
        })
    return {
        "patron": regla["patron"],
        "separacion": regla["separacion"],
        "max_s": regla["max_s"],
        "min_golpes": regla.get("min_golpes", 2),
        "lista": lista,
        "total": len(lista),
        "por_round": [
            {"round": r["round"], "total": sum(1 for c in lista if c["round"] == r["round"])}
            for r in rondas
        ],
    }
