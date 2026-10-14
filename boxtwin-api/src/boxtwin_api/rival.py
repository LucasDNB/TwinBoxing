"""
BoxTwin API - Rival Fantasma: ataque contra exposicion (cambios del 14-10, B18).

POR QUE EXISTE
  El Rival Fantasma tenia dos bloques que hablaban de lo mismo por separado: "donde ataca uno y
  donde recibe el otro" (sin construir) y "oportunidades y amenazas" (cinco reglas sobre
  metricas sueltas, en la web y sin validar). El pedido del 14-10 los unifica y los arma con lo
  que ya existe: la distribucion de ataque de cada uno por mano y altura, y su mapa de
  exposicion (B16). Las reglas viven aca y no en la web, para probarlas en un solo lugar.

  El cruce es directo, zona por zona, sin espejar (DC-2): "cabeza por la mano izquierda del
  rival" en la exposicion de B es la misma zona a la que A tira con su izquierda. Las dos
  mitades usan fracciones corregidas por el sesgo del detector (DC-4), para que el sesgo no
  aparezca como una diferencia entre peleadores.

  "Recibe" es "le dirigen" (DC-3). Ninguna lectura dice que algo va a entrar, ni da una
  probabilidad.

QUE HACE
  cruzar(perfil_a, perfil_b)  cruces en los dos sentidos, lecturas por peleador y faltantes

  Reglas (ampliacion-13-10/05 §2), para cada sentido atacante -> receptor:
    zona_coincidente  el atacante va a la zona >= 25% y al receptor le van ahi >= 25%
    zona_trabajada    el atacante va a la zona 10 puntos o mas por encima de lo que el receptor ve
    cuerpo            el atacante va al cuerpo >= 25% y al receptor le van al cuerpo < 15%
  Cada una es oportunidad para el atacante y amenaza para el receptor. Con menos de 50 golpes
  con altura en cualquiera de las cuatro mitades no se dispara ninguna.

USO
  from boxtwin_api.rival import cruzar
  r = cruzar(perfil_a, perfil_b)
"""

from __future__ import annotations

from boxtwin_api.zonas import ZONAS, sesgo_detector

__all__ = ["MINIMO_GOLPES", "UMBRAL_COINCIDENTE", "UMBRAL_CUERPO_ATAQUE", "UMBRAL_CUERPO_VISTO",
           "UMBRAL_TRABAJADA", "cruzar", "nombre_de_zona"]

MINIMO_GOLPES = 50
UMBRAL_COINCIDENTE = 0.25
UMBRAL_TRABAJADA = 0.10
UMBRAL_CUERPO_ATAQUE = 0.25
UMBRAL_CUERPO_VISTO = 0.15

_ALTURA = {"cabeza": "a la cabeza", "cuerpo": "al cuerpo"}
_MANO = {"izq": "con la izquierda", "der": "con la derecha"}


def nombre_de_zona(zona: str) -> str:
    """'cuerpo_mano_der' -> 'al cuerpo con la derecha', desde la mano del que tira."""
    altura, _, mano = zona.split("_")
    return f"{_ALTURA[altura]} {_MANO[mano]}"


def _fracciones(b: dict | None) -> dict[str, float] | None:
    # Sin ningun golpe con altura no hay fracciones: cero por ciento seria un dato inventado.
    if not b or not b.get("zonas") or not b.get("con_dato"):
        return None
    return {z: (b["zonas"][z].get("fraccion_corregida") or 0.0) for z in ZONAS}


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _lecturas(atacante: str, receptor: str, nombres: dict, at: dict, ex: dict) -> list[dict]:
    na, nr = nombres[atacante], nombres[receptor]
    salida = []

    def agregar(regla, zona, texto):
        for tipo, para in (("oportunidad", atacante), ("amenaza", receptor)):
            salida.append({"tipo": tipo, "para": para, "atacante": atacante, "regla": regla,
                           "zona": zona, "texto": texto})

    for z in ZONAS:
        a, e = at[z], ex[z]
        if a >= UMBRAL_COINCIDENTE and e >= UMBRAL_COINCIDENTE:
            agregar("zona_coincidente", z,
                    f"{na} suele ir {nombre_de_zona(z)} ({_pct(a)}), y ahí es donde a {nr} "
                    f"suelen buscarlo ({_pct(e)}).")
        if a - e >= UMBRAL_TRABAJADA - 1e-9:
            agregar("zona_trabajada", z,
                    f"{na} trabaja {nombre_de_zona(z)} ({_pct(a)}) más de lo que {nr} suele ver "
                    f"({_pct(e)}).")
    a_cuerpo = at["cuerpo_mano_izq"] + at["cuerpo_mano_der"]
    e_cuerpo = ex["cuerpo_mano_izq"] + ex["cuerpo_mano_der"]
    if a_cuerpo >= UMBRAL_CUERPO_ATAQUE and e_cuerpo < UMBRAL_CUERPO_VISTO:
        agregar("cuerpo", "cuerpo",
                f"A {nr} no suelen trabajarlo al cuerpo ({_pct(e_cuerpo)}) y {na} lo hace "
                f"({_pct(a_cuerpo)}).")
    return salida


def cruzar(pa: dict, pb: dict) -> dict:
    """
    Cruza los bloques `ataque` y `exposicion` de dos perfiles ya filtrados a sparring y combate.
    Sin altura (bandera apagada o perfiles sin datos), cruces y lecturas vacios.
    """
    nombres = {"A": pa["boxeador"]["nombre"], "B": pb["boxeador"]["nombre"]}
    bloques = {"A": (pa.get("ataque"), pa.get("exposicion")),
               "B": (pb.get("ataque"), pb.get("exposicion"))}
    base = {l: {"ataque": (at or {}).get("con_dato") or 0, "exposicion": (ex or {}).get("con_dato") or 0}
            for l, (at, ex) in bloques.items()}
    vacio = {"cruces": [], "lecturas": [], "base": base, "faltan": None,
             "sesgo_detector": sesgo_detector()}

    cruces = []
    lecturas = []
    for atacante, receptor in (("A", "B"), ("B", "A")):
        at_b, ex_b = bloques[atacante][0], bloques[receptor][1]
        at, ex = _fracciones(at_b), _fracciones(ex_b)
        if at is None or ex is None:
            continue
        for z in ZONAS:
            cruces.append({"atacante": atacante, "zona": z,
                           "ataque_fraccion": at[z], "exposicion_fraccion": ex[z],
                           "n_ataque": at_b["zonas"][z]["n"], "n_exposicion": ex_b["zonas"][z]["n"]})
    if not cruces:
        return vacio

    pocos = {l: v for l, v in base.items() if v["ataque"] < MINIMO_GOLPES or v["exposicion"] < MINIMO_GOLPES}
    if pocos:
        return {**vacio, "cruces": cruces, "faltan": {"minimo": MINIMO_GOLPES, "base": base}}

    for atacante, receptor in (("A", "B"), ("B", "A")):
        at = _fracciones(bloques[atacante][0])
        ex = _fracciones(bloques[receptor][1])
        if at is not None and ex is not None:
            lecturas += _lecturas(atacante, receptor, nombres, at, ex)
    return {**vacio, "cruces": cruces, "lecturas": lecturas}
