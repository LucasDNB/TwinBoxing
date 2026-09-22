"""
BoxTwin - La Fight-Card: lo que ve el entrenador.

POR QUE EXISTE
  Es la salida del producto y es donde se juega la honestidad del proyecto entero. Un
  sistema monocular no establece contacto fisico, asi que hay tres cosas que este archivo no
  puede contener y que estan prohibidas por diseno y no por olvido: golpes CONECTADOS,
  puntuacion de rounds y veredicto. Lo que hay es volumen DETECTADO, con la precision y el
  recall medidos del detector al lado del numero.

  El recall medido es 0,484. O sea que el conteo absoluto esta por debajo de lo real y hay
  que decirlo: lo que sostiene la lectura no es el numero suelto sino la comparacion adentro
  de la sesion -round 1 contra round 3, A contra B- donde el sesgo se cancela porque es el
  mismo detector midiendo las dos cosas.

  El tipo de golpe es distinto del resto y va separado: el clasificador no generaliza a
  fuentes nuevas, entra igual por decision de producto, y por eso cada tipo viaja con su
  confianza y con la version del checkpoint que lo produjo. Asi, cuando el modelo mejore, se
  puede decir cuanto mejoro y sobre que.

QUE HACE
  Junta los golpes del detector, los indicadores de guardia y la cobertura de identidad en
  un solo documento, agregado por round y por brazo, con cada evento enlazado a su instante.

USO
  from boxtwin.mvp.fightcard import construir
  fc = construir(sesion, golpes, eventos_guardia, cobertura, detector={...})
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "VERSION_FIGHTCARD",
    "PRECISION_DETECTOR",
    "RECALL_DETECTOR",
    "NOMBRE_EN_PANTALLA",
    "construir",
    "id_de_golpe",
]

VERSION_FIGHTCARD = "0.1"

# Lo medido del detector sobre fuente no vista, del reporte de folds y del sistema completo
# sobre sparring-3. Viaja adentro del documento porque RF6 pide que este a la vista del
# mismo numero que califica: un conteo sin su recall al lado se lee como un conteo exacto.
PRECISION_DETECTOR = 0.885
RECALL_DETECTOR = 0.484

# RF11: la nomenclatura de pantalla es rioplatense y se resuelve aca, en la presentacion,
# sin tocar el esquema de anotacion, que sigue en ingles porque es el que consumen los
# modelos y los exports.
#
# PENDIENTE de confirmar con Lucas como criterio de dominio: el spec fija "cross es el
# gancho" y eso esta puesto. Lo que queda abierto es como se llaman aca el hook y el
# uppercut, asi que hasta que lo diga alguien que boxea quedan con su nombre en ingles, que
# es lo que hoy se usa en el gimnasio, en vez de inventarles una traduccion.
NOMBRE_EN_PANTALLA = {
    "jab": "jab",
    "cross": "gancho",
    "hook": "hook",
    "uppercut": "uppercut",
}

BRAZO_EN_PANTALLA = {"left": "izq", "right": "der"}


def id_de_golpe(peleador: str, brazo: str, inicio: int) -> str:
    """
    Identificador estable de un golpe.

    Sale del carril y del cuadro de inicio y no de un contador, porque RF12 pide poder
    reclasificar una sesion vieja con un checkpoint nuevo sin repetir pose ni deteccion: si
    el id dependiera del orden de la lista, las correcciones que el entrenador ya hizo se
    pegarian a otros golpes.
    """
    return f"{peleador}-{BRAZO_EN_PANTALLA.get(brazo, brazo)}-{inicio}"


def construir(
    sesion,
    golpes: list,
    guardia: list | None = None,
    identidad: dict | None = None,
    detector: dict | None = None,
    clasificador: dict | None = None,
    tipos: dict[str, dict] | None = None,
) -> dict:
    """
    El documento completo. `tipos` es opcional: mapea id de golpe -> {tipo, confianza}.

    Se construye sin el clasificador y se le agrega despues, que es la misma operacion con
    la que se reclasifica una sesion vieja. Si el tipo viniera cableado adentro del detector,
    cambiar de checkpoint obligaria a reprocesar el video.
    """
    ventanas = list(sesion.ventanas)
    fps = float(sesion.video.get("fps") or 0.0)
    duracion = float(sesion.video.get("duracion_s") or 0.0)
    guardia = guardia or []
    tipos = tipos or {}

    por_peleador: dict[str, dict] = {}
    for p in ("A", "B"):
        suyos = [g for g in golpes if g.peleador == p]
        eventos = []
        for g in suyos:
            gid = id_de_golpe(g.peleador, g.brazo, g.inicio)
            t = tipos.get(gid, {})
            eventos.append(
                {
                    "id": gid,
                    "t_inicio": round(g.t_inicio, 3),
                    "t_fin": round(g.t_fin, 3),
                    "cuadro_inicio": int(g.inicio),
                    "cuadro_fin": int(g.fin),
                    "brazo": BRAZO_EN_PANTALLA.get(g.brazo, g.brazo),
                    "score": round(float(g.score), 4),
                    # RF9: el tipo es estimacion y viaja con su confianza. Sin clasificador
                    # corrido es None, que no es lo mismo que "no se pudo clasificar".
                    "tipo": t.get("tipo"),
                    "confianza_tipo": t.get("confianza"),
                    "tipo_clasificador": t.get("crudo"),
                    "corregido": None,
                }
            )
        por_peleador[p] = {
            "golpes": eventos,
            "por_round": _por_round(suyos, ventanas),
            "total": _conteo(suyos),
            "guardia": [e.a_dict() for e in guardia if e.peleador == p],
        }

    det = {
        "checkpoint": "",
        "umbral": None,
        "precision_medida": PRECISION_DETECTOR,
        "recall_medido": RECALL_DETECTOR,
        "medido_sobre": "sparring-3 entero, con el detector entrenado sin esa fuente",
        **(detector or {}),
    }
    cla = {
        "checkpoint": "",
        "exactitud_familia_fuente_no_vista": None,
        "estimado": True,
        **(clasificador or {}),
    }

    fc: dict[str, Any] = {
        "version": VERSION_FIGHTCARD,
        "video": {
            "nombre": sesion.video.get("nombre"),
            "duracion_s": round(duracion, 3),
            "fps": fps,
            "rounds": ventanas,
        },
        "identidad": identidad or {},
        "detector": det,
        "clasificador": cla,
        "peleadores": por_peleador,
        "nomenclatura": NOMBRE_EN_PANTALLA,
        # Las tres cosas que este documento no dice, escritas adentro del documento. No es
        # decoracion: es lo que impide que alguien lea el conteo como un marcador.
        "no_incluye": [
            "golpes conectados: un sistema monocular no establece contacto fisico",
            "puntuacion de rounds",
            "veredicto de combate",
        ],
        "avisos": list(getattr(sesion, "avisos", []) or []),
    }
    fc["lectura"] = _lectura(fc)
    return fc


def _conteo(golpes: list) -> dict:
    izq = sum(1 for g in golpes if g.brazo == "left")
    return {"total": len(golpes), "izq": izq, "der": len(golpes) - izq}


def _por_round(golpes: list, ventanas: list[dict]) -> list[dict]:
    """
    El conteo por round, con su ritmo por minuto.

    El ritmo se calcula sobre la duracion real de la ventana y no sobre la nominal: si la
    sesion corto a mitad del ultimo round, dividir por el round entero baja el ritmo de un
    round que en realidad fue igual de intenso.
    """
    salida = []
    for v in ventanas:
        dentro = [g for g in golpes if v["inicio_s"] <= g.t_inicio < v["fin_s"]]
        dur = max(v["fin_s"] - v["inicio_s"], 1e-9)
        c = _conteo(dentro)
        salida.append(
            {
                "round": v["round"],
                "inicio_s": v["inicio_s"],
                "fin_s": v["fin_s"],
                **c,
                "por_minuto": round(c["total"] * 60.0 / dur, 2),
            }
        )
    return salida


def _lectura(fc: dict) -> dict:
    """
    Las dos comparaciones que el recall medido si sostiene, calculadas una sola vez.

    Van resueltas en el documento y no en el frontend porque son la lectura que el producto
    propone, y dejarlas para la pantalla invita a que cada vista invente la suya.
    """
    a = fc["peleadores"]["A"]["total"]["total"]
    b = fc["peleadores"]["B"]["total"]["total"]
    caidas = {}
    for p in ("A", "B"):
        rondas = fc["peleadores"][p]["por_round"]
        if len(rondas) >= 2:
            primero, ultimo = rondas[0]["por_minuto"], rondas[-1]["por_minuto"]
            caidas[p] = {
                "primer_round_por_minuto": primero,
                "ultimo_round_por_minuto": ultimo,
                "variacion": round(ultimo - primero, 2),
            }
    return {
        "volumen_relativo": {
            "A": a,
            "B": b,
            "cociente_A_sobre_B": round(a / b, 3) if b else None,
        },
        "caida_entre_rounds": caidas,
        "advertencia": (
            "el conteo es de golpes DETECTADOS y esta por debajo del real: el recall medido "
            f"del detector es {fc['detector']['recall_medido']}. Lo que la medicion sostiene "
            "es la comparacion adentro de la sesion, no el numero absoluto"
        ),
    }
