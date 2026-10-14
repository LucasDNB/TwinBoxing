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
    "NO_GOLPE",
    "VERSION_FIGHTCARD",
    "PRECISION_DETECTOR",
    "RECALL_DETECTOR",
    "NOMBRE_EN_PANTALLA",
    "aplicar_correcciones",
    "aplicar_tipos",
    "construir",
    "correccion",
    "correccion_objetivo",
    "id_de_golpe",
]

# 0.2 (web entrenador, B3 y B4): suma `descartado` por golpe, `descartados` por peleador y
# los campos de tramos en `identidad`. Es compatible hacia atras: nada de 0.1 se movio.
# 0.3 (ampliacion del 13-10): suma `combinaciones` izq-der-izq por peleador (B10). Tambien
# compatible: nada de 0.2 se movio. La altura de cada golpe (B11) la agrega la etapa
# `enriquecer`, que lee poses.npz: construir no la conoce.
VERSION_FIGHTCARD = "0.3"
OBJETIVOS = ("cabeza", "cuerpo")

# "No fue un golpe" (B4). No es un tipo: es la etiqueta de falso positivo del detector, que
# hasta ahora no recibia ninguna senal del producto.
NO_GOLPE = "no_golpe"

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
    lados: tuple[str, ...] = ("A", "B"),
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

    # Una sesion de un solo peleador (bolsa, sombra; B8) trae solo A: un B en cero se leeria
    # como "no tiro golpes" y no como "no habia nadie".
    por_peleador: dict[str, dict] = {}
    for p in lados:
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
                    "descartado": False,
                }
            )
        por_peleador[p] = {
            "golpes": eventos,
            "por_round": _por_round(suyos, ventanas),
            "total": _conteo(suyos),
            "guardia": [e.a_dict() for e in guardia if e.peleador == p],
            "descartados": 0,
        }

    det = {
        "checkpoint": "",
        "umbral": None,
        "precision_medida": PRECISION_DETECTOR,
        "recall_medido": RECALL_DETECTOR,
        "medido_sobre": "sparring-3 entero, con el detector entrenado sin esa fuente",
        # B8b (ampliacion 02 §2): el contexto en que se midio el margen. En bolsa o sombra el
        # detector nunca se midio, y la web no puede mostrar este numero como si lo fuera.
        "medido_en": "sparring",
        **(detector or {}),
    }
    cla = {
        "checkpoint": "",
        "exactitud_familia_fuente_no_vista": None,
        "estimado": True,
        **(clasificador or {}),
    }

    ident = dict(identidad or {})
    # 0.2: los campos de tramos los completa la etapa que los mide (B3); mientras tanto
    # viajan en nulo, que no es lo mismo que cero. La separacion de color es la misma que
    # el diagnostico con siembra ya escribia con otro nombre.
    for campo in ("tramos_total", "tramos_asignados_auto", "tramos_asignados_manual"):
        ident.setdefault(campo, None)
    ident.setdefault("separacion_color", ident.get("separacion_de_perfiles"))

    fc: dict[str, Any] = {
        "version": VERSION_FIGHTCARD,
        "video": {
            "nombre": sesion.video.get("nombre"),
            "duracion_s": round(duracion, 3),
            "fps": fps,
            "rounds": ventanas,
        },
        "identidad": ident,
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
    _combinaciones(fc)
    return fc


def _combinaciones(fc: dict) -> dict:
    """
    Las ternas izq-der-izq de cada peleador, recalculadas sobre el estado actual del documento.

    Se llama al construir y despues de cada cambio que las puede mover: un "no fue un golpe"
    rompe una terna y una correccion de tipo cambia lo que se muestra al lado (01 §3).
    """
    from boxtwin.mvp.combinaciones import resumen

    rondas = (fc.get("video") or {}).get("rounds") or []
    for datos in fc["peleadores"].values():
        datos["combinaciones"] = resumen(datos["golpes"], rondas)
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
    b = fc["peleadores"]["B"]["total"]["total"] if "B" in fc["peleadores"] else None
    caidas = {}
    for p in fc["peleadores"]:
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


def aplicar_tipos(fc: dict, tipos: dict, checkpoint: str, exactitud: float | None = None) -> dict:
    """
    Pega los tipos que produjo la etapa de clasificacion, sobre la Fight-Card ya escrita.

    Es la misma operacion con la que se reclasifica una sesion vieja cuando sale un
    checkpoint nuevo (RF12), asi que no puede depender de nada que se haya perdido: entra
    por el id del golpe, que sale del carril y del cuadro y no de la posicion en la lista.

    Una correccion del entrenador NO se pisa. Un modelo nuevo puede mejorar el promedio y
    seguir equivocandose justo donde una persona ya miro y dijo otra cosa.
    """
    fc["clasificador"] = {
        **fc.get("clasificador", {}),
        "checkpoint": checkpoint,
        "exactitud_familia_fuente_no_vista": exactitud,
        "estimado": True,
    }
    for datos in fc["peleadores"].values():
        for g in datos["golpes"]:
            t = tipos.get(g["id"])
            if not t:
                continue
            if g.get("corregido"):
                # Se guarda igual, para poder medir despues cuanto mejoro el modelo sobre
                # los casos que una persona ya habia corregido.
                g["tipo_modelo_nuevo"] = t.get("tipo")
                continue
            g["tipo"] = t.get("tipo")
            g["confianza_tipo"] = t.get("confianza")
            g["tipo_clasificador"] = t.get("crudo")
    return _combinaciones(fc)


def aplicar_correcciones(fc: dict, correcciones: list[dict]) -> dict:
    """
    Aplica las correcciones del entrenador sin borrar lo que dijo el modelo.

    Cada correccion queda como un registro aparte -RF10- y aca solo se refleja en la
    vista: el golpe pasa a mostrar el tipo corregido y conserva cual era el original y que
    checkpoint lo produjo. Esa pareja es exactamente la etiqueta nueva que al clasificador
    le falta: material que no vio, juzgado por alguien que sabe.
    """
    por_id: dict[str, dict] = {}
    objetivos: dict[str, dict] = {}
    for c in correcciones:
        # La ultima correccion de un mismo golpe gana, y las anteriores siguen en el
        # registro. Cambiar de opinion es legitimo; perder el rastro no. Tipo y altura (B11)
        # son correcciones independientes: corregir una no deshace la otra.
        if c.get("campo", "tipo") == "objetivo":
            objetivos[c["golpe"]] = c
        else:
            por_id[c["golpe"]] = c
    for datos in fc["peleadores"].values():
        for g in datos["golpes"]:
            o = objetivos.get(g["id"])
            if o:
                g["objetivo_corregido"] = {
                    "objetivo": o["objetivo"],
                    "objetivo_original": o.get("objetivo_original"),
                    "cuando": o.get("cuando"),
                    "por": o.get("por"),
                }
                g["objetivo"] = o["objetivo"]
                g["objetivo_confianza"] = None
                g["objetivo_motivo"] = None
            c = por_id.get(g["id"])
            if not c:
                g.setdefault("descartado", False)
                continue
            g["corregido"] = {
                "tipo": c["tipo"],
                "tipo_original": c.get("tipo_original", g.get("tipo")),
                "checkpoint_original": c.get("checkpoint"),
                "cuando": c.get("cuando"),
                "por": c.get("por"),
            }
            g["confianza_tipo"] = None   # ya no es una estimacion del modelo
            if c["tipo"] == NO_GOLPE:
                # El golpe sigue en la lista y en el total: es lo que el sistema DETECTO, y
                # el conteo de detectados no cambia por una correccion (B4). Se marca y se
                # cuenta aparte. El tipo queda el del modelo, que no es una familia nueva.
                g["descartado"] = True
            else:
                g["descartado"] = False
                g["tipo"] = c["tipo"]
        datos["descartados"] = sum(1 for g in datos["golpes"] if g.get("descartado"))
    from boxtwin.mvp.objetivo import contar

    contar(fc)
    return _combinaciones(fc)


def correccion(
    golpe: str, tipo: str, fc: dict, por: str | None = None, cuando: str | None = None
) -> dict:
    """
    El registro de una correccion, con todo lo que hace falta para usarla como etiqueta.

    Lleva el video y el checkpoint porque sin eso no se puede saber sobre que material ni
    contra que modelo se midio, y una etiqueta sin procedencia no entra a un dataset.
    """
    from datetime import datetime

    original = None
    for datos in fc["peleadores"].values():
        for g in datos["golpes"]:
            if g["id"] == golpe:
                original = g
                break
    if original is None:
        raise ValueError(f"la Fight-Card no tiene un golpe {golpe!r}")
    if tipo not in NOMBRE_EN_PANTALLA and tipo != NO_GOLPE:
        raise ValueError(
            f"tipo desconocido: {tipo!r}, se esperaba {sorted(NOMBRE_EN_PANTALLA)} o {NO_GOLPE!r}"
        )
    return {
        "golpe": golpe,
        "campo": "tipo",
        "tipo": tipo,
        "tipo_original": original.get("tipo"),
        "tipo_clasificador": original.get("tipo_clasificador"),
        "confianza_original": original.get("confianza_tipo"),
        "checkpoint": fc.get("clasificador", {}).get("checkpoint"),
        "video": fc.get("video", {}).get("nombre"),
        "t_inicio": original.get("t_inicio"),
        "t_fin": original.get("t_fin"),
        "cuadro_inicio": original.get("cuadro_inicio"),
        "cuadro_fin": original.get("cuadro_fin"),
        "por": por,
        "cuando": cuando or datetime.now().astimezone().isoformat(),
    }


def correccion_objetivo(
    golpe: str, objetivo: str, fc: dict, por: str | None = None, cuando: str | None = None
) -> dict:
    """
    La correccion de altura (B11, 03 §5): cabeza o cuerpo. Como las de tipo, es una etiqueta
    para recalibrar tau, y por eso lleva la altura que habia calculado la regla.
    """
    from datetime import datetime

    if objetivo not in OBJETIVOS:
        raise ValueError(f"objetivo desconocido: {objetivo!r}, se esperaba {list(OBJETIVOS)}")
    original = next((g for d in fc["peleadores"].values() for g in d["golpes"] if g["id"] == golpe), None)
    if original is None:
        raise ValueError(f"la Fight-Card no tiene un golpe {golpe!r}")
    previo = original.get("objetivo_corregido") or {}
    return {
        "golpe": golpe,
        "campo": "objetivo",
        "objetivo": objetivo,
        "objetivo_original": previo.get("objetivo_original", original.get("objetivo")),
        "objetivo_h": original.get("objetivo_h"),
        "regla": ((fc.get("metodos") or {}).get("objetivo") or {}).get("regla"),
        "tau": ((fc.get("metodos") or {}).get("objetivo") or {}).get("tau"),
        "video": fc.get("video", {}).get("nombre"),
        "t_inicio": original.get("t_inicio"),
        "por": por,
        "cuando": cuando or datetime.now().astimezone().isoformat(),
    }
