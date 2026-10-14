"""
BoxTwin API - El perfil de un boxeador: sus sesiones puestas una al lado de la otra.

POR QUE EXISTE
  Una sesion suelta no contesta la pregunta que el entrenador tiene. No es "cuantos golpes
  tiro hoy" sino "esta mejorando", y eso necesita varias sesiones de la MISMA persona.

  El sistema no puede saber que el peleador A de hoy es el de la semana pasada: A y B se
  asignan por posicion en pantalla y no significan nada entre videos. Lo dice una persona,
  una vez por sesion, y a partir de ahi el perfil se arma solo.

QUE HACE
  Junta las Fight-Cards de las sesiones de un boxeador y devuelve cuatro cosas: volumen y
  ritmo por sesion, mezcla de golpes, evolucion en el tiempo y lateralidad.

  Cada una viaja con lo que la limita, y no como letra chica:

  El VOLUMEN es de golpes DETECTADOS y esta por debajo del real -el recall medido del
  detector ronda 0,48- asi que sirve para comparar sesiones entre si y no como cuenta
  absoluta. Eso vale mientras el detector no cambie: si cambia, las sesiones viejas y las
  nuevas dejan de ser comparables, y por eso el perfil devuelve con que detector se midio
  cada una.

  La MEZCLA de golpes sale de un clasificador que no generaliza: confunde 38 de 76 hooks
  con straight, y ese eje esta medido cinco veces por caminos independientes sin mejorar.
  Se devuelve con su exactitud y con cuantos golpes quedaron sin clasificar.

  B7 (web entrenador) suma la lectura por contexto -bolsa, sombra, sparring y combate miden
  cosas distintas y se comparan consigo mismos, no contra una linea de base comun- y el
  volumen por round. Las combinaciones (B10, con la regla del 14-10: cualquier secuencia de 2+ golpes)
  se cuentan con la regla del anotador. Lo que depende de un criterio
  pre-registrado sin aprobar (C2 para rounds, C3 para guardia) se devuelve en nulo aunque
  haya datos: asi ningun cliente lo puede mostrar por error.

  Los golpes que el entrenador marco "no fue un golpe" (B4) no entran en la mezcla ni en
  las combinaciones; los corregidos entran con su tipo corregido.

USO
  from boxtwin_api.perfil import construir_perfil
  perfil = construir_perfil(boxeador, sesiones_con_lado, dir_de_sesion)
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["CONTEXTOS", "combinaciones", "construir_perfil", "regla_de_combinacion",
           "resumen_de_sesion"]

CONTEXTOS = ("bolsa", "sombra", "sparring", "combate")
FAMILIAS = ("jab", "cross", "hook", "uppercut")


def _leer_fightcard(directorio: Path) -> dict | None:
    f = directorio / "fightcard.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def resumen_de_sesion(fc: dict, lado: str) -> dict | None:
    """
    Lo que una sesion aporta al perfil de uno de sus dos peleadores.

    Devuelve None si la Fight-Card no tiene ese lado, que pasa cuando la sesion fallo o
    quedo a medias: es mejor que el perfil ignore una sesion a que sume ceros, porque un
    cero se lee como "no tiro golpes" y no como "no se midio".
    """
    pel = (fc.get("peleadores") or {}).get(lado)
    if not pel:
        return None

    golpes = pel.get("golpes") or []
    duracion = float((fc.get("video") or {}).get("duracion_s") or 0.0)
    minutos = duracion / 60.0 if duracion else 0.0

    tipos: dict[str, int] = {}
    sin_tipo = 0
    confianzas = []
    # Lo descartado por el entrenador sigue contando como detectado, pero no es un golpe y
    # no entra en la mezcla.
    descartados = sum(1 for g in golpes if g.get("descartado"))
    for g in golpes:
        if g.get("descartado"):
            continue
        t = g.get("tipo")
        if t:
            tipos[t] = tipos.get(t, 0) + 1
            if g.get("confianza_tipo") is not None:
                confianzas.append(float(g["confianza_tipo"]))
        else:
            sin_tipo += 1

    brazos: dict[str, int] = {}
    for g in golpes:
        b = g.get("brazo")
        if b:
            brazos[b] = brazos.get(b, 0) + 1

    clf = fc.get("clasificador") or {}
    rondas = (fc.get("video") or {}).get("rounds") or []
    cobertura = (fc.get("identidad") or {}).get(f"cobertura_{lado}")
    return {
        "golpes": len(golpes),
        "descartados": descartados,
        "rounds": len(rondas) or 1,
        "minutos": round(minutos, 2) if minutos else None,
        # Los minutos en que el sistema lo vio: el denominador de lo que solo se puede contar
        # cuando el peleador esta identificado (combinaciones, ampliacion 01 §3).
        "minutos_con_identidad": (round(minutos * float(cobertura), 3)
                                  if minutos and cobertura is not None else None),
        "golpes_por_minuto": round(len(golpes) / minutos, 2) if minutos else None,
        "por_round": pel.get("por_round") or [],
        "mezcla": tipos,
        "sin_clasificar": sin_tipo,
        "confianza_media": (
            round(sum(confianzas) / len(confianzas), 3) if confianzas else None
        ),
        "brazos": brazos,
        "guardia": pel.get("guardia") or [],
        # Con que se midio. Sin esto, comparar dos sesiones puede ser comparar dos
        # detectores distintos y atribuirle al boxeador una mejora del software.
        "detector": (fc.get("detector") or {}).get("checkpoint"),
        "clasificador": clf.get("checkpoint") or None,
        "exactitud_familia": clf.get("exactitud_familia_fuente_no_vista"),
    }


def regla_de_combinacion() -> dict | None:
    """La regla de combinaciones del anotador, o None si esta instancia no lo tiene instalado."""
    try:
        from boxtwin.mvp.combinaciones import REGLA
    except ImportError:  # pragma: no cover - depende del entorno
        return None
    return dict(REGLA)


def combinaciones(entradas: list[dict], golpes_de: dict[str, list[dict]],
                  recall: float | None = None) -> dict | None:
    """
    Las combinaciones del peleador en las sesiones filtradas (ampliacion 01 §3, con la regla
    del 14-10: cualquier secuencia de dos o mas golpes con menos de 0,5 s entre cada uno).

    Usa la MISMA funcion que la Fight-Card, importada del anotador, y la recalcula sobre los
    golpes de cada sesion en vez de leer el bloque guardado: asi una Fight-Card 0.2, que no lo
    trae, cuenta igual que una 0.3, y no hay una segunda copia de la regla que se pueda
    separar de la primera. Sin el anotador instalado devuelve None: la API no inventa la regla.

    El ritmo se calcula sobre los minutos en que el sistema VIO al peleador (cobertura por
    duracion): si no lo vio, no puede decir que no combino. Una sesion sin cobertura medida no
    entra en el ritmo, aunque sus combinaciones si entran en el total.

    Las secuencias de brazos salen del detector y son medidas. Los tipos de cada combinacion
    van aparte, como estimacion, y solo las que tienen todos sus tipos estimados entran en ese
    conteo.

    `recall` es el que califica el numero en pantalla (01 §4): el de combinaciones si C11 se midio,
    si no el de golpes, de la Fight-Card mas reciente.
    """
    try:
        from boxtwin.mvp.combinaciones import REGLA, brazos_de, cadenas
    except ImportError:  # pragma: no cover - depende del entorno
        return None

    total = 0
    con = 0
    por_contexto: dict[str, int] = {}
    tipos: dict[tuple, int] = {}
    secuencias: dict[tuple, int] = {}
    por_largo: dict[str, int] = {}
    con_ritmo = 0
    minutos = 0.0
    for x in entradas:
        suyas = cadenas(golpes_de.get(x["sesion_id"], []), REGLA)
        total += len(suyas)
        con += 1 if suyas else 0
        por_contexto[x["contexto"]] = por_contexto.get(x["contexto"], 0) + len(suyas)
        if x.get("minutos_con_identidad"):
            minutos += x["minutos_con_identidad"]
            con_ritmo += len(suyas)
        for c in suyas:
            # Cuatro golpes o mas van juntos: son pocas y separarlas no le dice nada al
            # entrenador (en lo anotado, 26 de 179).
            largo = str(len(c)) if len(c) < 4 else "4+"
            por_largo[largo] = por_largo.get(largo, 0) + 1
            brazos = tuple(brazos_de(c))
            if all(brazos):
                secuencias[brazos] = secuencias.get(brazos, 0) + 1
            clave = tuple(_tipo_vigente(g) for g in c)
            if all(clave):
                tipos[clave] = tipos.get(clave, 0) + 1
    orden = sorted(tipos.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    orden_brazos = sorted(secuencias.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {
        "patron": REGLA["patron"],
        "separacion": REGLA["separacion"],
        "max_s": REGLA["max_s"],
        "total": total,
        "por_minuto": round(con_ritmo / minutos, 3) if minutos else None,
        "por_contexto": por_contexto,
        "por_largo": por_largo,
        "secuencias_frecuentes": [{"brazos": list(k), "n": n} for k, n in orden_brazos],
        "tipos_frecuentes": [{"tipos": list(k), "n": n, "estimado": True} for k, n in orden],
        "sesiones_con_combinaciones": con,
        "recall_medido": recall,
    }


def _tipo_vigente(g: dict) -> str | None:
    c = g.get("corregido") or {}
    if c.get("tipo") and c["tipo"] != "no_golpe":
        return c["tipo"]
    return g.get("tipo")


def _guardia_arriba(entradas: list[dict]) -> float | None:
    ev = [e for x in entradas for e in x["guardia"]
          if e.get("medible") is not False and e.get("mano_opuesta_caida") is not None]
    if not ev:
        return None
    return round(sum(1 for e in ev if e["mano_opuesta_caida"] is False) / len(ev), 3)


def _caida_ultimo_round(entradas: list[dict]) -> float | None:
    """Variacion relativa del ritmo del ultimo round contra el anterior, promedio entre las
    sesiones con tres rounds o mas."""
    variaciones = []
    for x in entradas:
        r = x["por_round"]
        if len(r) >= 3 and r[-2].get("por_minuto"):
            variaciones.append((r[-1]["por_minuto"] - r[-2]["por_minuto"]) / r[-2]["por_minuto"])
    return round(sum(variaciones) / len(variaciones), 3) if variaciones else None


def _por_contexto(entradas: list[dict], banderas: dict) -> list[dict]:
    salida = []
    for ctx in CONTEXTOS:
        suyas = [x for x in entradas if x["contexto"] == ctx]
        if not suyas:
            continue
        golpes = sum(x["golpes"] for x in suyas)
        minutos = sum(x["minutos"] or 0 for x in suyas)
        mezcla: dict[str, int] = {}
        for x in suyas:
            for t, n in x["mezcla"].items():
                mezcla[t] = mezcla.get(t, 0) + n
        clasificados = sum(mezcla.get(f, 0) for f in FAMILIAS)
        salida.append({
            "contexto": ctx,
            "sesiones": len(suyas),
            "golpes_por_minuto": round(golpes / minutos, 2) if minutos else None,
            "jab_fraccion": round(mezcla.get("jab", 0) / clasificados, 3) if clasificados else None,
            "guardia_opuesta_arriba": _guardia_arriba(suyas) if banderas.get("guardia") else None,
            "caida_ultimo_round": (_caida_ultimo_round(suyas)
                                   if banderas.get("comparacion_rounds") else None),
        })
    return salida


def _rounds_promedio(entradas: list[dict]) -> list[dict] | None:
    por_indice: dict[int, list[float]] = {}
    for x in entradas:
        for i, r in enumerate(x["por_round"], start=1):
            if r.get("por_minuto") is not None:
                por_indice.setdefault(i, []).append(float(r["por_minuto"]))
    if not por_indice:
        return None
    return [{"round": i, "golpes_por_minuto": round(sum(v) / len(v), 2)}
            for i, v in sorted(por_indice.items())]


def construir_perfil(boxeador, sesiones: list[tuple], dir_de_sesion, contexto: str = "todas",
                     banderas: dict | None = None) -> dict:
    """
    El perfil completo. `sesiones` son pares (sesion, lado) ya filtrados por boxeador.

    Las sesiones vienen de la mas vieja a la mas nueva: la evolucion se lee en ese orden y
    darla al reves invita a leer una mejora como un empeoramiento.

    `contexto` filtra las sesiones, la evolucion, los totales y las combinaciones; la tabla
    por contexto y la cantidad de sesiones por contexto se calculan siempre sobre todas, que
    es lo que la pantalla necesita para poner los filtros y comparar.
    """
    banderas = banderas or {}
    todas = []
    golpes_de: dict[str, list[dict]] = {}
    nomenclatura: dict[str, str] = {}
    recall: float | None = None
    for ses, lado in sorted(sesiones, key=lambda x: x[0].creada):
        fc = _leer_fightcard(Path(dir_de_sesion(ses.id)))
        if not fc:
            continue
        r = resumen_de_sesion(fc, lado)
        if r is None:
            continue
        golpes_de[ses.id] = ((fc.get("peleadores") or {}).get(lado) or {}).get("golpes") or []
        # La de la Fight-Card mas reciente: los nombres de golpe en pantalla los decide el
        # contrato (RF11, D7), no la web.
        nomenclatura = fc.get("nomenclatura") or nomenclatura
        det = fc.get("detector") or {}
        recall = (det.get("combinaciones") or {}).get("recall_medido") or det.get("recall_medido")
        todas.append({
            "sesion_id": ses.id,
            "nombre": ses.nombre,
            "fecha": ses.creada.isoformat(),
            "lado": lado,
            "contexto": getattr(ses, "contexto", None) or "sparring",
            **r,
        })
    entradas = todas if contexto == "todas" else [x for x in todas if x["contexto"] == contexto]

    total_golpes = sum(e["golpes"] for e in entradas)
    mezcla: dict[str, int] = {}
    brazos: dict[str, int] = {}
    sin_clasificar = 0
    detectores = set()
    for e in entradas:
        for t, n in e["mezcla"].items():
            mezcla[t] = mezcla.get(t, 0) + n
        for b, n in e["brazos"].items():
            brazos[b] = brazos.get(b, 0) + n
        sin_clasificar += e["sin_clasificar"]
        if e["detector"]:
            detectores.add(e["detector"])

    gpm = [e["golpes_por_minuto"] for e in entradas if e["golpes_por_minuto"] is not None]
    avisos = []
    if entradas:
        avisos.append(
            "El volumen es de golpes DETECTADOS y esta por debajo del real: el recall "
            "medido del detector ronda 0,48. Lo que sostiene es la comparacion entre "
            "sesiones, no el numero absoluto."
        )
    if len(detectores) > 1:
        avisos.append(
            f"Las sesiones de este perfil se midieron con {len(detectores)} detectores "
            "distintos, asi que una diferencia entre ellas puede ser del software y no del "
            "boxeador. Para comparar, volver a procesar con el mismo."
        )
    if sin_clasificar and total_golpes:
        avisos.append(
            f"{sin_clasificar} de {total_golpes} golpes no tienen tipo estimado: la mezcla "
            "esta calculada sobre el resto."
        )
    if mezcla:
        avisos.append(
            "La mezcla de golpes sale de un clasificador que no generaliza a material "
            "nuevo: confunde 38 de 76 hooks con straight. Leerla como tendencia, no como "
            "conteo."
        )

    sesiones_por_contexto = {c: sum(1 for x in todas if x["contexto"] == c) for c in CONTEXTOS}
    extendido = {
        "nomenclatura": nomenclatura,
        "contexto": contexto,
        "sesiones_por_contexto": sesiones_por_contexto,
        "rounds_total": sum(x["rounds"] for x in entradas),
        "por_contexto": _por_contexto(todas, banderas),
        # Sin C2 aprobado, nulo aunque haya datos: el recall puede variar entre rounds y una
        # caida del ultimo round podria ser del detector y no del boxeador (D3).
        "rounds_promedio": _rounds_promedio(entradas) if banderas.get("comparacion_rounds") else None,
        "combinaciones": combinaciones(entradas, golpes_de, recall),
    }

    return {
        **extendido,
        "boxeador": {
            "id": boxeador.id,
            "nombre": boxeador.nombre,
            "guardia": boxeador.guardia,
            "notas": boxeador.notas,
            "categoria": getattr(boxeador, "categoria", None),
            "altura_m": getattr(boxeador, "altura_m", None),
            "envergadura_m": getattr(boxeador, "envergadura_m", None),
        },
        "sesiones": entradas,
        "totales": {
            "sesiones": len(entradas),
            "golpes": total_golpes,
            "mezcla": mezcla,
            "brazos": brazos,
            "sin_clasificar": sin_clasificar,
            "golpes_por_minuto_medio": round(sum(gpm) / len(gpm), 2) if gpm else None,
        },
        # La evolucion es la misma metrica en el tiempo, que es lo que convierte esto en una
        # herramienta de entrenamiento y no en un informe suelto.
        "evolucion": [
            {"fecha": e["fecha"], "sesion_id": e["sesion_id"],
             "golpes": e["golpes"], "golpes_por_minuto": e["golpes_por_minuto"],
             # Con que detector: si cambia entre dos puntos, la linea se corta en la web.
             "detector": e["detector"], "contexto": e["contexto"]}
            for e in entradas
        ],
        "avisos": avisos,
    }
