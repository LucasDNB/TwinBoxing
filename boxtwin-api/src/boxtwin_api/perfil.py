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

USO
  from boxtwin_api.perfil import construir_perfil
  perfil = construir_perfil(boxeador, sesiones_con_lado, dir_de_sesion)
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["construir_perfil", "resumen_de_sesion"]


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
    for g in golpes:
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
    return {
        "golpes": len(golpes),
        "minutos": round(minutos, 2) if minutos else None,
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


def construir_perfil(boxeador, sesiones: list[tuple], dir_de_sesion) -> dict:
    """
    El perfil completo. `sesiones` son pares (sesion, lado) ya filtrados por boxeador.

    Las sesiones vienen de la mas vieja a la mas nueva: la evolucion se lee en ese orden y
    darla al reves invita a leer una mejora como un empeoramiento.
    """
    entradas = []
    for ses, lado in sorted(sesiones, key=lambda x: x[0].creada):
        fc = _leer_fightcard(Path(dir_de_sesion(ses.id)))
        if not fc:
            continue
        r = resumen_de_sesion(fc, lado)
        if r is None:
            continue
        entradas.append({
            "sesion_id": ses.id,
            "nombre": ses.nombre,
            "fecha": ses.creada.isoformat(),
            "lado": lado,
            **r,
        })

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

    return {
        "boxeador": {
            "id": boxeador.id,
            "nombre": boxeador.nombre,
            "guardia": boxeador.guardia,
            "notas": boxeador.notas,
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
             "golpes": e["golpes"], "golpes_por_minuto": e["golpes_por_minuto"]}
            for e in entradas
        ],
        "avisos": avisos,
    }
