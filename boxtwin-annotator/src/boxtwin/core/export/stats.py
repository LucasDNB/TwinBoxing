"""
BoxTwin - Reporte de distribucion del dataset.

POR QUE EXISTE
  Es el numero que va a la tesis, y por eso tiene que decir sobre que se calculo. La leccion
  esta documentada en el proyecto: sobre BoxingVI conviven dos manifests con seis clips de
  diferencia, y todo numero que se reporte tiene que declarar cual uso, porque si no dos
  cifras correctas se contradicen.

  Reporta el desbalance en los dos espacios de clases. El lado es lo que se anota, pero el
  desbalance que le importa al clasificador es el del espacio del export, y con dos
  peleadores de guardias distintas los dos espacios no coinciden.

  Reporta tambien el tiempo de anotacion y cuantos cuadros quedaron no confiables. El
  primero es lo que permite estimar cuanto falta; el segundo es lo que dice cuanto del video
  es inutilizable, y ese porcentaje es un resultado del metodo, no un detalle.

QUE HACE
  Escribe un json con la distribucion por clase en los dos espacios, duraciones, tiempos de
  anotacion y cobertura, mas un resumen legible en texto.

USO
  export stats
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from boxtwin.core.export.base import (
    ExportContext,
    ExportResult,
    base_metadata,
    escribir_json,
    registrar,
)
from boxtwin.core.export.labels import CLASS_SETS, LabelSpace, class_name
from boxtwin.core.types import Completeness, FighterId
from boxtwin.core.validation import validate_document

__all__ = ["exportar"]


def _duraciones(eventos) -> dict[str, Any]:
    if not eventos:
        return {"n": 0}
    d = [e.duration_frames for e in eventos]
    return {
        "n": len(d),
        "media": round(statistics.fmean(d), 2),
        "desvio": round(statistics.pstdev(d), 2) if len(d) > 1 else 0.0,
        "min": min(d),
        "max": max(d),
    }


@registrar("stats")
def exportar(ctx: ExportContext) -> ExportResult:
    doc = ctx.doc
    eventos = doc.events
    completos = [e for e in eventos if e.completeness is Completeness.FULL]

    # -- distribucion en los dos espacios y los tres conjuntos
    distribucion: dict[str, dict[str, dict[str, int]]] = {}
    for space in LabelSpace:
        distribucion[space.value] = {}
        for n in CLASS_SETS:
            cuenta = Counter()
            for e in eventos:
                nombre = class_name(e, space, n)
                if nombre is not None:
                    cuenta[nombre] += 1
            distribucion[space.value][str(n)] = dict(sorted(cuenta.items()))

    # -- duraciones por clase, en el espacio observado de 12
    por_clase: dict[str, list] = {}
    for e in completos:
        nombre = class_name(e, LabelSpace.SIDE, 12)
        if nombre:
            por_clase.setdefault(nombre, []).append(e)
    duraciones = {k: _duraciones(v) for k, v in sorted(por_clase.items())}

    # -- cobertura no confiable
    total_frames = doc.video.total_frames
    no_confiables: dict[str, dict[str, Any]] = {}
    for f in FighterId:
        cuadros = set()
        for seg in doc.unreliable_segments:
            if seg.fighter is f:
                cuadros.update(range(seg.start_frame, seg.end_frame_excl))
        no_confiables[f.value] = {
            "frames": len(cuadros),
            "porcentaje": round(100 * len(cuadros) / total_frames, 2) if total_frames else 0.0,
        }

    # -- tiempo de anotacion
    metricas = list(doc.process.event_metrics.values())
    tiempos = sorted(m.active_ms for m in metricas)
    proceso = {
        "sesiones": len(doc.process.sessions),
        "anotadores": [a.id for a in doc.process.annotators],
        "active_ms_total": sum(s.active_ms for s in doc.process.sessions),
        "eventos_con_metricas": len(metricas),
        "ms_por_evento": {
            "mediana": tiempos[len(tiempos) // 2] if tiempos else None,
            "media": round(statistics.fmean(tiempos), 1) if tiempos else None,
        },
        "replays": {
            "media": round(statistics.fmean([m.replays for m in metricas]), 2) if metricas else None,
            "max": max((m.replays for m in metricas), default=None),
        },
    }

    issues = validate_document(doc)
    reporte: dict[str, Any] = base_metadata(ctx, "stats")
    reporte["eventos"] = {
        "total": len(eventos),
        "por_completitud": dict(sorted(Counter(e.completeness.value for e in eventos).items())),
        "por_peleador": dict(sorted(Counter(e.fighter.value for e in eventos).items())),
        "por_resultado": dict(sorted(Counter(e.landed.value for e in eventos).items())),
        "por_calidad": dict(sorted(Counter(e.quality.value for e in eventos).items())),
    }
    reporte["distribucion"] = distribucion
    reporte["duraciones_por_clase"] = duraciones
    reporte["duracion_global"] = _duraciones(completos)
    reporte["identidad"] = {
        "assignments": len(doc.identity.assignments),
        "tracks_manuales": len(doc.identity.manual_tracks),
        "interpolaciones": len(doc.identity.interpolations),
        "frames_interpolados": sum(i.gap_len for i in doc.identity.interpolations),
    }
    reporte["no_confiable"] = no_confiables
    reporte["proceso"] = proceso
    reporte["validacion"] = dict(sorted(Counter(i.level.value for i in issues).items()))

    base = ctx.video_path.stem
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = ctx.out_dir / f"{base}.stats.json"
    escribir_json(json_path, reporte)

    txt_path = ctx.out_dir / f"{base}.stats.txt"
    txt_path.write_text(_texto(reporte), encoding="utf-8")

    avisos = []
    balance = distribucion[LabelSpace.SIDE.value]["12"]
    if balance:
        peor = max(balance.values()) / max(1, min(balance.values()))
        if peor >= 3:
            avisos.append(
                f"desbalance de {peor:.1f}:1 entre la clase mas y menos frecuente "
                "en side de 12"
            )
    if len(balance) < 12:
        avisos.append(f"{12 - len(balance)} de las 12 clases no tienen ningun ejemplo")

    return ExportResult(
        formato="stats",
        archivos=[json_path, txt_path],
        resumen={"eventos": len(eventos), "clases_con_ejemplos": len(balance)},
        avisos=avisos,
    )


def _texto(r: dict[str, Any]) -> str:
    """
    Resumen legible. Declara siempre sobre que se calculo cada numero: dos cifras correctas
    con denominadores distintos se contradicen y hacen perder horas.
    """
    lineas = [
        f"BoxTwin - distribucion de {r['video']['name']}",
        f"anotacion sha256 {r['annot_sha256'][:16]}",
        f"generado {r['created_at']}",
        "",
        f"eventos: {r['eventos']['total']}",
        f"  por completitud : {r['eventos']['por_completitud']}",
        f"  por peleador    : {r['eventos']['por_peleador']}",
        f"  por resultado   : {r['eventos']['por_resultado']}",
        "",
        "distribucion en side, 12 clases (solo golpes completos)",
    ]
    for k, v in r["distribucion"]["side"]["12"].items():
        dur = r["duraciones_por_clase"].get(k, {})
        media = f"  dur media {dur['media']} +- {dur['desvio']}" if dur.get("n") else ""
        lineas.append(f"  {k:28s} {v:5d}{media}")

    g = r["duracion_global"]
    if g.get("n"):
        lineas += [
            "",
            f"duracion de los golpes completos: media {g['media']} cuadros, "
            f"desvio {g['desvio']}, rango {g['min']}-{g['max']}",
        ]

    lineas += [
        "",
        "cuadros no confiables",
        *(f"  {k}: {v['frames']} ({v['porcentaje']}%)" for k, v in r["no_confiable"].items()),
        "",
        "identidad",
        f"  assignments {r['identidad']['assignments']} · "
        f"manuales {r['identidad']['tracks_manuales']} · "
        f"interpolaciones {r['identidad']['interpolaciones']} "
        f"({r['identidad']['frames_interpolados']} cuadros)",
        "",
        "proceso de anotacion",
        f"  sesiones {r['proceso']['sesiones']} · anotadores {r['proceso']['anotadores']}",
        f"  tiempo activo total {r['proceso']['active_ms_total'] / 1000:.1f} s",
        f"  mediana por evento  {r['proceso']['ms_por_evento']['mediana']} ms",
        f"  vueltas de preview  media {r['proceso']['replays']['media']}, "
        f"max {r['proceso']['replays']['max']}",
        "",
        f"validacion: {r['validacion'] or 'sin observaciones'}",
        "",
    ]
    return "\n".join(lineas)
