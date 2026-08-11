"""
BoxTwin - Chequeos a nivel documento.

POR QUE EXISTE
  Las invariantes estructurales ya las impone el esquema y hacen fallar el parseo. Lo que
  queda son las inconsistencias que no impiden abrir el archivo pero contaminan el
  dataset: un evento que se sale del video, un ID switch que dejo dos tracks con el mismo
  rol en el mismo cuadro, un golpe anotado sobre un tramo que se marco no confiable.
  Estos chequeos NO pueden vivir en el modelo pydantic. Si un archivo con advertencias no
  se pudiera abrir, un crash a mitad de sesion dejaria la anotacion inaccesible, que es
  exactamente el escenario que hay que evitar.
  El criterio de que avisar y que no importa tanto como los chequeos mismos. Con
  combinaciones, el solapamiento entre golpes de lados distintos es la norma y no un
  defecto: avisarlo haria saltar la advertencia todo el tiempo y entrenaria al anotador a
  ignorarlas, perdiendo tambien las que si importan.

QUE HACE
  Recorre el documento y devuelve una lista ordenada y determinista de Issues con nivel,
  codigo, mensaje y referencia al objeto que lo produjo.

USO
  from boxtwin.core.validation import validate_document, has_errors
  issues = validate_document(doc)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import FighterId, IssueLevel, Quality, TrackRole

__all__ = ["Issue", "validate_document", "has_errors", "ISSUE_CODES"]


@dataclass(frozen=True)
class Issue:
    level: IssueLevel
    code: str
    message: str
    ref: str | None = None
    frames: tuple[int, int] | None = None


ISSUE_CODES: dict[str, IssueLevel] = {
    "EV_OUT_OF_BOUNDS": IssueLevel.ERROR,
    "EV_TOO_LONG": IssueLevel.WARNING,
    "EV_OVERLAP_SAME_SIDE": IssueLevel.WARNING,
    "EV_OVERLAP_SAME_SIDE_LONG": IssueLevel.WARNING,
    "EV_REFIRE_TOO_FAST": IssueLevel.WARNING,
    "EV_GUARD_MISMATCH": IssueLevel.INFO,
    "EV_UNRELIABLE_BUT_CLEAN": IssueLevel.WARNING,
    "EV_NO_IDENTITY": IssueLevel.WARNING,
    "FG_GUARD_OVERRIDE_OVERLAP": IssueLevel.WARNING,
    "ID_ASSIGNMENT_OVERLAP": IssueLevel.ERROR,
    "ID_ROLE_COLLISION": IssueLevel.ERROR,
    "ID_INTERP_TOO_LONG": IssueLevel.WARNING,
    "ID_ORPHAN_INTERP": IssueLevel.ERROR,
    "CB_UNKNOWN_EVENT": IssueLevel.ERROR,
    "PM_MISSING_METRICS": IssueLevel.INFO,
}

_LEVEL_RANK = {IssueLevel.ERROR: 0, IssueLevel.WARNING: 1, IssueLevel.INFO: 2}


# ---------------------------------------------------------------------------
# Helpers de intervalos
# ---------------------------------------------------------------------------


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Une intervalos semiabiertos solapados o contiguos."""
    if not intervals:
        return []
    ordenados = sorted(intervals)
    out = [ordenados[0]]
    for start, end in ordenados[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


def _is_covered(merged: list[tuple[int, int]], start: int, end_excl: int) -> bool:
    """True si [start, end_excl) cae entero dentro de alguno de los intervalos unidos."""
    return any(s <= start and end_excl <= e for s, e in merged)


def _ranges_overlap(a_start: int, a_end_excl: int, b_start: int, b_end_excl: int) -> bool:
    return a_start < b_end_excl and b_start < a_end_excl


# ---------------------------------------------------------------------------
# Chequeos
# ---------------------------------------------------------------------------


def _check_events_bounds_and_duration(doc: AnnotationDoc, out: list[Issue]) -> None:
    total = doc.video.total_frames
    max_dur = doc.settings_snapshot.event_max_duration_frames
    for ev in doc.events:
        if ev.end_frame >= total:
            out.append(
                Issue(
                    IssueLevel.ERROR,
                    "EV_OUT_OF_BOUNDS",
                    f"el evento termina en el frame {ev.end_frame} y el video tiene {total} frames",
                    ref=ev.id,
                    frames=(ev.start_frame, ev.end_frame),
                )
            )
        if ev.duration_frames > max_dur:
            out.append(
                Issue(
                    IssueLevel.WARNING,
                    "EV_TOO_LONG",
                    f"el evento dura {ev.duration_frames} frames, sobre el maximo de {max_dur}",
                    ref=ev.id,
                    frames=(ev.start_frame, ev.end_frame),
                )
            )


def _check_event_pairs(doc: AnnotationDoc, out: list[Issue]) -> None:
    """
    Cruces entre eventos del mismo peleador.

    Los de lados distintos no generan ningun Issue: eso es una combinacion, el caso mas
    frecuente del boxeo. Solo se mira el mismo brazo, donde un cruce es fisicamente
    sospechoso salvo en el doble jab, que puede solapar unos pocos cuadros.
    """
    st = doc.settings_snapshot
    for fighter in (FighterId.A, FighterId.B):
        eventos = sorted(doc.events_of(fighter), key=lambda e: (e.start_frame, e.id))
        for a, b in combinations(eventos, 2):
            if a.side is not b.side:
                continue

            gap = abs(b.start_frame - a.start_frame)
            refire = gap < st.refire_min_gap_frames
            if refire:
                out.append(
                    Issue(
                        IssueLevel.WARNING,
                        "EV_REFIRE_TOO_FAST",
                        f"dos golpes del brazo {a.side.value} de {fighter.value} arrancan a "
                        f"{gap} frames de distancia, por debajo de {st.refire_min_gap_frames}; "
                        "un brazo no vuelve a salir tan rapido, probablemente sea el mismo golpe marcado dos veces",
                        ref=f"{a.id},{b.id}",
                        frames=(a.start_frame, b.end_frame),
                    )
                )

            if not a.overlaps(b):
                continue
            # El aviso de refire ya dice todo lo que este par tiene de raro; repetirlo
            # como solapamiento solo agrega ruido.
            if refire:
                continue

            solape = a.overlap_frames(b)
            if solape > st.same_side_overlap_max_frames:
                out.append(
                    Issue(
                        IssueLevel.WARNING,
                        "EV_OVERLAP_SAME_SIDE_LONG",
                        f"dos golpes del brazo {a.side.value} de {fighter.value} se solapan "
                        f"{solape} frames, sobre el maximo de {st.same_side_overlap_max_frames}",
                        ref=f"{a.id},{b.id}",
                        frames=(max(a.start_frame, b.start_frame), min(a.end_frame, b.end_frame)),
                    )
                )
            else:
                out.append(
                    Issue(
                        IssueLevel.WARNING,
                        "EV_OVERLAP_SAME_SIDE",
                        f"dos golpes del brazo {a.side.value} de {fighter.value} se solapan "
                        f"{solape} frames; compatible con un doble jab, revisar las fronteras",
                        ref=f"{a.id},{b.id}",
                        frames=(max(a.start_frame, b.start_frame), min(a.end_frame, b.end_frame)),
                    )
                )


def _check_event_guard(doc: AnnotationDoc, out: list[Issue]) -> None:
    for ev in doc.events:
        vigente = doc.guard_at(ev.fighter, ev.start_frame)
        if ev.guard is not vigente:
            out.append(
                Issue(
                    IssueLevel.INFO,
                    "EV_GUARD_MISMATCH",
                    f"el evento declara guardia {ev.guard.value} y la vigente para "
                    f"{ev.fighter.value} en el frame {ev.start_frame} es {vigente.value}; "
                    "el rol lead/rear se deriva de la del evento",
                    ref=ev.id,
                    frames=(ev.start_frame, ev.end_frame),
                )
            )


def _check_event_reliability(doc: AnnotationDoc, out: list[Issue]) -> None:
    for ev in doc.events:
        if ev.quality is not Quality.CLEAN:
            continue
        for seg in doc.unreliable_segments:
            if seg.fighter is ev.fighter and seg.overlaps_inclusive(ev.start_frame, ev.end_frame):
                out.append(
                    Issue(
                        IssueLevel.WARNING,
                        "EV_UNRELIABLE_BUT_CLEAN",
                        f"el evento cruza el tramo {seg.id} marcado {seg.reason.value} pero "
                        "esta anotado como clean",
                        ref=ev.id,
                        frames=(ev.start_frame, ev.end_frame),
                    )
                )
                break


def _coverage_by_fighter(doc: AnnotationDoc) -> dict[FighterId, list[tuple[int, int]]]:
    """Intervalos en que cada peleador tiene identidad resuelta, ya unidos."""
    cobertura: dict[FighterId, list[tuple[int, int]]] = {FighterId.A: [], FighterId.B: []}
    for asg in doc.identity.assignments:
        if asg.role is TrackRole.IGNORE:
            continue
        cobertura[FighterId(asg.role.value)].append((asg.start_frame, asg.end_frame_excl))
    for mt in doc.identity.manual_tracks:
        if mt.role is TrackRole.IGNORE:
            continue
        cobertura[FighterId(mt.role.value)].append((mt.start_frame, mt.end_frame_excl))
    return {f: _merge(iv) for f, iv in cobertura.items()}


def _check_event_identity(doc: AnnotationDoc, out: list[Issue]) -> None:
    cobertura = _coverage_by_fighter(doc)
    for ev in doc.events:
        # El rango del evento es inclusivo; se compara como semiabierto.
        if not _is_covered(cobertura[ev.fighter], ev.start_frame, ev.end_frame + 1):
            out.append(
                Issue(
                    IssueLevel.WARNING,
                    "EV_NO_IDENTITY",
                    f"ningun assignment cubre a {ev.fighter.value} en todo el rango del evento; "
                    "el export no va a poder sacar keypoints de esos frames",
                    ref=ev.id,
                    frames=(ev.start_frame, ev.end_frame),
                )
            )


def _check_guard_overrides(doc: AnnotationDoc, out: list[Issue]) -> None:
    for fighter, fd in doc.fighters.items():
        ovs = sorted(fd.guard_overrides, key=lambda o: o.start_frame)
        for a, b in zip(ovs, ovs[1:]):
            if _ranges_overlap(a.start_frame, a.end_frame_excl, b.start_frame, b.end_frame_excl):
                out.append(
                    Issue(
                        IssueLevel.WARNING,
                        "FG_GUARD_OVERRIDE_OVERLAP",
                        f"{fighter.value} tiene dos tramos de guardia solapados en "
                        f"[{b.start_frame}, {a.end_frame_excl}); gana el primero y la "
                        "derivacion lead/rear queda ambigua",
                        ref=fighter.value,
                        frames=(b.start_frame, a.end_frame_excl),
                    )
                )


def _check_identity(doc: AnnotationDoc, out: list[Issue]) -> None:
    asgs = doc.identity.assignments

    # Un mismo track no puede tener dos roles en el mismo frame.
    for a, b in combinations(asgs, 2):
        if a.track_id != b.track_id:
            continue
        if _ranges_overlap(a.start_frame, a.end_frame_excl, b.start_frame, b.end_frame_excl):
            out.append(
                Issue(
                    IssueLevel.ERROR,
                    "ID_ASSIGNMENT_OVERLAP",
                    f"el track {a.track_id} tiene dos assignments solapados, {a.id} y {b.id}",
                    ref=f"{a.id},{b.id}",
                    frames=(max(a.start_frame, b.start_frame), min(a.end_frame_excl, b.end_frame_excl)),
                )
            )

    # Dos tracks distintos no pueden ser el mismo peleador a la vez.
    for a, b in combinations(asgs, 2):
        if a.track_id == b.track_id or a.role is not b.role or a.role is TrackRole.IGNORE:
            continue
        if _ranges_overlap(a.start_frame, a.end_frame_excl, b.start_frame, b.end_frame_excl):
            out.append(
                Issue(
                    IssueLevel.ERROR,
                    "ID_ROLE_COLLISION",
                    f"los tracks {a.track_id} y {b.track_id} son los dos {a.role.value} al "
                    "mismo tiempo; el resolver tendria que desempatar por confianza",
                    ref=f"{a.id},{b.id}",
                    frames=(max(a.start_frame, b.start_frame), min(a.end_frame_excl, b.end_frame_excl)),
                )
            )

    conocidos = {a.track_id for a in asgs} | {m.track_id for m in doc.identity.manual_tracks}
    max_gap = doc.settings_snapshot.interp_max_gap_frames
    for interp in doc.identity.interpolations:
        if interp.gap_len > max_gap:
            out.append(
                Issue(
                    IssueLevel.WARNING,
                    "ID_INTERP_TOO_LONG",
                    f"la interpolacion cubre {interp.gap_len} frames, sobre el maximo de {max_gap}",
                    ref=interp.id,
                    frames=(interp.gap_start_frame, interp.gap_end_frame_excl),
                )
            )
        huerfanos = [t for t in (interp.from_track_id, interp.to_track_id) if t not in conocidos]
        if huerfanos:
            out.append(
                Issue(
                    IssueLevel.ERROR,
                    "ID_ORPHAN_INTERP",
                    f"la interpolacion referencia tracks sin assignment: {huerfanos}",
                    ref=interp.id,
                    frames=(interp.gap_start_frame, interp.gap_end_frame_excl),
                )
            )


def _check_combos_and_metrics(doc: AnnotationDoc, out: list[Issue]) -> None:
    ids = {e.id for e in doc.events}
    for co in doc.combo_overrides:
        faltantes = [eid for eid in co.event_ids if eid not in ids]
        if faltantes:
            out.append(
                Issue(
                    IssueLevel.ERROR,
                    "CB_UNKNOWN_EVENT",
                    f"la correccion de combinacion referencia eventos inexistentes: {faltantes}",
                    ref=co.id,
                )
            )

    for ev in doc.events:
        if ev.id not in doc.process.event_metrics:
            out.append(
                Issue(
                    IssueLevel.INFO,
                    "PM_MISSING_METRICS",
                    "el evento no tiene metricas de proceso; no va a entrar en el analisis "
                    "de tiempo de anotacion",
                    ref=ev.id,
                    frames=(ev.start_frame, ev.end_frame),
                )
            )


# ---------------------------------------------------------------------------
# Entrada publica
# ---------------------------------------------------------------------------


def validate_document(doc: AnnotationDoc) -> list[Issue]:
    """
    Devuelve todos los Issues del documento, ordenados y de forma determinista.

    El orden es por nivel, despues por frame de inicio y despues por codigo y referencia,
    para que dos corridas sobre el mismo archivo den exactamente la misma lista y se pueda
    diffear un reporte de validacion.
    """
    out: list[Issue] = []
    _check_events_bounds_and_duration(doc, out)
    _check_event_pairs(doc, out)
    _check_event_guard(doc, out)
    _check_event_reliability(doc, out)
    _check_event_identity(doc, out)
    _check_guard_overrides(doc, out)
    _check_identity(doc, out)
    _check_combos_and_metrics(doc, out)

    out.sort(
        key=lambda i: (
            _LEVEL_RANK[i.level],
            i.frames[0] if i.frames else -1,
            i.code,
            i.ref or "",
        )
    )
    return out


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level is IssueLevel.ERROR for i in issues)
