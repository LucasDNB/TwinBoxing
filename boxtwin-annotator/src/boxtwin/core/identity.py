"""
BoxTwin - Resolucion de identidad: de track_id a peleador.

POR QUE EXISTE
  El problema real de este dominio no es la caja mal ubicada, es el intercambio de
  identidad. Dos cuerpos parecidos, contacto permanente y oclusion mutua hacen que el
  tracker pierda y reasigne ids todo el tiempo, y ademas cada reanudacion del preproceso
  introduce un corte garantizado. Un track_id no significa nada estable por si solo.
  Por eso el rol no se guarda en el cache de pose, que es inmutable, sino como intervalos
  en el archivo de anotacion: el track 3 es fighter_A entre los cuadros 4120 y 9008, y
  despues quiza sea otra cosa.

  Este modulo es solo de LECTURA: las operaciones que modifican asignaciones viven en
  identity_ops.py. Aca esta lo que el overlay y los exports necesitan, que es decir de
  quien es cada deteccion de un cuadro.

  Los cuadros de un hueco interpolado se sintetizan al leer y nunca se escriben en el npz.
  El cache guarda lo que el modelo observo; una pose inventada no es observacion y mezclar
  las dos cosas en el mismo archivo haria imposible saber despues cual era cual. Salen
  marcadas con interpolated=True para que el export pueda decidir.

QUE HACE
  Indexa los intervalos por track y resuelve el rol en tiempo logaritmico. Marca ademas si
  la deteccion cae en un tramo declarado no confiable, sin borrarla: excluirla del export
  es una decision del export, no de la lectura.

  Regla de desempate, que hay que fijar porque el caso pasa seguido en clinch: si dos
  tracks resuelven al mismo peleador en el mismo cuadro, gana el de mayor confianza de
  deteccion y el otro queda marcado. Sin una regla explicita el resultado dependeria del
  orden de las detecciones y el export dejaria de ser determinista.

USO
  from boxtwin.core.identity import IdentityResolver
  res = IdentityResolver(doc, cache)
  for pose in res.resolve_frame(1234):
      pose.role, pose.reliable
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from boxtwin.core.posecache import PoseCache, PoseDetections
from boxtwin.core.schema import AnnotationDoc, Interpolation
from boxtwin.core.types import FighterId, TrackRole

__all__ = ["ResolvedPose", "IdentityResolver", "rol_de_track", "rol_en_track"]


def rol_de_track(doc: AnnotationDoc, track_id: int, frame: int) -> TrackRole | None:
    """El rol de un track en un cuadro, o None si ningun assignment lo cubre."""
    for a in doc.identity.assignments:
        if a.track_id == track_id and a.covers(frame):
            return a.role
    return None


def rol_en_track(doc: AnnotationDoc) -> Callable[[int, int], TrackRole | None]:
    """
    Consulta indexada de "que rol tiene este track en este cuadro".

    El cuadro no es un detalle. Un mismo track_id puede ser fighter_A un rato y fighter_B
    despues, y no es un error: es exactamente lo que deja un swap, que existe porque el
    tracker le puso el mismo id a dos personas distintas. La primera version de esto colapsaba
    los roles de cada track en un conjunto sin mirar cuando, y sobre Sparring.mp4 eso hacia
    que los tracks 3 y 12 figuraran los dos como {A, B} y no filtrara nada.
    """
    por_track: dict[int, list[tuple[int, int, TrackRole]]] = {}
    for a in doc.identity.assignments:
        por_track.setdefault(a.track_id, []).append((a.start_frame, a.end_frame_excl, a.role))

    def consultar(track_id: int, frame: int) -> TrackRole | None:
        for ini, fin, rol in por_track.get(track_id, ()):
            if ini <= frame < fin:
                return rol
        return None

    return consultar


@dataclass(frozen=True)
class ResolvedPose:
    """Una deteccion con su identidad resuelta. Los arrays son vistas del cache."""

    frame: int
    track_id: int
    role: TrackRole | None  # None = sin assignment que lo cubra
    bbox: np.ndarray
    det_conf: float
    keypoints: np.ndarray
    kp_score: np.ndarray
    reliable: bool = True
    interpolated: bool = False
    manual: bool = False
    shadowed: bool = False  # perdio el desempate contra otro track del mismo rol

    @property
    def fighter(self) -> FighterId | None:
        """El peleador, o None si el rol es ignore o no hay assignment."""
        if self.role in (TrackRole.A, TrackRole.B):
            return FighterId(self.role.value)
        return None


class IdentityResolver:
    """Indice de intervalos de identidad sobre un documento de anotacion."""

    def __init__(self, doc: AnnotationDoc, cache: PoseCache) -> None:
        self.doc = doc
        self.cache = cache
        self._rebuild()

    def refresh(self) -> None:
        """Se llama cuando cambian las asignaciones. El indice es barato de rehacer."""
        self._rebuild()

    def _rebuild(self) -> None:
        # Por cada track, los intervalos ordenados por inicio. Son decenas, no miles.
        por_track: dict[int, list[tuple[int, int, TrackRole]]] = {}
        for a in self.doc.identity.assignments:
            por_track.setdefault(a.track_id, []).append(
                (a.start_frame, a.end_frame_excl, a.role)
            )
        for mt in self.doc.identity.manual_tracks:
            por_track.setdefault(mt.track_id, []).append(
                (mt.start_frame, mt.end_frame_excl, mt.role)
            )

        self._tracks: dict[int, tuple[list[int], list[tuple[int, int, TrackRole]]]] = {}
        for tid, tramos in por_track.items():
            tramos.sort(key=lambda t: t[0])
            self._tracks[tid] = ([t[0] for t in tramos], tramos)

        # Tramos no confiables por peleador, ordenados.
        self._unreliable: dict[FighterId, list[tuple[int, int]]] = {
            FighterId.A: [],
            FighterId.B: [],
        }
        for seg in self.doc.unreliable_segments:
            self._unreliable[seg.fighter].append((seg.start_frame, seg.end_frame_excl))
        for lista in self._unreliable.values():
            lista.sort()

        self._manual_ids = {mt.track_id for mt in self.doc.identity.manual_tracks}

        # Huecos interpolados, con los extremos ya resueltos para no recalcularlos en cada
        # cuadro: en reproduccion esto se consulta treinta veces por segundo.
        self._interps: list[tuple[Interpolation, dict]] = []
        for interp in self.doc.identity.interpolations:
            extremos = self._extremos_de(interp)
            if extremos is not None:
                self._interps.append((interp, extremos))

    def _extremos_de(self, interp: Interpolation) -> dict | None:
        """Detecciones de los dos bordes del hueco, o None si el cache no las tiene."""
        ultimo = interp.gap_start_frame - 1
        primero = interp.gap_end_frame_excl
        if not (0 <= ultimo < len(self.cache) and 0 <= primero < len(self.cache)):
            return None
        da, db = self.cache.detections(ultimo), self.cache.detections(primero)
        ia = da.index_of_track(interp.from_track_id)
        ib = db.index_of_track(interp.to_track_id)
        if ia is None or ib is None:
            return None
        return {
            "ultimo": ultimo,
            "primero": primero,
            "bbox_a": da.bbox[ia], "bbox_b": db.bbox[ib],
            "kp_a": da.keypoints[ia], "kp_b": db.keypoints[ib],
            "score": np.minimum(da.kp_score[ia], db.kp_score[ib]),
        }

    # -- consultas puntuales ----------------------------------------------

    def role_of(self, frame: int, track_id: int) -> TrackRole | None:
        """Rol de un track en un cuadro, o None si ningun assignment lo cubre."""
        entrada = self._tracks.get(track_id)
        if entrada is None:
            return None
        inicios, tramos = entrada
        # El ultimo intervalo que arranca en o antes del cuadro es el unico candidato:
        # los intervalos de un mismo track nunca se solapan.
        i = bisect_right(inicios, frame) - 1
        if i < 0:
            return None
        start, end_excl, role = tramos[i]
        return role if start <= frame < end_excl else None

    def is_reliable(self, frame: int, fighter: FighterId) -> bool:
        for start, end_excl in self._unreliable[fighter]:
            if start <= frame < end_excl:
                return False
            if start > frame:
                break
        return True

    # -- resolucion de un cuadro ------------------------------------------

    def resolve_frame(self, frame: int) -> list[ResolvedPose]:
        """
        Todas las detecciones del cuadro con su identidad.

        Devuelve tambien las que no tienen rol asignado: el overlay las dibuja en el color
        de sin asignar y son justamente las que el anotador tiene que resolver.
        """
        dets = self.cache.detections(frame)
        poses = [self._build(frame, dets, i) for i in range(len(dets))]
        poses.extend(self._interpoladas(frame, poses))
        return self._resolve_collisions(poses)

    def _interpoladas(self, frame: int, ya: list[ResolvedPose]) -> list[ResolvedPose]:
        """
        Poses sinteticas de los huecos declarados.

        Si el rol ya esta cubierto por una deteccion real en ese cuadro no se agrega nada:
        lo observado siempre gana sobre lo inventado.
        """
        cubiertos = {p.role for p in ya if p.role is not None}
        salida: list[ResolvedPose] = []
        for interp, ext in self._interps:
            if not (interp.gap_start_frame <= frame < interp.gap_end_frame_excl):
                continue
            if interp.role in cubiertos:
                continue
            t = (frame - ext["ultimo"]) / (ext["primero"] - ext["ultimo"])
            fighter = (
                FighterId(interp.role.value) if interp.role in (TrackRole.A, TrackRole.B) else None
            )
            salida.append(
                ResolvedPose(
                    frame=frame,
                    track_id=interp.to_track_id,
                    role=interp.role,
                    bbox=((1 - t) * ext["bbox_a"] + t * ext["bbox_b"]).astype(np.float32),
                    det_conf=0.0,
                    keypoints=((1 - t) * ext["kp_a"] + t * ext["kp_b"]).astype(np.float32),
                    kp_score=ext["score"],
                    reliable=self.is_reliable(frame, fighter) if fighter else True,
                    interpolated=True,
                )
            )
        return salida

    def by_fighter(self, frame: int) -> dict[FighterId, ResolvedPose | None]:
        """La deteccion de cada peleador en el cuadro, ya desempatada."""
        salida: dict[FighterId, ResolvedPose | None] = {FighterId.A: None, FighterId.B: None}
        for pose in self.resolve_frame(frame):
            f = pose.fighter
            if f is not None and not pose.shadowed:
                salida[f] = pose
        return salida

    def _build(self, frame: int, dets: PoseDetections, i: int) -> ResolvedPose:
        tid = int(dets.track_id[i])
        role = self.role_of(frame, tid)
        fighter = FighterId(role.value) if role in (TrackRole.A, TrackRole.B) else None
        return ResolvedPose(
            frame=frame,
            track_id=tid,
            role=role,
            bbox=dets.bbox[i],
            det_conf=float(dets.det_conf[i]),
            keypoints=dets.keypoints[i],
            kp_score=dets.kp_score[i],
            reliable=self.is_reliable(frame, fighter) if fighter else True,
            manual=tid in self._manual_ids,
        )

    def _resolve_collisions(self, poses: list[ResolvedPose]) -> list[ResolvedPose]:
        """
        Si dos tracks son el mismo peleador en el mismo cuadro, gana el de mayor confianza.

        El caso pasa en cada clinch. Sin regla explicita el ganador dependeria del orden de
        las detecciones, que no esta garantizado, y el export dejaria de ser reproducible.
        """
        por_rol: dict[FighterId, list[int]] = {}
        for idx, p in enumerate(poses):
            f = p.fighter
            if f is not None:
                por_rol.setdefault(f, []).append(idx)

        for indices in por_rol.values():
            if len(indices) < 2:
                continue
            # Desempate por confianza y, si empatan, por track_id: determinista siempre.
            ganador = max(indices, key=lambda i: (poses[i].det_conf, -poses[i].track_id))
            for i in indices:
                if i != ganador:
                    poses[i] = _replace(poses[i], shadowed=True)
        return poses

    # -- utilidades para la GUI -------------------------------------------

    def assigned_tracks(self, frame: int) -> dict[int, TrackRole]:
        """Roles vigentes en un cuadro, para el panel de identidad."""
        salida: dict[int, TrackRole] = {}
        for tid in self._tracks:
            role = self.role_of(frame, tid)
            if role is not None:
                salida[tid] = role
        return salida

    def unassigned_tracks(self, frames: Iterable[int]) -> set[int]:
        """Tracks presentes en esos cuadros que todavia no tienen rol."""
        pendientes: set[int] = set()
        for f in frames:
            dets = self.cache.detections(f)
            for tid in dets.track_id.tolist():
                if self.role_of(f, int(tid)) is None:
                    pendientes.add(int(tid))
        return pendientes


def _replace(p: ResolvedPose, **kw) -> ResolvedPose:
    from dataclasses import replace

    return replace(p, **kw)
