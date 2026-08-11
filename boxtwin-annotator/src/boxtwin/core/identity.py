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

  Este modulo es solo de LECTURA. Las operaciones que modifican asignaciones (swap,
  re-seed, interpolacion) son del bloque 5. Aca vive lo que el overlay y los exports
  necesitan: dado un cuadro, decir de quien es cada deteccion.

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
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from boxtwin.core.posecache import PoseCache, PoseDetections
from boxtwin.core.schema import AnnotationDoc
from boxtwin.core.types import FighterId, TrackRole

__all__ = ["ResolvedPose", "IdentityResolver"]


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
        return self._resolve_collisions(poses)

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
