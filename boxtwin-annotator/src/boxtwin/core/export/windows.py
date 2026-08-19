"""
BoxTwin - Ventanas de cuadros: eventos y fondo.

POR QUE EXISTE
  Un evento se anota como un rango inclusivo de cuadros, pero lo que entra al modelo es una
  ventana de keypoints. Traducir una cosa en la otra tiene decisiones que, si quedan
  dispersas en cada exportador, terminan siendo tres decisiones distintas.

  El muestreo de fondo es la parte delicada. Un ejemplo de fondo que en realidad contiene la
  cola de un golpe le ensena al modelo que ese movimiento es fondo, y eso rompe justamente
  la frontera que el clasificador tiene que aprender. Por eso se exige un margen a los
  bordes de todo evento del peleador, se excluyen los tramos no confiables y se pide que el
  peleador tenga identidad resuelta en toda la ventana.

  El muestreo es determinista por semilla, y la semilla va en la metadata del export. Sin
  eso, dos corridas del mismo comando producen datasets distintos y ninguna comparacion
  entre modelos significa nada.

QUE HACE
  Convierte eventos en ventanas de cuadros y muestrea ventanas de fondo que no tocan ningun
  evento.

USO
  ventanas = ventanas_de_fondo(doc, resolver, cantidad=200, largo=20, seed=42)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from boxtwin.core.identity import IdentityResolver
from boxtwin.core.schema import AnnotationDoc, Event
from boxtwin.core.types import FighterId

__all__ = ["Ventana", "ventana_de_evento", "ventanas_de_fondo", "frames_ocupados"]


@dataclass(frozen=True)
class Ventana:
    """Rango inclusivo de cuadros con su peleador."""

    fighter: FighterId
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def ventana_de_evento(ev: Event, doc: AnnotationDoc, *, pad: int = 0) -> Ventana:
    """
    Ventana de un evento, con relleno opcional a los lados.

    El relleno se recorta contra los limites del video y nunca contra otros eventos: en una
    combinacion los golpes se solapan de todos modos, y recortarlos entre si haria que la
    ventana de un mismo golpe cambie segun que otro golpe haya cerca.
    """
    return Ventana(
        fighter=ev.fighter,
        start_frame=max(0, ev.start_frame - pad),
        end_frame=min(doc.video.total_frames - 1, ev.end_frame + pad),
    )


def frames_ocupados(doc: AnnotationDoc, fighter: FighterId, *, margen: int) -> set[int]:
    """
    Cuadros que no pueden ser fondo para ese peleador.

    Incluye todos los eventos suyos mas un margen a cada lado, y los tramos marcados no
    confiables. El margen es lo que evita que la cola de un golpe entre como fondo.
    """
    ocupados: set[int] = set()
    for ev in doc.events:
        if ev.fighter is not fighter:
            continue
        ocupados.update(range(max(0, ev.start_frame - margen), ev.end_frame + margen + 1))
    for seg in doc.unreliable_segments:
        if seg.fighter is fighter:
            ocupados.update(range(seg.start_frame, seg.end_frame_excl))
    return ocupados


def ventanas_de_fondo(
    doc: AnnotationDoc,
    resolver: IdentityResolver,
    *,
    cantidad: int,
    largo: int,
    seed: int,
    margen: int | None = None,
) -> list[Ventana]:
    """
    Muestrea ventanas sin ningun golpe del peleador.

    Se exige ademas que tenga identidad resuelta en TODA la ventana: un fondo sin keypoints
    no ensena nada y ademas entraria al dataset como ceros, que el modelo puede aprender a
    reconocer como una clase espuria.

    Determinista por semilla. Las candidatas se recorren en orden y se elige de a una sin
    reposicion, evitando solapamientos entre las elegidas para que el conjunto de fondo no
    sea el mismo tramo repetido.
    """
    if largo < 1:
        raise ValueError("el largo de la ventana tiene que ser al menos 1")
    margen = doc.settings_snapshot.background_margin_frames if margen is None else margen
    total = doc.video.total_frames

    candidatas: list[Ventana] = []
    for fighter in (FighterId.A, FighterId.B):
        ocupados = frames_ocupados(doc, fighter, margen=margen)
        inicio = 0
        while inicio + largo <= total:
            rango = range(inicio, inicio + largo)
            if any(f in ocupados for f in rango):
                inicio += 1
                continue
            if not all(resolver.by_fighter(f)[fighter] is not None for f in rango):
                inicio += 1
                continue
            candidatas.append(Ventana(fighter, inicio, inicio + largo - 1))
            inicio += largo  # sin solapamiento entre candidatas del mismo peleador

    if not candidatas:
        return []

    rng = random.Random(seed)
    elegidas = candidatas if cantidad >= len(candidatas) else rng.sample(candidatas, cantidad)
    return sorted(elegidas, key=lambda v: (v.fighter.value, v.start_frame))
