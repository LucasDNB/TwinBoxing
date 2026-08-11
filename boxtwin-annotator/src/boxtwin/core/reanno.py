"""
BoxTwin - Reanotacion ciega: muestra y archivo.

POR QUE EXISTE
  Un dataset propio no vale mas que uno publico solo por ser propio. Lo que lo hace
  defendible es poder decir cuanto se contradice a si mismo, y eso se mide reanotando a
  ciegas una muestra y comparando.

  La muestra se congela ANTES de reanotar y no se puede resortear. Resortear despues de ver
  resultados parciales convierte el numero en lo que uno quiera que sea, y con eso el
  reporte deja de poder defenderse. Es el mismo criterio que ya se aplico en la verificacion
  de BoxingVI: la muestra vivia en un commit anterior a toda anotacion.

  La ventana que se le muestra al reanotador lleva un relleno ALEATORIO a cada lado. Si el
  clip empezara justo en start_frame, las fronteras estarian dadas por los bordes del clip y
  el error absoluto medio de fronteras mediria cero por construccion. Con relleno variable,
  los bordes no dicen nada.

  El reanotador puede revelar la etiqueta original, pero ese intento queda marcado y no
  cuenta como ciego. Prohibirlo no serviria: el que quiera mirar mira igual. Registrarlo si.

QUE HACE
  Sortea la muestra de forma determinista, define la ventana de cada intento y persiste el
  archivo de reanotacion.

USO
  doc_re = sortear(doc, fraction=0.10, seed=42, annotator="lucas")
  guardar(doc_re, path)
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from boxtwin.core.annotations import dumps as dumps_annot
from boxtwin.core.schema import (
    AnnotationDoc,
    BoxTwinModel,
    Frame,
    Sha256,
    Timestamp,
)
from boxtwin.core.types import (
    Completeness,
    Landed,
    PunchType,
    Quality,
    Side,
    Target,
)

__all__ = [
    "REANNO_SCHEMA_VERSION",
    "REANNO_KIND",
    "TrialLabels",
    "ReannoTrial",
    "ReannoSample",
    "ReannoDoc",
    "sortear",
    "ventana_de_intento",
    "cargar",
    "guardar",
    "SampleFrozenError",
]

REANNO_SCHEMA_VERSION = 1
REANNO_KIND = "boxtwin.reanno"

# Relleno aleatorio a cada lado de la ventana del intento. El rango es amplio a proposito:
# con relleno fijo, restar el relleno recuperaria las fronteras originales.
PAD_MIN = 8
PAD_MAX = 24


class SampleFrozenError(RuntimeError):
    """Se intento resortear una muestra ya existente."""


class TrialLabels(BoxTwinModel):
    """Lo que el reanotador decidio, sin ver lo anterior."""

    start_frame: Frame
    end_frame: Frame
    peak_frame: Frame | None = None
    side: Side
    punch_type: PunchType
    target: Target
    completeness: Completeness
    landed: Landed = Landed.UNKNOWN
    quality: Quality = Quality.CLEAN


class ReannoTrial(BoxTwinModel):
    event_id: str
    annotator: str
    annotated_at: Timestamp
    active_ms: Annotated[int, Field(ge=0)] = 0
    replays: Annotated[int, Field(ge=0)] = 0
    # Un intento revelado no cuenta como ciego. No se prohibe, se registra.
    revealed: bool = False
    labels: TrialLabels


class ReannoSample(BoxTwinModel):
    """
    La muestra congelada.

    seed y fraction quedan guardados para que el sorteo se pueda reproducir y para que se
    vea, en el propio archivo, que no se resorteo.
    """

    seed: int
    fraction: float = Field(gt=0.0, le=1.0)
    n: int = Field(ge=0)
    drawn_at: Timestamp
    drawn_by: str
    event_ids: list[str] = Field(default_factory=list)
    # Ventana de cada intento, con su relleno propio. Va en la muestra y no se recalcula:
    # dos corridas del reanotador tienen que mostrar exactamente lo mismo.
    windows: dict[str, list[int]] = Field(default_factory=dict)


class ReannoDoc(BoxTwinModel):
    schema_version: int = Field(ge=1)
    kind: str = REANNO_KIND
    source_annot_sha256: Sha256
    video_sha256: Sha256
    sample: ReannoSample
    trials: list[ReannoTrial] = Field(default_factory=list)

    def pendientes(self) -> list[str]:
        hechos = {t.event_id for t in self.trials}
        return [e for e in self.sample.event_ids if e not in hechos]

    def ventana(self, event_id: str) -> tuple[int, int]:
        ini, fin = self.sample.windows[event_id]
        return ini, fin


def annot_sha(doc: AnnotationDoc) -> str:
    import hashlib

    return hashlib.sha256(dumps_annot(doc).encode("utf-8")).hexdigest()


def ventana_de_intento(
    doc: AnnotationDoc, event_id: str, rng: random.Random
) -> tuple[int, int]:
    """Ventana con relleno aleatorio, para que los bordes no revelen las fronteras."""
    ev = doc.event_by_id(event_id)
    if ev is None:
        raise KeyError(event_id)
    pre = rng.randint(PAD_MIN, PAD_MAX)
    post = rng.randint(PAD_MIN, PAD_MAX)
    return (
        max(0, ev.start_frame - pre),
        min(doc.video.total_frames - 1, ev.end_frame + post),
    )


def sortear(
    doc: AnnotationDoc,
    *,
    fraction: float = 0.10,
    seed: int = 42,
    annotator: str = "desconocido",
    ahora: datetime | None = None,
) -> ReannoDoc:
    """
    Sortea la muestra. Determinista por semilla.

    Se ordenan los ids antes de sortear para que el resultado no dependa del orden en que
    quedaron los eventos en memoria, que puede variar entre corridas.
    """
    if not 0 < fraction <= 1:
        raise ValueError("la fraccion tiene que estar en (0, 1]")

    ids = sorted(e.id for e in doc.events)
    if not ids:
        raise ValueError("no hay eventos para reanotar")

    n = max(1, round(len(ids) * fraction))
    rng = random.Random(seed)
    elegidos = sorted(rng.sample(ids, min(n, len(ids))))

    ventanas = {eid: list(ventana_de_intento(doc, eid, rng)) for eid in elegidos}
    ahora = ahora or datetime.now().astimezone()

    return ReannoDoc(
        schema_version=REANNO_SCHEMA_VERSION,
        source_annot_sha256=annot_sha(doc),
        video_sha256=doc.video.sha256,
        sample=ReannoSample(
            seed=seed,
            fraction=fraction,
            n=len(elegidos),
            drawn_at=ahora,
            drawn_by=annotator,
            event_ids=elegidos,
            windows=ventanas,
        ),
    )


def dumps(doc: ReannoDoc) -> str:
    return (
        json.dumps(doc.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=False)
        + "\n"
    )


def guardar(doc: ReannoDoc, path: Path) -> None:
    """Escritura atomica, igual que la del annot.json: un corte no puede truncar el archivo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    texto = dumps(doc)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(texto)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def cargar(path: Path) -> ReannoDoc:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != REANNO_KIND:
        raise ValueError(f"{path} no es un archivo de reanotacion")
    return ReannoDoc.model_validate(data)


def cargar_o_sortear(
    path: Path,
    doc: AnnotationDoc,
    *,
    fraction: float,
    seed: int,
    annotator: str,
    force: bool = False,
) -> tuple[ReannoDoc, bool]:
    """
    Devuelve la muestra existente o sortea una nueva. (documento, es_nueva)

    Se niega a resortear salvo force explicito: resortear despues de ver resultados
    parciales convierte el numero en lo que uno quiera que sea.
    """
    path = Path(path)
    if path.is_file() and not force:
        existente = cargar(path)
        if existente.source_annot_sha256 != annot_sha(doc):
            raise SampleFrozenError(
                f"{path.name} se sorteo sobre otra version de la anotacion.\n"
                "  Comparar contra el estado actual invalidaria el pre-registro.\n"
                "  Usar --force para sortear de nuevo, sabiendo que se descarta lo anterior."
            )
        return existente, False
    return sortear(doc, fraction=fraction, seed=seed, annotator=annotator), True
